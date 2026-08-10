# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model-independent MiniMax H3 multimodal trajectory helpers."""

import math
from typing import Dict, List, Mapping, Optional, Tuple, Union

import torch
from diffusers.utils.torch_utils import randn_tensor

from ...samples import (
    ComponentTimes,
    ComponentTrajectory,
    IndexedTrajectoryTensor,
    LatentState,
    MultiModalStepOutput,
    NoisedState,
    StructuredTrajectory,
)
from ...scheduler import SDESchedulerOutput

MINIMAX_H3_COMPONENT_ORDER: Tuple[str, ...] = ("video", "audio")
MINIMAX_H3_COMPONENT_WIDTHS = {"video": 96, "audio": 32}
_SIGNED_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)


def shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply the MiniMax H3 exponential sigma shift.

    Args:
        sigma: Unshifted sigma values in ``[0, 1]``.
        shift: Positive exponential shift.

    Returns:
        Shifted sigma values.
    """
    _validate_sigma_transform_inputs(sigma, shift, "shift_sigma")
    return shift * sigma / (1 + (shift - 1) * sigma)


def inverse_shift_sigma(shifted_sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """Recover the unshifted base quantile from a shifted sigma.

    Args:
        shifted_sigma: Shifted sigma values in ``[0, 1]``.
        shift: Positive exponential shift.

    Returns:
        Unshifted base quantiles.
    """
    _validate_sigma_transform_inputs(shifted_sigma, shift, "inverse_shift_sigma")
    return shifted_sigma / (shift - (shift - 1) * shifted_sigma)


def framework_sigma_to_model_time(sigma: torch.Tensor) -> torch.Tensor:
    """Convert Flow-Factory noise sigma to H3 clean time.

    Args:
        sigma: Framework sigma values in ``[0, 1]``.

    Returns:
        H3 clean-time values.
    """
    _validate_unit_interval_tensor(sigma, "framework sigma")
    return 1 - sigma


def model_time_to_framework_sigma(model_time: torch.Tensor) -> torch.Tensor:
    """Convert H3 clean time to Flow-Factory noise sigma.

    Args:
        model_time: H3 clean-time values in ``[0, 1]``.

    Returns:
        Framework sigma values.
    """
    _validate_unit_interval_tensor(model_time, "H3 model time")
    return 1 - model_time


def build_training_component_times(
    primary_video_timesteps: torch.Tensor,
    *,
    video_shift: float,
    audio_shift: float,
) -> ComponentTimes:
    """Map primary video coordinates onto aligned video and audio schedules.

    Args:
        primary_video_timesteps: Batched video scheduler coordinates.
        video_shift: Video sigma shift.
        audio_shift: Audio sigma shift.

    Returns:
        Aligned component coordinates with zero decoupled next times.
    """
    if not isinstance(primary_video_timesteps, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor primary_video_timesteps, received "
            f"{type(primary_video_timesteps).__name__}: {primary_video_timesteps!r}"
        )
    if primary_video_timesteps.ndim != 1:
        raise ValueError(
            "expected primary_video_timesteps shaped (B,), received "
            f"{tuple(primary_video_timesteps.shape)}"
        )
    video_sigma = primary_video_timesteps / 1000
    _validate_unit_interval_tensor(video_sigma, "primary video sigma")
    base_quantile = inverse_shift_sigma(video_sigma, video_shift)
    audio_sigma = shift_sigma(base_quantile, audio_shift)
    zero_timestep = torch.zeros_like(primary_video_timesteps)
    zero_sigma = torch.zeros_like(video_sigma)
    return ComponentTimes(
        timestep={
            "video": primary_video_timesteps,
            "audio": audio_sigma * 1000,
        },
        next_timestep={"video": zero_timestep, "audio": zero_timestep.clone()},
        sigma={"video": video_sigma, "audio": audio_sigma},
        next_sigma={"video": zero_sigma, "audio": zero_sigma.clone()},
    )


def validate_target_state(state: LatentState) -> None:
    """Validate target-only packed video and audio state invariants.

    Args:
        state: Heterogeneous video/audio target state.

    Returns:
        None.
    """
    if not isinstance(state, LatentState):
        raise TypeError(
            f"expected LatentState target state, received {type(state).__name__}: {state!r}"
        )
    if state.component_names != MINIMAX_H3_COMPONENT_ORDER:
        raise ValueError(
            "expected MiniMax H3 target state component order "
            f"{MINIMAX_H3_COMPONENT_ORDER}, received {state.component_names}"
        )
    video = state.components["video"]
    audio = state.components["audio"]
    for component, values in (("video", video), ("audio", audio)):
        if values.ndim != 3:
            raise ValueError(
                f"expected MiniMax H3 {component} target shaped (B, N, C), "
                f"received {tuple(values.shape)}"
            )
        expected_width = MINIMAX_H3_COMPONENT_WIDTHS[component]
        if values.shape[-1] != expected_width:
            raise ValueError(
                f"expected MiniMax H3 {component} row width {expected_width}, "
                f"received {values.shape[-1]} in shape {tuple(values.shape)}"
            )
        if not values.is_floating_point():
            raise TypeError(
                f"expected floating MiniMax H3 {component} state, received dtype {values.dtype}"
            )
        if values.shape[0] == 0:
            raise ValueError(
                f"expected MiniMax H3 {component} state with a non-empty batch, "
                f"received {tuple(values.shape)}"
            )
        if values.shape[1] == 0:
            raise ValueError(
                f"expected MiniMax H3 {component} state with non-empty generated rows, "
                f"received {tuple(values.shape)}"
            )
    if video.shape[0] != audio.shape[0]:
        raise ValueError(
            f"expected video/audio batch size {video.shape[0]}, received audio {audio.shape[0]}"
        )
    if video.dtype != audio.dtype:
        raise ValueError(
            f"expected dtype video={video.dtype} to match audio, received audio={audio.dtype}"
        )
    if video.device != audio.device:
        raise ValueError(
            f"expected device video={video.device} to match audio, received audio={audio.device}"
        )
    if state.active_masks is not None:
        raise ValueError(
            "expected target-only MiniMax H3 LatentState.active_masks=None, "
            f"received components {tuple(state.active_masks)}"
        )


def draw_forward_process_noise(
    clean_state: LatentState,
    times: ComponentTimes,
    *,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]],
) -> NoisedState:
    """Draw video then audio noise and return H3 data-ward velocity targets.

    Args:
        clean_state: Clean target-only video/audio state.
        times: Validated per-component current and next coordinates.
        generator: Generator or per-sample generators for ordered noise draws.

    Returns:
        Noised state, sampled noise, and H3 data-ward targets.
    """
    validate_target_state(clean_state)
    _validate_component_times(times, clean_state)
    noise: Dict[str, torch.Tensor] = {}
    noised: Dict[str, torch.Tensor] = {}
    target_velocity: Dict[str, torch.Tensor] = {}
    for component in MINIMAX_H3_COMPONENT_ORDER:
        clean = clean_state.components[component]
        component_noise = randn_tensor(
            clean.shape,
            generator=generator,
            device=clean.device,
            dtype=clean.dtype,
        )
        sigma = times.sigma[component].to(device=clean.device, dtype=clean.dtype)
        while sigma.ndim < clean.ndim:
            sigma = sigma.unsqueeze(-1)
        noise[component] = component_noise
        noised[component] = (1 - sigma) * clean + sigma * component_noise
        target_velocity[component] = clean - component_noise
    return NoisedState(
        state=LatentState(noised),
        target_velocity=LatentState(target_velocity),
        noise=LatentState(noise),
    )


def pack_video_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack video latents with the H3 patch size ``(1, 2, 2)``.

    Args:
        latents: Video latents shaped ``(B, 24, F, H, W)``.

    Returns:
        Packed rows shaped ``(B, F*(H/2)*(W/2), 96)``.
    """
    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"expected torch.Tensor video latents, received {type(latents).__name__}")
    if latents.ndim != 5 or latents.shape[1] != 24:
        raise ValueError(
            "expected video latents shaped (B, 24, F, H, W), " f"received {tuple(latents.shape)}"
        )
    batch_size, channels, frames, height, width = latents.shape
    if height % 2 or width % 2:
        raise ValueError(
            f"expected video height and width divisible by 2, received H={height}, W={width}"
        )
    return (
        latents.reshape(batch_size, channels, frames, height // 2, 2, width // 2, 2)
        .permute(0, 2, 3, 5, 1, 4, 6)
        .reshape(batch_size, frames * (height // 2) * (width // 2), 96)
    )


def unpack_video_latents(
    packed: torch.Tensor, *, frames: int, height: int, width: int
) -> torch.Tensor:
    """Unpack H3 video rows into channel-first latent geometry.

    Args:
        packed: Packed video rows shaped ``(B, N, 96)``.
        frames: Output frame count.
        height: Output latent height.
        width: Output latent width.

    Returns:
        Video latents shaped ``(B, 24, F, H, W)``.
    """
    if not isinstance(packed, torch.Tensor):
        raise TypeError(f"expected torch.Tensor packed video, received {type(packed).__name__}")
    if packed.ndim != 3 or packed.shape[-1] != 96:
        raise ValueError(
            f"expected packed video shaped (B, N, width 96), received {tuple(packed.shape)}"
        )
    for field, value in (("frames", frames), ("height", height), ("width", width)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"expected positive int {field}, received {value!r}")
    if height % 2 or width % 2:
        raise ValueError(
            f"expected video height and width divisible by 2, received H={height}, W={width}"
        )
    expected_rows = frames * (height // 2) * (width // 2)
    if packed.shape[1] != expected_rows:
        raise ValueError(
            f"expected {expected_rows} packed video rows for F={frames}, H={height}, W={width}, "
            f"received {packed.shape[1]}"
        )
    return (
        packed.reshape(packed.shape[0], frames, height // 2, width // 2, 24, 2, 2)
        .permute(0, 4, 1, 2, 5, 3, 6)
        .reshape(packed.shape[0], 24, frames, height, width)
    )


def pack_audio_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack stereo audio latents in channel-major frame order.

    Args:
        latents: Audio latents shaped ``(B, 2, 32, F)``.

    Returns:
        Packed audio rows shaped ``(B, 2*F, 32)``.
    """
    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"expected torch.Tensor audio latents, received {type(latents).__name__}")
    if latents.ndim != 4 or latents.shape[1] != 2 or latents.shape[2] != 32:
        raise ValueError(
            f"expected audio latents shaped (B, 2, 32, F), received {tuple(latents.shape)}"
        )
    return latents.permute(0, 1, 3, 2).reshape(latents.shape[0], 2 * latents.shape[3], 32)


def unpack_audio_latents(packed: torch.Tensor) -> torch.Tensor:
    """Unpack channel-major H3 audio rows into stereo latent geometry.

    Args:
        packed: Packed audio rows shaped ``(B, 2*F, 32)``.

    Returns:
        Audio latents shaped ``(B, 2, 32, F)``.
    """
    if not isinstance(packed, torch.Tensor):
        raise TypeError(f"expected torch.Tensor packed audio, received {type(packed).__name__}")
    if packed.ndim != 3 or packed.shape[-1] != 32 or packed.shape[1] % 2:
        raise ValueError(
            f"expected packed audio shaped (B, 2*F, 32), received {tuple(packed.shape)}"
        )
    frames = packed.shape[1] // 2
    return packed.reshape(packed.shape[0], 2, frames, 32).permute(0, 1, 3, 2)


def combine_component_log_probs(
    video_log_prob: torch.Tensor,
    audio_log_prob: torch.Tensor,
    *,
    video_dof: int,
    audio_dof: int,
) -> torch.Tensor:
    """Combine component means by generated scalar degrees of freedom.

    Args:
        video_log_prob: Per-sample video mean log probabilities.
        audio_log_prob: Per-sample audio mean log probabilities.
        video_dof: Generated video scalar count.
        audio_dof: Generated audio scalar count.

    Returns:
        Per-sample joint weighted mean log probability.
    """
    for field, value in (("video_dof", video_dof), ("audio_dof", audio_dof)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"expected positive int {field}, received {value!r}")
    if not isinstance(video_log_prob, torch.Tensor) or not isinstance(audio_log_prob, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor video_log_prob and audio_log_prob, received "
            f"{type(video_log_prob).__name__} and {type(audio_log_prob).__name__}"
        )
    if video_log_prob.shape != audio_log_prob.shape or video_log_prob.ndim != 1:
        raise ValueError(
            "expected matching per-sample component log_probs shaped (B,), received "
            f"video {tuple(video_log_prob.shape)} and audio {tuple(audio_log_prob.shape)}"
        )
    if (
        video_log_prob.dtype != audio_log_prob.dtype
        or video_log_prob.device != audio_log_prob.device
    ):
        raise ValueError(
            "expected video/audio log_probs with matching dtype/device, received "
            f"{video_log_prob.dtype}/{video_log_prob.device} and "
            f"{audio_log_prob.dtype}/{audio_log_prob.device}"
        )
    return (video_log_prob * video_dof + audio_log_prob * audio_dof) / (video_dof + audio_dof)


def build_component_step_output(
    video_output: SDESchedulerOutput,
    audio_output: SDESchedulerOutput,
) -> MultiModalStepOutput:
    """Build one heterogeneous step output from real component scheduler outputs.

    Args:
        video_output: Video scheduler transition output.
        audio_output: Audio scheduler transition output.

    Returns:
        Component-preserving multimodal transition output.
    """
    for component, output in (("video", video_output), ("audio", audio_output)):
        if not isinstance(output, SDESchedulerOutput):
            raise TypeError(
                f"expected SDESchedulerOutput for {component}, received {type(output).__name__}"
            )

    def paired(field: str) -> Optional[Dict[str, torch.Tensor]]:
        values = {
            "video": getattr(video_output, field),
            "audio": getattr(audio_output, field),
        }
        if (values["video"] is None) != (values["audio"] is None):
            raise ValueError(
                f"expected video/audio scheduler outputs to agree on {field!r} presence, "
                f"received video={values['video'] is not None}, audio={values['audio'] is not None}"
            )
        return None if values["video"] is None else values

    next_state_values = paired("next_latents")
    next_mean_values = paired("next_latents_mean")
    velocity_values = paired("velocity")
    component_log_probs = paired("log_prob")
    log_prob = None
    if component_log_probs is not None:
        reference_values = next_state_values or next_mean_values or velocity_values
        if reference_values is None:
            raise ValueError(
                "expected next_latents, next_latents_mean, or velocity to determine "
                "component scalar degrees of freedom when log_prob is present"
            )
        video = reference_values["video"]
        audio = reference_values["audio"]
        log_prob = combine_component_log_probs(
            component_log_probs["video"],
            component_log_probs["audio"],
            video_dof=video[0].numel(),
            audio_dof=audio[0].numel(),
        )
    return MultiModalStepOutput(
        next_state=(None if next_state_values is None else LatentState(next_state_values)),
        next_state_mean=(None if next_mean_values is None else LatentState(next_mean_values)),
        std_dev_t=paired("std_dev_t"),
        dt=paired("dt"),
        log_prob=log_prob,
        component_log_probs=component_log_probs,
        velocity=None if velocity_values is None else LatentState(velocity_values),
    )


def build_structured_trajectories(
    *,
    states: Mapping[str, torch.Tensor],
    state_index_map: Union[torch.Tensor, Mapping[str, torch.Tensor]],
    schedule: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
    log_probs: Optional[torch.Tensor] = None,
    component_log_probs: Optional[Mapping[str, torch.Tensor]] = None,
    log_prob_index_map: Optional[torch.Tensor] = None,
    callbacks: Optional[Mapping[str, Mapping[str, torch.Tensor]]] = None,
    callback_index_map: Optional[Union[torch.Tensor, Mapping[str, torch.Tensor]]] = None,
) -> List[StructuredTrajectory]:
    """Build per-sample trajectories from independent compact component tensors.

    Args:
        states: Compact batched states by component.
        state_index_map: Shared or independent state maps.
        schedule: Independent full timestep/sigma schedules.
        log_probs: Optional batched joint log probabilities.
        component_log_probs: Optional batched component log probabilities.
        log_prob_index_map: Shared transition map for all log probabilities.
        callbacks: Optional componentized batched callback tensors.
        callback_index_map: Shared or independent callback maps.

    Returns:
        One structured trajectory per batch sample.
    """
    _require_component_order(states, "states")
    _require_component_order(schedule, "schedule")
    state_maps = _normalize_component_maps(state_index_map, "state_index_map")
    batch_size: Optional[int] = None
    schedule_length: Optional[int] = None
    for component in MINIMAX_H3_COMPONENT_ORDER:
        values = states[component]
        expected_width = MINIMAX_H3_COMPONENT_WIDTHS[component]
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim != 4
            or values.shape[-1] != expected_width
        ):
            received = (
                tuple(values.shape) if isinstance(values, torch.Tensor) else type(values).__name__
            )
            raise ValueError(
                f"expected states[{component!r}] shaped (B, stored, N, {expected_width}), "
                f"received {received}"
            )
        if values.shape[0] == 0:
            raise ValueError(
                f"expected states[{component!r}] with a non-empty batch, "
                f"received {tuple(values.shape)}"
            )
        if values.shape[2] == 0:
            raise ValueError(
                f"expected component {component!r} with non-empty generated rows, "
                f"received {tuple(values.shape)}"
            )
        if values.shape[1] == 0:
            raise ValueError(
                f"expected states[{component!r}] stored dimension greater than zero, "
                f"received {tuple(values.shape)}"
            )
        batch_size = values.shape[0] if batch_size is None else batch_size
        if values.shape[0] != batch_size:
            raise ValueError(
                f"expected states[{component!r}] batch size {batch_size}, received {values.shape[0]}"
            )
        timesteps, sigmas = _validate_schedule_entry(component, schedule[component])
        schedule_length = len(timesteps) if schedule_length is None else schedule_length
        if len(timesteps) != schedule_length:
            raise ValueError(
                f"expected component {component!r} schedule length {schedule_length}, "
                f"received {len(timesteps)}"
            )
        _validate_index_map(
            state_maps[component],
            expected_length=schedule_length,
            num_stored=values.shape[1],
            field=f"state_index_map[{component!r}]",
        )
    num_transitions = schedule_length - 1
    _validate_optional_log_probs(
        log_probs,
        component_log_probs,
        log_prob_index_map,
        batch_size=batch_size,
        num_transitions=num_transitions,
    )
    if callbacks is None and callback_index_map is not None:
        raise ValueError("expected callback_index_map=None when callbacks=None, received a map")
    validated_callback_maps = (
        None
        if callbacks is None
        else _normalize_component_maps(callback_index_map, "callback_index_map")
    )
    validated_callbacks = _validate_callbacks(
        callbacks,
        validated_callback_maps,
        batch_size=batch_size,
        num_transitions=num_transitions,
        component_rows={
            component: states[component].shape[2] for component in MINIMAX_H3_COMPONENT_ORDER
        },
    )

    trajectories: List[StructuredTrajectory] = []
    for sample_index in range(batch_size):
        components = {
            component: ComponentTrajectory(
                states=states[component][sample_index],
                timesteps=schedule[component][0],
                sigmas=schedule[component][1],
                state_index_map=state_maps[component],
            )
            for component in MINIMAX_H3_COMPONENT_ORDER
        }
        trajectories.append(
            StructuredTrajectory(
                components=components,
                log_probs=None if log_probs is None else log_probs[sample_index],
                log_prob_index_map=log_prob_index_map,
                component_log_probs=(
                    None
                    if component_log_probs is None
                    else {
                        component: component_log_probs[component][sample_index]
                        for component in MINIMAX_H3_COMPONENT_ORDER
                    }
                ),
                callbacks=(
                    None
                    if not validated_callbacks
                    else {
                        field: {
                            component: IndexedTrajectoryTensor(
                                values=component_values[component][sample_index],
                                index_map=validated_callback_maps[component],
                            )
                            for component in MINIMAX_H3_COMPONENT_ORDER
                        }
                        for field, component_values in validated_callbacks.items()
                    }
                ),
            )
        )
    return trajectories


def _validate_sigma_transform_inputs(sigma: torch.Tensor, shift: float, function: str) -> None:
    if not isinstance(sigma, torch.Tensor):
        raise TypeError(
            f"expected torch.Tensor sigma for {function}, received {type(sigma).__name__}"
        )
    _validate_unit_interval_tensor(sigma, f"{function} sigma")
    if (
        not isinstance(shift, (int, float))
        or isinstance(shift, bool)
        or not math.isfinite(float(shift))
        or shift <= 0
    ):
        raise ValueError(f"expected finite positive shift for {function}, received {shift!r}")


def _validate_unit_interval_tensor(values: torch.Tensor, field: str) -> None:
    if not isinstance(values, torch.Tensor) or not values.is_floating_point():
        raise TypeError(
            f"expected floating torch.Tensor {field}, received "
            f"{type(values).__name__}/{getattr(values, 'dtype', None)}"
        )
    if (
        not bool(torch.isfinite(values).all())
        or bool((values < 0).any())
        or bool((values > 1).any())
    ):
        raise ValueError(f"expected {field} in [0, 1], received {values.tolist()}")


def _validate_component_times(times: ComponentTimes, state: LatentState) -> None:
    if not isinstance(times, ComponentTimes) or times.sigma is None:
        raise TypeError(
            f"expected ComponentTimes with sigma mapping, received {type(times).__name__}"
        )
    batch_size = state.components["video"].shape[0]
    for field in ("timestep", "next_timestep", "sigma", "next_sigma"):
        values = getattr(times, field)
        if values is None:
            raise ValueError(f"expected ComponentTimes.{field}, received None")
        _require_component_order(values, f"ComponentTimes.{field}")
        for component in MINIMAX_H3_COMPONENT_ORDER:
            coordinate = values[component]
            if not isinstance(coordinate, torch.Tensor):
                raise TypeError(
                    f"expected torch.Tensor ComponentTimes.{field}[{component!r}], "
                    f"received {type(coordinate).__name__}: {coordinate!r}"
                )
            if not coordinate.is_floating_point():
                raise TypeError(
                    f"expected floating ComponentTimes.{field}[{component!r}], "
                    f"received dtype {coordinate.dtype}"
                )
            if coordinate.shape != (batch_size,):
                raise ValueError(
                    f"expected ComponentTimes.{field}[{component!r}] shape ({batch_size},), "
                    f"received {tuple(coordinate.shape)}"
                )
            reference = state.components[component]
            if coordinate.device != reference.device:
                raise ValueError(
                    f"expected ComponentTimes.{field}[{component!r}] device to match "
                    f"state {reference.device}, received {coordinate.device}"
                )
            if not bool(torch.isfinite(coordinate).all()):
                raise ValueError(
                    f"expected ComponentTimes.{field}[{component!r}] finite in "
                    f"{'[0, 1]' if 'sigma' in field else '[0, 1000]'}, "
                    f"received {coordinate.tolist()}"
                )
            upper = 1.0 if "sigma" in field else 1000.0
            if bool((coordinate < 0).any()) or bool((coordinate > upper).any()):
                raise ValueError(
                    f"expected ComponentTimes.{field}[{component!r}] in [0, {upper:g}], "
                    f"received {coordinate.tolist()}"
                )
    for component in MINIMAX_H3_COMPONENT_ORDER:
        if not torch.allclose(
            times.timestep[component].float(),
            times.sigma[component].float() * 1000,
            rtol=0,
            atol=1e-5,
        ):
            raise ValueError(
                f"expected ComponentTimes.timestep[{component!r}] == "
                f"sigma[{component!r}] * 1000, received "
                f"{times.timestep[component].tolist()} and {times.sigma[component].tolist()}"
            )
        for field in ("next_timestep", "next_sigma"):
            if bool((getattr(times, field)[component] != 0).any()):
                raise ValueError(
                    f"expected decoupled ComponentTimes.{field}[{component!r}] to be zero, "
                    f"received {getattr(times, field)[component].tolist()}"
                )


def _require_component_order(values: Mapping[str, object], field: str) -> None:
    if not isinstance(values, Mapping):
        raise TypeError(f"expected Mapping for {field}, received {type(values).__name__}")
    if tuple(values) != MINIMAX_H3_COMPONENT_ORDER:
        raise ValueError(
            f"expected {field} component order {MINIMAX_H3_COMPONENT_ORDER}, "
            f"received {tuple(values)}"
        )


def _validate_schedule_entry(
    component: str, entry: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(entry, tuple) or len(entry) != 2:
        raise TypeError(
            f"expected (timesteps, sigmas) for component {component!r}, received {entry!r}"
        )
    timesteps, sigmas = entry
    if not isinstance(timesteps, torch.Tensor) or not isinstance(sigmas, torch.Tensor):
        raise TypeError(
            f"expected torch.Tensor schedule entries for component {component!r}, "
            f"received timesteps={type(timesteps).__name__}, sigmas={type(sigmas).__name__}"
        )
    for field, values in (("timesteps", timesteps), ("sigmas", sigmas)):
        if not values.is_floating_point():
            raise TypeError(
                f"expected component {component!r} {field} with floating dtype, "
                f"received {values.dtype}"
            )
        if not bool(torch.isfinite(values).all()):
            raise ValueError(
                f"expected component {component!r} {field} finite, received {values.tolist()}"
            )
    if (
        timesteps.ndim != 1
        or sigmas.ndim != 1
        or timesteps.shape != sigmas.shape
        or len(timesteps) < 2
    ):
        raise ValueError(
            f"expected component {component!r} schedule tensors shaped (T + 1,), "
            f"received timesteps={getattr(timesteps, 'shape', None)} and "
            f"sigmas={getattr(sigmas, 'shape', None)}"
        )
    if (
        bool((sigmas < 0).any())
        or bool((sigmas > 1).any())
        or not bool((sigmas[1:] < sigmas[:-1]).all())
    ):
        raise ValueError(
            f"expected component {component!r} sigmas strictly decreasing in [0, 1], "
            f"received {sigmas.tolist()}"
        )
    if not torch.allclose(timesteps, sigmas * 1000, rtol=0, atol=1e-5):
        raise ValueError(
            f"expected component {component!r} timesteps == sigmas * 1000, "
            f"received timesteps={timesteps.tolist()} and sigmas={sigmas.tolist()}"
        )
    if timesteps[-1].item() != 0.0 or sigmas[-1].item() != 0.0:
        raise ValueError(
            f"expected component {component!r} schedule terminal zero, received "
            f"timestep={timesteps[-1].item()} sigma={sigmas[-1].item()}"
        )
    return timesteps, sigmas


def _normalize_component_maps(
    index_map: Union[torch.Tensor, Mapping[str, torch.Tensor], None],
    field: str,
) -> Dict[str, torch.Tensor]:
    """Return independent component maps, cloning an explicitly shared tensor."""
    if isinstance(index_map, torch.Tensor):
        return {component: index_map.clone() for component in MINIMAX_H3_COMPONENT_ORDER}
    if index_map is None:
        raise ValueError(f"expected {field}, received None")
    _require_component_order(index_map, field)
    return {component: index_map[component] for component in MINIMAX_H3_COMPONENT_ORDER}


def _validate_index_map(
    index_map: torch.Tensor,
    *,
    expected_length: int,
    num_stored: int,
    field: str,
) -> None:
    if not isinstance(index_map, torch.Tensor):
        raise TypeError(f"expected torch.Tensor {field}, received {type(index_map).__name__}")
    if index_map.ndim != 1 or len(index_map) != expected_length:
        raise ValueError(
            f"expected {field} shape ({expected_length},), received {tuple(index_map.shape)}"
        )
    if index_map.dtype not in _SIGNED_INTEGER_DTYPES:
        raise TypeError(f"expected signed integer {field}, received {index_map.dtype}")
    if index_map.numel() and (int(index_map.min()) < -1 or int(index_map.max()) >= num_stored):
        raise ValueError(
            f"expected {field} values in [-1, {num_stored - 1}], received {index_map.tolist()}"
        )
    collected = index_map[index_map >= 0]
    if collected.numel() != num_stored or set(collected.tolist()) != set(range(num_stored)):
        raise ValueError(
            f"expected {field} to address each of {num_stored} stored entries exactly once, "
            f"received {index_map.tolist()}"
        )


def _validate_optional_log_probs(
    log_probs: Optional[torch.Tensor],
    component_log_probs: Optional[Mapping[str, torch.Tensor]],
    index_map: Optional[torch.Tensor],
    *,
    batch_size: int,
    num_transitions: int,
) -> None:
    if log_probs is None:
        if component_log_probs is not None or index_map is not None:
            raise ValueError(
                "expected component_log_probs and log_prob_index_map to be None when log_probs=None"
            )
        return
    if (
        not isinstance(log_probs, torch.Tensor)
        or log_probs.ndim != 2
        or log_probs.shape[0] != batch_size
    ):
        raise ValueError(
            f"expected log_probs shaped ({batch_size}, stored), received "
            f"{getattr(log_probs, 'shape', None)}"
        )
    if index_map is None:
        raise ValueError("expected log_prob_index_map alongside log_probs, received None")
    _validate_index_map(
        index_map,
        expected_length=num_transitions,
        num_stored=log_probs.shape[1],
        field="log_prob_index_map",
    )
    if component_log_probs is None:
        return
    _require_component_order(component_log_probs, "component_log_probs")
    for component in MINIMAX_H3_COMPONENT_ORDER:
        if component_log_probs[component].shape != log_probs.shape:
            raise ValueError(
                f"expected component_log_probs[{component!r}] shape {tuple(log_probs.shape)}, "
                f"received {tuple(component_log_probs[component].shape)}"
            )


def _validate_callbacks(
    callbacks: Optional[Mapping[str, Mapping[str, torch.Tensor]]],
    index_maps: Optional[Mapping[str, torch.Tensor]],
    *,
    batch_size: int,
    num_transitions: int,
    component_rows: Mapping[str, int],
) -> Dict[str, Dict[str, torch.Tensor]]:
    if callbacks is None:
        if index_maps is not None:
            raise ValueError(
                "expected callback_index_map=None when callbacks=None, "
                f"received components {tuple(index_maps)}"
            )
        return {}
    if not isinstance(callbacks, Mapping) or not callbacks:
        raise ValueError(
            f"expected non-empty callback mapping, received {type(callbacks).__name__}"
        )
    if index_maps is None:
        raise ValueError("expected callback_index_map alongside callbacks, received None")
    validated: Dict[str, Dict[str, torch.Tensor]] = {}
    stored_lengths: Dict[str, int] = {}
    for field, component_values in callbacks.items():
        if not isinstance(field, str) or not field:
            raise ValueError(f"expected non-empty callback field, received {field!r}")
        _require_component_order(component_values, f"callbacks[{field!r}]")
        validated[field] = {}
        for component in MINIMAX_H3_COMPONENT_ORDER:
            values = component_values[component]
            width = MINIMAX_H3_COMPONENT_WIDTHS[component]
            if (
                not isinstance(values, torch.Tensor)
                or values.ndim != 4
                or values.shape[0] != batch_size
                or values.shape[-1] != width
            ):
                raise ValueError(
                    f"expected callbacks[{field!r}][{component!r}] shaped "
                    f"({batch_size}, stored, N, {width}), received "
                    f"{getattr(values, 'shape', None)}"
                )
            if values.shape[2] != component_rows[component]:
                raise ValueError(
                    f"callback {field!r} component {component!r} row geometry expected "
                    f"{component_rows[component]}, received {values.shape[2]}"
                )
            stored_length = stored_lengths.setdefault(component, values.shape[1])
            if values.shape[1] != stored_length:
                raise ValueError(
                    f"expected callback stored length {stored_length}, received "
                    f"{values.shape[1]} for {field!r}/{component!r}"
                )
            validated[field][component] = values
    for component in MINIMAX_H3_COMPONENT_ORDER:
        _validate_index_map(
            index_maps[component],
            expected_length=num_transitions,
            num_stored=stored_lengths[component],
            field=f"callback_index_map[{component!r}]",
        )
    return validated
