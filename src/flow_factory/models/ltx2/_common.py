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

# src/flow_factory/models/ltx2/_common.py
"""Shared helpers for the LTX2 (T2AV / I2AV) adapters."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from diffusers.utils.torch_utils import randn_tensor

from ...samples import (
    ComponentTimes,
    LatentState,
    MultiModalStepOutput,
    NoisedState,
    StackedSampleBatch,
)
from ...scheduler import SDESchedulerOutput
from ...utils.base import filter_kwargs
from ...utils.noise_schedule import flow_match_sigma

# Both LTX2 adapters expose one joint video+audio policy, so the authoritative
# component order is fixed here rather than derived from any mapping iteration.
LTX2_COMPONENT_ORDER: Tuple[str, ...] = ("video", "audio")


def combine_modality_log_prob(
    video_log_prob: torch.Tensor,
    audio_log_prob: torch.Tensor,
    n_video: int,
    n_audio: int,
) -> torch.Tensor:
    """Element-weighted mean of the per-step video/audio log-probs.

    LTX2 steps video and audio with two scheduler instances, each returning a
    per-sample log_prob already meaned over its own latent dims. Weighting by the
    element counts ``n_video`` / ``n_audio`` reproduces the mean that a single
    scheduler over the concatenated ``[video|audio]`` latent would produce (mean
    over all latent dims), so the joint log_prob keeps the same scale as the
    original video-only path.

    Args:
        video_log_prob: Per-sample video log_prob, shape ``(B,)``.
        audio_log_prob: Per-sample audio log_prob, shape ``(B,)``.
        n_video: Number of video latent elements per sample (the tensor passed to
            the video ``step()``).
        n_audio: Number of audio latent elements per sample.

    Returns:
        Per-sample joint log_prob, shape ``(B,)``.
    """
    total = n_video + n_audio
    return (video_log_prob * n_video + audio_log_prob * n_audio) / total


def build_ltx2_training_component_times(
    adapter: Any, primary_timesteps: torch.Tensor
) -> ComponentTimes:
    """Map one sampled coordinate onto the video and audio training times.

    LTX2 currently runs twin schedules, so both components receive the same
    numeric timestep and sigma. Decoupled training keeps the next coordinate at
    zero, and the mapping consumes no randomness.

    Args:
        adapter: Adapter declaring ``trajectory_component_order``.
        primary_timesteps: Primary scheduler coordinates of shape ``(B,)``.

    Returns:
        Video/audio timesteps and sigmas in authoritative component order.
    """
    if not isinstance(primary_timesteps, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor primary_timesteps for build_training_component_times on "
            f"{type(adapter).__name__}, received {type(primary_timesteps).__name__}: "
            f"{primary_timesteps!r}"
        )
    if primary_timesteps.ndim != 1:
        raise ValueError(
            "expected primary_timesteps with one scheduler coordinate per sample, shape "
            f"(B,), received {tuple(primary_timesteps.shape)}"
        )
    _require_ltx2_component_order(adapter, adapter.trajectory_component_order, "adapter")
    sigma = flow_match_sigma(primary_timesteps)
    zero_timestep = torch.zeros_like(primary_timesteps)
    zero_sigma = torch.zeros_like(sigma)
    return ComponentTimes(
        timestep={"video": primary_timesteps, "audio": primary_timesteps},
        next_timestep={"video": zero_timestep, "audio": zero_timestep},
        sigma={"video": sigma, "audio": sigma},
        next_sigma={"video": zero_sigma, "audio": zero_sigma},
    )


def draw_ltx2_forward_process_noise(
    adapter: Any,
    clean_state: LatentState,
    times: ComponentTimes,
    *,
    generator: Optional[torch.Generator],
) -> NoisedState:
    """Draw video noise then audio noise, then apply it deterministically.

    Args:
        adapter: Adapter owning the deterministic application hook.
        clean_state: Clean latent state in authoritative component order.
        times: Component times including each component's current sigma.
        generator: Optional generator shared by both ordered draws.

    Returns:
        Noised state, target velocity, and the sampled noise.
    """
    _require_ltx2_component_order(adapter, clean_state.component_names, "clean_state")
    noise: Dict[str, torch.Tensor] = {}
    for name in LTX2_COMPONENT_ORDER:
        component = clean_state.components[name]
        noise[name] = randn_tensor(
            component.shape,
            generator=generator,
            device=component.device,
            dtype=component.dtype,
        )
    return adapter.apply_forward_process_noise(clean_state, times, LatentState(noise))


def build_ltx2_component_step_output(
    adapter: Any,
    *,
    video_output: SDESchedulerOutput,
    audio_output: SDESchedulerOutput,
    video_velocity: torch.Tensor,
    audio_velocity: torch.Tensor,
    n_video: int,
    n_audio: int,
    compute_log_prob: bool,
) -> MultiModalStepOutput:
    """Wrap the two scheduler outputs before the legacy concatenation runs.

    Args:
        adapter: Adapter whose forward produced both outputs.
        video_output: Video scheduler step output.
        audio_output: Audio scheduler step output.
        video_velocity: Guided video velocity fed to the video scheduler.
        audio_velocity: Guided audio velocity fed to the audio scheduler.
        n_video: Video elements per sample that carry stochastic DOF.
        n_audio: Audio elements per sample.
        compute_log_prob: Whether the step computed transition log probabilities.

    Returns:
        Ordered component step output holding the real per-modality statistics.
    """

    def paired(field: str) -> Optional[Dict[str, torch.Tensor]]:
        video_value = getattr(video_output, field)
        audio_value = getattr(audio_output, field)
        if (video_value is None) != (audio_value is None):
            raise ValueError(
                f"expected the {type(adapter).__name__} video and audio scheduler steps to agree "
                f"on {field!r}, received video={type(video_value).__name__} and "
                f"audio={type(audio_value).__name__}"
            )
        if video_value is None:
            return None
        return {"video": video_value, "audio": audio_value}

    def paired_statistic(field: str) -> Optional[Dict[str, torch.Tensor]]:
        values = paired(field)
        if values is None:
            return None
        reference = {"video": video_velocity, "audio": audio_velocity}
        return {
            name: _align_statistic_rank(
                values[name],
                reference[name],
                adapter=adapter,
                component=name,
                field=field,
            )
            for name in LTX2_COMPONENT_ORDER
        }

    next_state = paired("next_latents")
    next_state_mean = paired("next_latents_mean")
    component_log_probs = paired("log_prob")
    log_prob = None
    if compute_log_prob and component_log_probs is not None:
        log_prob = combine_modality_log_prob(
            component_log_probs["video"],
            component_log_probs["audio"],
            n_video=n_video,
            n_audio=n_audio,
        )
    return MultiModalStepOutput(
        next_state=None if next_state is None else LatentState(next_state),
        next_state_mean=None if next_state_mean is None else LatentState(next_state_mean),
        std_dev_t=paired_statistic("std_dev_t"),
        dt=paired_statistic("dt"),
        log_prob=log_prob,
        component_log_probs=component_log_probs,
        velocity=LatentState({"video": video_velocity, "audio": audio_velocity}),
    )


def build_ltx2_joint_forward_kwargs(
    adapter: Any,
    *,
    batch: StackedSampleBatch,
    state: LatentState,
    times: ComponentTimes,
    next_state: Optional[LatentState],
    compute_log_prob: bool,
    return_fields: Tuple[str, ...],
    noise_level: Optional[float],
    forward_kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Pack an ordered component state for the legacy concatenated ``forward``.

    The caller runs the full input contract first (``validate_ltx2_forward_state_inputs``
    or its I2AV superset); only the component order is re-checked here, so an unvalidated
    caller still gets the contextual error instead of a ``KeyError``.

    Args:
        adapter: Adapter whose ``forward`` will receive the packed arguments.
        batch: Collated batch supplying conditioning and ``video_seq_len``.
        state: Current state in authoritative component order.
        times: Current and next times in authoritative component order.
        next_state: Optional stored next state in authoritative component order.
        compute_log_prob: Whether to compute transition log probabilities.
        return_fields: Scheduler output fields requested from ``forward``.
        noise_level: Scheduler noise-level override.
        forward_kwargs: Model-conditioning arguments resolved by the wrapper.

    Returns:
        Keyword arguments for ``adapter.forward`` in component-return mode.
    """
    _require_ltx2_component_orders(adapter, state=state, times=times, next_state=next_state)

    timestep = times.timestep["video"]
    next_timestep = times.next_timestep["video"]
    video = state.components["video"]
    audio = state.components["audio"]
    video_seq_len = video.shape[1]
    stored_seq_len = batch.get("video_seq_len")
    if not isinstance(stored_seq_len, int) or isinstance(stored_seq_len, bool):
        raise TypeError(
            f"expected int batch video_seq_len for {type(adapter).__name__} forward_state, "
            f"received {type(stored_seq_len).__name__}: {stored_seq_len!r}"
        )
    if stored_seq_len != video_seq_len:
        raise ValueError(
            f"expected batch video_seq_len {stored_seq_len} to match the video component "
            f"sequence length {video_seq_len} for {type(adapter).__name__} forward_state"
        )

    call_kwargs = filter_kwargs(adapter.forward, **forward_kwargs)
    call_kwargs.pop("video_seq_len", None)
    return {
        "t": timestep,
        "t_next": next_timestep,
        "latents": torch.cat([video, audio], dim=1),
        "next_latents": (
            None
            if next_state is None
            else torch.cat([next_state.components["video"], next_state.components["audio"]], dim=1)
        ),
        "video_seq_len": video_seq_len,
        "compute_log_prob": compute_log_prob,
        "return_kwargs": list(return_fields),
        "noise_level": noise_level,
        "_return_components": True,
        **call_kwargs,
    }


def validate_ltx2_forward_state_inputs(
    adapter: Any,
    *,
    state: LatentState,
    times: ComponentTimes,
    next_state: Optional[LatentState],
) -> None:
    """Check every component input the joint LTX2 forward is about to consume.

    The concatenated ``forward`` cannot see component boundaries, so a mismatch in
    channel count, dtype, device or stored next state would either raise deep inside
    ``torch.cat`` or silently mis-attribute one modality's values to the other.

    Args:
        adapter: Adapter running the component forward.
        state: Current state in authoritative component order.
        times: Current and next times in authoritative component order.
        next_state: Optional stored next state in authoritative component order.

    Raises:
        ValueError: If any order, shape, dtype, device or mask expectation fails.
    """
    _require_ltx2_component_orders(adapter, state=state, times=times, next_state=next_state)

    name = type(adapter).__name__
    video = state.components["video"]
    audio = state.components["audio"]
    for component, values in (("video", video), ("audio", audio)):
        if values.ndim != 3:
            raise ValueError(
                f"expected {name} state component {component!r} packed as (B, S, C), received "
                f"shape {tuple(values.shape)}"
            )
    if audio.shape[0] != video.shape[0]:
        raise ValueError(
            f"expected {name} batch size of state component 'audio' to match 'video' "
            f"({video.shape[0]}), received {audio.shape[0]}"
        )
    if audio.shape[2] != video.shape[2]:
        raise ValueError(
            f"expected {name} channel dimension of state component 'audio' to match 'video' "
            f"({video.shape[2]}), received {audio.shape[2]}"
        )
    if audio.dtype is not video.dtype:
        raise ValueError(
            f"expected {name} dtype of state component 'audio' to match 'video' "
            f"({video.dtype}), received {audio.dtype}"
        )
    if audio.device != video.device:
        raise ValueError(
            f"expected {name} device of state component 'audio' to match 'video' "
            f"({video.device}), received {audio.device}"
        )

    batch_size = video.shape[0]
    for field in ("timestep", "next_timestep"):
        values = getattr(times, field)
        for component in LTX2_COMPONENT_ORDER:
            coordinate = values[component]
            identifier = f"times.{field}[{component!r}]"
            if not isinstance(coordinate, torch.Tensor):
                raise TypeError(
                    f"expected {name} {identifier} as a torch.Tensor, received "
                    f"{type(coordinate).__name__}: {coordinate!r}"
                )
            if coordinate.shape != (batch_size,):
                raise ValueError(
                    f"expected {name} {identifier} with one coordinate per sample, shape "
                    f"({batch_size},), received {tuple(coordinate.shape)}"
                )
            if coordinate.device != video.device:
                raise ValueError(
                    f"expected {name} {identifier} on the state device {video.device}, received "
                    f"{coordinate.device}"
                )
    _require_joint_coordinate(adapter, times.timestep, "timestep")
    _require_joint_coordinate(adapter, times.next_timestep, "next_timestep")

    if next_state is None:
        return
    for component in LTX2_COMPONENT_ORDER:
        current = state.components[component]
        stored = next_state.components[component]
        identifier = f"next_state component {component!r}"
        if stored.shape != current.shape:
            raise ValueError(
                f"expected {name} {identifier} to match the state shape "
                f"{tuple(current.shape)}, received {tuple(stored.shape)}"
            )
        if stored.dtype is not current.dtype:
            raise ValueError(
                f"expected {name} {identifier} dtype {current.dtype}, received {stored.dtype}"
            )
        if stored.device != current.device:
            raise ValueError(
                f"expected {name} {identifier} device {current.device}, received {stored.device}"
            )
    _require_matching_state_masks(adapter, state=state, next_state=next_state)


def _require_matching_state_masks(
    adapter: Any, *, state: LatentState, next_state: LatentState
) -> None:
    """Require the stored next state to carry exactly the current static masks."""
    name = type(adapter).__name__
    if state.active_masks is None:
        if next_state.active_masks is not None:
            raise ValueError(
                f"expected {name} next_state active_masks to be None because the state carries "
                f"none, received {tuple(next_state.active_masks)}"
            )
        return
    if next_state.active_masks is None:
        raise ValueError(
            f"expected {name} next_state active_masks to be present because the state carries "
            "masks, received None"
        )
    for component in LTX2_COMPONENT_ORDER:
        current = state.active_masks[component]
        stored = next_state.active_masks[component]
        identifier = f"next_state active_masks[{component!r}]"
        if stored.shape != current.shape or not bool(torch.equal(stored, current)):
            raise ValueError(
                f"expected {name} {identifier} to equal the state mask of shape "
                f"{tuple(current.shape)} with {int(current.sum().item())} active entries, "
                f"received shape {tuple(stored.shape)} with {int(stored.sum().item())}"
            )


def attach_ltx2_state_masks(
    state: LatentState, output: MultiModalStepOutput
) -> MultiModalStepOutput:
    """Carry the input active masks onto every shape-compatible returned state.

    Args:
        state: Input state supplying the static component masks.
        output: Component step output produced by ``forward``.

    Returns:
        The same output, with masks attached where they broadcast.
    """
    if not isinstance(output, MultiModalStepOutput):
        raise TypeError(
            "expected MultiModalStepOutput from the LTX2 component forward, "
            f"received {type(output).__name__}"
        )
    if state.active_masks is None:
        return output
    for field in ("next_state", "next_state_mean", "velocity"):
        component_state = getattr(output, field)
        if component_state is None or component_state.active_masks is not None:
            continue
        if not all(
            _mask_broadcasts(state.active_masks[name], component_state.components[name])
            for name in component_state.component_names
        ):
            continue
        setattr(
            output,
            field,
            LatentState(dict(component_state.components), active_masks=state.active_masks),
        )
    return output


def validate_i2av_forward_state_inputs(
    adapter: Any,
    *,
    batch: StackedSampleBatch,
    state: LatentState,
    times: ComponentTimes,
    next_state: Optional[LatentState],
) -> None:
    """Check the shared component contract, then the I2AV conditioning contract.

    The shared contract runs first so a missing or reordered component is reported
    with its expected/received context instead of failing where this function reads
    ``active_masks['video']``.

    Args:
        adapter: Adapter running the component forward.
        batch: Collated batch that must carry ``conditioning_mask``.
        state: Current state whose masks describe the stochastic video tokens.
        times: Current and next times in authoritative component order.
        next_state: Optional stored next state in authoritative component order.

    Raises:
        ValueError: If the shared contract fails or the masks disagree with the
            batch conditioning mask.
    """
    validate_ltx2_forward_state_inputs(adapter, state=state, times=times, next_state=next_state)
    conditioning_mask = batch.get("conditioning_mask")
    if conditioning_mask is None:
        raise ValueError(
            f"expected batch conditioning_mask for {type(adapter).__name__} forward_state, "
            f"received None with batch keys {tuple(sorted(batch.keys()))}"
        )
    if not isinstance(conditioning_mask, torch.Tensor):
        raise TypeError(
            f"expected torch.Tensor batch conditioning_mask for {type(adapter).__name__} "
            f"forward_state, received {type(conditioning_mask).__name__}"
        )
    if state.active_masks is None:
        raise ValueError(
            f"expected LatentState.active_masks marking the generated video tokens for "
            f"{type(adapter).__name__} forward_state, received None"
        )
    expected = ~conditioning_mask.bool()
    video_mask = state.active_masks["video"]
    if (
        video_mask.shape[: expected.ndim] != expected.shape
        or video_mask.numel() != expected.numel()
    ):
        raise ValueError(
            f"expected LatentState.active_masks['video'] to cover the conditioning_mask shape "
            f"{tuple(expected.shape)} (a broadcast channel axis is allowed), received "
            f"{tuple(video_mask.shape)}"
        )
    if not bool(torch.equal(video_mask.reshape(expected.shape), expected)):
        raise ValueError(
            "expected LatentState.active_masks['video'] to equal ~conditioning_mask, received "
            f"{int(video_mask.sum().item())} active tokens against "
            f"{int(expected.sum().item())} generated tokens"
        )
    audio_mask = state.active_masks["audio"]
    if not bool(audio_mask.all()):
        raise ValueError(
            "expected LatentState.active_masks['audio'] to be all active because LTX2 audio "
            f"carries no fixed conditioning, received {int(audio_mask.sum().item())} of "
            f"{audio_mask.numel()} active entries"
        )


def _align_statistic_rank(
    statistic: torch.Tensor,
    reference: torch.Tensor,
    *,
    adapter: Any,
    component: str,
    field: str,
) -> torch.Tensor:
    """Re-rank a per-sample scheduler statistic onto the packed component layout.

    I2AV steps the video scheduler on unpacked ``(B, C, F, H, W)`` frames, so its
    statistics carry that rank while the component latents stay packed
    ``(B, S, C)``. Trailing singleton axes are dropped so the same value keeps
    broadcasting against the component it describes instead of silently producing
    a higher-rank result.
    """
    if statistic.shape == reference.shape:
        return statistic
    if statistic.shape[0] != reference.shape[0] or statistic.numel() != statistic.shape[0]:
        raise ValueError(
            f"expected {type(adapter).__name__} {field!r} for component {component!r} to be "
            f"per-sample or shaped like the component {tuple(reference.shape)}, received "
            f"{tuple(statistic.shape)}"
        )
    return statistic.reshape((statistic.shape[0],) + (1,) * (reference.ndim - 1))


def _require_ltx2_component_orders(
    adapter: Any,
    *,
    state: LatentState,
    times: ComponentTimes,
    next_state: Optional[LatentState],
) -> None:
    """Check every component mapping the joint forward indexes by name."""
    _require_ltx2_component_order(adapter, state.component_names, "state")
    _require_ltx2_component_order(adapter, tuple(times.timestep), "times.timestep")
    _require_ltx2_component_order(adapter, tuple(times.next_timestep), "times.next_timestep")
    if next_state is not None:
        _require_ltx2_component_order(adapter, next_state.component_names, "next_state")


def _require_ltx2_component_order(adapter: Any, received: Tuple[str, ...], field: str) -> None:
    if tuple(received) != LTX2_COMPONENT_ORDER:
        raise ValueError(
            f"expected {type(adapter).__name__} {field} component order "
            f"{LTX2_COMPONENT_ORDER}, received {tuple(received)}"
        )


def _require_joint_coordinate(
    adapter: Any, values: Mapping[str, torch.Tensor], field: str
) -> torch.Tensor:
    """Return the shared coordinate the joint LTX2 transformer can accept."""
    video = values["video"]
    audio = values["audio"]
    if video.shape != audio.shape or not bool(torch.equal(video, audio)):
        raise ValueError(
            f"expected {type(adapter).__name__} {field} entries for 'video' and 'audio' to be "
            f"equal because the LTX2 transformer takes one joint time coordinate, received "
            f"video {video.tolist()} and audio {audio.tolist()}"
        )
    return video


def _mask_broadcasts(mask: torch.Tensor, values: torch.Tensor) -> bool:
    if mask.ndim != values.ndim:
        return False
    return all(mask_dim in (1, value_dim) for mask_dim, value_dim in zip(mask.shape, values.shape))
