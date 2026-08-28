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

"""Model-agnostic flow-matching primitives for finite offline objectives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Tuple, Union

import torch

from ...models.output_state import EncodedOutputState
from ...samples import ComponentTimes, LatentState, NoisedState
from ...utils.noise_schedule import TimeSampler


def sample_offline_timesteps(
    training_args: Any,
    *,
    batch_size: int,
    device: Union[torch.device, str],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw one independent scheduler coordinate per loss term and sample.

    ``num_train_timesteps`` is the number of Monte Carlo terms averaged inside
    one dataloader microstep. It does not alter gradient accumulation.
    """
    _require_positive_int(batch_size, "batch_size")
    num_timesteps = getattr(training_args, "num_train_timesteps", None)
    _require_positive_int(num_timesteps, "training_args.num_train_timesteps")
    scheme = getattr(training_args, "weighting_scheme", None)
    common = {
        "batch_size": batch_size,
        "num_timesteps": num_timesteps,
        "timestep_range": getattr(training_args, "timestep_range", None),
        "time_shift": getattr(training_args, "time_shift", None),
        "device": torch.device(device),
        "generator": generator,
    }
    if scheme == "logit_normal":
        return TimeSampler.independent_logit_normal_shifted(
            **common,
            logit_mean=getattr(training_args, "logit_mean", None),
            logit_std=getattr(training_args, "logit_std", None),
        )
    if scheme == "uniform":
        return TimeSampler.independent_uniform(**common)
    raise ValueError(
        "offline weighting_scheme must be 'logit_normal' or 'uniform', " f"received {scheme!r}"
    )


def build_noised_output_state(
    adapter: Any,
    clean_state: LatentState,
    primary_timesteps: torch.Tensor,
    *,
    batch: Mapping[str, Any],
    generator: Optional[torch.Generator] = None,
    noise: Optional[LatentState] = None,
) -> Tuple[ComponentTimes, NoisedState]:
    """Map scheduler coordinates and apply either fresh or explicitly shared noise."""
    if not isinstance(clean_state, LatentState):
        raise TypeError(
            "expected clean_state to be LatentState, "
            f"received {type(clean_state).__name__}: {clean_state!r}"
        )
    if not isinstance(primary_timesteps, torch.Tensor):
        raise TypeError(
            "expected primary_timesteps to be torch.Tensor, "
            f"received {type(primary_timesteps).__name__}: {primary_timesteps!r}"
        )
    if not isinstance(batch, Mapping):
        raise TypeError(f"expected batch to be Mapping, received {type(batch).__name__}: {batch!r}")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError(
            "expected generator to be torch.Generator or None, "
            f"received {type(generator).__name__}: {generator!r}"
        )
    if noise is not None and not isinstance(noise, LatentState):
        raise TypeError(
            f"expected noise to be LatentState or None, received {type(noise).__name__}"
        )
    if noise is not None and generator is not None:
        raise ValueError(
            "generator and explicit noise are mutually exclusive; explicit shared noise "
            "must not consume another RNG stream"
        )

    times = adapter.build_training_component_times(primary_timesteps, batch=batch)
    if not isinstance(times, ComponentTimes):
        raise TypeError(
            "adapter.build_training_component_times must return ComponentTimes, "
            f"received {type(times).__name__}"
        )
    if noise is None:
        noised = adapter.add_forward_process_noise(
            clean_state,
            times,
            generator=generator,
        )
    else:
        noised = adapter.apply_forward_process_noise(clean_state, times, noise)
    if not isinstance(noised, NoisedState):
        raise TypeError(
            "adapter forward-process hook must return NoisedState, "
            f"received {type(noised).__name__}"
        )
    return times, noised


def flow_matching_per_sample_loss(
    adapter: Any,
    predicted_velocity: LatentState,
    noised: NoisedState,
) -> torch.Tensor:
    """Return fp32 velocity MSE reduced over active latent elements per sample."""
    if not isinstance(predicted_velocity, LatentState):
        raise TypeError(
            "expected predicted_velocity to be LatentState, "
            f"received {type(predicted_velocity).__name__}"
        )
    if not isinstance(noised, NoisedState):
        raise TypeError(f"expected noised to be NoisedState, received {type(noised).__name__}")
    target_velocity = noised.target_velocity
    expected_names = target_velocity.component_names
    if predicted_velocity.component_names != expected_names:
        raise ValueError(
            "predicted and target velocity component order mismatch: "
            f"expected {expected_names}, received {predicted_velocity.component_names}"
        )

    squared_errors = {}
    for name in expected_names:
        predicted = predicted_velocity.components[name]
        target = target_velocity.components[name]
        if predicted.shape != target.shape:
            raise ValueError(
                f"velocity shape mismatch for component {name!r}: expected "
                f"{tuple(target.shape)}, received {tuple(predicted.shape)}"
            )
        if predicted.device != target.device:
            raise ValueError(
                f"velocity device mismatch for component {name!r}: expected "
                f"{target.device}, received {predicted.device}"
            )
        if not predicted.is_floating_point() or not target.is_floating_point():
            raise TypeError(
                f"velocity component {name!r} must use floating tensors, received "
                f"predicted={predicted.dtype}, target={target.dtype}"
            )
        squared_errors[name] = (predicted.float() - target.float()).square()

    reduced = adapter.reduce_latent_values(squared_errors, state=noised.state)
    if not isinstance(reduced, torch.Tensor):
        raise TypeError(
            "adapter.reduce_latent_values must return torch.Tensor, "
            f"received {type(reduced).__name__}"
        )
    batch_size = next(iter(squared_errors.values())).shape[0]
    if reduced.shape != (batch_size,):
        raise ValueError(
            "adapter.reduce_latent_values must return one value per sample with shape "
            f"({batch_size},), received {tuple(reduced.shape)}"
        )
    if reduced.dtype is not torch.float32:
        raise TypeError(
            "offline flow-matching reduction must preserve fp32 errors, "
            f"received {reduced.dtype}"
        )
    return reduced


def validate_preference_output_states(
    chosen: EncodedOutputState,
    rejected: EncodedOutputState,
) -> None:
    """Require pairwise arms to support the same forward process and reduction."""
    for name, value in (("chosen", chosen), ("rejected", rejected)):
        if not isinstance(value, EncodedOutputState):
            raise TypeError(
                f"expected {name} output to be EncodedOutputState, "
                f"received {type(value).__name__}"
            )
    chosen_state = chosen.clean_state
    rejected_state = rejected.clean_state
    if chosen_state.component_names != rejected_state.component_names:
        raise ValueError(
            "preference arm component order mismatch: "
            f"chosen={chosen_state.component_names}, rejected={rejected_state.component_names}"
        )
    for name in chosen_state.component_names:
        chosen_component = chosen_state.components[name]
        rejected_component = rejected_state.components[name]
        if chosen_component.shape != rejected_component.shape:
            raise ValueError(
                f"preference arm shape mismatch for component {name!r}: "
                f"chosen={tuple(chosen_component.shape)}, "
                f"rejected={tuple(rejected_component.shape)}"
            )
        if chosen_component.dtype != rejected_component.dtype:
            raise TypeError(
                f"preference arm dtype mismatch for component {name!r}: "
                f"chosen={chosen_component.dtype}, rejected={rejected_component.dtype}"
            )
        if chosen_component.device != rejected_component.device:
            raise ValueError(
                f"preference arm device mismatch for component {name!r}: "
                f"chosen={chosen_component.device}, rejected={rejected_component.device}"
            )
    if chosen.geometry_signatures != rejected.geometry_signatures:
        raise ValueError(
            "preference arms must use identical geometry signatures before shared-noise "
            f"training, received chosen={chosen.geometry_signatures}, "
            f"rejected={rejected.geometry_signatures}"
        )
    _validate_matching_masks(chosen_state, rejected_state)
    _validate_matching_context_structure(
        chosen.forward_context,
        rejected.forward_context,
        identifier="forward_context",
    )


def validate_preference_component_times(
    chosen: ComponentTimes,
    rejected: ComponentTimes,
) -> None:
    """Require pairwise arms to resolve one identical forward-process schedule.

    Offline DPO supplies the same primary scheduler coordinates to both output
    arms. An adapter may still derive component-specific coordinates from output
    context, so equality has to be proven after that model-owned mapping rather
    than inferred from the shared primary tensor.

    Args:
        chosen: Component schedule resolved for the chosen output.
        rejected: Component schedule resolved for the rejected output.

    Raises:
        TypeError: If either value is not ``ComponentTimes`` or optional fields differ.
        ValueError: If field metadata, component order, or tensor values differ.
    """
    for name, value in (("chosen", chosen), ("rejected", rejected)):
        if not isinstance(value, ComponentTimes):
            raise TypeError(
                f"expected {name} component times to be ComponentTimes, "
                f"received {type(value).__name__}: {value!r}"
            )

    for field_name in ("timestep", "next_timestep", "sigma", "next_sigma"):
        chosen_values = getattr(chosen, field_name)
        rejected_values = getattr(rejected, field_name)
        if (chosen_values is None) != (rejected_values is None):
            raise TypeError(
                "preference arm component times optional-field mismatch for " f"{field_name!r}"
            )
        if chosen_values is None:
            continue
        if tuple(chosen_values) != tuple(rejected_values):
            raise ValueError(
                "preference arm component times order mismatch for "
                f"{field_name!r}: chosen={tuple(chosen_values)}, "
                f"rejected={tuple(rejected_values)}"
            )
        for component_name in chosen_values:
            chosen_tensor = chosen_values[component_name]
            rejected_tensor = rejected_values[component_name]
            if (
                chosen_tensor.shape != rejected_tensor.shape
                or chosen_tensor.dtype != rejected_tensor.dtype
                or chosen_tensor.device != rejected_tensor.device
            ):
                raise ValueError(
                    "preference arm component times tensor metadata mismatch for "
                    f"{field_name}[{component_name!r}]: "
                    f"chosen=({tuple(chosen_tensor.shape)}, {chosen_tensor.dtype}, "
                    f"{chosen_tensor.device}), rejected=({tuple(rejected_tensor.shape)}, "
                    f"{rejected_tensor.dtype}, {rejected_tensor.device})"
                )
            if not torch.equal(chosen_tensor, rejected_tensor):
                raise ValueError(
                    "preference arm component times values mismatch for "
                    f"{field_name}[{component_name!r}]"
                )


def _validate_matching_masks(chosen: LatentState, rejected: LatentState) -> None:
    if (chosen.active_masks is None) != (rejected.active_masks is None):
        raise ValueError("preference arms must either both define active masks or both omit them")
    if chosen.active_masks is None:
        return
    if tuple(chosen.active_masks) != tuple(rejected.active_masks):
        raise ValueError("preference arm active-mask component order mismatch")
    for name in chosen.component_names:
        chosen_mask = chosen.active_masks[name]
        rejected_mask = rejected.active_masks[name]
        if (
            chosen_mask.shape != rejected_mask.shape
            or chosen_mask.device != rejected_mask.device
            or not torch.equal(chosen_mask, rejected_mask)
        ):
            raise ValueError(
                f"preference arms require identical active masks for component {name!r}"
            )


def _validate_matching_context_structure(chosen: Any, rejected: Any, *, identifier: str) -> None:
    if isinstance(chosen, torch.Tensor) or isinstance(rejected, torch.Tensor):
        if not isinstance(chosen, torch.Tensor) or not isinstance(rejected, torch.Tensor):
            raise TypeError(f"preference arm {identifier} tensor structure mismatch")
        if (
            chosen.shape != rejected.shape
            or chosen.dtype != rejected.dtype
            or chosen.device != rejected.device
        ):
            raise ValueError(
                f"preference arm {identifier} tensor metadata mismatch: "
                f"chosen=({tuple(chosen.shape)}, {chosen.dtype}, {chosen.device}), "
                f"rejected=({tuple(rejected.shape)}, {rejected.dtype}, {rejected.device})"
            )
        return
    if isinstance(chosen, Mapping) or isinstance(rejected, Mapping):
        if not isinstance(chosen, Mapping) or not isinstance(rejected, Mapping):
            raise TypeError(f"preference arm {identifier} mapping structure mismatch")
        if tuple(chosen) != tuple(rejected):
            raise ValueError(
                f"preference arm {identifier} key/order mismatch: "
                f"chosen={tuple(chosen)}, rejected={tuple(rejected)}"
            )
        for key in chosen:
            _validate_matching_context_structure(
                chosen[key],
                rejected[key],
                identifier=f"{identifier}[{key!r}]",
            )
        return
    if isinstance(chosen, (list, tuple)) or isinstance(rejected, (list, tuple)):
        if type(chosen) is not type(rejected) or len(chosen) != len(rejected):
            raise TypeError(f"preference arm {identifier} sequence structure mismatch")
        for index, (chosen_item, rejected_item) in enumerate(zip(chosen, rejected)):
            _validate_matching_context_structure(
                chosen_item,
                rejected_item,
                identifier=f"{identifier}[{index}]",
            )
        return
    if type(chosen) is not type(rejected):
        raise TypeError(
            f"preference arm {identifier} scalar type mismatch: "
            f"chosen={type(chosen).__name__}, rejected={type(rejected).__name__}"
        )


def _require_positive_int(value: object, identifier: str) -> None:
    if type(value) is not int:
        raise TypeError(
            f"expected positive int for {identifier}, received "
            f"{type(value).__name__}: {value!r}"
        )
    if value < 1:
        raise ValueError(f"expected {identifier} >= 1, received {value}")


__all__ = [
    "build_noised_output_state",
    "flow_matching_per_sample_loss",
    "sample_offline_timesteps",
    "validate_preference_component_times",
    "validate_preference_output_states",
]
