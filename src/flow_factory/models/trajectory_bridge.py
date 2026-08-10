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

from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from ..samples import (
    ComponentTimes,
    LatentState,
    MultiModalStepOutput,
    NoisedState,
    ReplayStep,
    StackedSampleBatch,
    StructuredTrajectory,
)
from ..scheduler import SDESchedulerOutput
from ..utils.base import filter_kwargs
from .latent_geometry import LatentAxes

_STORAGE_KEYS = {
    "trajectory",
    "timesteps",
    "all_latents",
    "latent_index_map",
    "log_probs",
    "log_prob_index_map",
}
_STATE_OWNED_FORWARD_KEYS = {
    "t",
    "t_next",
    "latents",
    "next_latents",
    "compute_log_prob",
    "return_kwargs",
    "noise_level",
}
_SIGNED_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _require_legacy_tensors(
    batch: StackedSampleBatch,
    names: Tuple[str, ...],
    *,
    operation: str,
    step_index: Optional[int] = None,
) -> None:
    missing = tuple(name for name in names if batch.get(name) is None)
    context = f" at step_index={step_index}" if step_index is not None else ""
    if missing:
        raise ValueError(
            f"expected legacy fields {names} for {operation}{context}, received keys "
            f"{tuple(sorted(batch.keys()))} with missing values {missing}"
        )
    received_types = {
        name: type(batch[name]).__name__
        for name in names
        if not isinstance(batch[name], torch.Tensor)
    }
    if received_types:
        raise TypeError(
            f"expected torch.Tensor legacy fields {names} for {operation}{context}, "
            f"received types {received_types}"
        )


def _map_position(
    index_map: torch.Tensor,
    map_name: str,
    rollout_position: int,
    *,
    upper_bound: int,
) -> int:
    if index_map.ndim != 1 or index_map.dtype not in _SIGNED_INTEGER_DTYPES:
        raise TypeError(
            f"expected signed integer {map_name} with shape (T,), received "
            f"dtype {index_map.dtype} and shape {tuple(index_map.shape)}"
        )
    if rollout_position >= index_map.shape[0]:
        raise ValueError(
            f"expected rollout position below {index_map.shape[0]} for {map_name}, "
            f"received {rollout_position} with map contents {index_map.tolist()}"
        )
    stored_position = int(index_map[rollout_position].item())
    if stored_position == -1:
        raise ValueError(
            f"{map_name} at rollout position {rollout_position} received uncollected "
            f"sentinel -1; map contents {index_map.tolist()}"
        )
    if stored_position < 0 or stored_position >= upper_bound:
        raise ValueError(
            f"{map_name} at rollout position {rollout_position} expected stored index in "
            f"[0, {upper_bound - 1}], received {stored_position}; map contents "
            f"{index_map.tolist()}"
        )
    return stored_position


def _trajectory_value_at(values: torch.Tensor, position: int, *, identifier: str) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(
            f"expected batched {identifier} shape (B, T), received {tuple(values.shape)}"
        )
    if position >= values.shape[1]:
        raise ValueError(
            f"expected {identifier} position below {values.shape[1]}, received {position}"
        )
    return values[:, position]


def _legacy_next_timestep(timesteps: torch.Tensor, step_index: int) -> torch.Tensor:
    """Return the stored next timestep, or the terminal ``0`` the legacy replay used.

    Adapters store one timestep per denoising step ``(B, T)`` while latents keep
    ``T + 1`` rollout positions, so the final transition has no stored ``t_next``.
    """
    if timesteps.ndim != 2:
        raise ValueError(
            f"expected batched legacy timesteps shape (B, T), received {tuple(timesteps.shape)}"
        )
    if step_index + 1 >= timesteps.shape[1]:
        return torch.tensor(0, device=timesteps.device)
    return timesteps[:, step_index + 1]


def _validate_structured_trajectory(
    adapter: Any,
    batch: StackedSampleBatch,
    trajectory: StructuredTrajectory,
) -> None:
    if not isinstance(batch, StackedSampleBatch):
        raise TypeError(
            "expected StackedSampleBatch for structured trajectory bridge, "
            f"received {type(batch).__name__}"
        )
    expected_names = adapter.trajectory_component_order
    if trajectory.component_names != expected_names:
        raise ValueError(
            f"expected trajectory_component_order {expected_names}, received structured "
            f"component order {trajectory.component_names}"
        )
    expected_batch_size = len(batch.samples)
    for name in expected_names:
        component = trajectory.components[name]
        states_shape = tuple(component.states.shape)
        timesteps_shape = tuple(component.timesteps.shape)
        if (
            component.states.ndim < 2
            or component.timesteps.ndim != 2
            or component.states.shape[0] != expected_batch_size
            or component.timesteps.shape[0] != expected_batch_size
        ):
            raise ValueError(
                f"expected batched StructuredTrajectory component {name!r} with states "
                f"shape (B, stored, ...) and timesteps shape (B, T), received states "
                f"{states_shape} and timesteps {timesteps_shape} for batch size "
                f"{expected_batch_size}"
            )
        if component.sigmas is not None and component.sigmas.shape != component.timesteps.shape:
            raise ValueError(
                f"expected batched sigmas matching timesteps for component {name!r}, "
                f"received sigmas {tuple(component.sigmas.shape)} and timesteps "
                f"{timesteps_shape}"
            )
    if trajectory.log_probs is not None and (
        trajectory.log_probs.ndim != 2 or trajectory.log_probs.shape[0] != expected_batch_size
    ):
        raise ValueError(
            "expected batched StructuredTrajectory.log_probs shape (B, T), received "
            f"{tuple(trajectory.log_probs.shape)} for batch size {expected_batch_size}"
        )


def _get_terminal_state(adapter: Any, batch: StackedSampleBatch) -> LatentState:
    trajectory = batch.get("trajectory")
    if trajectory is not None:
        if not isinstance(trajectory, StructuredTrajectory):
            raise TypeError(
                "expected StructuredTrajectory or None for batch['trajectory'], "
                f"received {type(trajectory).__name__}"
            )
        _validate_structured_trajectory(adapter, batch, trajectory)
        terminal: Dict[str, torch.Tensor] = {}
        for name in adapter.trajectory_component_order:
            component = trajectory.components[name]
            terminal_position = _map_position(
                component.state_index_map,
                f"component {name!r} state_index_map",
                component.state_index_map.shape[0] - 1,
                upper_bound=component.states.shape[1],
            )
            terminal[name] = component.states[:, terminal_position]
        return LatentState(terminal)

    _require_legacy_tensors(
        batch,
        ("all_latents", "latent_index_map"),
        operation="get_terminal_state",
    )
    terminal_position = _map_position(
        batch["latent_index_map"],
        "latent_index_map",
        batch["latent_index_map"].shape[0] - 1,
        upper_bound=batch["all_latents"].shape[1],
    )
    return LatentState({"latent": batch["all_latents"][:, terminal_position]})


def _get_replay_step(
    adapter: Any,
    batch: StackedSampleBatch,
    step_index: int,
) -> ReplayStep:
    if not isinstance(step_index, int):
        raise TypeError(
            f"expected int step_index for get_replay_step, received "
            f"{type(step_index).__name__}: {step_index!r}"
        )
    if step_index < 0:
        raise ValueError(f"expected non-negative step_index, received {step_index}")

    trajectory = batch.get("trajectory")
    if trajectory is not None:
        if not isinstance(trajectory, StructuredTrajectory):
            raise TypeError(
                "expected StructuredTrajectory or None for batch['trajectory'], "
                f"received {type(trajectory).__name__} at step_index={step_index}"
            )
        _validate_structured_trajectory(adapter, batch, trajectory)
        state: Dict[str, torch.Tensor] = {}
        next_state: Dict[str, torch.Tensor] = {}
        timestep: Dict[str, torch.Tensor] = {}
        next_timestep: Dict[str, torch.Tensor] = {}
        sigma: Dict[str, torch.Tensor] = {}
        next_sigma: Dict[str, torch.Tensor] = {}
        sigma_presence = []
        for name in adapter.trajectory_component_order:
            component = trajectory.components[name]
            state_position = _map_position(
                component.state_index_map,
                f"component {name!r} state_index_map",
                step_index,
                upper_bound=component.states.shape[1],
            )
            next_state_position = _map_position(
                component.state_index_map,
                f"component {name!r} state_index_map",
                step_index + 1,
                upper_bound=component.states.shape[1],
            )
            state[name] = component.states[:, state_position]
            next_state[name] = component.states[:, next_state_position]
            timestep[name] = _trajectory_value_at(
                component.timesteps,
                step_index,
                identifier=f"component {name!r} timesteps",
            )
            next_timestep[name] = _trajectory_value_at(
                component.timesteps,
                step_index + 1,
                identifier=f"component {name!r} timesteps",
            )
            sigma_presence.append(component.sigmas is not None)
            if component.sigmas is not None:
                sigma[name] = _trajectory_value_at(
                    component.sigmas,
                    step_index,
                    identifier=f"component {name!r} sigmas",
                )
                next_sigma[name] = _trajectory_value_at(
                    component.sigmas,
                    step_index + 1,
                    identifier=f"component {name!r} sigmas",
                )
        if len(set(sigma_presence)) != 1:
            raise ValueError(
                "expected all structured trajectory components to provide sigmas or none, "
                f"received {dict(zip(adapter.trajectory_component_order, sigma_presence))} "
                f"at step_index={step_index}"
            )
        log_prob = None
        component_log_probs: Optional[Dict[str, torch.Tensor]] = None
        if trajectory.log_probs is not None:
            if trajectory.log_prob_index_map is None:
                log_prob_position = step_index
            else:
                log_prob_position = _map_position(
                    trajectory.log_prob_index_map,
                    "structured log_prob_index_map",
                    step_index,
                    upper_bound=trajectory.log_probs.shape[1],
                )
            log_prob = _trajectory_value_at(
                trajectory.log_probs,
                log_prob_position,
                identifier="structured log_probs",
            )
            if trajectory.component_log_probs is not None:
                component_log_probs = {
                    name: _trajectory_value_at(
                        trajectory.component_log_probs[name],
                        log_prob_position,
                        identifier=f"structured component {name!r} log_probs",
                    )
                    for name in adapter.trajectory_component_order
                }
        return ReplayStep(
            state=LatentState(state),
            next_state=LatentState(next_state),
            times=ComponentTimes(
                timestep=timestep,
                next_timestep=next_timestep,
                sigma=sigma if sigma_presence[0] else None,
                next_sigma=next_sigma if sigma_presence[0] else None,
            ),
            log_prob=log_prob,
            component_log_probs=component_log_probs,
        )

    _require_legacy_tensors(
        batch,
        ("all_latents", "latent_index_map", "timesteps"),
        operation="get_replay_step",
        step_index=step_index,
    )
    state_position = _map_position(
        batch["latent_index_map"],
        "latent_index_map",
        step_index,
        upper_bound=batch["all_latents"].shape[1],
    )
    next_state_position = _map_position(
        batch["latent_index_map"],
        "latent_index_map",
        step_index + 1,
        upper_bound=batch["all_latents"].shape[1],
    )
    log_prob = None
    if batch.get("log_probs") is not None:
        _require_legacy_tensors(
            batch,
            ("log_probs", "log_prob_index_map"),
            operation="get_replay_step log probability",
            step_index=step_index,
        )
        log_prob_position = _map_position(
            batch["log_prob_index_map"],
            "log_prob_index_map",
            step_index,
            upper_bound=batch["log_probs"].shape[1],
        )
        log_prob = _trajectory_value_at(
            batch["log_probs"],
            log_prob_position,
            identifier="legacy log_probs",
        )
    return ReplayStep(
        state=LatentState({"latent": batch["all_latents"][:, state_position]}),
        next_state=LatentState({"latent": batch["all_latents"][:, next_state_position]}),
        times=ComponentTimes(
            timestep={
                "latent": _trajectory_value_at(
                    batch["timesteps"], step_index, identifier="legacy timesteps"
                )
            },
            next_timestep={"latent": _legacy_next_timestep(batch["timesteps"], step_index)},
        ),
        log_prob=log_prob,
        component_log_probs=None if log_prob is None else {"latent": log_prob},
    )


def _shared_callback_index_map(batch: StackedSampleBatch, field: str) -> torch.Tensor:
    index_map = batch["callback_index_map"]
    if index_map.ndim == 1:
        return index_map
    if index_map.ndim != 2:
        raise ValueError(
            f"expected legacy callback_index_map shape (T,) or (B, T) for callback {field!r}, "
            f"received {tuple(index_map.shape)}"
        )
    if not bool(torch.equal(index_map, index_map[:1].expand_as(index_map))):
        raise ValueError(
            f"expected one shared legacy callback_index_map across the batch for callback "
            f"{field!r}, received per-sample maps {index_map.tolist()}"
        )
    return index_map[0]


def _get_replay_callback(
    adapter: Any,
    batch: StackedSampleBatch,
    step_index: int,
    field: str,
) -> LatentState:
    if not isinstance(step_index, int) or isinstance(step_index, bool):
        raise TypeError(
            f"expected int step_index for get_replay_callback, received "
            f"{type(step_index).__name__}: {step_index!r}"
        )
    if step_index < 0:
        raise ValueError(f"expected non-negative step_index, received {step_index}")
    if not isinstance(field, str) or not field:
        raise ValueError(
            f"expected a non-empty callback field name, received {field!r} "
            f"at step_index={step_index}"
        )

    trajectory = batch.get("trajectory")
    if trajectory is not None:
        if not isinstance(trajectory, StructuredTrajectory):
            raise TypeError(
                "expected StructuredTrajectory or None for batch['trajectory'], "
                f"received {type(trajectory).__name__} at step_index={step_index}"
            )
        _validate_structured_trajectory(adapter, batch, trajectory)
        if trajectory.callbacks is None or field not in trajectory.callbacks:
            raise ValueError(
                f"expected structured callback field {field!r} in stored fields "
                f"{trajectory.callback_fields} at step_index={step_index}"
            )
        stored = trajectory.callbacks[field]
        expected_names = adapter.trajectory_component_order
        if tuple(stored) != expected_names:
            raise ValueError(
                f"expected callback {field!r} component order {expected_names}, received "
                f"{tuple(stored)} at step_index={step_index}"
            )
        expected_batch_size = len(batch.samples)
        components: Dict[str, torch.Tensor] = {}
        for name in expected_names:
            indexed = stored[name]
            if not indexed.batched or indexed.values.shape[0] != expected_batch_size:
                raise ValueError(
                    f"expected batched callback {field!r} component {name!r} with batch size "
                    f"{expected_batch_size}, received batched={indexed.batched} and values "
                    f"shape {tuple(indexed.values.shape)}"
                )
            components[name] = indexed.at(
                step_index, identifier=f"callback {field!r} component {name!r}"
            )
        return LatentState(components)

    _require_legacy_tensors(
        batch,
        (field, "callback_index_map"),
        operation="get_replay_callback",
        step_index=step_index,
    )
    index_map = _shared_callback_index_map(batch, field)
    stored_position = _map_position(
        index_map,
        f"callback {field!r} callback_index_map",
        step_index,
        upper_bound=batch[field].shape[1],
    )
    return LatentState({"latent": batch[field][:, stored_position]})


def _get_state_active_numel(adapter: Any, state: LatentState) -> Dict[str, int]:
    if not isinstance(state, LatentState):
        raise TypeError(
            f"expected LatentState for get_state_active_numel, received {type(state).__name__}"
        )
    expected_names = adapter.trajectory_component_order
    if state.component_names != expected_names:
        raise ValueError(
            "expected state component keys/order to match trajectory_component_order "
            f"{expected_names}, received {state.component_names}"
        )
    active_numel: Dict[str, int] = {}
    for name in expected_names:
        values = state.components[name]
        if values.ndim < 2 or values.shape[0] <= 0:
            raise ValueError(
                f"expected batched state component {name!r} with shape (B, ...) and a positive "
                f"batch size, received {tuple(values.shape)}"
            )
        active_numel[name] = int(values.numel() // values.shape[0])
    return active_numel


def _get_train_step_indices(adapter: Any) -> torch.Tensor:
    group = adapter.scheduler_group
    primary_name = group.primary_name
    indices = group.primary.train_timesteps
    if not isinstance(indices, torch.Tensor) or indices.ndim != 1:
        raise TypeError(
            f"expected 1-D torch.Tensor train step indices from scheduler component "
            f"{primary_name!r}, received {type(indices).__name__}"
        )
    if indices.dtype not in _SIGNED_INTEGER_DTYPES:
        raise TypeError(
            f"expected signed integer train step indices from scheduler component "
            f"{primary_name!r}, received dtype {indices.dtype}"
        )
    for name in group.names:
        if name == primary_name:
            continue
        other = group[name].train_timesteps
        if not isinstance(other, torch.Tensor) or not torch.equal(
            other.to(indices.device), indices
        ):
            received = other.tolist() if isinstance(other, torch.Tensor) else other
            raise ValueError(
                "expected aligned scheduler-group train step indices; component "
                f"{name!r} received {received} but primary {primary_name!r} declares "
                f"{indices.tolist()}"
            )
    return indices


def _add_forward_process_noise(
    clean_state: LatentState,
    times: ComponentTimes,
    *,
    generator: Optional[torch.Generator],
) -> NoisedState:
    expected_names = ("latent",)
    if clean_state.component_names != expected_names:
        raise ValueError(
            f"expected exactly components {expected_names} for default noising, "
            f"received {clean_state.component_names}"
        )
    if times.sigma is None or tuple(times.sigma) != expected_names:
        received = None if times.sigma is None else tuple(times.sigma)
        raise ValueError(
            f"expected sigma components {expected_names} for default noising, "
            f"received {received}"
        )
    clean_latents = clean_state.components["latent"]
    sigma = times.sigma["latent"]
    if sigma.ndim > clean_latents.ndim:
        raise ValueError(
            f"expected sigma ndim <= latent ndim {clean_latents.ndim}, received sigma "
            f"shape {tuple(sigma.shape)} for latent shape {tuple(clean_latents.shape)}"
        )
    sigma = sigma.reshape(sigma.shape + (1,) * (clean_latents.ndim - sigma.ndim))
    noise = torch.randn(
        clean_latents.shape,
        dtype=clean_latents.dtype,
        device=clean_latents.device,
        generator=generator,
    )
    return NoisedState(
        state=LatentState({"latent": (1 - sigma) * clean_latents + sigma * noise}),
        target_velocity=LatentState({"latent": noise - clean_latents}),
        noise=LatentState({"latent": noise}),
    )


def _forward_state(
    adapter: Any,
    *,
    batch: StackedSampleBatch,
    state: LatentState,
    times: ComponentTimes,
    next_state: Optional[LatentState],
    compute_log_prob: bool,
    return_fields: Tuple[str, ...],
    noise_level: Optional[float],
    kwargs: Mapping[str, Any],
) -> MultiModalStepOutput:
    collisions = tuple(name for name in _STATE_OWNED_FORWARD_KEYS if name in kwargs)
    if collisions:
        raise ValueError(
            f"explicit forward_state kwargs collide with state-owned arguments {collisions}"
        )
    expected_names = ("latent",)
    received = {
        "state": state.component_names,
        "timestep": tuple(times.timestep),
        "next_timestep": tuple(times.next_timestep),
        "next_state": None if next_state is None else next_state.component_names,
    }
    if (
        state.component_names != expected_names
        or tuple(times.timestep) != expected_names
        or tuple(times.next_timestep) != expected_names
        or (next_state is not None and next_state.component_names != expected_names)
    ):
        raise ValueError(
            f"expected exactly component order {expected_names} for default forward_state, "
            f"received {received}"
        )
    forward_kwargs = {
        key: value
        for key, value in batch.items()
        if key not in _STORAGE_KEYS and key not in _STATE_OWNED_FORWARD_KEYS
    }
    forward_kwargs.update(kwargs)
    forward_kwargs = filter_kwargs(adapter.forward, **forward_kwargs)
    output = adapter.forward(
        t=times.timestep["latent"],
        t_next=times.next_timestep["latent"],
        latents=state.components["latent"],
        next_latents=None if next_state is None else next_state.components["latent"],
        compute_log_prob=compute_log_prob,
        return_kwargs=return_fields,
        noise_level=noise_level,
        **forward_kwargs,
    )
    if not isinstance(output, SDESchedulerOutput):
        raise TypeError(
            "expected adapter.forward to return SDESchedulerOutput in forward_state, "
            f"received {type(output).__name__}"
        )

    def wrap(value: Optional[torch.Tensor]) -> Optional[LatentState]:
        return None if value is None else LatentState({"latent": value})

    def wrap_statistic(value: Optional[torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        return None if value is None else {"latent": value}

    return MultiModalStepOutput(
        next_state=wrap(output.next_latents),
        next_state_mean=wrap(output.next_latents_mean),
        std_dev_t=wrap_statistic(output.std_dev_t),
        dt=wrap_statistic(output.dt),
        log_prob=output.log_prob,
        component_log_probs=wrap_statistic(output.log_prob),
        velocity=wrap(output.velocity),
    )


def _resolve_component_latent_axes(
    adapter: Any,
    component: str,
    latents: torch.Tensor,
) -> LatentAxes:
    if component != "latent":
        raise ValueError(
            "expected component 'latent' for default latent-axis resolution, "
            f"received {component!r}"
        )
    return adapter.resolve_latent_axes(latents)


def _reduce_component_latent_values(
    adapter: Any,
    values: Mapping[str, torch.Tensor],
    *,
    state: Optional[LatentState],
) -> Dict[str, torch.Tensor]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(
            f"expected non-empty Mapping[str, torch.Tensor] for values, received "
            f"{type(values).__name__}"
        )
    expected_names = adapter.trajectory_component_order
    if tuple(values) != expected_names:
        raise ValueError(
            f"expected values keys/order to match trajectory_component_order "
            f"{expected_names}, received {tuple(values)}"
        )
    if state is not None and state.component_names != expected_names:
        raise ValueError(
            f"expected state component order {expected_names}, received {state.component_names}"
        )
    reduced: Dict[str, torch.Tensor] = {}
    for name in expected_names:
        component_values = values[name]
        if not isinstance(component_values, torch.Tensor) or component_values.ndim < 1:
            raise ValueError(
                f"expected values[{name!r}] to be a batched torch.Tensor with a leading "
                f"batch dimension, received shape "
                f"{tuple(getattr(component_values, 'shape', ()))}"
            )
        batch_size = component_values.shape[0]
        reduced[name] = component_values.reshape(batch_size, -1).mean(dim=1)
    return reduced


def _reduce_latent_values(
    adapter: Any,
    values: Mapping[str, torch.Tensor],
    *,
    active_numel: Optional[Mapping[str, int]],
) -> torch.Tensor:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(
            f"expected non-empty Mapping[str, torch.Tensor] for values, received "
            f"{type(values).__name__}"
        )
    expected_names = adapter.trajectory_component_order
    received_names = tuple(values)
    if received_names != expected_names:
        raise ValueError(
            f"expected values keys/order to match trajectory_component_order "
            f"{expected_names}, received {received_names}"
        )
    if active_numel is not None:
        if not isinstance(active_numel, Mapping):
            raise TypeError(
                f"expected Mapping[str, int] or None for active_numel, "
                f"received {type(active_numel).__name__}"
            )
        unknown_active_names = tuple(name for name in active_numel if name not in expected_names)
        if unknown_active_names:
            raise ValueError(
                f"active_numel received unknown keys {unknown_active_names}; expected a subset "
                f"of trajectory_component_order {expected_names}"
            )

    first = values[expected_names[0]]
    if not isinstance(first, torch.Tensor) or first.ndim < 1:
        raise TypeError(
            f"expected batched torch.Tensor for values[{expected_names[0]!r}], "
            f"received {type(first).__name__}"
        )
    batch_size = first.shape[0]
    expected_device = first.device
    expected_dtype = first.dtype
    if len(expected_names) == 1 and first.ndim == 1:
        override = None if active_numel is None else active_numel.get(expected_names[0])
        if override is not None:
            if not isinstance(override, int) or isinstance(override, bool) or override <= 0:
                raise ValueError(
                    f"expected active_numel[{expected_names[0]!r}] to be a positive int, "
                    f"received {override!r}"
                )
        return first

    weighted_sum: Optional[torch.Tensor] = None
    total_weight = 0
    for name in expected_names:
        component_values = values[name]
        if not isinstance(component_values, torch.Tensor) or component_values.ndim < 1:
            raise TypeError(
                f"expected batched torch.Tensor for values[{name!r}], "
                f"received {type(component_values).__name__}"
            )
        if component_values.shape[0] != batch_size:
            raise ValueError(
                f"expected batch size {batch_size} for values[{name!r}], received "
                f"shape {tuple(component_values.shape)}"
            )
        if component_values.device != expected_device or component_values.dtype != expected_dtype:
            raise ValueError(
                f"expected compatible dtype/device from component "
                f"{expected_names[0]!r} ({expected_device}, {expected_dtype}), received "
                f"{name!r} ({component_values.device}, {component_values.dtype})"
            )
        override = None if active_numel is None else active_numel.get(name)
        if override is not None:
            if not isinstance(override, int) or isinstance(override, bool) or override <= 0:
                raise ValueError(
                    f"expected active_numel[{name!r}] to be a positive int, "
                    f"received {override!r}"
                )
            if component_values.ndim != 1:
                raise ValueError(
                    f"expected already-reduced values[{name!r}] shape (B,) when "
                    f"active_numel is provided, received {tuple(component_values.shape)}"
                )
            component_sum = component_values * override
            component_weight = override
        else:
            flattened = component_values.reshape(batch_size, -1)
            component_sum = flattened.sum(dim=1)
            component_weight = flattened.shape[1]
        if component_weight <= 0:
            raise ValueError(
                f"expected positive element weight for component {name!r}, "
                f"received {component_weight}"
            )
        weighted_sum = component_sum if weighted_sum is None else weighted_sum + component_sum
        total_weight += component_weight
    if total_weight <= 0:
        raise ValueError(f"expected positive total latent weight, received {total_weight}")
    return weighted_sum / total_weight
