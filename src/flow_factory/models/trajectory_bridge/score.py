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

from typing import Any, Dict

import torch

from ...samples import ComponentTimes, LatentState
from ...utils.base import to_broadcast_tensor


def _require_floating_score_tensor(
    tensor: torch.Tensor,
    *,
    field: str,
    component: str,
) -> None:
    if not tensor.is_floating_point():
        raise TypeError(
            f"project_velocity_to_score_state {field} component {component!r} expected "
            f"floating dtype, received {tensor.dtype}"
        )


def validate_score_projection_state(
    adapter: Any,
    value: LatentState,
    *,
    field: str,
) -> LatentState:
    if not isinstance(value, LatentState):
        raise TypeError(
            f"project_velocity_to_score_state {field} expected LatentState, "
            f"received {type(value).__name__}"
        )
    expected_names = adapter.trajectory_component_order
    if value.component_names != expected_names:
        raise ValueError(
            f"project_velocity_to_score_state {field} expected component order "
            f"{expected_names}, received {value.component_names}"
        )
    for name in expected_names:
        _require_floating_score_tensor(
            value.components[name],
            field=field,
            component=name,
        )
    return value


def validate_score_projection_inputs(
    adapter: Any,
    state: LatentState,
    times: ComponentTimes,
    velocity: LatentState,
) -> None:
    validate_score_projection_state(adapter, state, field="state")
    validate_score_projection_state(adapter, velocity, field="velocity")
    expected_names = adapter.trajectory_component_order
    if not isinstance(times, ComponentTimes):
        raise TypeError(
            "project_velocity_to_score_state times expected ComponentTimes, "
            f"received {type(times).__name__}"
        )
    if times.sigma is None or tuple(times.sigma) != expected_names:
        received = None if times.sigma is None else tuple(times.sigma)
        raise ValueError(
            "project_velocity_to_score_state sigma expected component order "
            f"{expected_names}, received {received}"
        )
    for name in expected_names:
        _require_floating_score_tensor(
            times.sigma[name],
            field="sigma",
            component=name,
        )


def project_clean_to_score_state(
    adapter: Any,
    state: LatentState,
    times: ComponentTimes,
    clean_state: LatentState,
) -> LatentState:
    """Apply the default single-component flow-match clean-to-score projection."""
    expected_names = ("latent",)
    received = {
        "adapter": adapter.trajectory_component_order,
        "state": state.component_names,
        "clean_state": clean_state.component_names,
        "sigma": None if times.sigma is None else tuple(times.sigma),
    }
    if (
        adapter.trajectory_component_order != expected_names
        or state.component_names != expected_names
        or clean_state.component_names != expected_names
        or times.sigma is None
        or tuple(times.sigma) != expected_names
    ):
        raise ValueError(
            f"expected exactly component order {expected_names} for the default "
            f"project_clean_to_score_state, received {received}; structured adapters must "
            "override the protected hook for their component schedules"
        )
    return project_flow_match_clean_to_score_state(adapter, state, times, clean_state)


def project_flow_match_clean_to_score_state(
    adapter: Any,
    state: LatentState,
    times: ComponentTimes,
    clean_state: LatentState,
) -> LatentState:
    """Project clean predictions with each declared flow-match component schedule."""
    expected_names = adapter.trajectory_component_order
    received = {
        "state": state.component_names,
        "clean_state": clean_state.component_names,
        "sigma": None if times.sigma is None else tuple(times.sigma),
    }
    if (
        state.component_names != expected_names
        or clean_state.component_names != expected_names
        or times.sigma is None
        or tuple(times.sigma) != expected_names
    ):
        raise ValueError(
            "expected flow-match clean-to-score component order "
            f"{expected_names}, received {received} from {type(adapter).__name__}"
        )
    projected: Dict[str, torch.Tensor] = {}
    for name in expected_names:
        component_state = state.components[name]
        component_clean = clean_state.components[name]
        if component_state.shape != component_clean.shape:
            raise ValueError(
                f"expected clean_state component {name!r} to match state shape "
                f"{tuple(component_state.shape)}, received {tuple(component_clean.shape)}"
            )
        if component_state.device != component_clean.device:
            raise ValueError(
                f"expected clean_state component {name!r} on state device "
                f"{component_state.device}, received {component_clean.device}"
            )
        sigma = times.sigma[name]
        batch_size = component_state.shape[0]
        if sigma.ndim > 1 or sigma.numel() not in (1, batch_size):
            raise ValueError(
                f"expected sigma for component {name!r} to hold one value per sample with "
                f"shape ({batch_size},), received {tuple(sigma.shape)}"
            )
        zero_samples = torch.nonzero(sigma.reshape(-1) == 0, as_tuple=False).flatten().tolist()
        if zero_samples:
            raise ValueError(
                "project_velocity_to_score_state expected non-zero sigma for component "
                f"{name!r}, received zero at sample indices {zero_samples} with sigma "
                f"{sigma.tolist()}"
            )
        compute_dtype = torch.promote_types(
            torch.promote_types(component_state.dtype, component_clean.dtype),
            torch.float32,
        )
        compute_state = component_state.to(dtype=compute_dtype)
        compute_clean = component_clean.to(dtype=compute_dtype)
        sigma = to_broadcast_tensor(sigma, compute_state)
        alpha = 1 - sigma
        projected[name] = (alpha * compute_clean - compute_state) / sigma.square()
    return LatentState(projected, active_masks=state.active_masks)


def validate_projected_score_state(
    adapter: Any,
    state: LatentState,
    score_state: LatentState,
) -> LatentState:
    validate_score_projection_state(adapter, score_state, field="score_state")
    expected_names = adapter.trajectory_component_order
    for name in expected_names:
        expected = state.components[name]
        received = score_state.components[name]
        if received.shape != expected.shape or received.device != expected.device:
            raise ValueError(
                f"expected projected score component {name!r} with shape/device "
                f"({tuple(expected.shape)}, {expected.device}), received "
                f"({tuple(received.shape)}, {received.device}) from {type(adapter).__name__}"
            )
    expected_masks = state.active_masks
    received_masks = score_state.active_masks
    if (expected_masks is None) != (received_masks is None) or (
        expected_masks is not None
        and (
            tuple(received_masks) != tuple(expected_masks)
            or any(received_masks[name] is not expected_masks[name] for name in expected_names)
        )
    ):
        raise ValueError(
            "expected project_clean_to_score_state to preserve the input active_masks tensors, "
            f"received different masks from {type(adapter).__name__}"
        )
    return score_state
