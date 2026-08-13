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

from typing import Any, Dict, Mapping, Optional, Union

import torch

from ...samples import (
    LatentState,
)
from ..latent_geometry import LatentAxes
from .masks import _active_element_counts, _expand_active_mask


def resolve_component_latent_axes(
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


def validate_reduction_inputs(
    adapter: Any,
    values: Mapping[str, torch.Tensor],
    state: Optional[LatentState],
) -> int:
    """Validate the shared reduction input contract and return the batch size."""
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
    first = values[expected_names[0]]
    if not isinstance(first, torch.Tensor):
        raise TypeError(
            f"expected batched torch.Tensor for values[{expected_names[0]!r}], "
            f"received {type(first).__name__}"
        )
    if first.ndim < 1:
        raise ValueError(
            f"expected values[{expected_names[0]!r}] to be a batched torch.Tensor with a "
            f"leading batch dimension, received shape {tuple(first.shape)}"
        )
    batch_size = first.shape[0]
    if state is not None:
        if not isinstance(state, LatentState):
            raise TypeError(
                f"expected LatentState or None for state, received {type(state).__name__}"
            )
        if state.component_names != expected_names:
            raise ValueError(
                f"expected state component order {expected_names}, "
                f"received {state.component_names}"
            )
        for name, component_state in state.components.items():
            if component_state.ndim < 1 or component_state.shape[0] != batch_size:
                raise ValueError(
                    f"expected state[{name!r}] to use the values batch size {batch_size}, "
                    f"received shape {tuple(component_state.shape)}"
                )
    return batch_size


def validate_reduced_component_values(
    adapter: Any,
    reduced: Mapping[str, torch.Tensor],
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    """Validate a per-component reduction result before any trainer consumes it."""
    identifier = f"reduce_component_latent_values on {type(adapter).__name__}"
    if not isinstance(reduced, Mapping):
        raise TypeError(
            f"expected {identifier} to return Mapping[str, torch.Tensor], "
            f"received {type(reduced).__name__}"
        )
    expected_names = adapter.trajectory_component_order
    if tuple(reduced) != expected_names:
        raise ValueError(
            f"expected {identifier} to return component order {expected_names}, "
            f"received {tuple(reduced)}"
        )
    expected_device: Optional[torch.device] = None
    expected_dtype: Optional[torch.dtype] = None
    for name in expected_names:
        component_values = reduced[name]
        if not isinstance(component_values, torch.Tensor):
            raise TypeError(
                f"expected {identifier} result [{name!r}] to be a torch.Tensor, "
                f"received {type(component_values).__name__}"
            )
        if component_values.shape != (batch_size,):
            raise ValueError(
                f"expected {identifier} result [{name!r}] shape {(batch_size,)}, "
                f"received {tuple(component_values.shape)}"
            )
        if expected_device is None:
            expected_device = component_values.device
            expected_dtype = component_values.dtype
        elif component_values.device != expected_device or component_values.dtype != expected_dtype:
            raise ValueError(
                f"expected {identifier} result [{name!r}] to match component "
                f"{expected_names[0]!r} ({expected_device}, {expected_dtype}), received "
                f"({component_values.device}, {component_values.dtype})"
            )
    return dict(reduced)


def validate_reduced_latent_values(
    adapter: Any,
    reduced: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Validate a global reduction result before any trainer consumes it."""
    identifier = f"reduce_latent_values on {type(adapter).__name__}"
    if not isinstance(reduced, torch.Tensor):
        raise TypeError(
            f"expected {identifier} to return a torch.Tensor, received {type(reduced).__name__}"
        )
    if reduced.shape != (batch_size,):
        raise ValueError(
            f"expected {identifier} to return one scalar per sample with shape "
            f"{(batch_size,)}, received {tuple(reduced.shape)}"
        )
    return reduced


def default_reduce_component_latent_values(
    adapter: Any,
    values: Mapping[str, torch.Tensor],
    *,
    state: Optional[LatentState],
) -> Dict[str, torch.Tensor]:
    active_masks = None if state is None else state.active_masks
    reduced: Dict[str, torch.Tensor] = {}
    for name in adapter.trajectory_component_order:
        component_values = values[name]
        if not isinstance(component_values, torch.Tensor) or component_values.ndim < 1:
            raise ValueError(
                f"expected values[{name!r}] to be a batched torch.Tensor with a leading "
                f"batch dimension, received shape "
                f"{tuple(getattr(component_values, 'shape', ()))}"
            )
        batch_size = component_values.shape[0]
        if active_masks is None:
            reduced[name] = component_values.reshape(batch_size, -1).mean(dim=1)
            continue
        mask = _expand_active_mask(
            active_masks[name],
            component_values,
            component=name,
            operation="reduce_component_latent_values",
        )
        counts = _active_element_counts(mask, component=name)
        active_sum = (
            torch.where(mask, component_values, torch.zeros_like(component_values))
            .reshape(batch_size, -1)
            .sum(dim=1)
        )
        reduced[name] = active_sum / counts.to(active_sum.dtype)
    return reduced


def default_reduce_latent_values(
    adapter: Any,
    values: Mapping[str, torch.Tensor],
    *,
    active_numel: Optional[Mapping[str, int]],
    state: Optional[LatentState],
) -> torch.Tensor:
    expected_names = adapter.trajectory_component_order
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

    active_masks = None if state is None else state.active_masks
    weighted_sum: Optional[torch.Tensor] = None
    total_weight: Union[int, torch.Tensor] = 0
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
            component_weight: Union[int, torch.Tensor] = override
        elif active_masks is not None:
            mask = _expand_active_mask(
                active_masks[name],
                component_values,
                component=name,
                operation="reduce_latent_values",
            )
            component_weight = _active_element_counts(mask, component=name)
            component_sum = (
                torch.where(mask, component_values, torch.zeros_like(component_values))
                .reshape(batch_size, -1)
                .sum(dim=1)
            )
        else:
            flattened = component_values.reshape(batch_size, -1)
            component_sum = flattened.sum(dim=1)
            component_weight = flattened.shape[1]
        if not isinstance(component_weight, torch.Tensor) and component_weight <= 0:
            raise ValueError(
                f"expected positive element weight for component {name!r}, "
                f"received {component_weight}"
            )
        weighted_sum = component_sum if weighted_sum is None else weighted_sum + component_sum
        total_weight = total_weight + component_weight
    if isinstance(total_weight, torch.Tensor):
        total_weight = total_weight.to(weighted_sum.dtype)
    elif total_weight <= 0:
        raise ValueError(f"expected positive total latent weight, received {total_weight}")
    return weighted_sum / total_weight
