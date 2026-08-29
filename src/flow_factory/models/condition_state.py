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

"""Adapter-owned runtime realization of cached model-input conditions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional, Protocol, Tuple, runtime_checkable

import torch

from ..contracts import NON_MODEL_CONDITION_KEYS


@dataclass(frozen=True, slots=True)
class PreparedConditionState:
    """Bundle cached conditions with one realized model-input state.

    Args:
        condition: Cached input-only fields retained for model forward.
        forward_context: Runtime input-owned fields added to model forward.
        output_context: Runtime input-owned fields consumed only while encoding
            and binding an output target.

    Note:
        The ownership shell and outer mappings are copied and frozen. Tensor
        leaves are retained without cloning so online rollout and offline DPO can
        reuse one exact stochastic condition realization.
    """

    condition: Mapping[str, Any]
    forward_context: Mapping[str, Any]
    output_context: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Freeze mappings and reject ambiguous field ownership."""
        condition = _freeze_string_mapping(self.condition, "PreparedConditionState.condition")
        forward_context = _freeze_string_mapping(
            self.forward_context,
            "PreparedConditionState.forward_context",
        )
        output_context = _freeze_string_mapping(
            self.output_context,
            "PreparedConditionState.output_context",
        )
        _reject_non_model_keys(condition, "PreparedConditionState.condition")
        _reject_non_model_keys(forward_context, "PreparedConditionState.forward_context")

        forward_collisions = tuple(sorted(set(condition).intersection(forward_context)))
        if forward_collisions:
            raise ValueError(
                "prepared condition forward context collides with cached condition keys "
                f"{forward_collisions}; every model-forward field must have one owner"
            )
        output_collisions = tuple(sorted(set(condition).intersection(output_context)))
        if output_collisions:
            raise ValueError(
                "prepared condition output context collides with cached condition keys "
                f"{output_collisions}; every output-binding field must have one owner"
            )

        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "forward_context", forward_context)
        object.__setattr__(self, "output_context", output_context)

    @classmethod
    def identity(cls, condition: Mapping[str, Any]) -> "PreparedConditionState":
        """Create a realization that preserves the cached condition unchanged.

        Args:
            condition: Cached input-only model fields.

        Returns:
            Identity prepared condition with no runtime-owned contexts.
        """
        return cls(condition=condition, forward_context={}, output_context={})

    def model_forward_condition(self) -> Mapping[str, Any]:
        """Return the collision-free model-forward condition mapping."""
        return MappingProxyType({**self.condition, **self.forward_context})

    def output_codec_condition(self) -> Mapping[str, Any]:
        """Return cached and output-binding fields for target encoding."""
        return MappingProxyType({**self.condition, **self.output_context})


@runtime_checkable
class ConditionStatePreparer(Protocol):
    """Define adapter-owned per-request condition realization."""

    @property
    def required_components(self) -> Tuple[str, ...]:
        """Return adapter component names required while realizing conditions."""
        ...

    def prepare_condition_state(
        self,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> PreparedConditionState:
        """Prepare one exact model-input state from cached conditions."""
        ...


def validate_condition_preparer_required_components(
    preparer: object,
    available_components: Sequence[str],
) -> Tuple[str, ...]:
    """Validate preparer component requirements against an adapter runtime.

    Args:
        preparer: Structural condition-state preparer instance.
        available_components: Canonical component names exposed by the runtime.

    Returns:
        The preparer's validated required component tuple.
    """
    prepare = getattr(preparer, "prepare_condition_state", None)
    if not callable(prepare):
        raise TypeError(
            "expected condition-state preparer with callable prepare_condition_state, "
            f"received {type(preparer).__name__}"
        )
    required_components = getattr(preparer, "required_components", None)
    _validate_component_names(
        required_components,
        "condition preparer.required_components",
        allow_empty=True,
    )
    if isinstance(available_components, (str, bytes)) or not isinstance(
        available_components, Sequence
    ):
        raise TypeError(
            "expected available_components to be a sequence of strings, "
            f"received {type(available_components).__name__}: {available_components!r}"
        )
    available = tuple(available_components)
    _validate_component_names(available, "available_components", allow_empty=True)
    unknown = tuple(name for name in required_components if name not in available)
    if unknown:
        raise ValueError(
            "condition-state preparer requires unknown adapter components "
            f"{unknown}; available components={available}"
        )
    return tuple(required_components)


def _freeze_string_mapping(value: object, identifier: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"expected Mapping[str, Any] for {identifier}, received "
            f"{type(value).__name__}: {value!r}"
        )
    invalid_keys = tuple(key for key in value if not isinstance(key, str) or not key)
    if invalid_keys:
        raise TypeError(f"expected non-empty string keys for {identifier}, got {invalid_keys!r}")
    return MappingProxyType(dict(value))


def _reject_non_model_keys(value: Mapping[str, Any], identifier: str) -> None:
    rejected = tuple(sorted(set(value).intersection(NON_MODEL_CONDITION_KEYS)))
    if rejected:
        raise ValueError(
            f"{identifier} contains fields that cannot enter model forward: {rejected}"
        )


def _validate_component_names(
    value: object,
    identifier: str,
    *,
    allow_empty: bool,
) -> None:
    if type(value) is not tuple:
        raise TypeError(f"expected tuple[str, ...] for {identifier}, received {value!r}")
    if not value and not allow_empty:
        raise ValueError(f"expected {identifier} to contain at least one component")
    invalid = tuple(name for name in value if not isinstance(name, str) or not name)
    if invalid:
        raise TypeError(f"expected non-empty strings for {identifier}, received {invalid!r}")
    if len(set(value)) != len(value):
        raise ValueError(f"expected unique component names for {identifier}, received {value!r}")


__all__ = [
    "ConditionStatePreparer",
    "PreparedConditionState",
    "validate_condition_preparer_required_components",
]
