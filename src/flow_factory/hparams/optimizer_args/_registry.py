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

"""Optimizer arguments registry and lookup."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Type

from ._base import OptimizerArguments
from .adamw import AdamWOptimizerArguments
from .muon import MuonOptimizerArguments

_OPTIMIZER_ARGS_REGISTRY: Dict[str, Type[OptimizerArguments]] = {
    "adamw": AdamWOptimizerArguments,
    "muon": MuonOptimizerArguments,
}


def get_optimizer_args_class(identifier: str) -> Type[OptimizerArguments]:
    """Resolve the arguments subclass for an optimizer key.

    Args:
        identifier: Optimizer key, for example ``adamw`` or ``muon``.

    Returns:
        The matching arguments subclass.

    Raises:
        ValueError: If the key is not registered.
    """
    key = identifier.lower()
    if key not in _OPTIMIZER_ARGS_REGISTRY:
        raise ValueError(
            f"expected optimizer in {sorted(_OPTIMIZER_ARGS_REGISTRY)}, received {identifier!r}"
        )
    return _OPTIMIZER_ARGS_REGISTRY[key]


def build_optimizer_args(config: Mapping[str, Any]) -> OptimizerArguments:
    """Build one optimizer configuration, dispatching on its ``optimizer`` key.

    Args:
        config: Raw mapping from one entry of the YAML ``optimizers`` list.

    Returns:
        An instance of the arguments subclass the key selects.
    """
    if not isinstance(config, Mapping):
        raise TypeError(
            "expected a mapping for one optimizer configuration, received "
            f"{type(config).__name__}: {config!r}"
        )
    return get_optimizer_args_class(str(config.get("optimizer", "adamw"))).from_dict(dict(config))


def register_optimizer_args(identifier: str, args_class: Type[OptimizerArguments]) -> None:
    """Register an arguments subclass for a custom optimizer key.

    Args:
        identifier: Optimizer key used in YAML.
        args_class: ``OptimizerArguments`` subclass carrying its hyperparameters.
    """
    if not issubclass(args_class, OptimizerArguments):
        raise TypeError(
            "expected an OptimizerArguments subclass for optimizer "
            f"{identifier!r}, received {args_class!r}"
        )
    _OPTIMIZER_ARGS_REGISTRY[identifier.lower()] = args_class
