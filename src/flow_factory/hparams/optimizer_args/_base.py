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

"""Optimizer arguments shared by every optimizer type."""

from dataclasses import dataclass, field
from typing import Mapping

from ..abc import ArgABC


@dataclass
class OptimizerArguments(ArgABC):
    """Configure one optimizer, addressed by the name of what it optimizes.

    Every optimizer type shares these fields; type-specific hyperparameters live on
    the subclass selected by ``optimizer`` (see ``optimizer_args/_registry.py``).

    Attributes:
        name: Trainable variant this configures. A single-policy run leaves the
            default, and a multi-variant run names one config per variant.
        optimizer: Registry key selecting both the argument subclass and the
            optimizer implementation.
        learning_rate: Base learning rate.
        weight_decay: Decoupled weight decay.
        max_grad_norm: Gradient-norm clip applied before this optimizer steps.
        update_frequency: Optimizer steps of other variants between this one's
            steps. ``1`` means it steps every round.
    """

    name: str = "default"
    optimizer: str = "adamw"
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    update_frequency: int = 1

    def __post_init__(self) -> None:
        """Validate the fields every optimizer type shares."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"expected a non-empty string optimizer name, received {self.name!r}")
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.max_grad_norm = float(self.max_grad_norm)
        if self.learning_rate <= 0:
            raise ValueError(
                f"expected a positive learning_rate for optimizer {self.name!r}, "
                f"received {self.learning_rate}"
            )
        if self.weight_decay < 0:
            raise ValueError(
                f"expected a non-negative weight_decay for optimizer {self.name!r}, "
                f"received {self.weight_decay}"
            )
        if self.max_grad_norm <= 0:
            raise ValueError(
                f"expected a positive max_grad_norm for optimizer {self.name!r}, "
                f"received {self.max_grad_norm}"
            )
        if not isinstance(self.update_frequency, int) or isinstance(self.update_frequency, bool):
            raise TypeError(
                f"expected int update_frequency for optimizer {self.name!r}, received "
                f"{type(self.update_frequency).__name__}: {self.update_frequency!r}"
            )
        if self.update_frequency < 1:
            raise ValueError(
                f"expected update_frequency of at least 1 for optimizer {self.name!r}, "
                f"received {self.update_frequency}"
            )


@dataclass
class MultiOptimizerArguments(ArgABC):
    """Hold one optimizer configuration per trainable variant.

    Mirrors ``MultiRewardArguments``: a list that also supports lookup by name, so a
    trainer can ask for the config belonging to one variant without knowing the
    order the user wrote them in.

    YAML Configuration Example:
        ```yaml
        optimizers:
          - name: "generator"
            optimizer: "muon"
            learning_rate: 2.0e-5

          - name: "fake"
            optimizer: "adamw"
            learning_rate: 1.0e-5
            update_frequency: 5
        ```
    """

    optimizer_configs: list = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject duplicate names, which would make lookup ambiguous."""
        names = [config.name for config in self.optimizer_configs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"expected unique optimizer names, received duplicates {duplicates} in {names}"
            )

    @classmethod
    def from_dict(cls, args_input):
        """Build the container from a YAML list or a single-optimizer mapping.

        Each entry dispatches on its ``optimizer`` key to the matching arguments
        subclass, so AdamW and Muon hyperparameters never share one class.

        Args:
            args_input: List of optimizer configurations, or one mapping.

        Returns:
            A populated container.

        Raises:
            TypeError: If the input is neither a list nor a mapping.
        """
        from ._registry import build_optimizer_args

        if isinstance(args_input, list):
            return cls(optimizer_configs=[build_optimizer_args(entry) for entry in args_input])
        if isinstance(args_input, Mapping):
            return cls(optimizer_configs=[build_optimizer_args(args_input)])
        raise TypeError(
            "expected a list of optimizer configurations or a single mapping, received "
            f"{type(args_input).__name__}: {args_input!r}"
        )

    def get_by_name(self, name: str):
        """Return the configuration for one variant, or ``None`` when absent.

        Args:
            name: Variant name to look up.

        Returns:
            The matching configuration, or ``None``.
        """
        for config in self.optimizer_configs:
            if config.name == name:
                return config
        return None

    def __iter__(self):
        """Iterate configurations in declaration order."""
        return iter(self.optimizer_configs)

    def __len__(self) -> int:
        """Return the number of configurations."""
        return len(self.optimizer_configs)

    def __getitem__(self, index):
        """Return one configuration by position."""
        return self.optimizer_configs[index]
