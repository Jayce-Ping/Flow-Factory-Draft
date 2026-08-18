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

"""Training arguments shared by distribution-matching distillation trainers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Tuple, cast

from ..optimizer_args import AdamWOptimizerArguments
from ._base import TrainingArguments

if TYPE_CHECKING:
    from ...models.roles import RoleName


def _finite_float(value: object, field_name: str, *, allow_zero: bool) -> float:
    """Convert and validate one finite optimizer scalar."""
    if isinstance(value, bool):
        raise TypeError(
            f"expected numeric {field_name}, received {type(value).__name__}: {value!r}"
        )
    try:
        converted = float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"expected numeric {field_name}, received {type(value).__name__}: {value!r}"
        ) from error
    if not math.isfinite(converted) or converted < 0 or (converted == 0 and not allow_zero):
        comparator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"expected finite {field_name} {comparator}, received {value!r}")
    return converted


@dataclass
class DMD2TrainingArguments(TrainingArguments):
    """Configure data-free DMD2 distribution matching."""

    gradient_step_per_epoch: int = field(
        default=1,
        metadata={"help": "DMD2 requires one generator optimizer step per rollout."},
    )
    ttur_fake_updates: int = 5
    perturbation_timestep_range: Tuple[float, float] = (0.02, 0.98)
    # The boundary replay re-runs the generator forward with gradients enabled, and
    # autocast is free to pick different kernels than the no-grad rollout did. Under bf16
    # that lands a few ULPs apart -- 9.8e-04 measured on 8 GPUs -- which no tolerance
    # below bf16 resolution can accept, so the window is configurable rather than fixed.
    replay_rtol: float = 1e-4
    replay_atol: float = 1e-4

    def __post_init__(self) -> None:
        """Validate DMD2 controls."""
        super().__post_init__()
        self.replay_rtol = self._validate_replay_tolerance(self.replay_rtol, "train.replay_rtol")
        self.replay_atol = self._validate_replay_tolerance(self.replay_atol, "train.replay_atol")
        if (
            not isinstance(self.ttur_fake_updates, int)
            or isinstance(self.ttur_fake_updates, bool)
            or self.ttur_fake_updates < 1
        ):
            raise ValueError(
                "expected train.ttur_fake_updates as an int >= 1, "
                f"received {self.ttur_fake_updates!r}"
            )
        if (
            not isinstance(self.perturbation_timestep_range, (tuple, list))
            or len(self.perturbation_timestep_range) != 2
        ):
            raise TypeError(
                "expected train.perturbation_timestep_range as a two-item tuple or list, "
                f"received {type(self.perturbation_timestep_range).__name__}: "
                f"{self.perturbation_timestep_range!r}"
            )
        lower, upper = (
            _finite_float(
                self.perturbation_timestep_range[0],
                "train.perturbation_timestep_range[0]",
                allow_zero=True,
            ),
            _finite_float(
                self.perturbation_timestep_range[1],
                "train.perturbation_timestep_range[1]",
                allow_zero=True,
            ),
        )
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError(
                "expected train.perturbation_timestep_range to satisfy "
                f"0 <= lower < upper <= 1, received {(lower, upper)!r}"
            )
        self.perturbation_timestep_range = (lower, upper)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> DMD2TrainingArguments:
        """Build strict algorithm arguments from a mapping.

        Args:
            values: User-provided training field mapping.

        Returns:
            Parsed DMD2 training arguments.
        """
        if not isinstance(values, Mapping):
            raise TypeError(
                "expected training arguments as a mapping, "
                f"received {type(values).__name__}: {values!r}"
            )
        expected_fields = {config_field.name for config_field in fields(cls) if config_field.init}
        unknown_fields = set(values) - expected_fields
        retired_fields = unknown_fields & {
            "dfake_gen_update_ratio",
            "fake_updates_per_generator",
            "dm_loss_type",
            "dm_step_scale",
            "pseudo_huber_c_scale",
        }
        if retired_fields:
            raise ValueError(
                f"{cls.__name__} retired field(s) {tuple(sorted(retired_fields))!r}; "
                "use train.ttur_fake_updates for the fake-first TTUR count"
            )
        if unknown_fields:
            raise ValueError(
                f"unknown {cls.__name__} field(s) {tuple(sorted(unknown_fields))!r}; "
                f"expected {tuple(sorted(expected_fields))!r}"
            )
        explicit_extras = values.get("extra_kwargs")
        if explicit_extras:
            raise ValueError(
                f"{cls.__name__} does not accept extra_kwargs; received {explicit_extras!r}"
            )
        return cls(**dict(values))

    def role_update_plan(self) -> "RoleUpdatePlan":
        """Return fake TTUR phases followed by one generator phase."""
        # Constraint #22(c): trainers.role_optimization imports training args.
        from ...trainers.role_optimization import RolePhase, RoleUpdatePlan

        return RoleUpdatePlan(
            phases=(
                RolePhase("fake", repeats=self.ttur_fake_updates),
                RolePhase("generator"),
            )
        )

    @property
    def required_trainable_roles(self) -> Tuple[RoleName, ...]:
        """Return trainable roles in canonical materialization order."""
        return ("generator", "fake")

    @property
    def requires_ref_model(self) -> bool:
        """Require the frozen pretrained score reference."""
        return True

    @staticmethod
    def _validate_replay_tolerance(value: object, field_name: str) -> float:
        """Convert and validate one non-negative finite replay tolerance."""
        if isinstance(value, bool):
            raise TypeError(
                f"expected numeric {field_name}, received {type(value).__name__}: {value!r}"
            )
        try:
            converted = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"expected numeric {field_name}, received {type(value).__name__}: {value!r}"
            ) from error
        if not math.isfinite(converted) or converted < 0:
            raise ValueError(f"expected finite {field_name} >= 0, received {value!r}")
        return converted


DMD2_DEFAULT_OPTIMIZERS: Tuple[AdamWOptimizerArguments, ...] = (
    AdamWOptimizerArguments(name="generator", learning_rate=1e-5),
    AdamWOptimizerArguments(name="fake", learning_rate=1e-5),
)
"""Per-role defaults used when a config file declares no ``optimizers`` list.

These are the algorithm's own numbers, so they live beside its training arguments
rather than in the framework. A config file that declares ``optimizers`` overrides
them entirely, which is also how a run selects Muon for a role.
"""
