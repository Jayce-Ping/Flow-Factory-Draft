# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training arguments for TDM-R1 fake-surrogate-generator distillation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, Tuple

from ..optimizer_args import AdamWOptimizerArguments
from .dmd2 import _finite_float
from .tdm import TDMTrainingArguments

if TYPE_CHECKING:
    from ...models.roles import RoleName
    from ...trainers.role_optimization import RoleUpdatePlan


@dataclass
class TDMR1TrainingArguments(TDMTrainingArguments):
    """Configure TDM-R1 with a learned surrogate and frozen reference."""

    advantage_aggregation: Literal["sum", "gdpo"] = "gdpo"
    tdm_weight: float = 0.3
    surrogate_preference_beta: float = 1.0
    advantage_clip_range: float = 5.0
    use_huber: bool = False

    def __post_init__(self) -> None:
        """Validate reward and generator objective controls."""
        super().__post_init__()
        if self.advantage_aggregation not in ("sum", "gdpo"):
            raise ValueError(
                "expected train.advantage_aggregation in ('sum', 'gdpo'), "
                f"received {self.advantage_aggregation!r}"
            )
        self.tdm_weight = _finite_float(self.tdm_weight, "train.tdm_weight", allow_zero=False)
        self.surrogate_preference_beta = _finite_float(
            self.surrogate_preference_beta,
            "train.surrogate_preference_beta",
            allow_zero=False,
        )
        self.advantage_clip_range = _finite_float(
            self.advantage_clip_range,
            "train.advantage_clip_range",
            allow_zero=False,
        )

    def role_update_plan(self) -> RoleUpdatePlan:
        """Return fake TTUR phases, then one surrogate and one generator phase."""
        # Constraint #22(c): trainers.role_optimization imports training args.
        from ...trainers.role_optimization import RolePhase, RoleUpdatePlan

        return RoleUpdatePlan(
            phases=(
                RolePhase("fake", repeats=self.ttur_fake_updates),
                RolePhase("surrogate"),
                RolePhase("generator"),
            )
        )

    @property
    def required_trainable_roles(self) -> Tuple[RoleName, ...]:
        """Return trainable roles in canonical materialization order."""
        return ("generator", "fake", "surrogate")


TDM_R1_DEFAULT_OPTIMIZERS: Tuple[AdamWOptimizerArguments, ...] = (
    AdamWOptimizerArguments(name="generator", learning_rate=7.5e-5, betas=(0.0, 0.999)),
    AdamWOptimizerArguments(name="fake", learning_rate=3e-4, betas=(0.0, 0.999)),
    AdamWOptimizerArguments(name="surrogate", learning_rate=3e-4, betas=(0.9, 0.999)),
)
"""Per-role defaults used when a config file declares no ``optimizers`` list.

The generator and fake score run with a zeroed first moment, following TDM-R1;
the surrogate keeps the usual AdamW momentum.
"""
