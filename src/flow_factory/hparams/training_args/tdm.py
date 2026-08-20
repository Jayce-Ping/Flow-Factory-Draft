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

"""Training arguments for deterministic trajectory distribution matching."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .dmd2 import DMD2TrainingArguments, _finite_float


@dataclass
class TDMTrainingArguments(DMD2TrainingArguments):
    """Configure deterministic few-step trajectory distribution matching."""

    gradient_step_per_epoch: int = field(
        default=1,
        metadata={"help": "TDM requires one generator optimizer step per rollout."},
    )
    num_inference_steps: int = 4
    trajectory_steps: int = 4
    use_huber: bool = True
    huber_c: float = 1e-3
    tdm_snr_gamma: float = 5.0
    tdm_importance_clip: float = 20.0

    def __post_init__(self) -> None:
        """Validate trajectory count, replay tolerances, and Huber controls."""
        super().__post_init__()
        if (
            not isinstance(self.trajectory_steps, int)
            or isinstance(self.trajectory_steps, bool)
            or self.trajectory_steps < 1
        ):
            raise ValueError(
                "expected train.trajectory_steps as an int >= 1, "
                f"received {self.trajectory_steps!r}"
            )
        if not isinstance(self.use_huber, bool):
            raise TypeError(
                f"expected train.use_huber as a bool, received {type(self.use_huber).__name__}: "
                f"{self.use_huber!r}"
            )
        self.huber_c = _finite_float(self.huber_c, "train.huber_c", allow_zero=False)
        self.tdm_snr_gamma = _finite_float(
            self.tdm_snr_gamma,
            "train.tdm_snr_gamma",
            allow_zero=False,
        )
        self.tdm_importance_clip = _finite_float(
            self.tdm_importance_clip,
            "train.tdm_importance_clip",
            allow_zero=False,
        )

    def get_num_train_timesteps(self, args: Any) -> int:
        """Count one accumulation unit per rollout; K boundaries are averaged.

        Args:
            args: Parent arguments object; unused because TDM owns the unit count.

        Returns:
            ``1`` so auto GAS stays one generator step per outer iteration.
        """
        del args
        return 1

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
