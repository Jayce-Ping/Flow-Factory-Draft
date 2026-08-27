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

"""Training arguments shared by finite offline dataset algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Tuple, Union

from ...contracts.execution import OFFLINE_EXECUTION_CONTRACT, ExecutionContract
from ._base import TrainingArguments, _standardize_timestep_range


@dataclass
class OfflineTrainingArguments(TrainingArguments):
    """Configure one or more complete finite dataloader traversals."""

    execution_contract: ClassVar[ExecutionContract] = OFFLINE_EXECUTION_CONTRACT

    trainer_type: str = field(
        default="offline",
        metadata={"help": "Offline trainer identifier."},
    )
    max_epochs: int | None = field(
        default=1,
        metadata={
            "help": (
                "Maximum number of complete offline dataloader traversals. "
                "Each clean traversal is one data epoch."
            )
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": (
                "Explicit number of offline dataloader batches per optimizer update. "
                "Must be an integer >= 1; automatic online geometry derivation is unsupported."
            )
        },
    )
    num_batches_per_epoch: int = field(init=False, default=0)


@dataclass
class OfflineFlowMatchingTrainingArguments(OfflineTrainingArguments):
    """Configure flow-matching losses over finite target-media datasets.

    ``num_train_timesteps`` controls how many independently sampled loss terms
    are averaged inside one dataloader-batch microstep. It never changes
    gradient accumulation or the definition of a data epoch.
    """

    weighting_scheme: Literal["logit_normal", "uniform"] = field(
        default="logit_normal",
        metadata={"help": "Timestep sampling distribution for offline flow matching."},
    )
    logit_mean: float = field(
        default=0.0,
        metadata={"help": "Mean for logit-normal timestep sampling."},
    )
    logit_std: float = field(
        default=1.0,
        metadata={"help": "Standard deviation for logit-normal timestep sampling."},
    )
    num_train_timesteps: int = field(
        default=1,
        metadata={
            "help": (
                "Training timesteps averaged within each dataloader-batch microstep. "
                "Non-positive values derive the count from num_inference_steps and "
                "timestep_range; this value never multiplies gradient accumulation."
            )
        },
    )
    time_shift: float = field(
        default=1.0,
        metadata={"help": "Time shift for logit-normal timestep sampling."},
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.99,
        metadata={
            "help": ("Fractional denoising-axis range used for offline flow-matching training.")
        },
    )

    def __post_init__(self) -> None:
        """Normalize the shared offline flow-matching timestep controls."""
        super().__post_init__()
        self.timestep_range = _standardize_timestep_range(self.timestep_range)
        if not self.num_train_timesteps or self.num_train_timesteps <= 0:
            self.num_train_timesteps = max(
                1,
                int(self.num_inference_steps * (self.timestep_range[1] - self.timestep_range[0])),
            )

    def get_num_train_timesteps(self, args: Any) -> int:
        """Return the number of loss terms averaged inside each microstep."""
        del args
        return self.num_train_timesteps
