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

"""Training arguments for supervised flow-matching fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from ...contracts.execution import OFFLINE_EXECUTION_CONTRACT, ExecutionContract
from ._offline import OfflineFlowMatchingTrainingArguments


@dataclass
class SFTTrainingArguments(OfflineFlowMatchingTrainingArguments):
    """Configure finite-dataset supervised flow-matching training."""

    execution_contract: ClassVar[ExecutionContract] = OFFLINE_EXECUTION_CONTRACT

    trainer_type: Literal["sft"] = field(
        default="sft",
        metadata={"help": "Select the offline supervised fine-tuning trainer."},
    )

    def __post_init__(self) -> None:
        """Validate the fixed SFT trainer identity."""
        super().__post_init__()
        if self.trainer_type != "sft":
            raise ValueError(
                "SFTTrainingArguments requires train.trainer_type='sft', "
                f"received {self.trainer_type!r}"
            )

    @property
    def requires_ref_model(self) -> bool:
        """Return false because supervised flow matching has no reference branch."""
        return False


__all__ = ["SFTTrainingArguments"]
