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

"""Training arguments for supervised fine-tuning on demonstrations."""

from __future__ import annotations

from dataclasses import dataclass, field

from ._offline import OfflineFlowMatchingTrainingArguments


@dataclass
class SFTTrainingArguments(OfflineFlowMatchingTrainingArguments):
    """Configure offline supervised fine-tuning over demonstration records."""

    trainer_type: str = field(
        default="sft",
        metadata={"help": "Select the offline SFT trainer."},
    )
