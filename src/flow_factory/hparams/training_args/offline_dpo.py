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

"""Training arguments for offline diffusion preference optimization."""

from __future__ import annotations

from dataclasses import dataclass, field

from ._offline import OfflineFlowMatchingTrainingArguments


@dataclass
class OfflineDPOTrainingArguments(OfflineFlowMatchingTrainingArguments):
    """Configure DPO from chosen/rejected media stored in a finite dataset."""

    trainer_type: str = field(
        default="offline-dpo",
        metadata={"help": "Select the offline DPO trainer."},
    )
    beta: float = field(
        default=2000.0,
        metadata={"help": "DPO temperature parameter controlling preference sharpness."},
    )

    @property
    def requires_ref_model(self) -> bool:
        """Offline DPO always compares the policy against a frozen reference."""
        return True
