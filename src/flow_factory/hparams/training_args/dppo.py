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

"""Training arguments for Flow-DPPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .grpo import GRPOTrainingArguments


@dataclass
class DPPOTrainingArguments(GRPOTrainingArguments):
    r"""Training arguments for Flow-DPPO.

    DPPO is a strict Flow-GRPO variant: it keeps GRPO's group advantages and the
    optional KL-vs-reference penalty, but replaces the PPO ratio-clip with a KL
    trust-region mask. A sample's gradient is zeroed when its per-step
    KL(current || rollout-old) exceeds ``kl_mask_threshold`` and the update would
    push further in the wrong direction.
    """

    kl_mask_threshold: float = field(
        default=1.0e-6,
        metadata={
            "help": "Mask (zero-gradient) samples whose per-step KL(current || old) "
            "exceeds this threshold and push the wrong way."
        },
    )
    kl_guidance_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "CFG scale for the KL-vs-reference forward. None uses the training "
            "guidance_scale; >1.0 enables CFG on the frozen reference model."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # Guard against scientific-notation strings from CLI/YAML overrides.
        self.kl_mask_threshold = float(self.kl_mask_threshold)
        if self.kl_guidance_scale is not None:
            self.kl_guidance_scale = float(self.kl_guidance_scale)

    def get_preprocess_guidance_scale(self) -> float:
        """Ensure negative prompts are encoded when the KL-ref branch needs CFG."""
        return max(self.guidance_scale, self.kl_guidance_scale or 0.0)
