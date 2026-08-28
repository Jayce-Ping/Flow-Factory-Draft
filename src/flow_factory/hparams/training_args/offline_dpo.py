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

"""Training arguments for finite-dataset diffusion DPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Mapping

from ...contracts.execution import OFFLINE_EXECUTION_CONTRACT, ExecutionContract
from ._offline import OfflineFlowMatchingTrainingArguments, _finite_float


@dataclass
class OfflineDPOTrainingArguments(OfflineFlowMatchingTrainingArguments):
    """Configure reference-based DPO over an offline preference dataset."""

    execution_contract: ClassVar[ExecutionContract] = OFFLINE_EXECUTION_CONTRACT

    trainer_type: Literal["offline-dpo"] = field(
        default="offline-dpo",
        metadata={"help": "Select the offline preference DPO trainer."},
    )
    beta: float = field(
        default=2000.0,
        metadata={
            "help": (
                "Positive DPO temperature multiplying the policy-versus-reference "
                "chosen/rejected flow-matching loss delta."
            )
        },
    )

    def __post_init__(self) -> None:
        """Validate the fixed trainer identity and reference-based DPO scale."""
        super().__post_init__()
        if self.trainer_type != "offline-dpo":
            raise ValueError(
                "OfflineDPOTrainingArguments requires train.trainer_type='offline-dpo', "
                f"received {self.trainer_type!r}"
            )
        self.beta = _finite_float(
            self.beta,
            "train.beta",
            strictly_positive=True,
        )

    @classmethod
    def from_dict(cls, args_dict: Mapping[str, Any]) -> "OfflineDPOTrainingArguments":
        """Parse only the reference-based semantics implemented by the DPO objective.

        Args:
            args_dict: User training configuration.

        Returns:
            Reference-based offline-DPO arguments.
        """
        explicit_extras = args_dict.get("extra_kwargs") if isinstance(args_dict, Mapping) else None
        if isinstance(args_dict, Mapping) and (
            "reference_free" in args_dict
            or (isinstance(explicit_extras, Mapping) and "reference_free" in explicit_extras)
        ):
            raise ValueError(
                "offline-dpo currently requires frozen reference losses; "
                "train.reference_free is not implemented"
            )
        return super().from_dict(args_dict)

    @property
    def requires_ref_model(self) -> bool:
        """Return true because the shared DPO objective consumes reference losses."""
        return True


__all__ = ["OfflineDPOTrainingArguments"]
