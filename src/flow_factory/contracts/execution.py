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

"""Pure execution semantics shared by configuration and trainer runtimes."""

from dataclasses import dataclass
from enum import Enum


class AcquisitionMode(str, Enum):
    """Describe how a trainer acquires optimization examples."""

    ROLLOUT = "rollout"
    DATASET = "dataset"


class CycleUnit(str, Enum):
    """Describe the unit that bounds one outer training cycle."""

    ROLLOUT_ITERATION = "rollout_iteration"
    DATA_EPOCH = "data_epoch"


class FeedbackMode(str, Enum):
    """Describe whether execution requires runtime reward feedback."""

    REWARD = "reward"
    NONE = "none"


class LoaderKind(str, Enum):
    """Distinguish rollout sampling from finite distributed epoch loading."""

    GROUPED_ROLLOUT = "grouped_rollout"
    DISTRIBUTED_EPOCH = "distributed_epoch"


@dataclass(frozen=True)
class ExecutionContract:
    """Declare the acquisition, feedback, loader, and cycle semantics of an algorithm."""

    acquisition: AcquisitionMode
    cycle_unit: CycleUnit
    feedback: FeedbackMode
    loader_kind: LoaderKind

    def __post_init__(self) -> None:
        """Validate field types and coherent execution semantics."""
        _require_enum(self.acquisition, AcquisitionMode, "acquisition")
        _require_enum(self.cycle_unit, CycleUnit, "cycle_unit")
        _require_enum(self.feedback, FeedbackMode, "feedback")
        _require_enum(self.loader_kind, LoaderKind, "loader_kind")

        expected_cycle_unit, expected_loader_kind = {
            AcquisitionMode.ROLLOUT: (
                CycleUnit.ROLLOUT_ITERATION,
                LoaderKind.GROUPED_ROLLOUT,
            ),
            AcquisitionMode.DATASET: (
                CycleUnit.DATA_EPOCH,
                LoaderKind.DISTRIBUTED_EPOCH,
            ),
        }[self.acquisition]
        if (
            self.cycle_unit is not expected_cycle_unit
            or self.loader_kind is not expected_loader_kind
        ):
            raise ValueError(
                f"expected acquisition={self.acquisition.value!r} to use "
                f"cycle_unit={expected_cycle_unit.value!r} and "
                f"loader_kind={expected_loader_kind.value!r}, received "
                f"cycle_unit={self.cycle_unit.value!r} and "
                f"loader_kind={self.loader_kind.value!r}"
            )
        if self.acquisition is AcquisitionMode.DATASET and self.feedback is FeedbackMode.REWARD:
            raise ValueError(
                "dataset acquisition does not support runtime reward feedback; offline SFT and "
                "preference training must consume supervision from each dataset batch"
            )


def _require_enum(value: object, enum_type: type[Enum], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(
            f"expected {field_name} to be {enum_type.__name__}, received "
            f"{type(value).__name__}: {value!r}"
        )


ONLINE_EXECUTION_CONTRACT = ExecutionContract(
    acquisition=AcquisitionMode.ROLLOUT,
    cycle_unit=CycleUnit.ROLLOUT_ITERATION,
    feedback=FeedbackMode.REWARD,
    loader_kind=LoaderKind.GROUPED_ROLLOUT,
)

ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT = ExecutionContract(
    acquisition=AcquisitionMode.ROLLOUT,
    cycle_unit=CycleUnit.ROLLOUT_ITERATION,
    feedback=FeedbackMode.NONE,
    loader_kind=LoaderKind.GROUPED_ROLLOUT,
)

OFFLINE_EXECUTION_CONTRACT = ExecutionContract(
    acquisition=AcquisitionMode.DATASET,
    cycle_unit=CycleUnit.DATA_EPOCH,
    feedback=FeedbackMode.NONE,
    loader_kind=LoaderKind.DISTRIBUTED_EPOCH,
)


__all__ = [
    "AcquisitionMode",
    "CycleUnit",
    "ExecutionContract",
    "FeedbackMode",
    "LoaderKind",
    "OFFLINE_EXECUTION_CONTRACT",
    "ONLINE_EXECUTION_CONTRACT",
    "ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT",
]
