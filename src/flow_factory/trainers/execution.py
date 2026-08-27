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

"""Runtime drivers and progress counters for online and offline trainers.

Pure execution declarations live in :mod:`flow_factory.contracts.execution` so
configuration code can select semantics without importing the trainer package.
They are re-exported here to preserve the existing public import path.
"""

from dataclasses import dataclass, replace
from typing import Any, Protocol

from torch.utils.data import DataLoader, DistributedSampler

from ..contracts.execution import (
    OFFLINE_EXECUTION_CONTRACT,
    ONLINE_EXECUTION_CONTRACT,
    ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
    AcquisitionMode,
    CycleUnit,
    ExecutionContract,
    FeedbackMode,
    LoaderKind,
)


@dataclass(frozen=True)
class TrainingProgress:
    """Track optimizer updates independently from online and offline cycles."""

    optimizer_step: int = 0
    rollout_iteration: int = 0
    data_epoch: int = 0

    def __post_init__(self) -> None:
        """Reject invalid counter state at its point of construction."""
        _require_non_negative_int(self.optimizer_step, "optimizer_step")
        _require_non_negative_int(self.rollout_iteration, "rollout_iteration")
        _require_non_negative_int(self.data_epoch, "data_epoch")

    def cycle_index(self, cycle_unit: CycleUnit) -> int:
        """Return the completed-cycle count for the requested unit.

        Args:
            cycle_unit: Outer-cycle unit declared by the execution contract.

        Returns:
            Number of completed cycles for the requested unit.
        """
        _require_cycle_unit(cycle_unit, "cycle_unit")
        if cycle_unit is CycleUnit.ROLLOUT_ITERATION:
            return self.rollout_iteration
        return self.data_epoch

    def advance_cycle(self, cycle_unit: CycleUnit, *, completed: bool) -> "TrainingProgress":
        """Advance one outer cycle after its work completed successfully.

        Args:
            cycle_unit: Outer-cycle unit declared by the execution contract.
            completed: Whether the rollout iteration or finite dataloader pass completed.

        Returns:
            New progress with the requested completed-cycle counter advanced once.

        Raises:
            RuntimeError: If the cycle did not complete. An offline data epoch is complete only
                after its finite dataloader is exhausted.
        """
        _require_cycle_unit(cycle_unit, "cycle_unit")
        if type(completed) is not bool:
            raise TypeError(
                f"expected completed to be bool, received {type(completed).__name__}: {completed!r}"
            )
        if not completed:
            raise RuntimeError(
                f"cannot advance {cycle_unit.value!r} because the execution cycle did not "
                "complete"
            )
        if cycle_unit is CycleUnit.ROLLOUT_ITERATION:
            return replace(self, rollout_iteration=self.rollout_iteration + 1)
        return replace(self, data_epoch=self.data_epoch + 1)

    def advance_optimizer_step(self, count: int = 1) -> "TrainingProgress":
        """Advance the optimizer-step counter independently of outer cycles.

        Args:
            count: Positive number of completed optimizer updates to record.

        Returns:
            New progress with the optimizer-step counter advanced by ``count``.
        """
        _require_positive_int(count, "count")
        return replace(self, optimizer_step=self.optimizer_step + count)


class ExecutionHost(Protocol):
    """Define the trainer hooks consumed by online and offline execution drivers."""

    dataloader: DataLoader

    def set_trajectory_seed(self, seed: int) -> None:
        """Set the trajectory seed used by the next online cycle.

        Args:
            seed: Effective seed for the next rollout iteration.
        """
        ...

    def run_online_cycle(self) -> None:
        """Run one complete rollout, feedback, and optimization cycle."""
        ...

    def optimize_batch(self, batch: Any) -> None:
        """Optimize one batch acquired from an offline dataloader.

        Args:
            batch: Collated batch yielded by the offline dataloader.
        """
        ...


class ExecutionDriver(Protocol):
    """Define the common outer-cycle driver interface."""

    def prepare_cycle(
        self,
        host: ExecutionHost,
        progress: TrainingProgress,
        *,
        seed: int,
    ) -> None:
        """Prepare one cycle before shared checkpoint and evaluation boundaries.

        Args:
            host: Trainer hooks consumed by the selected driver.
            progress: Immutable progress at the start of the cycle.
            seed: Base training seed used by online execution.
        """
        ...

    def run_cycle(
        self,
        host: ExecutionHost,
        progress: TrainingProgress,
    ) -> None:
        """Run one outer cycle without advancing progress counters.

        Args:
            host: Trainer hooks and dataloader consumed by the selected driver.
            progress: Immutable progress at the start of the cycle.
        """
        ...


class OnlineExecutionDriver:
    """Drive one online rollout iteration through trainer-owned hooks."""

    def prepare_cycle(
        self,
        host: ExecutionHost,
        progress: TrainingProgress,
        *,
        seed: int,
    ) -> None:
        """Seed one online cycle before shared save and evaluation boundaries.

        Args:
            host: Trainer hook that owns trajectory seeding.
            progress: Immutable progress at the start of the rollout iteration.
            seed: Base training seed combined with the rollout-iteration index.
        """
        _require_progress(progress)
        _require_int(seed, "seed")
        host.set_trajectory_seed(seed + progress.rollout_iteration)

    def run_cycle(
        self,
        host: ExecutionHost,
        progress: TrainingProgress,
    ) -> None:
        """Run one prepared online cycle without advancing progress.

        Args:
            host: Trainer hook that owns the online cycle implementation.
            progress: Immutable progress at the start of the rollout iteration.
        """
        _require_progress(progress)
        host.run_online_cycle()


class OfflineExecutionDriver:
    """Drive one complete offline data epoch with PyTorch distribution semantics."""

    def prepare_cycle(
        self,
        host: ExecutionHost,
        progress: TrainingProgress,
        *,
        seed: int,
    ) -> None:
        """Validate common cycle inputs without touching trajectory state.

        Args:
            host: Offline trainer host, unused until dataloader traversal.
            progress: Immutable progress at the start of the data epoch.
            seed: Base training seed retained by the common driver interface.
        """
        _require_progress(progress)
        _require_int(seed, "seed")
        _require_distributed_sampler(host)

    def run_cycle(
        self,
        host: ExecutionHost,
        progress: TrainingProgress,
    ) -> None:
        """Exhaust one distributed dataloader epoch without advancing progress.

        Args:
            host: Trainer hooks and finite offline dataloader to execute.
            progress: Immutable progress at the start of the data epoch.

        Raises:
            TypeError: If the offline dataloader does not use PyTorch's official
                ``DistributedSampler``. This requirement also applies to single-process runs.

        Note:
            Exceptions from dataloader iteration or ``optimize_batch`` propagate unchanged. The
            caller advances ``data_epoch`` only after this method returns normally.
        """
        _require_progress(progress)
        sampler = _require_distributed_sampler(host)
        sampler.set_epoch(progress.data_epoch)
        for batch in host.dataloader:
            host.optimize_batch(batch)


def build_execution_driver(contract: ExecutionContract) -> ExecutionDriver:
    """Build the driver selected by an execution contract.

    Args:
        contract: Validated acquisition and loader semantics declared by an algorithm.

    Returns:
        Online or offline driver matching the contract's acquisition and loader kind.

    Raises:
        TypeError: If ``contract`` is not an ``ExecutionContract``.
        ValueError: If no driver implements the requested acquisition and loader combination.
    """
    if not isinstance(contract, ExecutionContract):
        raise TypeError(
            f"expected contract to be ExecutionContract, received "
            f"{type(contract).__name__}: {contract!r}"
        )
    if (
        contract.acquisition is AcquisitionMode.ROLLOUT
        and contract.loader_kind is LoaderKind.GROUPED_ROLLOUT
    ):
        return OnlineExecutionDriver()
    if (
        contract.acquisition is AcquisitionMode.DATASET
        and contract.loader_kind is LoaderKind.DISTRIBUTED_EPOCH
    ):
        return OfflineExecutionDriver()
    raise ValueError(
        f"no execution driver for acquisition={contract.acquisition.value!r} and "
        f"loader_kind={contract.loader_kind.value!r}"
    )


def _require_cycle_unit(value: object, field_name: str) -> None:
    """Require one exact enum member for an execution-contract field."""
    if not isinstance(value, CycleUnit):
        raise TypeError(
            f"expected {field_name} to be CycleUnit, received " f"{type(value).__name__}: {value!r}"
        )


def _require_distributed_sampler(host: ExecutionHost) -> DistributedSampler:
    """Return the official sampler required by offline execution."""
    sampler = getattr(host.dataloader, "sampler", None)
    if not isinstance(sampler, DistributedSampler):
        sampler_name = type(sampler).__name__ if sampler is not None else "None"
        raise TypeError(
            "expected offline dataloader.sampler to be "
            "torch.utils.data.DistributedSampler, received "
            f"{sampler_name}; use DistributedSampler even when num_replicas=1"
        )
    return sampler


def _require_non_negative_int(value: object, field_name: str) -> None:
    """Require a non-negative integer counter without accepting booleans."""
    if type(value) is not int:
        raise TypeError(
            f"expected {field_name} to be int, received {type(value).__name__}: {value!r}"
        )
    if value < 0:
        raise ValueError(f"expected {field_name} >= 0, received {value}")


def _require_positive_int(value: object, field_name: str) -> None:
    """Require a positive integer without accepting booleans."""
    if type(value) is not int:
        raise TypeError(
            f"expected {field_name} to be int, received {type(value).__name__}: {value!r}"
        )
    if value < 1:
        raise ValueError(f"expected {field_name} >= 1, received {value}")


def _require_int(value: object, field_name: str) -> None:
    """Require an integer without accepting booleans."""
    if type(value) is not int:
        raise TypeError(
            f"expected {field_name} to be int, received {type(value).__name__}: {value!r}"
        )


def _require_progress(progress: object) -> None:
    """Require immutable training progress at a driver boundary."""
    if not isinstance(progress, TrainingProgress):
        raise TypeError(
            f"expected progress to be TrainingProgress, received "
            f"{type(progress).__name__}: {progress!r}"
        )


__all__ = [
    "AcquisitionMode",
    "CycleUnit",
    "ExecutionContract",
    "ExecutionDriver",
    "ExecutionHost",
    "FeedbackMode",
    "LoaderKind",
    "OFFLINE_EXECUTION_CONTRACT",
    "ONLINE_EXECUTION_CONTRACT",
    "ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT",
    "OfflineExecutionDriver",
    "OnlineExecutionDriver",
    "TrainingProgress",
    "build_execution_driver",
]
