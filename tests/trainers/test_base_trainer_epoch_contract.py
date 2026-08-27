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

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator, List, Optional

import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler

from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.execution import (
    OFFLINE_EXECUTION_CONTRACT,
    ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
)


class OnlineCycleTrainerFake(BaseTrainer):
    """Trainer exercising only the shared online rollout loop."""

    paradigm = "decoupled"

    def __init__(self, total_epochs: int) -> None:
        self.events: List[str] = []
        self.epoch = 0
        self.step = 0
        self._total_epochs = total_epochs
        self.adapter = SimpleNamespace(
            set_trajectory_seed=lambda seed: self.events.append(f"seed:{seed}"),
            ema_step=lambda step: self.events.append(f"ema:{step}"),
        )
        self.training_args = SimpleNamespace(seed=100)
        self.log_args = SimpleNamespace(save_freq=0, save_dir=None, run_name="run")
        self.eval_args = SimpleNamespace(eval_freq=0)

    def should_continue_training(self) -> bool:
        """Stop after the configured number of rollout iterations."""
        return self.epoch < self._total_epochs

    def sample(self) -> List[Any]:
        """Record the rollout stage."""
        self.events.append("sample")
        return []

    def prepare_feedback(self, samples: List[Any]) -> None:
        """Record the feedback stage."""
        self.events.append("feedback")

    def optimize(self, samples: List[Any]) -> None:
        """Record the optimization stage."""
        self.events.append("optimize")


class ScopedOnlineCycleTrainerFake(OnlineCycleTrainerFake):
    """Trainer that rolls out under a swapped parameter scope."""

    @contextmanager
    def sampling_context(self) -> Iterator[None]:
        """Record entering and leaving the sampling scope."""
        self.events.append("scope:enter")
        yield
        self.events.append("scope:exit")

    def _after_training_cycle(self) -> None:
        """Record an algorithm-owned auxiliary update."""
        self.events.append("after_step")


class BoundaryOnlineCycleTrainerFake(OnlineCycleTrainerFake):
    """Record seed, checkpoint, and evaluation boundaries around one online cycle."""

    def __init__(self) -> None:
        super().__init__(total_epochs=1)
        self.log_args = SimpleNamespace(save_freq=1, save_dir="unused", run_name="run")
        self.eval_args = SimpleNamespace(eval_freq=1)

    def save_checkpoint(self, save_directory: str, epoch: Optional[int] = None) -> None:
        """Record the checkpoint boundary without writing files.

        Args:
            save_directory: Resolved checkpoint root, unused by the fake.
            epoch: Legacy cycle index attached to the checkpoint.
        """
        del save_directory
        self.events.append(f"save:{epoch}")

    def evaluate(self) -> None:
        """Record the evaluation boundary."""
        self.events.append("eval")


class RewardFreeOnlineCycleTrainerFake(OnlineCycleTrainerFake):
    """Online trainer whose contract removes the feedback stage."""

    execution_contract = ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT

    def prepare_feedback(self, samples: List[Any]) -> None:
        """Reject feedback dispatch for a reward-free execution contract."""
        del samples
        raise AssertionError("reward-free execution must skip prepare_feedback")


def test_online_rollout_iterations_run_the_stages_in_order() -> None:
    """Every online algorithm inherits one seed, rollout, feedback, optimize sequence."""
    trainer = OnlineCycleTrainerFake(total_epochs=2)

    trainer.start()

    assert trainer.events == [
        "seed:100",
        "sample",
        "feedback",
        "optimize",
        "ema:0",
        "seed:101",
        "sample",
        "feedback",
        "optimize",
        "ema:1",
    ]
    assert trainer.epoch == 2
    assert trainer.progress.rollout_iteration == 2
    assert trainer.progress.data_epoch == 0


def test_sampling_scope_wraps_only_the_rollout() -> None:
    """An EMA or snapshot swap must close before feedback and the policy update."""
    trainer = ScopedOnlineCycleTrainerFake(total_epochs=1)

    trainer.start()

    assert trainer.events == [
        "seed:100",
        "scope:enter",
        "sample",
        "scope:exit",
        "feedback",
        "optimize",
        "ema:0",
        "after_step",
    ]


def test_online_seed_precedes_checkpoint_and_evaluation_boundaries() -> None:
    """The execution refactor preserves the existing online pre-cycle event order."""
    trainer = BoundaryOnlineCycleTrainerFake()

    trainer.start()

    assert trainer.events == [
        "seed:100",
        "save:0",
        "eval",
        "sample",
        "feedback",
        "optimize",
        "ema:0",
    ]


def test_feedback_contract_removes_reward_and_advantage_dispatch() -> None:
    """Reward-free online execution still rolls out and optimizes without a fake stage."""
    trainer = RewardFreeOnlineCycleTrainerFake(total_epochs=1)

    trainer.start()

    assert trainer.events == ["seed:100", "sample", "optimize", "ema:0"]


def test_on_policy_sampling_needs_no_scope_override() -> None:
    """The default scope is a no-op, so a coupled trainer inherits it unchanged."""
    trainer = OnlineCycleTrainerFake(total_epochs=1)

    with trainer.sampling_context():
        pass

    assert trainer.events == []


def test_coupled_paradigm_rejects_ode_dynamics() -> None:
    """A coupled algorithm on ODE dynamics has no transition density to differentiate."""
    trainer = object.__new__(OnlineCycleTrainerFake)
    type(trainer).paradigm = "coupled"
    trainer.adapter = SimpleNamespace(scheduler_group=_SchedulerGroupFake("Flow-ODE"))

    try:
        with pytest.raises(ValueError, match="requires stochastic dynamics"):
            trainer._validate_paradigm_dynamics()
    finally:
        type(trainer).paradigm = "decoupled"


def test_coupled_paradigm_accepts_stochastic_dynamics() -> None:
    """An SDE scheduler is exactly what a coupled algorithm needs."""
    trainer = object.__new__(OnlineCycleTrainerFake)
    type(trainer).paradigm = "coupled"
    trainer.adapter = SimpleNamespace(scheduler_group=_SchedulerGroupFake("Flow-SDE"))

    try:
        trainer._validate_paradigm_dynamics()
    finally:
        type(trainer).paradigm = "decoupled"


class _RecordingDistributedSampler(DistributedSampler):
    """Record official distributed sampler epoch updates."""

    def __init__(self, dataset: List[int], events: List[str]) -> None:
        super().__init__(dataset, num_replicas=1, rank=0, shuffle=False)
        self._events = events

    def set_epoch(self, epoch: int) -> None:
        """Record and apply one data epoch.

        Args:
            epoch: Completed data-epoch count used to seed this traversal.
        """
        self._events.append(f"set_epoch:{epoch}")
        super().set_epoch(epoch)


class OfflineEpochTrainerFake(BaseTrainer):
    """Trainer exercising complete finite offline dataloader traversals."""

    paradigm = "decoupled"
    execution_contract = OFFLINE_EXECUTION_CONTRACT

    def __init__(self, total_epochs: int, fail_on_batch: Optional[int] = None) -> None:
        self.events: List[str] = []
        self.epoch = 0
        self.step = 0
        self._total_epochs = total_epochs
        self._fail_on_batch = fail_on_batch
        dataset = list(range(5))
        sampler = _RecordingDistributedSampler(dataset, self.events)
        self.dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)
        self.adapter = SimpleNamespace(
            set_trajectory_seed=lambda seed: self.events.append(f"unexpected_seed:{seed}"),
            ema_step=lambda step: self.events.append(f"ema:{step}"),
        )
        self.training_args = SimpleNamespace(seed=100)
        self.log_args = SimpleNamespace(save_freq=0, save_dir=None, run_name="run")
        self.eval_args = SimpleNamespace(eval_freq=0)

    def should_continue_training(self) -> bool:
        """Stop after the configured number of complete data epochs."""
        return self.epoch < self._total_epochs

    def optimize_batch(self, batch: torch.Tensor) -> None:
        """Record and optimize one offline batch.

        Args:
            batch: Tensor batch yielded by the finite dataloader.
        """
        values = batch.tolist()
        self.events.append(f"batch:{values}")
        if self._fail_on_batch is not None and values[0] == self._fail_on_batch:
            raise RuntimeError("offline batch failed")
        self.step += 1


class InvalidSamplerOfflineTrainerFake(OfflineEpochTrainerFake):
    """Offline trainer whose loader violates the official sampler contract."""

    def __init__(self) -> None:
        super().__init__(total_epochs=1)
        self.events.clear()
        self.dataloader = DataLoader(list(range(4)), batch_size=2)
        self.log_args = SimpleNamespace(save_freq=1, save_dir="unused", run_name="run")
        self.eval_args = SimpleNamespace(eval_freq=1)

    def save_checkpoint(self, save_directory: str, epoch: Optional[int] = None) -> None:
        """Record a checkpoint attempt without writing files."""
        del save_directory, epoch
        self.events.append("save")

    def evaluate(self) -> None:
        """Record an evaluation attempt."""
        self.events.append("eval")


class BoundaryOfflineEpochTrainerFake(OfflineEpochTrainerFake):
    """Record offline boundaries after a complete finite traversal."""

    def __init__(self, fail_on_batch: Optional[int] = None) -> None:
        super().__init__(total_epochs=1, fail_on_batch=fail_on_batch)
        self.log_args = SimpleNamespace(save_freq=1, save_dir="unused", run_name="run")
        self.eval_args = SimpleNamespace(eval_freq=1)

    def save_checkpoint(self, save_directory: str, epoch: Optional[int] = None) -> None:
        """Record the completed epoch attached to the checkpoint.

        Args:
            save_directory: Resolved checkpoint root, unused by the fake.
            epoch: Completed data-epoch count attached to the checkpoint.
        """
        del save_directory
        self.events.append(f"save:{epoch}")

    def evaluate(self) -> None:
        """Record the post-epoch evaluation boundary."""
        self.events.append("eval")


class MissingOfflineHookTrainer(BaseTrainer):
    """Offline trainer intentionally missing its required batch hook."""

    execution_contract = OFFLINE_EXECUTION_CONTRACT


class MissingOnlineHookTrainer(BaseTrainer):
    """Online trainer intentionally missing its required rollout optimization hook."""


def test_offline_epoch_means_one_complete_dataloader_traversal() -> None:
    """Offline data epochs advance after all distributed batches are consumed."""
    trainer = OfflineEpochTrainerFake(total_epochs=2)

    trainer.start()

    assert trainer.events == [
        "set_epoch:0",
        "batch:[0, 1]",
        "batch:[2, 3]",
        "batch:[4]",
        "set_epoch:1",
        "batch:[0, 1]",
        "batch:[2, 3]",
        "batch:[4]",
    ]
    assert trainer.epoch == 2
    assert trainer.step == 6
    assert trainer.progress.data_epoch == 2
    assert trainer.progress.rollout_iteration == 0


def test_offline_boundaries_run_after_completed_epoch_progress() -> None:
    """Checkpoint and evaluation observe trained weights and data_epoch=1."""
    trainer = BoundaryOfflineEpochTrainerFake()

    trainer.start()

    assert trainer.events == [
        "set_epoch:0",
        "batch:[0, 1]",
        "batch:[2, 3]",
        "batch:[4]",
        "save:1",
        "eval",
    ]
    assert trainer.progress.data_epoch == 1


@pytest.mark.parametrize(
    ("trainer_class", "expected_ema_events"),
    [
        (OfflineEpochTrainerFake, ["ema:0"]),
        (OnlineCycleTrainerFake, []),
    ],
)
def test_shared_ema_uses_optimizer_step_cadence_only_for_offline_acquisition(
    trainer_class: type[BaseTrainer],
    expected_ema_events: List[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline EMA follows successful updates while online EMA remains cycle-owned."""
    trainer = object.__new__(trainer_class)
    events: List[str] = []
    trainer.progress = trainer._get_progress()
    trainer.training_args = SimpleNamespace(max_grad_norm=1.0)
    trainer.model_bundle = SimpleNamespace(parameters=lambda: [])
    trainer.accelerator = SimpleNamespace(
        clip_grad_norm_=lambda parameters, max_norm: torch.tensor(0.5)
    )
    trainer.optimizer = SimpleNamespace(
        step=lambda: events.append("optimizer"),
        zero_grad=lambda: events.append("zero_grad"),
    )
    trainer.adapter = SimpleNamespace(
        ema_step=lambda step: events.append(f"ema:{step}"),
    )
    trainer.log_data = lambda data, step: events.append(f"log:{step}")
    monkeypatch.setattr(
        "flow_factory.trainers.abc.reduce_loss_info",
        lambda accelerator, loss_info: {"loss": torch.tensor(1.0)},
    )

    trainer._apply_optimizer_step({"loss": [torch.tensor(1.0)]})

    assert [event for event in events if event.startswith("ema:")] == expected_ema_events
    assert events == ["optimizer", *expected_ema_events, "zero_grad", "log:0"]
    assert trainer.step == 1


def test_incomplete_offline_traversal_does_not_advance_data_epoch() -> None:
    """A dataloader or optimization error leaves the current data epoch incomplete."""
    trainer = OfflineEpochTrainerFake(total_epochs=1, fail_on_batch=2)

    with pytest.raises(RuntimeError, match="offline batch failed"):
        trainer.start()

    assert trainer.events == ["set_epoch:0", "batch:[0, 1]", "batch:[2, 3]"]
    assert trainer.epoch == 0
    assert trainer.step == 1


def test_incomplete_offline_traversal_does_not_run_cycle_boundaries() -> None:
    """A failed batch cannot publish a checkpoint labeled as a complete epoch."""
    trainer = BoundaryOfflineEpochTrainerFake(fail_on_batch=2)

    with pytest.raises(RuntimeError, match="offline batch failed"):
        trainer.start()

    assert trainer.events == ["set_epoch:0", "batch:[0, 1]", "batch:[2, 3]"]
    assert trainer.progress.data_epoch == 0


def test_invalid_offline_sampler_fails_before_shared_cycle_boundaries() -> None:
    """Sampler validation must precede checkpoint, evaluation, and optimization effects."""
    trainer = InvalidSamplerOfflineTrainerFake()

    with pytest.raises(TypeError, match="torch.utils.data.DistributedSampler"):
        trainer.start()

    assert trainer.events == []
    assert trainer.epoch == 0
    assert trainer.step == 0


@pytest.mark.parametrize(
    ("trainer_class", "message"),
    [
        (MissingOnlineHookTrainer, "must override optimize"),
        (MissingOfflineHookTrainer, "must override optimize_batch"),
    ],
)
def test_execution_contract_requires_the_matching_optimization_hook(
    trainer_class: type[BaseTrainer], message: str
) -> None:
    """A trainer cannot inherit the fail-fast hook selected by its execution mode."""
    trainer = object.__new__(trainer_class)

    with pytest.raises(TypeError, match=message):
        trainer._validate_execution_hooks()


class _SchedulerGroupFake:
    """Scheduler group exposing one component with a fixed dynamics type."""

    def __init__(self, dynamics_type: str) -> None:
        self.names = ("latent",)
        self._dynamics_type = dynamics_type

    def __getitem__(self, name: str) -> SimpleNamespace:
        """Return the single component scheduler."""
        return SimpleNamespace(dynamics_type=self._dynamics_type)
