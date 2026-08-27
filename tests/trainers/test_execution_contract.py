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

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler

from flow_factory.contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaBinding,
    InputMediaOrder,
    InputMediaSpec,
    MediaFormat,
    MediaType,
    NegativePromptPolicy,
    OutputMediaSequence,
    PipelineIOContract,
    RateRequirement,
)
from flow_factory.models.abc import BaseAdapter
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.execution import (
    OFFLINE_EXECUTION_CONTRACT,
    ONLINE_EXECUTION_CONTRACT,
    ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
    AcquisitionMode,
    CycleUnit,
    ExecutionContract,
    FeedbackMode,
    LoaderKind,
    OfflineExecutionDriver,
    OnlineExecutionDriver,
    TrainingProgress,
    build_execution_driver,
)
from flow_factory.trainers.registry import get_trainer_class

_IMAGE_OUTPUT_CONTRACT = PipelineIOContract(
    input_media=InputMediaSpec(
        rules=(),
        binding=InputMediaBinding.GROUPED_BY_TYPE,
        order=InputMediaOrder.INSENSITIVE,
    ),
    negative_prompt=NegativePromptPolicy.OPTIONAL,
    output_media=OutputMediaSequence(
        items=(
            MediaFormat(
                type=MediaType.IMAGE,
                fps=RateRequirement.NOT_APPLICABLE,
                sample_rate=RateRequirement.NOT_APPLICABLE,
            ),
        )
    ),
    geometry_source=GeometrySource.CONFIGURED,
    batch_capability=BatchCapability.UNIFORM,
)


class _OnlineHostFake:
    """Record online driver calls in execution order."""

    def __init__(self) -> None:
        self.dataloader = None
        self.events: List[Tuple[str, Optional[int]]] = []

    def set_trajectory_seed(self, seed: int) -> None:
        """Record the effective trajectory seed."""
        self.events.append(("seed", seed))

    def run_online_cycle(self) -> None:
        """Record one complete online-cycle dispatch."""
        self.events.append(("online_cycle", None))

    def optimize_batch(self, batch: Any) -> None:
        """Reject an offline hook on the online-only fake."""
        raise AssertionError("online execution must not call optimize_batch")


class _RecordingDistributedSampler(DistributedSampler):
    """Record epoch seeding while retaining official sampler behavior."""

    def __init__(self, dataset: List[int], events: List[Tuple[str, Any]]) -> None:
        super().__init__(dataset, num_replicas=1, rank=0, shuffle=False)
        self._events = events

    def set_epoch(self, epoch: int) -> None:
        """Record and apply one official distributed epoch index."""
        self._events.append(("set_epoch", epoch))
        super().set_epoch(epoch)


class _OfflineHostFake:
    """Record every batch consumed by the offline driver."""

    def __init__(self, dataloader: DataLoader, events: List[Tuple[str, Any]]) -> None:
        self.dataloader = dataloader
        self.events = events
        self.fail_on_batch: Optional[int] = None
        self._batch_count = 0

    def set_trajectory_seed(self, seed: int) -> None:
        """Reject trajectory seeding on the offline path."""
        raise AssertionError("offline execution must not set a trajectory seed")

    def run_online_cycle(self) -> None:
        """Reject online-cycle dispatch on the offline path."""
        raise AssertionError("offline execution must not run an online cycle")

    def optimize_batch(self, batch: torch.Tensor) -> None:
        """Record one batch and optionally raise a trainer error."""
        self._batch_count += 1
        self.events.append(("optimize_batch", batch.tolist()))
        if self._batch_count == self.fail_on_batch:
            raise RuntimeError("optimization failed")


def test_predefined_online_contract_uses_grouped_rollout_execution() -> None:
    """Online execution keeps the existing custom grouped-sampler semantics."""
    assert ONLINE_EXECUTION_CONTRACT == ExecutionContract(
        acquisition=AcquisitionMode.ROLLOUT,
        cycle_unit=CycleUnit.ROLLOUT_ITERATION,
        feedback=FeedbackMode.REWARD,
        loader_kind=LoaderKind.GROUPED_ROLLOUT,
    )


def test_predefined_offline_contract_uses_distributed_epoch_execution() -> None:
    """Offline execution uses a finite epoch loader distributed by PyTorch."""
    assert OFFLINE_EXECUTION_CONTRACT == ExecutionContract(
        acquisition=AcquisitionMode.DATASET,
        cycle_unit=CycleUnit.DATA_EPOCH,
        feedback=FeedbackMode.NONE,
        loader_kind=LoaderKind.DISTRIBUTED_EPOCH,
    )


def test_offline_trainer_requires_adapter_pipeline_and_output_codec_before_initialization() -> None:
    """Dataset acquisition fails before preprocessing when model output support is absent."""
    trainer_class = get_trainer_class("sft")
    trainer = object.__new__(trainer_class)
    trainer.adapter = SimpleNamespace(pipeline_io_contract=None, output_state_codec=None)

    with pytest.raises(TypeError, match="requires adapter.*PipelineIOContract"):
        BaseTrainer._validate_adapter_execution_contract(trainer)

    trainer.adapter = SimpleNamespace(
        pipeline_io_contract=_IMAGE_OUTPUT_CONTRACT,
        output_state_codec=None,
    )
    with pytest.raises(TypeError, match="requires adapter.*output-state codec"):
        BaseTrainer._validate_adapter_execution_contract(trainer)

    trainer.adapter = SimpleNamespace(
        pipeline_io_contract=_IMAGE_OUTPUT_CONTRACT,
        output_state_codec=object(),
    )
    BaseTrainer._validate_adapter_execution_contract(trainer)


def test_offline_trainer_surfaces_declared_adapter_blocker_before_generic_contract_errors() -> None:
    trainer_class = get_trainer_class("sft")
    trainer = object.__new__(trainer_class)
    trainer.adapter = SimpleNamespace(
        pipeline_io_contract=None,
        output_state_codec=None,
        output_state_codec_unavailable_reason=(
            "Condition pixels are missing; extend the condition projection."
        ),
    )

    with pytest.raises(
        NotImplementedError,
        match=r"Condition pixels are missing.*extend the condition projection",
    ):
        BaseTrainer._validate_adapter_execution_contract(trainer)


def test_offline_trainer_class_preflight_requires_contract_and_codec_builder() -> None:
    trainer_class = get_trainer_class("sft")

    class MissingContractAdapter(BaseAdapter):
        pass

    class MissingCodecBuilderAdapter(BaseAdapter):
        pipeline_io_contract = _IMAGE_OUTPUT_CONTRACT

    class DeclaredCodecBuilderAdapter(BaseAdapter):
        pipeline_io_contract = _IMAGE_OUTPUT_CONTRACT

        def build_output_state_codec(self) -> Any:
            return None

        def _validate_encoded_output_geometry(self, *args: Any, **kwargs: Any) -> None:
            pass

    class MissingGeometryValidationAdapter(BaseAdapter):
        pipeline_io_contract = _IMAGE_OUTPUT_CONTRACT

        def build_output_state_codec(self) -> Any:
            return None

    with pytest.raises(TypeError, match=r"MissingContractAdapter.*PipelineIOContract"):
        trainer_class.validate_adapter_class_execution_contract(MissingContractAdapter)
    with pytest.raises(TypeError, match=r"MissingCodecBuilderAdapter.*build_output_state_codec"):
        trainer_class.validate_adapter_class_execution_contract(MissingCodecBuilderAdapter)
    with pytest.raises(
        TypeError,
        match=r"MissingGeometryValidationAdapter.*_validate_encoded_output_geometry",
    ):
        trainer_class.validate_adapter_class_execution_contract(MissingGeometryValidationAdapter)

    trainer_class.validate_adapter_class_execution_contract(DeclaredCodecBuilderAdapter)


def test_online_trainer_does_not_require_offline_output_codec() -> None:
    """Rollout acquisition remains compatible with existing online-only adapters."""
    trainer_class = get_trainer_class("dpo")
    trainer = object.__new__(trainer_class)
    trainer.adapter = SimpleNamespace(
        pipeline_io_contract=None,
        output_state_codec=None,
        output_state_codec_unavailable_reason="Offline targets are not implemented.",
    )

    BaseTrainer._validate_adapter_execution_contract(trainer)

    class KnownOfflineBlockerAdapter(BaseAdapter):
        output_state_codec_unavailable_reason = "Offline targets are not implemented."

    trainer_class.validate_adapter_class_execution_contract(KnownOfflineBlockerAdapter)


def test_feedback_mode_is_independent_from_rollout_acquisition() -> None:
    """Rollout algorithms such as distillation may omit reward feedback."""
    contract = ExecutionContract(
        acquisition=AcquisitionMode.ROLLOUT,
        cycle_unit=CycleUnit.ROLLOUT_ITERATION,
        feedback=FeedbackMode.NONE,
        loader_kind=LoaderKind.GROUPED_ROLLOUT,
    )

    assert contract.feedback is FeedbackMode.NONE


def test_offline_runtime_reward_feedback_is_rejected_until_it_has_a_batch_seam() -> None:
    """A declared offline reward stage cannot be silently ignored by the epoch driver."""
    with pytest.raises(ValueError, match="does not support runtime reward feedback"):
        ExecutionContract(
            acquisition=AcquisitionMode.DATASET,
            cycle_unit=CycleUnit.DATA_EPOCH,
            feedback=FeedbackMode.REWARD,
            loader_kind=LoaderKind.DISTRIBUTED_EPOCH,
        )


@pytest.mark.parametrize(
    ("overrides", "field_name"),
    [
        ({"acquisition": "rollout"}, "acquisition"),
        ({"cycle_unit": "rollout_iteration"}, "cycle_unit"),
        ({"feedback": "reward"}, "feedback"),
        ({"loader_kind": "grouped_rollout"}, "loader_kind"),
    ],
)
def test_execution_contract_rejects_raw_strings(overrides, field_name) -> None:
    """Contract construction fails fast instead of coercing ambiguous raw strings."""
    fields = {
        "acquisition": AcquisitionMode.ROLLOUT,
        "cycle_unit": CycleUnit.ROLLOUT_ITERATION,
        "feedback": FeedbackMode.REWARD,
        "loader_kind": LoaderKind.GROUPED_ROLLOUT,
    }
    fields.update(overrides)

    with pytest.raises(TypeError, match=field_name):
        ExecutionContract(**fields)


@pytest.mark.parametrize(
    ("acquisition", "cycle_unit", "loader_kind"),
    [
        (
            AcquisitionMode.ROLLOUT,
            CycleUnit.DATA_EPOCH,
            LoaderKind.GROUPED_ROLLOUT,
        ),
        (
            AcquisitionMode.ROLLOUT,
            CycleUnit.ROLLOUT_ITERATION,
            LoaderKind.DISTRIBUTED_EPOCH,
        ),
        (
            AcquisitionMode.DATASET,
            CycleUnit.ROLLOUT_ITERATION,
            LoaderKind.DISTRIBUTED_EPOCH,
        ),
        (
            AcquisitionMode.DATASET,
            CycleUnit.DATA_EPOCH,
            LoaderKind.GROUPED_ROLLOUT,
        ),
    ],
)
def test_execution_contract_rejects_incoherent_cycle_or_loader_semantics(
    acquisition, cycle_unit, loader_kind
) -> None:
    """Acquisition mode determines its coherent cycle and distribution semantics."""
    with pytest.raises(ValueError, match=f"acquisition={acquisition.value!r}"):
        ExecutionContract(
            acquisition=acquisition,
            cycle_unit=cycle_unit,
            feedback=FeedbackMode.NONE,
            loader_kind=loader_kind,
        )


def test_execution_contract_is_immutable() -> None:
    """Algorithms cannot mutate an execution contract after initialization."""
    with pytest.raises(FrozenInstanceError):
        OFFLINE_EXECUTION_CONTRACT.feedback = FeedbackMode.REWARD


def test_training_progress_tracks_cycle_units_independently() -> None:
    """Online and offline cycle counters do not alias one generic epoch field."""
    progress = TrainingProgress(optimizer_step=4, rollout_iteration=2, data_epoch=3)

    assert progress.cycle_index(CycleUnit.ROLLOUT_ITERATION) == 2
    assert progress.cycle_index(CycleUnit.DATA_EPOCH) == 3


def test_data_epoch_advances_only_after_finite_loader_exhaustion() -> None:
    """A partial offline dataloader pass is not counted as an epoch."""
    progress = TrainingProgress(optimizer_step=2)

    with pytest.raises(RuntimeError, match="execution cycle did not complete"):
        progress.advance_cycle(CycleUnit.DATA_EPOCH, completed=False)

    assert progress.data_epoch == 0
    completed = progress.advance_cycle(CycleUnit.DATA_EPOCH, completed=True)
    assert completed.data_epoch == 1
    assert completed.rollout_iteration == 0
    assert completed.optimizer_step == 2


def test_rollout_iteration_advances_only_after_complete_execution() -> None:
    """A completed online rollout cycle advances only its own counter."""
    progress = TrainingProgress(data_epoch=2)

    completed = progress.advance_cycle(CycleUnit.ROLLOUT_ITERATION, completed=True)

    assert completed.rollout_iteration == 1
    assert completed.data_epoch == 2
    assert progress.rollout_iteration == 0


def test_optimizer_steps_advance_independently_from_cycles() -> None:
    """Multiple optimizer updates may occur within one offline data epoch."""
    progress = TrainingProgress(data_epoch=1)

    advanced = progress.advance_optimizer_step(3)

    assert advanced == TrainingProgress(optimizer_step=3, data_epoch=1)
    assert progress.optimizer_step == 0


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"optimizer_step": True}, TypeError, "optimizer_step"),
        ({"rollout_iteration": 1.0}, TypeError, "rollout_iteration"),
        ({"data_epoch": -1}, ValueError, "data_epoch >= 0"),
    ],
)
def test_training_progress_rejects_invalid_counter_state(kwargs, error_type, message) -> None:
    """Invalid progress cannot enter checkpoint or loop state."""
    with pytest.raises(error_type, match=message):
        TrainingProgress(**kwargs)


@pytest.mark.parametrize("count", [0, -1])
def test_optimizer_progress_rejects_non_positive_updates(count) -> None:
    """Optimizer progress records completed updates, never zero or negative deltas."""
    with pytest.raises(ValueError, match="count >= 1"):
        TrainingProgress().advance_optimizer_step(count)


def test_cycle_progress_rejects_untyped_completion_signal() -> None:
    """Cycle completion must be an explicit boolean signal from the driver."""
    with pytest.raises(TypeError, match="completed to be bool"):
        TrainingProgress().advance_cycle(CycleUnit.DATA_EPOCH, completed=1)


def test_cycle_index_rejects_raw_string() -> None:
    """Progress lookup requires the same typed cycle unit as the contract."""
    with pytest.raises(TypeError, match="cycle_unit"):
        TrainingProgress().cycle_index("data_epoch")


def test_online_driver_seeds_before_dispatching_the_cycle() -> None:
    """Online execution derives its seed from the completed rollout count."""
    host = _OnlineHostFake()
    progress = TrainingProgress(optimizer_step=7, rollout_iteration=3)
    driver = OnlineExecutionDriver()

    driver.prepare_cycle(host, progress, seed=100)
    assert host.events == [("seed", 103)]
    driver.run_cycle(host, progress)

    assert host.events == [("seed", 103), ("online_cycle", None)]
    assert progress == TrainingProgress(optimizer_step=7, rollout_iteration=3)


def test_offline_driver_sets_epoch_and_exhausts_distributed_dataloader() -> None:
    """One offline cycle processes every batch after setting the data epoch."""
    events: List[Tuple[str, Any]] = []
    dataset = list(range(5))
    sampler = _RecordingDistributedSampler(dataset, events)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)
    host = _OfflineHostFake(dataloader, events)
    progress = TrainingProgress(optimizer_step=4, data_epoch=2)
    driver = OfflineExecutionDriver()

    driver.prepare_cycle(host, progress, seed=100)
    driver.run_cycle(host, progress)

    assert events == [
        ("set_epoch", 2),
        ("optimize_batch", [0, 1]),
        ("optimize_batch", [2, 3]),
        ("optimize_batch", [4]),
    ]
    assert sampler.epoch == 2
    assert progress == TrainingProgress(optimizer_step=4, data_epoch=2)


def test_offline_driver_propagates_batch_errors_without_finishing_the_loader() -> None:
    """A failed batch aborts the cycle so its caller cannot count a full epoch."""
    events: List[Tuple[str, Any]] = []
    dataset = list(range(5))
    sampler = _RecordingDistributedSampler(dataset, events)
    host = _OfflineHostFake(DataLoader(dataset, batch_size=2, sampler=sampler), events)
    host.fail_on_batch = 2
    progress = TrainingProgress(data_epoch=4)
    driver = OfflineExecutionDriver()
    driver.prepare_cycle(host, progress, seed=100)

    with pytest.raises(RuntimeError, match="optimization failed"):
        driver.run_cycle(host, progress)

    assert events == [
        ("set_epoch", 4),
        ("optimize_batch", [0, 1]),
        ("optimize_batch", [2, 3]),
    ]
    assert progress.data_epoch == 4


def test_offline_driver_requires_official_distributed_sampler_on_one_process() -> None:
    """Single-process offline runs still use the same DistributedSampler contract."""
    events: List[Tuple[str, Any]] = []
    host = _OfflineHostFake(DataLoader(list(range(4)), batch_size=2), events)
    driver = OfflineExecutionDriver()
    progress = TrainingProgress()

    with pytest.raises(TypeError, match="DistributedSampler even when num_replicas=1"):
        driver.prepare_cycle(host, progress, seed=100)

    assert events == []


def test_build_execution_driver_uses_acquisition_and_loader_semantics() -> None:
    """Driver selection follows acquisition after contract combinations validate."""
    online_without_reward = ExecutionContract(
        acquisition=AcquisitionMode.ROLLOUT,
        cycle_unit=CycleUnit.ROLLOUT_ITERATION,
        feedback=FeedbackMode.NONE,
        loader_kind=LoaderKind.GROUPED_ROLLOUT,
    )
    assert isinstance(build_execution_driver(online_without_reward), OnlineExecutionDriver)
    assert isinstance(build_execution_driver(OFFLINE_EXECUTION_CONTRACT), OfflineExecutionDriver)


def test_build_execution_driver_rejects_untyped_contract() -> None:
    """Driver selection cannot silently infer semantics from a raw mapping."""
    with pytest.raises(TypeError, match="contract to be ExecutionContract"):
        build_execution_driver({"acquisition": "dataset"})


def test_builtin_trainers_declare_feedback_independently_from_paradigm() -> None:
    """Reward-free distillation and reward-based TDM-R1 select explicit contracts."""
    for trainer_name in ("diffusion-opd", "dmd2", "tdm"):
        trainer_class = get_trainer_class(trainer_name)
        assert trainer_class.execution_contract is ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT

    tdm_r1_class = get_trainer_class("tdm-r1")
    assert tdm_r1_class.execution_contract is ONLINE_EXECUTION_CONTRACT


@pytest.mark.parametrize("driver", [OnlineExecutionDriver(), OfflineExecutionDriver()])
def test_execution_drivers_reject_untyped_progress(driver) -> None:
    """Cycle preparation requires the immutable progress object used by the outer loop."""
    host = _OnlineHostFake()

    with pytest.raises(TypeError, match="progress to be TrainingProgress"):
        driver.prepare_cycle(host, object(), seed=100)


@pytest.mark.parametrize("driver", [OnlineExecutionDriver(), OfflineExecutionDriver()])
def test_execution_driver_run_rejects_untyped_progress(driver) -> None:
    """Cycle dispatch cannot proceed with an untyped progress object."""
    host = _OnlineHostFake()

    with pytest.raises(TypeError, match="progress to be TrainingProgress"):
        driver.run_cycle(host, object())


@pytest.mark.parametrize("driver", [OnlineExecutionDriver(), OfflineExecutionDriver()])
def test_execution_drivers_reject_boolean_seed(driver) -> None:
    """The common driver boundary rejects bool masquerading as an integer seed."""
    host = _OnlineHostFake()

    with pytest.raises(TypeError, match="seed to be int"):
        driver.prepare_cycle(host, TrainingProgress(), seed=True)
