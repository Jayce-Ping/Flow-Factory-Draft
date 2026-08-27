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

"""Lifecycle tests for trainer progress, EMA, and reference state resume."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType

from flow_factory.contracts.execution import OFFLINE_EXECUTION_CONTRACT
from flow_factory.hparams.training_args.dpo import DPOTrainingArguments
from flow_factory.hparams.training_args.offline_dpo import OfflineDPOTrainingArguments
from flow_factory.models.abc import BaseAdapter
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.common.runtime_state import (
    TRAINER_RUNTIME_METADATA_FILENAME,
    TrainerRuntimeState,
)
from flow_factory.trainers.execution import TrainingProgress


class _LifecycleAdapter(BaseAdapter):
    """Exercise BaseAdapter's real state-resume and EMA/reference lifecycle."""

    def __init__(
        self,
        accelerator: Accelerator,
        training_args: OfflineDPOTrainingArguments,
        model_args: SimpleNamespace,
        *,
        initial_policy: float,
    ) -> None:
        self.accelerator = accelerator
        self.training_args = training_args
        self.model_args = model_args
        self.policy = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.policy.weight.fill_(initial_policy)
        self.reference_before_attach: torch.Tensor | None = None

    def load_pipeline(self) -> object:
        """Satisfy the adapter abstract contract."""
        raise NotImplementedError

    def decode_latents(self, *args: Any, **kwargs: Any) -> object:
        """Satisfy the adapter abstract contract."""
        raise NotImplementedError

    def inference(self, *args: Any, **kwargs: Any) -> object:
        """Satisfy the adapter abstract contract."""
        raise NotImplementedError

    def forward(self, *args: Any, **kwargs: Any) -> object:
        """Satisfy the adapter abstract contract."""
        raise NotImplementedError

    def _ema_tracked_parameters(self) -> list[torch.nn.Parameter]:
        """Track the prepared policy parameter used by this test."""
        return list(self.policy.parameters())

    def runtime_state_children(self):
        """Record the temporary reference before deferred checkpoint restoration."""
        children = super().runtime_state_children()
        reference = children.get("reference")
        if reference is not None:
            self.reference_before_attach = reference.ema_parameters[0].detach().clone()
        return children

    def log_trainable_parameters(self) -> None:
        """Skip production logging for the tiny policy."""


class _LifecycleTrainer(BaseTrainer):
    """Run BaseTrainer initialization while replacing unrelated production stages."""

    paradigm = "decoupled"
    execution_contract = OFFLINE_EXECUTION_CONTRACT

    def _validate_adapter_execution_contract(self) -> None:
        """Keep this checkpoint-only fixture independent from model output codecs."""

    def _initialization(self) -> None:
        """Prepare one policy and optimizer as Accelerate training-state roots."""
        self.optimizer = torch.optim.AdamW(self.adapter.policy.parameters(), lr=0.01)
        self.model_bundle, self.optimizer = self.accelerator.prepare(
            self.adapter.policy,
            self.optimizer,
        )
        self.adapter.policy = self.model_bundle

    def _apply_shared_acceleration(self) -> None:
        """Skip acceleration plugins in the lifecycle fixture."""

    def _init_logging_backend(self) -> None:
        """Skip external logger initialization."""
        self.logger = None

    @staticmethod
    def _patch_deepspeed_autocast(accelerator: Accelerator) -> None:
        """Skip backend patching for a CPU fixture."""
        del accelerator

    def optimize_batch(self, batch: Any) -> None:
        """Satisfy the offline execution hook contract."""
        del batch


class _TwoRoleRegistrationTrainer(_LifecycleTrainer):
    """Declare two roles solely to test custom checkpoint registration order."""

    def _required_trainable_roles(self) -> tuple[str, ...]:
        """Return a deterministic two-role layout."""
        return ("base", "fake")


class _OnlineLifecycleTrainer(_LifecycleTrainer):
    """Exercise the legacy online lifecycle without offline runtime registration."""

    execution_contract = DPOTrainingArguments.execution_contract

    def optimize(self, samples: Any) -> None:
        """Satisfy the online execution hook contract."""
        del samples


def _config(
    *,
    resume_path: str | None = None,
    ema_decay: float = 0.9,
) -> SimpleNamespace:
    """Build the narrow configuration surface consumed by BaseTrainer."""
    training_args = OfflineDPOTrainingArguments(
        gradient_accumulation_steps=1,
        ema_decay=ema_decay,
        ema_update_interval=1,
        ema_device="cpu",
        ref_param_device="cpu",
    )
    model_args = SimpleNamespace(
        finetune_type="full",
        resume_path=resume_path,
        resume_type="state" if resume_path is not None else None,
    )
    return SimpleNamespace(
        training_args=training_args,
        model_args=model_args,
        log_args=SimpleNamespace(verbose=False),
        eval_args=SimpleNamespace(),
        reward_args=(),
        eval_reward_args=(),
    )


def _trainer(
    *,
    initial_policy: float,
    resume_path: str | None = None,
    ema_decay: float = 0.9,
) -> _LifecycleTrainer:
    """Construct one complete tiny trainer through BaseTrainer.__init__."""
    accelerator = Accelerator(cpu=True)
    config = _config(resume_path=resume_path, ema_decay=ema_decay)
    adapter = _LifecycleAdapter(
        accelerator,
        config.training_args,
        config.model_args,
        initial_policy=initial_policy,
    )
    return _LifecycleTrainer(accelerator=accelerator, config=config, adapter=adapter)


def _parameter_value(parameter: torch.Tensor) -> float:
    """Return the scalar carried by the tiny one-parameter policy."""
    return parameter.detach().cpu().item()


def _flip_first_byte(path: Path) -> None:
    """Corrupt one artifact without changing its recorded file size."""
    payload = bytearray(path.read_bytes())
    assert payload
    payload[0] ^= 1
    path.write_bytes(payload)


def test_full_offline_dpo_state_resume_restores_progress_ema_and_frozen_reference(
    tmp_path: Path,
) -> None:
    """Reference state comes from the checkpoint, never the restored live policy."""
    checkpoint_dir = tmp_path / "state"
    source = _trainer(initial_policy=2.0)
    assert source._runtime_state.child_names == ("ema", "reference")
    assert source._runtime_state.pending_child_names == ()
    assert "_lightweight_progress" not in source.__dict__

    with torch.no_grad():
        source.adapter.policy.weight.fill_(9.0)
        source.adapter.ema_wrapper.ema_parameters[0].fill_(5.0)
        source.adapter._ref_ema.ema_parameters[0].fill_(2.0)
    source.adapter.ema_wrapper.num_updates = 11
    source.step = 7
    source.epoch = 3
    source.adapter.save_checkpoint(checkpoint_dir, model_only=False)

    assert (checkpoint_dir / TRAINER_RUNTIME_METADATA_FILENAME).is_file()
    assert not tuple(checkpoint_dir.glob("custom_checkpoint_*.pkl"))
    assert not tuple(tmp_path.glob(".state.incomplete-*"))
    target = _trainer(initial_policy=100.0, resume_path=str(checkpoint_dir))

    assert _parameter_value(target.adapter.policy.weight) == pytest.approx(9.0)
    # Offline preflight constructs a shape/dtype-compatible child before policy
    # restore, then deferred attachment overwrites it from the checkpoint payload.
    assert _parameter_value(target.adapter.reference_before_attach) == pytest.approx(100.0)
    assert _parameter_value(target.adapter.ema_wrapper.ema_parameters[0]) == pytest.approx(5.0)
    assert target.adapter.ema_wrapper.num_updates == 11
    assert _parameter_value(target.adapter._ref_ema.ema_parameters[0]) == pytest.approx(2.0)
    assert target.progress == TrainingProgress(optimizer_step=7, data_epoch=3)
    assert target.progress is target._runtime_state.progress
    assert target._runtime_state.pending_child_names == ()
    assert "_lightweight_progress" not in target.__dict__


def test_legacy_state_checkpoint_is_rejected_before_policy_mutation(tmp_path: Path) -> None:
    """Pre-runtime state checkpoints require model-only resume instead of unsafe fallback."""
    checkpoint_dir = tmp_path / "legacy"
    source_accelerator = Accelerator(cpu=True)
    source_model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        source_model.weight.fill_(7.0)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.01)
    source_model, source_optimizer = source_accelerator.prepare(
        source_model,
        source_optimizer,
    )
    source_accelerator.save_state(checkpoint_dir)

    target_accelerator = Accelerator(cpu=True)
    config = _config(resume_path=str(checkpoint_dir))
    adapter = _LifecycleAdapter(
        target_accelerator,
        config.training_args,
        config.model_args,
        initial_policy=100.0,
    )
    with pytest.raises(
        RuntimeError,
        match="before safe runtime-state v1.*resume their model weights",
    ):
        _LifecycleTrainer(
            accelerator=target_accelerator,
            config=config,
            adapter=adapter,
        )

    assert _parameter_value(adapter.policy.weight) == pytest.approx(100.0)


def test_missing_rng_state_is_rejected_before_policy_mutation(tmp_path: Path) -> None:
    """Exact resume cannot silently continue with a fresh random stream."""
    checkpoint_dir = tmp_path / "state"
    source = _trainer(initial_policy=4.0)
    source.adapter.save_checkpoint(checkpoint_dir, model_only=False)
    (checkpoint_dir / "random_states_0.pkl").unlink()

    target_accelerator = Accelerator(cpu=True)
    config = _config(resume_path=str(checkpoint_dir))
    adapter = _LifecycleAdapter(
        target_accelerator,
        config.training_args,
        config.model_args,
        initial_policy=100.0,
    )
    with pytest.raises(RuntimeError, match="state artifact is missing"):
        _LifecycleTrainer(
            accelerator=target_accelerator,
            config=config,
            adapter=adapter,
        )

    assert _parameter_value(adapter.policy.weight) == pytest.approx(100.0)


def test_same_size_optimizer_corruption_is_rejected_before_policy_mutation(
    tmp_path: Path,
) -> None:
    """Artifact digests prevent Accelerate from partially restoring a corrupt state."""
    checkpoint_dir = tmp_path / "state"
    source = _trainer(initial_policy=4.0)
    source.adapter.save_checkpoint(checkpoint_dir, model_only=False)
    _flip_first_byte(checkpoint_dir / "optimizer.bin")

    target_accelerator = Accelerator(cpu=True)
    config = _config(resume_path=str(checkpoint_dir))
    adapter = _LifecycleAdapter(
        target_accelerator,
        config.training_args,
        config.model_args,
        initial_policy=100.0,
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch before resume"):
        _LifecycleTrainer(
            accelerator=target_accelerator,
            config=config,
            adapter=adapter,
        )

    assert _parameter_value(adapter.policy.weight) == pytest.approx(100.0)


def test_same_size_runtime_tensor_corruption_is_rejected_before_policy_mutation(
    tmp_path: Path,
) -> None:
    """EMA/reference tensor corruption is detected before the policy is restored."""
    checkpoint_dir = tmp_path / "state"
    source = _trainer(initial_policy=4.0)
    source.adapter.save_checkpoint(checkpoint_dir, model_only=False)
    metadata = json.loads(
        (checkpoint_dir / TRAINER_RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    _flip_first_byte(checkpoint_dir / metadata["tensor_file"]["path"])

    target_accelerator = Accelerator(cpu=True)
    config = _config(resume_path=str(checkpoint_dir))
    adapter = _LifecycleAdapter(
        target_accelerator,
        config.training_args,
        config.model_args,
        initial_policy=100.0,
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch before resume"):
        _LifecycleTrainer(
            accelerator=target_accelerator,
            config=config,
            adapter=adapter,
        )

    assert _parameter_value(adapter.policy.weight) == pytest.approx(100.0)


def test_pickle_custom_state_is_rejected_before_policy_mutation(tmp_path: Path) -> None:
    """Offline state resume never delegates a custom payload to unsafe pickle load."""
    checkpoint_dir = tmp_path / "state"
    source = _trainer(initial_policy=4.0)
    source.adapter.save_checkpoint(checkpoint_dir, model_only=False)
    (checkpoint_dir / "custom_checkpoint_0.pkl").write_bytes(b"not deserialized")

    target_accelerator = Accelerator(cpu=True)
    config = _config(resume_path=str(checkpoint_dir))
    adapter = _LifecycleAdapter(
        target_accelerator,
        config.training_args,
        config.model_args,
        initial_policy=100.0,
    )
    with pytest.raises(RuntimeError, match="pickle custom state"):
        _LifecycleTrainer(
            accelerator=target_accelerator,
            config=config,
            adapter=adapter,
        )

    assert _parameter_value(adapter.policy.weight) == pytest.approx(100.0)


def test_runtime_metadata_rejects_child_config_drift_before_policy_mutation(
    tmp_path: Path,
) -> None:
    """A changed EMA/reference declaration cannot load custom files by position."""
    checkpoint_dir = tmp_path / "state"
    source = _trainer(initial_policy=4.0)
    source.adapter.save_checkpoint(checkpoint_dir, model_only=False)
    metadata_path = checkpoint_dir / TRAINER_RUNTIME_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["child_names"] = ["ema"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    target_accelerator = Accelerator(cpu=True)
    config = _config(resume_path=str(checkpoint_dir))
    adapter = _LifecycleAdapter(
        target_accelerator,
        config.training_args,
        config.model_args,
        initial_policy=100.0,
    )
    with pytest.raises(ValueError, match="child_names mismatch.*ema.*reference"):
        _LifecycleTrainer(
            accelerator=target_accelerator,
            config=config,
            adapter=adapter,
        )

    assert _parameter_value(adapter.policy.weight) == pytest.approx(100.0)


def test_runtime_payload_rejects_ema_config_drift_before_policy_mutation(
    tmp_path: Path,
) -> None:
    """Child payload compatibility is preflighted before Accelerate restores policy state."""
    checkpoint_dir = tmp_path / "state"
    source = _trainer(initial_policy=4.0, ema_decay=0.9)
    source.adapter.save_checkpoint(checkpoint_dir, model_only=False)

    target_accelerator = Accelerator(cpu=True)
    config = _config(resume_path=str(checkpoint_dir), ema_decay=0.8)
    adapter = _LifecycleAdapter(
        target_accelerator,
        config.training_args,
        config.model_args,
        initial_policy=100.0,
    )
    with pytest.raises(ValueError, match="EMA state decay mismatch"):
        _LifecycleTrainer(
            accelerator=target_accelerator,
            config=config,
            adapter=adapter,
        )

    assert _parameter_value(adapter.policy.weight) == pytest.approx(100.0)


def test_runtime_metadata_rejects_algorithm_identity_drift_before_policy_mutation(
    tmp_path: Path,
) -> None:
    """A state checkpoint cannot silently cross an algorithm boundary."""
    checkpoint_dir = tmp_path / "state"
    source = _trainer(initial_policy=4.0)
    source.adapter.save_checkpoint(checkpoint_dir, model_only=False)
    metadata_path = checkpoint_dir / TRAINER_RUNTIME_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["identity"]["algorithm"] = "sft"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    target_accelerator = Accelerator(cpu=True)
    config = _config(resume_path=str(checkpoint_dir))
    adapter = _LifecycleAdapter(
        target_accelerator,
        config.training_args,
        config.model_args,
        initial_policy=100.0,
    )
    with pytest.raises(ValueError, match="metadata identity mismatch"):
        _LifecycleTrainer(
            accelerator=target_accelerator,
            config=config,
            adapter=adapter,
        )

    assert _parameter_value(adapter.policy.weight) == pytest.approx(100.0)


def test_runtime_metadata_rejects_parameter_schema_drift_before_policy_mutation(
    tmp_path: Path,
) -> None:
    """Same-shaped parameters cannot be rebound after canonical schema drift."""
    checkpoint_dir = tmp_path / "state"
    source = _trainer(initial_policy=4.0)
    source.adapter.save_checkpoint(checkpoint_dir, model_only=False)
    metadata_path = checkpoint_dir / TRAINER_RUNTIME_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["identity"]["parameter_schema_digest"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    target_accelerator = Accelerator(cpu=True)
    config = _config(resume_path=str(checkpoint_dir))
    adapter = _LifecycleAdapter(
        target_accelerator,
        config.training_args,
        config.model_args,
        initial_policy=100.0,
    )
    with pytest.raises(ValueError, match="metadata identity mismatch"):
        _LifecycleTrainer(
            accelerator=target_accelerator,
            config=config,
            adapter=adapter,
        )

    assert _parameter_value(adapter.policy.weight) == pytest.approx(100.0)


def test_adapter_state_save_runs_runtime_preflight_before_accelerator_write(
    tmp_path: Path,
) -> None:
    """Framework state saves cannot leave a backend checkpoint after preflight failure."""
    adapter = object.__new__(_LifecycleAdapter)
    events: list[str] = []
    adapter.accelerator = SimpleNamespace(
        is_main_process=False,
        save_state=lambda *args, **kwargs: events.append("accelerator_save"),
    )

    def reject_preflight(output_dir: str) -> None:
        events.append(f"preflight:{output_dir}")
        raise RuntimeError("incompatible runtime state")

    adapter._trainer_runtime_save_preflight = reject_preflight

    with pytest.raises(RuntimeError, match="incompatible runtime state"):
        adapter.save_checkpoint(str(tmp_path / "state"), model_only=False)

    assert events == [f"preflight:{tmp_path / 'state'}"]


def test_offline_state_save_publishes_only_after_accelerator_success(
    tmp_path: Path,
) -> None:
    """A partial backend write never appears under the final checkpoint path."""
    destination = tmp_path / "state"
    source = _trainer(initial_policy=4.0)

    def fail_after_partial_write(output_dir: str, **kwargs: Any) -> None:
        del kwargs
        partial_dir = Path(output_dir)
        partial_dir.mkdir(parents=True, exist_ok=True)
        (partial_dir / "partial.bin").write_bytes(b"partial")
        raise OSError("simulated backend write failure")

    source.accelerator.save_state = fail_after_partial_write

    with pytest.raises(OSError, match="simulated backend write failure"):
        source.adapter.save_checkpoint(destination, model_only=False)

    assert not destination.exists()
    assert tuple(tmp_path.glob(".state.incomplete-*"))


def test_resumed_cycle_skips_republishing_its_immutable_checkpoint(
    tmp_path: Path,
) -> None:
    """Resume enters the saved cycle without attempting to overwrite its source."""
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_dir = checkpoint_root / "checkpoint-3"
    source = _trainer(initial_policy=4.0)
    source.epoch = 3
    source.adapter.save_checkpoint(checkpoint_dir, model_only=False)

    target = _trainer(initial_policy=100.0, resume_path=str(checkpoint_dir))
    target.log_args.save_model_only = False

    target.save_checkpoint(str(checkpoint_root), epoch=3)

    assert checkpoint_dir.is_dir()
    assert target.progress == TrainingProgress(data_epoch=3)


def test_online_lifecycle_does_not_register_partial_runtime_state() -> None:
    """Online snapshots remain on the existing all-algorithm migration boundary."""
    accelerator = Accelerator(cpu=True)
    training_args = DPOTrainingArguments(
        gradient_accumulation_steps=1,
        ema_decay=0.0,
    )
    config = _config()
    config.training_args = training_args
    adapter = _LifecycleAdapter(
        accelerator,
        training_args,
        config.model_args,
        initial_policy=3.0,
    )

    trainer = _OnlineLifecycleTrainer(
        accelerator=accelerator,
        config=config,
        adapter=adapter,
    )

    assert "_runtime_state" not in trainer.__dict__
    assert accelerator._custom_objects == []
    assert trainer.progress == TrainingProgress()


def test_offline_runtime_rejects_multirole_pickle_custom_state() -> None:
    """Future offline multi-role state needs migration to the safe runtime format."""
    trainer = object.__new__(_TwoRoleRegistrationTrainer)
    trainer.accelerator = Accelerator(cpu=True)
    trainer.adapter = SimpleNamespace()
    trainer.model_args = SimpleNamespace(resume_path=None, resume_type=None)
    trainer._runtime_state = TrainerRuntimeState()

    trainer._register_runtime_checkpointing()
    trainer._register_multirole_checkpointing()
    with pytest.raises(RuntimeError, match="custom_objects.*pickle"):
        trainer._finalize_runtime_checkpointing()

    assert trainer.accelerator._custom_objects == [trainer._multirole_checkpoint_state]


def test_single_process_backend_allows_safe_runtime_state() -> None:
    """Safe runtime-state v1 supports the complete local atomic path."""
    trainer = object.__new__(_LifecycleTrainer)
    trainer._runtime_state = TrainerRuntimeState(child_names=("ema", "reference"))
    trainer.accelerator = SimpleNamespace(
        distributed_type=DistributedType.NO,
        num_processes=1,
    )

    trainer._validate_runtime_child_checkpoint_backend("resume training state")


@pytest.mark.parametrize(
    "distributed_type",
    [DistributedType.MULTI_CPU, DistributedType.FSDP, DistributedType.DEEPSPEED],
)
def test_distributed_backends_require_model_only_checkpoint(
    distributed_type: DistributedType,
) -> None:
    """Distributed training remains supported without partial exact-state claims."""
    trainer = object.__new__(_LifecycleTrainer)
    trainer._runtime_state = TrainerRuntimeState(child_names=("ema", "reference"))
    trainer.accelerator = SimpleNamespace(
        distributed_type=distributed_type,
        num_processes=2,
    )

    with pytest.raises(RuntimeError, match="single-process only.*model-only"):
        trainer._validate_runtime_child_checkpoint_backend("resume training state")


def test_lightweight_trainer_uses_one_lazy_progress_fallback() -> None:
    """__new__ fixtures keep legacy ergonomics without duplicating normal state."""
    trainer = object.__new__(_LifecycleTrainer)

    trainer.step = 4
    trainer.epoch = 2

    assert trainer.progress == TrainingProgress(optimizer_step=4, data_epoch=2)
    assert trainer.progress is trainer.__dict__["_lightweight_progress"]
    assert "_runtime_state" not in trainer.__dict__
