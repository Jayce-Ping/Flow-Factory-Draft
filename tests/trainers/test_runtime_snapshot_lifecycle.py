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

"""Regression coverage for algorithm-owned exact-resume snapshots."""

from types import SimpleNamespace
from typing import Any, List

import torch
from accelerate.utils import DistributedType

from flow_factory.models.abc import BaseAdapter
from flow_factory.trainers.common.runtime_state import TrainerRuntimeState
from flow_factory.trainers.distillation.opd.trainer import DiffusionOPDTrainer
from flow_factory.trainers.rl.crd import CRDTrainer
from flow_factory.trainers.rl.dgpo import DGPOTrainer


class _SnapshotModule(torch.nn.Module):
    """Expose one trainable parameter to the real snapshot implementation."""

    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))


class _SnapshotAdapter(BaseAdapter):
    """Provide the minimal component runtime needed by named snapshots."""

    def load_pipeline(self) -> Any:
        """Satisfy the adapter abstract contract."""
        raise NotImplementedError

    def decode_latents(self, *args: Any, **kwargs: Any) -> Any:
        """Satisfy the adapter abstract contract."""
        raise NotImplementedError

    def inference(self, *args: Any, **kwargs: Any) -> List[Any]:
        """Satisfy the adapter abstract contract."""
        raise NotImplementedError

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Satisfy the adapter abstract contract."""
        raise NotImplementedError

    def has_component(self, name: str) -> bool:
        """Declare the one snapshot-backed component."""
        return name == "transformer"

    def get_component(self, name: str) -> torch.nn.Module:
        """Return the one snapshot-backed component."""
        if name != "transformer":
            raise KeyError(f"expected component 'transformer', received {name!r}")
        return self.snapshot_module


def _adapter(value: float = 1.0) -> _SnapshotAdapter:
    """Build a lightweight adapter using the production snapshot methods."""
    adapter = object.__new__(_SnapshotAdapter)
    adapter.snapshot_module = _SnapshotModule(value)
    adapter.target_module_map = {"transformer": ["weight"]}
    adapter._named_parameters = {}
    return adapter


def _snapshot_host(
    trainer_type: type,
    training_args: Any,
    *,
    model_args: Any | None = None,
) -> Any:
    """Build a trainer host immediately before its snapshot initialization hook."""
    trainer = object.__new__(trainer_type)
    trainer.training_args = training_args
    trainer.model_args = model_args or SimpleNamespace(
        finetune_type="lora",
        resume_path=None,
        resume_type=None,
    )
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.adapter = _adapter()
    trainer._runtime_children_attached = False
    trainer.runtime_state = TrainerRuntimeState(
        child_names=trainer._algorithm_runtime_child_names()
    )
    return trainer


def _attach_snapshot_children(trainer: Any) -> None:
    """Attach all initialized children as BaseTrainer does after safe resume."""
    trainer._attach_runtime_children(trainer._runtime_checkpoint_children())


def test_crd_named_snapshots_round_trip_through_runtime_state() -> None:
    """CRD old and rollout copies remain exact across runtime child restore."""
    args = SimpleNamespace(ref_param_device="cpu")
    source = _snapshot_host(CRDTrainer, args)
    source._initialize_snapshots()
    with torch.no_grad():
        source.adapter.get_named_parameters(CRDTrainer._OLD_PARAMS_NAME)[0].fill_(2.0)
        source.adapter.get_named_parameters(CRDTrainer._SAMPLING_PARAMS_NAME)[0].fill_(3.0)
    _attach_snapshot_children(source)
    payload = source.runtime_state.state_dict()

    target = _snapshot_host(CRDTrainer, args)
    target._initialize_snapshots()
    target.runtime_state.load_state_dict(payload)
    _attach_snapshot_children(target)

    assert target.runtime_state.child_names == CRDTrainer.runtime_child_names
    torch.testing.assert_close(
        target.adapter.get_named_parameters(CRDTrainer._OLD_PARAMS_NAME)[0],
        torch.tensor(2.0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        target.adapter.get_named_parameters(CRDTrainer._SAMPLING_PARAMS_NAME)[0],
        torch.tensor(3.0),
        rtol=0,
        atol=0,
    )


def test_weight_resume_runs_before_crd_snapshot_initialization() -> None:
    """Model-only resume seeds CRD snapshots from the loaded policy, not init weights."""
    args = SimpleNamespace(ref_param_device="cpu")
    model_args = SimpleNamespace(
        finetune_type="lora",
        resume_path="checkpoint",
        resume_type="lora",
    )
    trainer = _snapshot_host(CRDTrainer, args, model_args=model_args)

    def load_resumed_policy() -> None:
        with torch.no_grad():
            trainer.adapter.snapshot_module.weight.fill_(7.0)

    trainer.adapter.post_init = load_resumed_policy
    trainer._initialize_adapter_runtime()
    trainer._initialize_snapshots()

    for name in CRDTrainer.runtime_child_names:
        torch.testing.assert_close(
            trainer.adapter.get_named_parameters(name)[0],
            torch.tensor(7.0),
            rtol=0,
            atol=0,
        )


def test_dgpo_declares_only_the_configuration_active_ema_reference() -> None:
    """Disabled DGPO clipping pays no snapshot cost; enabled clipping is tracked."""
    disabled_args = SimpleNamespace(
        clip_dsm=False,
        clip_kl=False,
        use_ema_ref=False,
        ema_ref_device="cpu",
        ema_ref_max_decay=0.99,
        ema_ref_ramp_rate=0.01,
    )
    disabled = _snapshot_host(DGPOTrainer, disabled_args)
    disabled._initialize_snapshots()
    assert disabled.runtime_state.child_names == ()
    assert disabled.adapter.list_named_parameters() == []

    enabled_args = SimpleNamespace(**vars(disabled_args))
    enabled_args.clip_dsm = True
    enabled = _snapshot_host(DGPOTrainer, enabled_args)
    enabled._initialize_snapshots()

    assert enabled.runtime_state.child_names == ("ema_ref",)
    assert enabled.adapter.list_named_parameters() == ["ema_ref"]
    assert (
        enabled._runtime_checkpoint_children()["ema_ref"]
        is enabled.adapter._named_parameters["ema_ref"].ema_wrapper
    )
    enabled._validate_runtime_child_coverage()

    enabled.accelerator.distributed_type = DistributedType.DEEPSPEED
    enabled._validate_distributed_runtime_children()
    enabled.accelerator.distributed_type = DistributedType.FSDP
    try:
        enabled._validate_distributed_runtime_children()
    except RuntimeError as error:
        assert "FSDP" in str(error) and "ema_ref" in str(error)
    else:
        raise AssertionError("FSDP named snapshots must fail before core state mutation")


def test_opd_state_resume_predeclares_teachers_without_loading_external_weights() -> None:
    """Teacher child schemas exist before preflight without mutating the student."""
    training_args = SimpleNamespace(
        teachers=[
            SimpleNamespace(name="teacher_a", path="unavailable-a"),
            SimpleNamespace(name="teacher_b", path="unavailable-b"),
        ],
        teacher_param_device="cpu",
    )
    model_args = SimpleNamespace(
        finetune_type="lora",
        resume_path="checkpoint",
        resume_type="state",
    )
    trainer = _snapshot_host(
        DiffusionOPDTrainer,
        training_args,
        model_args=model_args,
    )
    student_before = trainer.adapter.snapshot_module.weight.detach().clone()

    trainer._initialize_snapshots()

    assert trainer.runtime_state.child_names == ("teacher_a", "teacher_b")
    assert trainer._teacher_names == ["teacher_a", "teacher_b"]
    assert trainer.adapter.list_named_parameters() == ["teacher_a", "teacher_b"]
    torch.testing.assert_close(
        trainer.adapter.snapshot_module.weight,
        student_before,
        rtol=0,
        atol=0,
    )
    trainer._validate_runtime_child_coverage()


def test_algorithm_snapshot_names_cannot_shadow_framework_children() -> None:
    """A configured OPD teacher cannot alias EMA/reference/multi-role state."""
    training_args = SimpleNamespace(
        teachers=[SimpleNamespace(name="multirole", path="teacher")],
        teacher_param_device="cpu",
        ema_decay=0.0,
        requires_ref_model=False,
    )
    trainer = object.__new__(DiffusionOPDTrainer)
    trainer.training_args = training_args
    trainer.model_args = SimpleNamespace(finetune_type="lora")
    trainer._required_trainable_roles = lambda: ("base",)

    try:
        trainer._declared_runtime_child_names()
    except ValueError as error:
        assert "framework-reserved" in str(error) and "multirole" in str(error)
    else:
        raise AssertionError("algorithm snapshots must not shadow framework children")
