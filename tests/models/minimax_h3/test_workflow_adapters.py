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

from types import SimpleNamespace
from typing import Any, ClassVar, Dict, List

import pytest
import torch
import torch.nn as nn
from accelerate import DistributedType

from flow_factory.models.abc import BaseAdapter
from flow_factory.models.minimax_h3.adapters import (
    MiniMaxH3FL2VAAdapter,
    MiniMaxH3Ref2VAAdapter,
    MiniMaxH3T2VAAdapter,
)
from flow_factory.models.runtime import ModularPipelineRuntime
from flow_factory.scheduler import MiniMaxH3SDEScheduler


class UpstreamSchedulerFake:
    """Represent the lazy upstream scheduler replaced by Flow-Factory."""


class TransformerFake(nn.Module):
    """Provide one parameter for BaseAdapter freeze and precision setup."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class WorkflowPipelineFake:
    """Record exact workflow loading and named modular materialization."""

    calls: ClassVar[List[tuple[str, str]]] = []
    component_overrides: ClassVar[Dict[str, Any]] = {}

    def __init__(self, workflow: str) -> None:
        transformer_name = "transformer_ref" if workflow == "ref2va" else "transformer"
        self.workflow = workflow
        self.components = {
            "scheduler": "scheduler spec",
            "text_encoder": "text encoder spec",
            "tokenizer": "tokenizer spec",
            "processor": "processor spec",
            "image_processor": "image processor spec",
            "video_processor": "video processor spec",
            "vae": "video vae spec",
            "audio_vae": "audio vae spec",
            transformer_name: "transformer spec",
            "unrelated": "must stay lazy",
            **self.component_overrides,
        }
        self.load_calls: List[List[str]] = []

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *, workflow: str) -> "WorkflowPipelineFake":
        cls.calls.append((model_name_or_path, workflow))
        return cls(workflow)

    def load_components(self, names: List[str]) -> None:
        self.load_calls.append(list(names))
        for name in names:
            if name == "scheduler":
                setattr(self, name, UpstreamSchedulerFake())
            elif name in ("transformer", "transformer_ref"):
                setattr(self, name, TransformerFake())
            else:
                setattr(self, name, SimpleNamespace())


class AcceleratorFake:
    """Provide the BaseAdapter construction surface without distributed setup."""

    device = torch.device("cpu")
    distributed_type = DistributedType.NO
    is_fsdp2 = False
    mixed_precision = "no"

    def unwrap_model(self, module: nn.Module) -> nn.Module:
        return module


def _config(target_components: List[str]) -> SimpleNamespace:
    return SimpleNamespace(
        model_args=SimpleNamespace(
            model_name_or_path="MiniMaxAI/MiniMax-H3",
            resume_path=None,
            resume_type=None,
            finetune_type="full",
            target_components=target_components,
            target_modules="all",
            trainable_parameters_dtype=torch.float32,
            frozen_parameters_dtype=None,
        ),
        training_args=SimpleNamespace(
            enable_gradient_checkpointing=False,
            latent_storage_dtype=None,
        ),
        eval_args=SimpleNamespace(),
        scheduler_args=SimpleNamespace(
            dynamics_type="Flow-SDE",
            noise_level=0.7,
            sde_steps=[0, 2],
            num_sde_steps=1,
            seed=17,
        ),
        mixed_precision="no",
    )


@pytest.fixture(autouse=True)
def _fake_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    WorkflowPipelineFake.calls.clear()
    WorkflowPipelineFake.component_overrides = {}
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.require_minimax_h3_support",
        lambda: SimpleNamespace(MiniMaxH3ModularPipeline=WorkflowPipelineFake),
    )


@pytest.mark.parametrize(
    ("adapter_class", "workflow", "transformer_name", "preprocessing_names"),
    [
        (
            MiniMaxH3T2VAAdapter,
            "t2va",
            "transformer",
            ["text_encoder", "tokenizer", "processor"],
        ),
        (
            MiniMaxH3FL2VAAdapter,
            "fl2va",
            "transformer",
            ["image_processor", "text_encoder", "tokenizer", "processor", "vae"],
        ),
        (
            MiniMaxH3Ref2VAAdapter,
            "ref2va",
            "transformer_ref",
            [
                "image_processor",
                "text_encoder",
                "tokenizer",
                "processor",
                "vae",
                "audio_vae",
            ],
        ),
    ],
)
def test_workflow_adapter_loads_pruned_runtime_and_exact_setup_components(
    adapter_class: type[BaseAdapter],
    workflow: str,
    transformer_name: str,
    preprocessing_names: List[str],
) -> None:
    adapter = adapter_class(_config([transformer_name]), AcceleratorFake())

    assert adapter_class.__bases__ == (BaseAdapter,)
    assert WorkflowPipelineFake.calls == [("MiniMaxAI/MiniMax-H3", workflow)]
    assert isinstance(adapter.component_runtime, ModularPipelineRuntime)
    assert adapter.pipeline.load_calls == [[transformer_name], ["scheduler"]]
    assert not hasattr(adapter.pipeline, "unrelated")
    assert transformer_name in adapter.component_runtime.materialized_component_names
    opposite = "transformer_ref" if transformer_name == "transformer" else "transformer"
    assert opposite not in adapter.component_runtime.declared_component_names

    adapter.on_load_components(adapter.preprocessing_modules, device="cpu")

    assert adapter.preprocessing_modules == preprocessing_names
    assert adapter.pipeline.load_calls[-1] == preprocessing_names
    assert not hasattr(adapter.pipeline, "unrelated")


@pytest.mark.parametrize(
    ("adapter_class", "workflow", "required_name", "opposite_name"),
    [
        (MiniMaxH3T2VAAdapter, "t2va", "transformer", "transformer_ref"),
        (MiniMaxH3FL2VAAdapter, "fl2va", "transformer", "transformer_ref"),
        (MiniMaxH3Ref2VAAdapter, "ref2va", "transformer_ref", "transformer"),
    ],
)
def test_workflow_adapter_rejects_missing_or_opposite_transformer_partition(
    adapter_class: type[BaseAdapter],
    workflow: str,
    required_name: str,
    opposite_name: str,
) -> None:
    WorkflowPipelineFake.component_overrides = {
        required_name: None,
        opposite_name: "opposite transformer spec",
    }

    with pytest.raises(
        ValueError,
        match=rf"workflow='{workflow}'.*required.*{required_name}.*opposite.*{opposite_name}",
    ):
        adapter_class(_config([required_name]), AcceleratorFake())


@pytest.mark.parametrize(
    ("adapter_class", "workflow", "expected_target", "received_targets"),
    [
        (MiniMaxH3T2VAAdapter, "t2va", "transformer", ["transformer_ref"]),
        (
            MiniMaxH3FL2VAAdapter,
            "fl2va",
            "transformer",
            ["transformer", "audio_vae"],
        ),
        (
            MiniMaxH3Ref2VAAdapter,
            "ref2va",
            "transformer_ref",
            ["transformer", "transformer_ref"],
        ),
    ],
)
def test_target_partition_fails_before_checkpoint_or_lora_setup(
    monkeypatch: pytest.MonkeyPatch,
    adapter_class: type[BaseAdapter],
    workflow: str,
    expected_target: str,
    received_targets: List[str],
) -> None:
    checkpoint_calls: List[Any] = []
    lora_calls: List[Any] = []
    monkeypatch.setattr(
        BaseAdapter,
        "load_checkpoint",
        lambda *args, **kwargs: checkpoint_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        BaseAdapter,
        "apply_lora",
        lambda *args, **kwargs: lora_calls.append((args, kwargs)),
    )
    config = _config(received_targets)
    config.model_args.resume_path = "checkpoint"
    config.model_args.resume_type = "lora"
    config.model_args.finetune_type = "lora"

    with pytest.raises(
        ValueError,
        match=rf"workflow='{workflow}'.*target_components.*\['{expected_target}'\].*{received_targets!r}",
    ):
        adapter_class(config, AcceleratorFake())

    assert WorkflowPipelineFake.calls == []
    assert checkpoint_calls == []
    assert lora_calls == []


@pytest.mark.parametrize(
    ("adapter_class", "transformer_name"),
    [
        (MiniMaxH3T2VAAdapter, "transformer"),
        (MiniMaxH3FL2VAAdapter, "transformer"),
        (MiniMaxH3Ref2VAAdapter, "transformer_ref"),
    ],
)
def test_adapter_builds_fresh_ordered_video_audio_schedulers(
    adapter_class: type[BaseAdapter], transformer_name: str
) -> None:
    adapter = adapter_class(_config([transformer_name]), AcceleratorFake())

    assert isinstance(adapter.scheduler, MiniMaxH3SDEScheduler)
    assert isinstance(adapter.audio_scheduler, MiniMaxH3SDEScheduler)
    assert adapter.scheduler is not adapter.audio_scheduler
    assert adapter.scheduler.shift == 12.0
    assert adapter.audio_scheduler.shift == 3.0
    assert adapter.scheduler_group.names == ("video", "audio")
    assert adapter.scheduler_group.primary_name == "video"
    assert adapter.scheduler_group.primary is adapter.scheduler
    for scheduler in (adapter.scheduler, adapter.audio_scheduler):
        assert scheduler.noise_level == 0.7
        assert scheduler.seed == 17
        assert scheduler.dynamics_type == "Flow-SDE"
