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
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn
from accelerate import DistributedType

from flow_factory.models.abc import BaseAdapter
from flow_factory.models.model_bundle import ModelBundle, RoutedComponentProxy
from flow_factory.models.runtime import (
    ClassicPipelineRuntime,
    ModularPipelineRuntime,
    PseudoPipelineRuntime,
)


class TrackingModule(nn.Module):
    """Small module that records requested device moves."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.moves: List[str] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Scale an input by the tracked parameter."""
        return value * self.weight

    def to(self, *args: Any, **kwargs: Any) -> "TrackingModule":
        """Record a device move without requiring accelerator hardware."""
        device = kwargs.get("device", args[0] if args else None)
        self.moves.append(str(device))
        return self


class ClassicPipelineFake:
    """Eager pipeline-like container used by classic runtime tests."""

    def __init__(self) -> None:
        self.text_encoder = TrackingModule()
        self.text_encoder_2 = TrackingModule()
        self.transformer = TrackingModule()
        self.vae = TrackingModule()
        self.scheduler = object()

    @property
    def components(self) -> Dict[str, Any]:
        """Expose canonical eager components."""
        return {
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "transformer": self.transformer,
            "vae": self.vae,
            "scheduler": self.scheduler,
        }


class ModularPipelineFake:
    """Lazy modular pipeline that materializes requested component names."""

    def __init__(self, unavailable: List[str] | None = None) -> None:
        self.components = {
            "text_encoder": "text encoder spec",
            "transformer": "transformer spec",
            "vae": "vae spec",
        }
        self.unavailable = set(unavailable or [])
        self.load_calls: List[List[str]] = []

    def load_components(self, names: List[str]) -> None:
        """Materialize every available requested component."""
        self.load_calls.append(list(names))
        for name in names:
            if name not in self.unavailable:
                setattr(self, name, TrackingModule())


def test_classic_runtime_prefers_prepared_component_over_canonical() -> None:
    pipeline = ClassicPipelineFake()
    runtime = ClassicPipelineRuntime(pipeline)
    prepared = TrackingModule()

    runtime.set_prepared_component("transformer", prepared)

    assert runtime.get_component("transformer") is prepared
    assert runtime.get_canonical_component("transformer") is pipeline.transformer
    assert runtime.component_names == [
        "text_encoder",
        "text_encoder_2",
        "transformer",
        "vae",
    ]


def test_pseudo_runtime_uses_explicit_components_and_expands_groups() -> None:
    pipeline = SimpleNamespace()
    components = {
        "text_encoder": TrackingModule(),
        "text_encoder_2": TrackingModule(),
        "transformer": TrackingModule(),
        "transformer_2": TrackingModule(),
        "vae": TrackingModule(),
    }
    runtime = PseudoPipelineRuntime(pipeline, components)

    assert runtime.get_component("vae") is components["vae"]
    assert runtime.resolve_component_names(["text_encoders", "transformers", "vae"]) == [
        "text_encoder",
        "text_encoder_2",
        "transformer",
        "transformer_2",
        "vae",
    ]


def test_modular_runtime_materializes_only_selected_names() -> None:
    pipeline = ModularPipelineFake()
    runtime = ModularPipelineRuntime(pipeline)

    runtime.materialize_components(["vae"])

    assert pipeline.load_calls == [["vae"]]
    assert runtime.get_component("vae") is pipeline.vae
    assert not hasattr(pipeline, "transformer")


def test_modular_runtime_reports_materialization_failure_context() -> None:
    pipeline = ModularPipelineFake(unavailable=["transformer"])
    runtime = ModularPipelineRuntime(pipeline)

    with pytest.raises(
        RuntimeError,
        match=r"expected.*transformer.*received.*text_encoder.*vae",
    ):
        runtime.materialize_components(["transformer"])


def test_stage_load_and_unload_move_only_non_prepared_modules() -> None:
    pipeline = ClassicPipelineFake()
    runtime = ClassicPipelineRuntime(pipeline)
    prepared = TrackingModule()
    runtime.set_prepared_component("transformer", prepared)

    runtime.load_components(["transformer", "vae"], device="stage-device")
    runtime.unload_components(["transformer", "vae"])

    assert prepared.moves == []
    assert pipeline.transformer.moves == []
    assert pipeline.vae.moves == ["stage-device", "cpu"]


def test_prepared_modular_component_never_loads_or_moves_canonical_component() -> None:
    pipeline = ModularPipelineFake()
    runtime = ModularPipelineRuntime(pipeline)
    prepared = TrackingModule()
    runtime.set_prepared_component("transformer", prepared)

    runtime.load_components(["transformer"], device="stage-device")
    runtime.unload_components(["transformer"])

    assert pipeline.load_calls == []
    assert prepared.moves == []
    assert not hasattr(pipeline, "transformer")


class AcceleratorFake:
    """Minimal accelerator surface used during adapter construction."""

    device = torch.device("cpu")
    distributed_type = DistributedType.NO
    is_fsdp2 = False
    mixed_precision = "no"

    def unwrap_model(self, module: nn.Module) -> nn.Module:
        """Return an unwrapped module unchanged."""
        return module


class ExistingStyleAdapterFake(BaseAdapter):
    """Adapter implementing only the existing four abstract methods."""

    def load_pipeline(self) -> ClassicPipelineFake:
        """Return a small eager pipeline."""
        return ClassicPipelineFake()

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Return fake latents unchanged."""
        return latents

    def inference(self, **kwargs: Any) -> List[Any]:
        """Return no generated samples."""
        return []

    def forward(self, **kwargs: Any) -> Any:
        """Return no fake model output."""
        return None


def _adapter_config() -> SimpleNamespace:
    model_args = SimpleNamespace(
        resume_path=None,
        resume_type=None,
        finetune_type="full",
        target_components=["transformer"],
        target_modules="all",
        trainable_parameters_dtype=torch.float32,
        frozen_parameters_dtype=None,
    )
    training_args = SimpleNamespace(
        enable_gradient_checkpointing=False,
        latent_storage_dtype=None,
    )
    return SimpleNamespace(
        model_args=model_args,
        training_args=training_args,
        eval_args=SimpleNamespace(),
        scheduler_args=SimpleNamespace(),
        mixed_precision="no",
    )


def test_base_adapter_default_runtime_preserves_existing_subclass_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flow_factory.models.abc._load_scheduler",
        lambda pipeline_scheduler, scheduler_args: pipeline_scheduler,
    )

    adapter = ExistingStyleAdapterFake(_adapter_config(), AcceleratorFake())

    assert isinstance(adapter.component_runtime, ClassicPipelineRuntime)
    assert adapter.pipeline is adapter.component_runtime.pipeline
    assert adapter._components is adapter.component_runtime.prepared_components
    assert adapter.get_component("transformer") is adapter.pipeline.transformer


def test_bundle_proxy_installation_resolves_through_adapter_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flow_factory.models.abc._load_scheduler",
        lambda pipeline_scheduler, scheduler_args: pipeline_scheduler,
    )
    adapter = ExistingStyleAdapterFake(_adapter_config(), AcceleratorFake())
    canonical = adapter.get_component("transformer")
    bundle = ModelBundle({"transformer": canonical})
    proxy = RoutedComponentProxy(bundle, "transformer", canonical)

    adapter.set_component("transformer", proxy)

    assert adapter.get_component("transformer") is proxy
    assert adapter.get_component_unwrapped("transformer") is canonical
    assert torch.equal(
        adapter.get_component("transformer")(torch.tensor([2.0])), torch.tensor([2.0])
    )
