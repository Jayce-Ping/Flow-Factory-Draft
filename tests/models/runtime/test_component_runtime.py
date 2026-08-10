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

import inspect
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn
from accelerate import DistributedType
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline

from flow_factory.models.abc import BaseAdapter
from flow_factory.models.model_bundle import ModelBundle, RoutedComponentProxy
from flow_factory.models.runtime import (
    ClassicPipelineRuntime,
    ModularPipelineRuntime,
    PseudoPipelineRuntime,
)
from flow_factory.trainers.abc import BaseTrainer


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


class SchedulerFake:
    """Small scheduler-like object used by adapter construction tests."""

    def step(self) -> None:
        """Provide the scheduler step surface."""

    def eval(self) -> None:
        """Provide evaluation mode compatibility."""

    def train(self, mode: bool = True) -> None:
        """Provide training mode compatibility."""

    def rollout(self, mode: bool = True) -> None:
        """Provide rollout mode compatibility."""

    def set_seed(self, seed: int) -> None:
        """Provide trajectory seed compatibility."""


class ClassicPipelineFake:
    """Eager pipeline-like container used by classic runtime tests."""

    def __init__(self) -> None:
        self.text_encoder = TrackingModule()
        self.text_encoder_2 = TrackingModule()
        self.transformer = TrackingModule()
        self.vae = TrackingModule()
        self.scheduler = SchedulerFake()

    @property
    def components(self) -> Dict[str, Any]:
        """Expose canonical eager components."""
        components = {
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "transformer": self.transformer,
            "vae": self.vae,
            "scheduler": self.scheduler,
        }
        if hasattr(self, "optional_component"):
            components["optional_component"] = self.optional_component
        return components


class OptionalTransformerPipelineFake(ClassicPipelineFake):
    """Classic pipeline with a legal absent secondary transformer."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer_2 = None

    @property
    def components(self) -> Dict[str, Any]:
        """Expose the absent secondary transformer declaration."""
        return {**super().components, "transformer_2": self.transformer_2}


class CountingClassicPipelineFake(ClassicPipelineFake):
    """Classic pipeline that counts expensive component-map reconstruction."""

    def __init__(self) -> None:
        super().__init__()
        self.component_map_reads = 0

    @property
    def components(self) -> Dict[str, Any]:
        """Count component map access."""
        self.component_map_reads += 1
        return super().components


class BagelContainerFake(nn.Module):
    """Small parent module with a nested transformer alias."""

    def __init__(self) -> None:
        super().__init__()
        self.language_model = TrackingModule()
        self.moves: List[str] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Route a value through the nested language model."""
        return self.language_model(value)

    def to(self, *args: Any, **kwargs: Any) -> "BagelContainerFake":
        """Record parent device movement."""
        device = kwargs.get("device", args[0] if args else None)
        self.moves.append(str(device))
        return self


class ModularPipelineFake:
    """Match pinned ModularPipeline's public spec and materialized-value APIs."""

    def __init__(self, unavailable: List[str] | None = None) -> None:
        self.pretrained_specs = {
            "text_encoder": "text encoder spec",
            "transformer": "transformer spec",
            "vae": "vae spec",
        }
        self.config_specs: Dict[str, Any] = {}
        self.unavailable = set(unavailable or [])
        self.load_calls: List[List[str]] = []

    @property
    def component_names(self) -> List[str]:
        """Expose declared from-pretrained names like pinned ModularPipeline."""
        return list(self.pretrained_specs)

    @property
    def config_component_names(self) -> List[str]:
        """Expose declared from-config names like pinned ModularPipeline."""
        return list(self.config_specs)

    @property
    def components(self) -> Dict[str, Any]:
        """Expose only materialized values, never the complete lazy spec table."""
        names = [*self.component_names, *self.config_component_names]
        return {
            name: getattr(self, name) for name in names if getattr(self, name, None) is not None
        }

    def get_component_spec(self, name: str) -> Any:
        """Return one declared spec through the pinned public API."""
        return {**self.pretrained_specs, **self.config_specs}[name]

    def load_components(self, names: List[str]) -> None:
        """Materialize every available requested component."""
        self.load_calls.append(list(names))
        for name in names:
            if name not in self.unavailable:
                setattr(self, name, TrackingModule())


def test_pinned_modular_pipeline_public_spec_api_shape() -> None:
    assert isinstance(inspect.getattr_static(ModularPipeline, "component_names"), property)
    assert isinstance(inspect.getattr_static(ModularPipeline, "config_component_names"), property)
    assert isinstance(inspect.getattr_static(ModularPipeline, "components"), property)
    assert tuple(inspect.signature(ModularPipeline.get_component_spec).parameters) == (
        "self",
        "name",
    )
    assert "names" in inspect.signature(ModularPipeline.load_components).parameters


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


def test_pseudo_alias_is_addressable_but_excluded_from_device_enumeration() -> None:
    bagel = BagelContainerFake()
    vae = TrackingModule()
    runtime = PseudoPipelineRuntime(
        SimpleNamespace(),
        {"bagel": bagel, "vae": vae},
        aliases={"transformer": bagel.language_model},
    )

    assert runtime.get_component("transformer") is bagel.language_model
    assert runtime.resolve_component_names() == ["bagel", "vae"]
    assert runtime.resolve_component_names("transformers") == ["transformer"]

    runtime.load_components(["bagel", "transformer", "vae"], device="stage-device")
    runtime.unload_components(["bagel", "transformer", "vae"])

    assert bagel.moves == ["stage-device", "cpu"]
    assert bagel.language_model.moves == []
    assert vae.moves == ["stage-device", "cpu"]


def test_pseudo_runtime_rejects_alias_name_collision() -> None:
    module = TrackingModule()

    with pytest.raises(ValueError, match=r"aliases.*duplicate.*transformer"):
        PseudoPipelineRuntime(
            SimpleNamespace(),
            {"transformer": module},
            aliases={"transformer": module},
        )


def test_pseudo_runtime_rejects_non_module_alias() -> None:
    with pytest.raises(TypeError, match=r"torch\.nn\.Module.*transformer.*str"):
        PseudoPipelineRuntime(
            SimpleNamespace(),
            {"bagel": TrackingModule()},
            aliases={"transformer": "not a module"},
        )


def test_modular_runtime_materializes_only_selected_names() -> None:
    pipeline = ModularPipelineFake()
    runtime = ModularPipelineRuntime(pipeline)

    assert pipeline.components == {}
    assert runtime.declared_component_names == ["text_encoder", "transformer", "vae"]

    runtime.materialize_components(["vae"])

    assert pipeline.load_calls == [["vae"]]
    assert runtime.get_component("vae") is pipeline.vae
    assert not hasattr(pipeline, "transformer")


def test_modular_runtime_uses_pinned_public_component_spec_api_shape() -> None:
    pipeline = ModularPipelineFake()
    pipeline.config_specs["scheduler"] = "scheduler config spec"
    pipeline.transformer = TrackingModule()
    runtime = ModularPipelineRuntime(pipeline)

    assert pipeline.components == {"transformer": pipeline.transformer}
    assert runtime.canonical_components == {
        "text_encoder": "text encoder spec",
        "transformer": "transformer spec",
        "vae": "vae spec",
        "scheduler": "scheduler config spec",
    }
    assert runtime.declared_component_names == [
        "scheduler",
        "text_encoder",
        "transformer",
        "vae",
    ]
    assert runtime.get_component("transformer") is pipeline.transformer


def test_modular_runtime_reports_materialization_failure_context() -> None:
    pipeline = ModularPipelineFake(unavailable=["transformer"])
    runtime = ModularPipelineRuntime(pipeline)

    with pytest.raises(
        RuntimeError,
        match=r"expected.*transformer.*received.*text_encoder.*vae",
    ):
        runtime.materialize_components(["transformer"])


def test_modular_materialize_none_preserves_all_declared_specs_lazily() -> None:
    pipeline = ModularPipelineFake()
    pipeline.pretrained_specs.update(
        {
            "scheduler": "scheduler spec",
            "tokenizer": "tokenizer spec",
            "processor": "processor spec",
        }
    )
    runtime = ModularPipelineRuntime(pipeline)

    runtime.materialize_components()

    assert pipeline.load_calls == []
    assert runtime.materialized_component_names == []
    assert not hasattr(pipeline, "scheduler")
    assert not hasattr(pipeline, "tokenizer")
    assert not hasattr(pipeline, "processor")


def test_modular_none_enumerates_only_materialized_modules_without_loading_specs() -> None:
    pipeline = ModularPipelineFake()
    pipeline.pretrained_specs.update(
        {
            "scheduler": "scheduler spec",
            "tokenizer": "tokenizer spec",
            "processor": "processor spec",
        }
    )
    pipeline.transformer = TrackingModule()
    runtime = ModularPipelineRuntime(pipeline)

    assert runtime.declared_component_names == [
        "processor",
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    ]
    assert runtime.resolve_component_names() == ["transformer"]

    runtime.load_components(device="stage-device")
    runtime.unload_components()

    assert pipeline.load_calls == []
    assert pipeline.transformer.moves == ["stage-device", "cpu"]


def test_classic_optional_none_is_addressable_and_skipped_by_stage_lifecycle() -> None:
    pipeline = ClassicPipelineFake()
    pipeline.optional_component = None
    pipeline.transformer_aux = TrackingModule()
    runtime = ClassicPipelineRuntime(pipeline)

    assert runtime.get_component("optional_component") is None
    assert runtime.resolve_component_names() == [
        "text_encoder",
        "text_encoder_2",
        "transformer",
        "vae",
    ]
    assert "transformer_aux" not in runtime.resolve_component_names()
    assert runtime.resolve_component_names("transformers") == [
        "transformer",
        "transformer_aux",
    ]

    runtime.load_components(["optional_component"], device="stage-device")
    runtime.unload_components(["optional_component"])

    assert pipeline.transformer_aux.moves == []


def test_runtime_rejects_unknown_unload_and_missing_device() -> None:
    runtime = ClassicPipelineRuntime(ClassicPipelineFake())

    with pytest.raises(ValueError, match=r"unknown.*missing.*received"):
        runtime.unload_components(["missing"])
    with pytest.raises(ValueError, match=r"device.*None"):
        runtime.load_components(["vae"], device=None)


def test_runtime_uses_unambiguous_private_device_lifecycle_name() -> None:
    runtime = ClassicPipelineRuntime(ClassicPipelineFake())

    assert runtime._owns_device_lifecycle("vae")
    assert not hasattr(runtime, "_should_manage_device")


def test_classic_materialized_lookup_avoids_component_map_reconstruction() -> None:
    pipeline = CountingClassicPipelineFake()
    runtime = ClassicPipelineRuntime(pipeline)

    assert runtime.get_component("transformer") is pipeline.transformer
    assert runtime.get_canonical_component("vae") is pipeline.vae
    assert pipeline.component_map_reads == 0


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


def test_component_override_vocabulary_preserves_prepared_compatibility_aliases() -> None:
    pipeline = ClassicPipelineFake()
    runtime = ClassicPipelineRuntime(pipeline)
    replacement = TrackingModule()

    runtime.set_component_override("transformer", replacement)

    assert runtime.get_component("transformer") is replacement
    assert runtime.has_component_override("transformer")
    assert runtime.is_prepared("transformer")
    assert runtime.override_components is runtime.prepared_components

    runtime.load_components(["transformer"], device="stage-device")
    runtime.unload_components(["transformer"])

    assert replacement.moves == []
    assert pipeline.transformer.moves == []


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


class OptionalTransformerAdapterFake(ExistingStyleAdapterFake):
    """Existing-style adapter with an absent optional transformer."""

    def load_pipeline(self) -> OptionalTransformerPipelineFake:
        """Return a classic pipeline with ``transformer_2=None``."""
        return OptionalTransformerPipelineFake()

    @property
    def transformer_2(self) -> Any:
        """Expose the optional secondary transformer like a real adapter."""
        return self.get_component("transformer_2")


class ModularAdapterFake(ExistingStyleAdapterFake):
    """Adapter fake that selects a lazy modular runtime."""

    def load_pipeline(self) -> ModularPipelineFake:
        """Return a lazy pipeline with a declared scheduler spec."""
        pipeline = ModularPipelineFake()
        pipeline.config_specs["scheduler"] = "scheduler spec"
        return pipeline

    def build_component_runtime(self) -> ModularPipelineRuntime:
        """Build the lazy runtime used by this adapter."""
        return ModularPipelineRuntime(self.load_pipeline())


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
    assert adapter.scheduler_group.names == ("latent",)
    assert adapter.scheduler_group.primary is adapter.scheduler


def test_base_adapter_constructs_with_optional_transformer_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flow_factory.models.abc._load_scheduler",
        lambda pipeline_scheduler, scheduler_args: pipeline_scheduler,
    )

    adapter = OptionalTransformerAdapterFake(_adapter_config(), AcceleratorFake())

    assert adapter.pipeline.transformer_2 is None
    assert adapter.transformer_names == ["transformer"]


def test_base_adapter_materializes_declared_modular_scheduler_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = SchedulerFake()

    def load_components(self: ModularPipelineFake, names: List[str]) -> None:
        self.load_calls.append(list(names))
        for name in names:
            setattr(self, name, scheduler if name == "scheduler" else TrackingModule())

    monkeypatch.setattr(ModularPipelineFake, "load_components", load_components)
    monkeypatch.setattr(
        "flow_factory.models.abc._load_scheduler",
        lambda pipeline_scheduler, scheduler_args: pipeline_scheduler,
    )

    adapter = ModularAdapterFake(_adapter_config(), AcceleratorFake())

    assert adapter.scheduler is scheduler
    assert adapter.pipeline.load_calls[0] == ["scheduler"]


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


class LifecycleAdapterFake:
    """Adapter-like fake that records public lifecycle override calls."""

    preprocessing_modules = ["text_encoders", "vae"]
    inference_modules = ["transformer", "vae"]

    def __init__(self) -> None:
        self.calls: List[Any] = []
        self.preprocess_func = lambda **kwargs: kwargs

    def on_load_components(self, components: Any, device: Any) -> None:
        """Record public load routing."""
        self.calls.append(("load", components, device))

    def off_load_components(self, components: Any) -> None:
        """Record public unload routing."""
        self.calls.append(("unload", components))

    def _resolve_component_names(self, components: Any = None) -> List[str]:
        """Resolve the groups needed by the trainer regression test."""
        self.calls.append(("resolve", components))
        return ["transformer", "vae"]


class TrainerAcceleratorFake:
    """Minimal trainer accelerator fake."""

    device = torch.device("cpu")

    def wait_for_everyone(self) -> None:
        """Record no-op synchronization."""


def test_trainer_preprocessing_routes_through_adapter_public_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = LifecycleAdapterFake()
    trainer = SimpleNamespace(
        adapter=adapter,
        accelerator=TrainerAcceleratorFake(),
        config=SimpleNamespace(data_args=SimpleNamespace(eval_datasets=[])),
    )
    monkeypatch.setattr(
        "flow_factory.trainers.abc.get_train_dataloader",
        lambda **kwargs: ("train-loader", {}),
    )
    monkeypatch.setattr("flow_factory.trainers.abc.get_eval_dataloaders", lambda **kwargs: {})

    BaseTrainer._init_dataloader(trainer)

    assert adapter.calls == [
        ("load", adapter.preprocessing_modules, trainer.accelerator.device),
        ("unload", adapter.preprocessing_modules),
    ]


def test_trainer_inference_load_routes_through_adapter_public_lifecycle() -> None:
    adapter = LifecycleAdapterFake()
    trainer = SimpleNamespace(
        adapter=adapter,
        accelerator=TrainerAcceleratorFake(),
        config=SimpleNamespace(data_args=SimpleNamespace(enable_preprocess=True)),
    )

    BaseTrainer._load_inference_components(trainer, trainable_module_names=[])

    assert adapter.calls == [
        ("resolve", adapter.inference_modules),
        ("load", ["transformer", "vae"], trainer.accelerator.device),
    ]
