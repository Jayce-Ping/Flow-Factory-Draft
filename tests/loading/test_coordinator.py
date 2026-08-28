from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import torch

from flow_factory.loading import ComponentDescriptor, ComponentRole, LoadPlanner
from flow_factory.loading.coordinator import ModelLoadCoordinator, build_adapter_load_plan
from flow_factory.models.abc import BaseAdapter
from flow_factory.models.runtime import ClassicPipelineRuntime, PseudoPipelineRuntime


def test_coordinator_promotes_pseudo_alias_physical_root_to_target() -> None:
    bagel = torch.nn.Linear(2, 2)
    vae = torch.nn.Linear(2, 2)
    runtime = PseudoPipelineRuntime(
        SimpleNamespace(),
        {"bagel": bagel, "vae": vae},
        aliases={"transformer": bagel},
        alias_routes={"transformer": ("bagel", ("language_model",))},
    )
    adapter = SimpleNamespace(
        component_runtime=runtime,
        model_args=SimpleNamespace(target_components=["transformer"]),
        preprocessing_modules=["vae"],
        inference_modules=["transformer", "vae"],
        _resolve_component_names=runtime.resolve_component_names,
    )

    plan = build_adapter_load_plan(adapter)

    assert plan.request_for_component("transformer").root == "bagel"
    assert plan.request_for_component("bagel").role is ComponentRole.TARGET
    assert plan.request_for_component("vae").role is ComponentRole.AUXILIARY


def test_classic_runtime_collapses_same_object_aliases() -> None:
    encoder = torch.nn.Linear(2, 2)
    pipeline = SimpleNamespace(
        components={"text_encoder_2": encoder},
        text_encoder_2=encoder,
        text_encoder=encoder,
    )
    runtime = ClassicPipelineRuntime(pipeline)

    assert runtime.physical_route("text_encoder") == ("text_encoder_2", ())


def test_coordinator_expands_target_component_groups() -> None:
    transformer = torch.nn.Linear(2, 2)
    runtime = PseudoPipelineRuntime(
        SimpleNamespace(),
        {"transformer": transformer},
    )
    adapter = SimpleNamespace(
        component_runtime=runtime,
        model_args=SimpleNamespace(target_components=["transformers"]),
        preprocessing_modules=[],
        inference_modules=["transformer"],
        _resolve_component_names=runtime.resolve_component_names,
    )

    plan = build_adapter_load_plan(adapter)

    assert plan.request_for_component("transformer").role is ComponentRole.TARGET


def test_base_freezing_uses_materialized_physical_roots_then_logical_target() -> None:
    root = torch.nn.Module()
    root.language_model = torch.nn.Linear(2, 2)
    root.vision_model = torch.nn.Linear(2, 2)
    vae = torch.nn.Linear(2, 2)
    runtime = PseudoPipelineRuntime(
        SimpleNamespace(),
        {"bagel": root, "vae": vae},
        aliases={"transformer": root.language_model},
        alias_routes={"transformer": ("bagel", ("language_model",))},
    )
    adapter = SimpleNamespace(
        component_runtime=runtime,
        model_args=SimpleNamespace(
            target_components=["transformer"],
            finetune_type="full",
        ),
        target_module_map={"transformer": "all"},
        has_component=lambda name: name in runtime.declared_component_names,
        _require_component=lambda name: runtime.get_component(name),
    )
    adapter._freeze_component = MethodType(BaseAdapter._freeze_component, adapter)

    BaseAdapter._freeze_components(adapter)

    assert all(parameter.requires_grad for parameter in root.language_model.parameters())
    assert not any(parameter.requires_grad for parameter in root.vision_model.parameters())
    assert not any(parameter.requires_grad for parameter in vae.parameters())


def test_coordinator_never_moves_a_target_owned_physical_root_as_auxiliary() -> None:
    plan = LoadPlanner().build(
        [
            ComponentDescriptor(
                name="bagel",
                root="bagel",
                role=ComponentRole.AUXILIARY,
            ),
            ComponentDescriptor(
                name="transformer",
                root="bagel",
                path=("language_model",),
                role=ComponentRole.TARGET,
            ),
            ComponentDescriptor(
                name="vae",
                root="vae",
                role=ComponentRole.AUXILIARY,
            ),
        ]
    )
    calls = []
    coordinator = object.__new__(ModelLoadCoordinator)
    coordinator.plan = plan
    coordinator.adapter = SimpleNamespace(
        _resolve_component_names=lambda components: list(components),
        on_load_components=lambda components, device: calls.append(("load", components, device)),
    )
    coordinator.load_scope = lambda role: nullcontext()
    coordinator.components_loaded = lambda components: calls.append(("finalize", components))

    coordinator.load_components(
        ["bagel", "transformer", "vae"],
        device=torch.device("cpu"),
    )

    assert calls == [
        ("load", ["vae"], torch.device("cpu")),
        ("finalize", ["vae"]),
    ]
