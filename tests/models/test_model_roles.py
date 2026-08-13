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
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Iterator

import pytest
import torch
from peft import LoraConfig, get_peft_model

from flow_factory.models.abc import BaseAdapter
from flow_factory.models.model_bundle import ModelBundle, RoutedComponentProxy
from flow_factory.models.roles import ModelRoleRegistry, ModelRoleSpec, RoleParameter


def _registry() -> ModelRoleRegistry:
    return ModelRoleRegistry(
        SimpleNamespace(trainable_component_names=["transformer", "transformer_2"])
    )


def _generator_spec() -> ModelRoleSpec:
    return ModelRoleSpec(
        name="generator",
        trainable=True,
        storage_mode="lora",
        component_routes={
            "transformer": "transformer",
            "transformer_2": "transformer_2",
        },
        adapter_name="generator",
    )


def _declare_generator(registry: ModelRoleRegistry) -> torch.nn.Parameter:
    registry.declare(_generator_spec())
    parameter = torch.nn.Parameter(torch.ones(1))
    registry.register_parameter("generator", "transformer", "weight", parameter)
    return parameter


def test_declares_roles_in_exact_order_and_resolves_routes() -> None:
    registry = _registry()
    _declare_generator(registry)
    registry.declare(
        ModelRoleSpec(
            name="fake",
            trainable=True,
            storage_mode="full",
            component_routes={
                "transformer": "fake__transformer",
                "transformer_2": "fake__transformer_2",
            },
        )
    )
    fake_parameter = torch.nn.Parameter(torch.zeros(1))
    registry.register_parameter("fake", "transformer", "weight", fake_parameter)
    registry.declare(
        ModelRoleSpec(
            name="reference",
            trainable=False,
            storage_mode="snapshot",
            component_routes={
                "transformer": "reference__transformer",
                "transformer_2": "reference__transformer_2",
            },
        )
    )

    assert registry.role_names == ("generator", "fake", "reference")
    assert registry.resolve_route("generator", "transformer") == "transformer"
    assert registry.resolve_route("fake", "transformer_2") == "fake__transformer_2"
    assert registry.parameters("fake") == (fake_parameter,)
    assert registry.parameter_records("fake")[0].parameter_name == "weight"


def test_requires_generator_as_first_declaration() -> None:
    registry = _registry()

    with pytest.raises(ValueError, match="first.*generator.*received.*fake"):
        registry.declare(
            ModelRoleSpec(
                name="fake",
                trainable=True,
                storage_mode="full",
                component_routes={"transformer": "fake__transformer"},
            )
        )


def test_rejects_duplicate_and_illegal_role_names_with_details() -> None:
    registry = _registry()
    registry.declare(_generator_spec())

    with pytest.raises(ValueError, match="already declared.*generator"):
        registry.declare(_generator_spec())
    with pytest.raises(
        ValueError,
        match="expected one of.*generator.*fake.*surrogate.*reference.*received.*critic",
    ):
        registry.declare(
            ModelRoleSpec(
                name="critic",  # type: ignore[arg-type]
                trainable=True,
                storage_mode="full",
                component_routes={"transformer": "critic__transformer"},
            )
        )


def test_generator_maps_every_trainable_component_to_canonical_route() -> None:
    registry = _registry()

    with pytest.raises(
        ValueError,
        match="generator.*canonical.*transformer_2.*received",
    ):
        registry.declare(
            ModelRoleSpec(
                name="generator",
                trainable=True,
                storage_mode="lora",
                component_routes={"transformer": "renamed_transformer"},
            )
        )


def test_rejects_route_collisions_with_role_and_component_details() -> None:
    registry = _registry()
    registry.declare(_generator_spec())

    with pytest.raises(
        ValueError,
        match="route collision.*transformer.*generator.*transformer.*fake.*transformer_2",
    ):
        registry.declare(
            ModelRoleSpec(
                name="fake",
                trainable=True,
                storage_mode="full",
                component_routes={"transformer_2": "transformer"},
            )
        )


def test_lora_roles_can_share_a_route_for_the_same_canonical_component() -> None:
    registry = _registry()
    registry.declare(_generator_spec())

    registry.declare(
        ModelRoleSpec(
            name="fake",
            trainable=True,
            storage_mode="lora",
            component_routes={"transformer": "transformer"},
            adapter_name="fake",
        )
    )

    assert registry.resolve_route("generator", "transformer") == "transformer"
    assert registry.resolve_route("fake", "transformer") == "transformer"


@pytest.mark.parametrize("role_name", ["fake", "surrogate"])
def test_full_trainable_roles_cannot_alias_the_generator_route(role_name: str) -> None:
    registry = _registry()
    registry.declare(_generator_spec())

    with pytest.raises(
        ValueError,
        match=f"shared route.*transformer.*{role_name}.*storage_mode.*full.*generator",
    ):
        registry.declare(
            ModelRoleSpec(
                name=role_name,  # type: ignore[arg-type]
                trainable=True,
                storage_mode="full",
                component_routes={"transformer": "transformer"},
            )
        )


def test_reference_snapshot_can_share_the_generator_route() -> None:
    registry = _registry()
    registry.declare(_generator_spec())

    registry.declare(
        ModelRoleSpec(
            name="reference",
            trainable=False,
            storage_mode="snapshot",
            component_routes={"transformer": "transformer"},
        )
    )

    assert registry.resolve_route("reference", "transformer") == "transformer"


def test_parameter_identity_belongs_to_only_one_trainable_role() -> None:
    registry = _registry()
    generator_parameter = _declare_generator(registry)
    registry.declare(
        ModelRoleSpec(
            name="fake",
            trainable=True,
            storage_mode="full",
            component_routes={"transformer": "fake__transformer"},
        )
    )

    with pytest.raises(
        ValueError,
        match="parameter identity.*generator.*fake.*transformer.*weight",
    ):
        registry.register_parameter("fake", "transformer", "weight", generator_parameter)


def test_reference_is_non_trainable_and_cannot_own_parameters() -> None:
    registry = _registry()
    _declare_generator(registry)

    with pytest.raises(ValueError, match="reference.*non-trainable.*received.*True"):
        registry.declare(
            ModelRoleSpec(
                name="reference",
                trainable=True,
                storage_mode="snapshot",
                component_routes={"transformer": "reference__transformer"},
            )
        )

    registry.declare(
        ModelRoleSpec(
            name="reference",
            trainable=False,
            storage_mode="snapshot",
            component_routes={"transformer": "reference__transformer"},
        )
    )
    with pytest.raises(ValueError, match="reference.*non-trainable.*parameter"):
        registry.register_parameter(
            "reference",
            "transformer",
            "weight",
            torch.nn.Parameter(torch.ones(1)),
        )


def test_freeze_rejects_empty_trainable_role_with_role_name() -> None:
    registry = _registry()
    registry.declare(_generator_spec())

    with pytest.raises(ValueError, match="trainable role.*generator.*at least one parameter"):
        registry.freeze()


def test_freeze_blocks_declaration_and_ownership_mutation_with_details() -> None:
    registry = _registry()
    _declare_generator(registry)
    registry.freeze()

    assert registry.is_frozen
    with pytest.raises(RuntimeError, match="declare.*fake.*frozen"):
        registry.declare(
            ModelRoleSpec(
                name="fake",
                trainable=True,
                storage_mode="full",
                component_routes={"transformer": "fake__transformer"},
            )
        )
    with pytest.raises(RuntimeError, match="register parameter.*generator.*frozen"):
        registry.register_parameter(
            "generator",
            "transformer",
            "bias",
            torch.nn.Parameter(torch.zeros(1)),
        )


def test_spec_routes_and_metadata_are_immutable_snapshots() -> None:
    component_routes = {
        "transformer": "transformer",
        "transformer_2": "transformer_2",
    }
    spec = ModelRoleSpec(
        name="generator",
        trainable=True,
        storage_mode="lora",
        component_routes=component_routes,
        adapter_name="generator",
    )
    component_routes["transformer"] = "mutated"
    with pytest.raises(TypeError):
        spec.component_routes["transformer"] = "mutated"  # type: ignore[index]

    registry = _registry()
    registry.declare(spec)
    _declare_parameter = torch.nn.Parameter(torch.ones(1))
    registry.register_parameter("generator", "transformer", "weight", _declare_parameter)
    metadata = registry.metadata()
    metadata["roles"][0]["component_routes"]["transformer"] = "mutated"
    metadata["roles"].append({"name": "fake"})

    assert spec.component_routes["transformer"] == "transformer"
    assert registry.resolve_route("generator", "transformer") == "transformer"
    assert registry.role_names == ("generator",)
    assert registry.metadata()["roles"][0]["component_routes"]["transformer"] == "transformer"


def test_registration_validates_declared_route_and_parameter_type() -> None:
    registry = _registry()
    registry.declare(_generator_spec())

    with pytest.raises(KeyError, match="generator.*vae.*declared components"):
        registry.register_parameter("generator", "vae", "weight", torch.nn.Parameter(torch.ones(1)))
    with pytest.raises(TypeError, match="torch.nn.Parameter.*generator.*weight.*Tensor"):
        registry.register_parameter(
            "generator",
            "transformer",
            "weight",
            torch.ones(1),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("component_names", [7, "transformer"])
def test_registry_rejects_malformed_adapter_component_collections(component_names: object) -> None:
    adapter = SimpleNamespace(trainable_component_names=component_names)

    with pytest.raises(
        TypeError,
        match=f"non-string iterable.*{type(component_names).__name__}.*{component_names!r}",
    ):
        ModelRoleRegistry(adapter)


@pytest.mark.parametrize(
    ("spec_kwargs", "expected_detail"),
    [
        (
            {"component_routes": {1: "transformer"}},
            "component name.*int",
        ),
        (
            {"component_routes": {"transformer": 1}},
            "route.*transformer.*int",
        ),
        (
            {"adapter_name": 1},
            "adapter_name.*int",
        ),
    ],
)
def test_spec_rejects_wrong_value_types(
    spec_kwargs: dict[str, object], expected_detail: str
) -> None:
    kwargs = {
        "name": "fake",
        "trainable": True,
        "storage_mode": "full",
        "component_routes": {"transformer": "fake__transformer"},
        **spec_kwargs,
    }

    with pytest.raises(TypeError, match=expected_detail):
        ModelRoleSpec(**kwargs)  # type: ignore[arg-type]


def test_rejects_out_of_order_roles_and_same_spec_route_collisions() -> None:
    registry = _registry()
    registry.declare(_generator_spec())
    registry.declare(
        ModelRoleSpec(
            name="surrogate",
            trainable=True,
            storage_mode="full",
            component_routes={"transformer": "surrogate__transformer"},
        )
    )

    with pytest.raises(ValueError, match="fake.*surrogate.*canonical order"):
        registry.declare(
            ModelRoleSpec(
                name="fake",
                trainable=True,
                storage_mode="full",
                component_routes={"transformer": "fake__transformer"},
            )
        )

    collision_registry = _registry()
    collision_registry.declare(_generator_spec())
    with pytest.raises(ValueError, match="route collision.*shared.*transformer.*transformer_2"):
        collision_registry.declare(
            ModelRoleSpec(
                name="fake",
                trainable=True,
                storage_mode="full",
                component_routes={
                    "transformer": "shared",
                    "transformer_2": "shared",
                },
            )
        )


def test_parameter_records_are_frozen_and_metadata_excludes_parameter_objects() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    record = RoleParameter(
        role_name="generator",
        component_name="transformer",
        parameter_name="weight",
        parameter=parameter,
    )

    with pytest.raises(FrozenInstanceError):
        record.parameter_name = "bias"  # type: ignore[misc]

    registry = _registry()
    registry.declare(_generator_spec())
    registry.register_parameter("generator", "transformer", "weight", parameter)
    metadata_text = repr(registry.metadata())

    assert "Parameter containing" not in metadata_text
    assert metadata_text.count("weight") == 1


class _TinyRoleModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.target = torch.nn.Linear(2, 2, bias=False)
        self.frozen = torch.nn.Linear(2, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.target(inputs) + self.frozen(inputs)


class _TinyRoleAdapter(BaseAdapter):
    def __init__(self, finetune_type: str) -> None:
        self.model_args = SimpleNamespace(finetune_type=finetune_type)
        self.target_module_map = {"transformer": ["target"]}
        component = _TinyRoleModule()
        component.requires_grad_(False)
        if finetune_type == "lora":
            component = get_peft_model(
                component,
                LoraConfig(
                    r=1,
                    lora_alpha=1,
                    init_lora_weights="gaussian",
                    target_modules=["target"],
                ),
            )
        else:
            for parameter_name, parameter in component.named_parameters():
                parameter.requires_grad = "target" in parameter_name
        self._role_components = {"transformer": component}
        self.reference_is_active = False

    @property
    def trainable_component_names(self) -> list[str]:
        return ["transformer"]

    def get_component(self, name: str) -> torch.nn.Module:
        return self._role_components[name]

    @contextmanager
    def use_ref_parameters(self) -> Iterator[None]:
        previous_state = self.reference_is_active
        self.reference_is_active = True
        try:
            yield
        finally:
            self.reference_is_active = previous_state

    def load_pipeline(self) -> object:
        raise NotImplementedError

    def decode_latents(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def inference(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def forward(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError


def _adapter_parameter_ids(adapter: _TinyRoleAdapter) -> set[int]:
    return {id(parameter) for parameter in adapter.get_component("transformer").parameters()}


def _assert_all_owned_role_parameters_trainable(registry: ModelRoleRegistry) -> None:
    for role_name in registry.role_names:
        if registry.get_spec(role_name).trainable:
            assert registry.parameters(role_name)
            assert all(parameter.requires_grad for parameter in registry.parameters(role_name))


def test_lora_materialization_uses_named_adapters_and_exact_new_parameter_identity() -> None:
    adapter = _TinyRoleAdapter("lora")
    ids_before_materialization = _adapter_parameter_ids(adapter)

    adapter.configure_model_roles(["generator", "fake", "surrogate"])

    registry = adapter.model_role_registry
    component = adapter.get_component("transformer")
    assert set(component.peft_config) == {"default", "fake", "surrogate"}
    assert registry.role_names == ("generator", "fake", "surrogate", "reference")
    assert registry.get_spec("generator").adapter_name == "default"
    assert registry.get_spec("fake").adapter_name == "fake"
    assert registry.get_spec("surrogate").adapter_name == "surrogate"
    assert registry.bundle_members() == {"transformer": component}

    generator_ids = {id(parameter) for parameter in registry.parameters("generator")}
    fake_ids = {id(parameter) for parameter in registry.parameters("fake")}
    surrogate_ids = {id(parameter) for parameter in registry.parameters("surrogate")}
    ids_after_materialization = _adapter_parameter_ids(adapter)
    assert generator_ids <= ids_before_materialization
    assert fake_ids | surrogate_ids == ids_after_materialization - ids_before_materialization
    assert generator_ids
    assert fake_ids
    assert surrogate_ids
    assert generator_ids.isdisjoint(fake_ids)
    assert generator_ids.isdisjoint(surrogate_ids)
    assert fake_ids.isdisjoint(surrogate_ids)
    _assert_all_owned_role_parameters_trainable(registry)


def test_lora_role_contexts_activate_named_adapters_and_restore_after_exception() -> None:
    adapter = _TinyRoleAdapter("lora")
    adapter.configure_model_roles(["generator", "fake"])
    registry = adapter.model_role_registry
    component = adapter.get_component("transformer")

    assert registry.active_role == "generator"
    assert component.active_adapter == "default"
    _assert_all_owned_role_parameters_trainable(registry)
    with adapter.use_model_role("fake"):
        assert registry.active_role == "fake"
        assert component.active_adapter == "fake"
        _assert_all_owned_role_parameters_trainable(registry)
        with pytest.raises(RuntimeError, match="inner failure"):
            with adapter.use_model_role("reference"):
                assert registry.active_role == "reference"
                _assert_all_owned_role_parameters_trainable(registry)
                assert all(
                    module.disable_adapters
                    for module in component.modules()
                    if hasattr(module, "disable_adapters")
                )
                with adapter.use_model_role("reference"):
                    assert all(
                        module.disable_adapters
                        for module in component.modules()
                        if hasattr(module, "disable_adapters")
                    )
                    _assert_all_owned_role_parameters_trainable(registry)
                _assert_all_owned_role_parameters_trainable(registry)
                raise RuntimeError("inner failure")
        assert registry.active_role == "fake"
        assert component.active_adapter == "fake"
        _assert_all_owned_role_parameters_trainable(registry)
    assert registry.active_role == "generator"
    assert component.active_adapter == "default"
    _assert_all_owned_role_parameters_trainable(registry)


@pytest.mark.parametrize("required_roles", [None, "generator", b"generator", 7, {"generator"}])
def test_configure_model_roles_rejects_non_sequence_inputs(required_roles: object) -> None:
    adapter = _TinyRoleAdapter("full")

    with pytest.raises(
        TypeError,
        match=(
            "expected required_trainable_roles to be a non-string sequence.*"
            f"received {type(required_roles).__name__}: {required_roles!r}"
        ),
    ):
        adapter.configure_model_roles(required_roles)  # type: ignore[arg-type]


@pytest.mark.parametrize("required_roles", [None, "generator", b"generator", 7, {"generator"}])
def test_registry_materialize_rejects_non_sequence_inputs(required_roles: object) -> None:
    registry = _registry()

    with pytest.raises(
        TypeError,
        match=(
            "expected required_trainable_roles to be a non-string sequence.*"
            f"received {type(required_roles).__name__}: {required_roles!r}"
        ),
    ):
        registry.materialize(required_roles)  # type: ignore[arg-type]


def test_configure_model_roles_rejects_non_string_unhashable_entries() -> None:
    adapter = _TinyRoleAdapter("full")
    required_roles = ["generator", ["fake"]]

    with pytest.raises(
        TypeError,
        match="expected string role name.*index 1.*received list.*fake",
    ):
        adapter.configure_model_roles(required_roles)  # type: ignore[arg-type]


def test_registry_materialize_rejects_non_string_unhashable_entries() -> None:
    registry = _registry()
    required_roles = ["generator", ["fake"]]

    with pytest.raises(
        TypeError,
        match="expected string role name.*index 1.*received list.*fake",
    ):
        registry.materialize(required_roles)  # type: ignore[arg-type]


def test_full_materialization_copies_routes_and_preserves_target_module_freezing() -> None:
    adapter = _TinyRoleAdapter("full")
    generator = adapter.get_component("transformer")
    generator_state = {
        parameter_name: parameter.detach().clone()
        for parameter_name, parameter in generator.named_parameters()
    }

    adapter.configure_model_roles(["generator", "fake", "surrogate"])

    registry = adapter.model_role_registry
    members = registry.bundle_members()
    assert tuple(members) == (
        "transformer",
        "fake__transformer",
        "surrogate__transformer",
    )
    for role_name in ("fake", "surrogate"):
        route_name = f"{role_name}__transformer"
        replica = members[route_name]
        assert replica is not generator
        assert registry.resolve_route(role_name, "transformer") == route_name
        assert {
            parameter_name: parameter.detach()
            for parameter_name, parameter in replica.named_parameters()
        }.keys() == generator_state.keys()
        for parameter_name, parameter in replica.named_parameters():
            assert torch.equal(parameter, generator_state[parameter_name])
            assert parameter.requires_grad is ("target" in parameter_name)

    role_parameter_ids = [
        {id(parameter) for parameter in registry.parameters(role_name)}
        for role_name in ("generator", "fake", "surrogate")
    ]
    assert all(role_parameter_ids)
    assert role_parameter_ids[0].isdisjoint(role_parameter_ids[1])
    assert role_parameter_ids[0].isdisjoint(role_parameter_ids[2])
    assert role_parameter_ids[1].isdisjoint(role_parameter_ids[2])


def test_full_reference_uses_snapshot_context_and_nested_roles_restore_exactly() -> None:
    adapter = _TinyRoleAdapter("full")
    adapter.configure_model_roles(["generator", "fake"])
    registry = adapter.model_role_registry

    with registry.use("fake"):
        with registry.use("reference"):
            assert registry.active_role == "reference"
            assert adapter.reference_is_active
        assert registry.active_role == "fake"
        assert not adapter.reference_is_active
    assert registry.active_role == "generator"


def test_role_materialization_is_one_shot_and_rejected_after_registry_freeze() -> None:
    adapter = _TinyRoleAdapter("full")
    adapter.configure_model_roles(["generator", "fake"])

    assert adapter.model_role_registry.is_frozen
    with pytest.raises(RuntimeError, match="materialize.*frozen"):
        adapter.model_role_registry.materialize(["generator"])
    with pytest.raises(RuntimeError, match="configure.*frozen"):
        adapter.configure_model_roles(["generator"])


def _fill_role_module(component: _TinyRoleModule, target: float, frozen: float = 0.0) -> None:
    component.target.weight.data.fill_(target)
    component.frozen.weight.data.fill_(frozen)


def test_routed_proxy_dispatches_full_roles_and_attributes_through_one_bundle() -> None:
    adapter = _TinyRoleAdapter("full")
    adapter.configure_model_roles(["generator", "fake", "surrogate"])
    registry = adapter.model_role_registry
    members = registry.bundle_members()
    _fill_role_module(members["transformer"], 1.0)
    _fill_role_module(members["fake__transformer"], 2.0)
    _fill_role_module(members["surrogate__transformer"], 3.0)
    bundle = ModelBundle(members)
    proxy = RoutedComponentProxy(bundle, "transformer", registry, bundle.members)
    inputs = torch.ones(1, 2)

    assert not isinstance(proxy, torch.nn.Module)
    assert proxy.inner is members["transformer"]
    assert torch.equal(proxy(inputs), torch.full((1, 2), 2.0))
    assert proxy.target is members["transformer"].target
    with adapter.use_model_role("fake"):
        assert proxy.inner is members["fake__transformer"]
        assert proxy.target is members["fake__transformer"].target
        assert torch.equal(proxy(inputs), torch.full((1, 2), 4.0))
        with adapter.use_model_role("surrogate"):
            assert proxy.inner is members["surrogate__transformer"]
            assert torch.equal(proxy(inputs), torch.full((1, 2), 6.0))
        assert proxy.inner is members["fake__transformer"]
    assert proxy.inner is members["transformer"]
    assert torch.equal(proxy(inputs), torch.full((1, 2), 2.0))


def test_routed_proxy_uses_active_lora_adapter_on_canonical_bundle_route() -> None:
    adapter = _TinyRoleAdapter("lora")
    adapter.configure_model_roles(["generator", "fake"])
    registry = adapter.model_role_registry
    members = registry.bundle_members()
    bundle = ModelBundle(members)
    proxy = RoutedComponentProxy(bundle, "transformer", registry, bundle.members)
    inputs = torch.ones(1, 2)
    for parameter in registry.parameters("generator"):
        parameter.data.fill_(1.0)
    for parameter in registry.parameters("fake"):
        parameter.data.fill_(2.0)

    assert proxy.active_adapter == "default"
    assert registry.resolve_route("generator", "transformer") == "transformer"
    generator_output = proxy(inputs)
    with adapter.use_model_role("fake"):
        assert proxy.active_adapter == "fake"
        assert registry.resolve_route("fake", "transformer") == "transformer"
        assert proxy.inner is members["transformer"]
        fake_output = proxy(inputs)
        assert not torch.equal(fake_output, generator_output)
    assert proxy.active_adapter == "default"
    assert torch.equal(proxy(inputs), generator_output)


def test_routed_proxy_preserves_default_role_after_unknown_role_error() -> None:
    adapter = _TinyRoleAdapter("full")
    adapter.configure_model_roles(["generator", "fake"])
    registry = adapter.model_role_registry
    bundle = ModelBundle(registry.bundle_members())
    proxy = RoutedComponentProxy(bundle, "transformer", registry, bundle.members)

    with pytest.raises(KeyError, match="critic.*available roles"):
        with adapter.use_model_role("critic"):  # type: ignore[arg-type]
            pass

    assert registry.active_role == "generator"
    assert proxy.inner is bundle.members["transformer"]


def test_base_adapter_keeps_four_method_abstract_contract() -> None:
    assert BaseAdapter.__abstractmethods__ == {
        "load_pipeline",
        "decode_latents",
        "inference",
        "forward",
    }
