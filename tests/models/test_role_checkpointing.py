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

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, List, Tuple

import pytest
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file

from flow_factory.models.abc import BaseAdapter
from flow_factory.models.model_bundle import ModelBundle, RoutedComponentProxy
from flow_factory.models.roles import ModelRoleRegistry
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.role_optimization import (
    OptimizationRole,
    RoleOptimizationCoordinator,
    RoleOptimizerConfig,
)

_METADATA_FILENAME = "flow_factory_multirole_metadata.json"


class TinyRoleModule(torch.nn.Module):
    """Expose one trainable target and one frozen parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.target = torch.nn.Linear(2, 2, bias=False)
        self.frozen = torch.nn.Linear(2, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run both tiny projections."""
        return self.target(inputs) + self.frozen(inputs)

    def save_pretrained(
        self,
        save_directory: str,
        max_shard_size: str = "10GB",
        safe_serialization: bool = True,
    ) -> None:
        """Write a minimal Diffusers-compatible full checkpoint."""
        del max_shard_size
        if not safe_serialization:
            raise ValueError(f"expected safe_serialization=True, received {safe_serialization!r}")
        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), output / "diffusion_pytorch_model.safetensors")


class TinyRoleAdapter(BaseAdapter):
    """Provide the model-role and export surfaces without heavy pipeline setup."""

    def __init__(self, accelerator: Accelerator, finetune_type: str) -> None:
        self.accelerator = accelerator
        self.model_args = SimpleNamespace(
            finetune_type=finetune_type,
            target_components=["transformer"],
        )
        self.target_module_map = {"transformer": ["target"]}
        component: torch.nn.Module = TinyRoleModule()
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
        self.ema_wrapper = None
        self._ref_ema = None

    @property
    def trainable_component_names(self) -> List[str]:
        """Return the sole canonical trainable component."""
        return ["transformer"]

    def has_component(self, name: str) -> bool:
        """Declare the tiny role components without a real component runtime."""
        return name in self._role_components

    def get_component(self, name: str) -> torch.nn.Module:
        """Return a tiny role component."""
        return self._role_components[name]

    def set_component(self, name: str, module: torch.nn.Module) -> None:
        """Replace a tiny role component."""
        self._role_components[name] = module

    @contextmanager
    def use_ref_parameters(self) -> Iterator[None]:
        """Provide a no-op full-reference context for registry activation."""
        yield

    def load_pipeline(self) -> object:
        """Satisfy the adapter abstract contract."""
        raise NotImplementedError

    def decode_latents(self, *args: object, **kwargs: object) -> object:
        """Satisfy the adapter abstract contract."""
        raise NotImplementedError

    def inference(self, *args: object, **kwargs: object) -> object:
        """Satisfy the adapter abstract contract."""
        raise NotImplementedError

    def forward(self, *args: object, **kwargs: object) -> object:
        """Satisfy the adapter abstract contract."""
        raise NotImplementedError


class TinyTrainer(BaseTrainer):
    """Provide concrete abstract hooks for checkpoint seam tests."""

    def start(self) -> None:
        """Provide an unused training hook."""

    def prepare_feedback(self, samples: List[Any]) -> None:
        """Provide an unused feedback hook."""

    def optimize(self, *args: Any, **kwargs: Any) -> None:
        """Provide an unused optimization hook."""


class MutatingStateAccelerator:
    """Record whether canonical callbacks run before state mutation."""

    def __init__(self) -> None:
        self.is_main_process = True
        self.is_local_main_process = True
        self.project_configuration = SimpleNamespace(save_on_each_node=False)
        self.model_value = 1
        self.optimizer_value = 2
        self.save_calls = 0
        self.load_calls = 0

    def save_state(self, output_dir: str, **kwargs: Any) -> None:
        """Mutate fake state only after observing metadata on disk."""
        del kwargs
        assert (Path(output_dir) / _METADATA_FILENAME).is_file()
        self.save_calls += 1
        self.model_value = 10
        self.optimizer_value = 20

    def load_state(self, input_dir: str) -> None:
        """Represent backend model/optimizer mutation."""
        del input_dir
        self.load_calls += 1
        self.model_value = 10
        self.optimizer_value = 20

    def wait_for_everyone(self) -> None:
        """Provide the adapter checkpoint synchronization surface."""


def _role_config(role_name: str, update_frequency: int = 1) -> RoleOptimizerConfig:
    return RoleOptimizerConfig(
        role_name=role_name,  # type: ignore[arg-type]
        learning_rate=0.01,
        adam_betas=(0.9, 0.99),
        adam_weight_decay=0.0,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        update_frequency=update_frequency,
    )


def _trainer_runtime(
    finetune_type: str,
    *,
    role_names: Tuple[str, ...] = ("generator", "fake"),
) -> TinyTrainer:
    accelerator = Accelerator(cpu=True)
    adapter = TinyRoleAdapter(accelerator, finetune_type)
    adapter.configure_model_roles(role_names)
    registry = adapter.model_role_registry
    bundle = ModelBundle(registry.bundle_members())
    role_configs = {
        role_name: _role_config(role_name, update_frequency=2 if role_name == "fake" else 1)
        for role_name in role_names
    }
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(registry.parameters(role_name)),
                "role_name": role_name,
                "lr": role_configs[role_name].learning_rate,
                "betas": role_configs[role_name].adam_betas,
                "weight_decay": role_configs[role_name].adam_weight_decay,
                "eps": role_configs[role_name].adam_epsilon,
            }
            for role_name in role_names
        ]
    )
    prepared_bundle, prepared_optimizer = accelerator.prepare(bundle, optimizer)
    optimization_roles = {
        role_name: OptimizationRole(
            config=role_configs[role_name],
            parameters=tuple(prepared_optimizer.param_groups[group_id]["params"]),
            optimizer_group_ids=(group_id,),
        )
        for group_id, role_name in enumerate(role_names)
    }
    trainer = object.__new__(TinyTrainer)
    trainer.accelerator = accelerator
    trainer.adapter = adapter
    trainer.model_bundle = prepared_bundle
    trainer.optimizer = prepared_optimizer
    trainer.optimization_roles = optimization_roles
    trainer.role_optimization = RoleOptimizationCoordinator(
        accelerator,
        prepared_bundle,
        prepared_optimizer,
        optimization_roles,  # type: ignore[arg-type]
    )
    trainer.training_args = SimpleNamespace(
        trainer_type="dmd2",
        required_trainable_roles=role_names,
    )
    trainer.step = 0
    return trainer


def _step_fake_role(trainer: TinyTrainer) -> None:
    coordinator = trainer.role_optimization
    fake_parameter = trainer.optimization_roles["fake"].parameters[0]
    with coordinator.phase("fake"):
        with coordinator.microbatch():
            coordinator.backward(fake_parameter.square().sum())
            coordinator.finish_microbatch()


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_accelerate_round_trip_restores_all_roles_one_optimizer_and_counters(
    tmp_path: Path,
    finetune_type: str,
) -> None:
    trainer = _trainer_runtime(finetune_type)
    trainer._register_multirole_checkpointing()
    _step_fake_role(trainer)
    trainer.optimization_roles["generator"].step = 7
    trainer.step = 7
    expected_parameters = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model_bundle.named_parameters()
    }
    expected_optimizer_state_count = len(trainer.optimizer.state)

    trainer.accelerator.save_state(str(tmp_path))
    assert (tmp_path / _METADATA_FILENAME).is_file()
    assert len(trainer.accelerator._optimizers) == 1

    for parameter in trainer.model_bundle.parameters():
        parameter.data.add_(100)
    trainer.optimizer.state.clear()
    trainer.optimization_roles["fake"].step = 0
    trainer.optimization_roles["generator"].step = 0
    trainer.step = 0
    trainer.accelerator.load_state(str(tmp_path))

    for name, parameter in trainer.model_bundle.named_parameters():
        torch.testing.assert_close(parameter, expected_parameters[name])
    assert len(trainer.optimizer.state) == expected_optimizer_state_count == 1
    assert trainer.optimization_roles["fake"].step == 1
    assert trainer.optimization_roles["generator"].step == 7
    assert trainer.step == 7
    assert trainer.adapter.model_role_registry.active_role == "generator"


def test_metadata_pre_hook_rejects_incompatible_state_before_model_mutation(
    tmp_path: Path,
) -> None:
    source = _trainer_runtime("full")
    source._register_multirole_checkpointing()
    source.accelerator.save_state(str(tmp_path))
    metadata_path = tmp_path / _METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["roles"][1]["component_routes"]["transformer"] = "changed__transformer"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    target = _trainer_runtime("full")
    target._register_multirole_checkpointing()
    parameters_before = [
        parameter.detach().clone() for parameter in target.model_bundle.parameters()
    ]
    with pytest.raises(
        ValueError,
        match="multi-role metadata.*component_routes.*expected.*fake__transformer.*received.*changed",
    ):
        target.accelerator.load_state(str(tmp_path))

    for parameter, before in zip(target.model_bundle.parameters(), parameters_before):
        torch.testing.assert_close(parameter, before)


def test_multirole_resume_requires_dedicated_metadata_before_mutation(tmp_path: Path) -> None:
    source = _trainer_runtime("full")
    source._register_multirole_checkpointing()
    source.accelerator.save_state(str(tmp_path))
    (tmp_path / _METADATA_FILENAME).unlink()

    target = _trainer_runtime("full")
    target._register_multirole_checkpointing()
    parameters_before = [
        parameter.detach().clone() for parameter in target.model_bundle.parameters()
    ]
    with pytest.raises(FileNotFoundError, match="multi-role metadata.*expected.*received missing"):
        target.accelerator.load_state(str(tmp_path))

    for parameter, before in zip(target.model_bundle.parameters(), parameters_before):
        torch.testing.assert_close(parameter, before)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda state: state.update(version=99), "version.*expected 1.*received 99"),
        (
            lambda state: state["roles"].reverse(),
            "role order.*expected.*generator.*fake.*reference.*received.*reference.*fake.*generator",
        ),
        (
            lambda state: state["roles"][1].update(storage_mode="lora"),
            "storage_mode.*fake.*expected.*full.*received.*lora",
        ),
        (
            lambda state: state["roles"][1]["parameters"][0].update(shape=[999]),
            "parameters.*fake.*expected.*received.*999",
        ),
        (
            lambda state: state.update(optimizer_group_roles=["fake", "generator"]),
            "optimizer_group_roles.*expected.*generator.*fake.*received.*fake.*generator",
        ),
        (
            lambda state: state["update_plan"][0].update(repeats=99),
            "update_plan.*expected.*received.*99",
        ),
    ],
)
def test_metadata_validation_reports_expected_and_received_contract(
    mutate: Any,
    match: str,
) -> None:
    trainer = _trainer_runtime("full")
    metadata = trainer._multirole_metadata()
    mutate(metadata)

    with pytest.raises(ValueError, match=match):
        trainer._validate_multirole_metadata(metadata)


def test_registered_custom_state_defensively_revalidates_metadata() -> None:
    trainer = _trainer_runtime("full")
    state = trainer._multirole_state_dict()
    state["metadata"]["optimizer_group_roles"] = ["fake", "generator"]

    with pytest.raises(ValueError, match="optimizer_group_roles.*expected.*received"):
        trainer._load_multirole_state_dict(state)


def test_canonical_save_writes_metadata_before_accelerate_mutates_state(tmp_path: Path) -> None:
    trainer = _trainer_runtime("full")
    trainer._register_multirole_checkpointing()
    fake_accelerator = MutatingStateAccelerator()
    trainer.accelerator = fake_accelerator  # type: ignore[assignment]
    trainer.adapter.accelerator = fake_accelerator  # type: ignore[assignment]

    trainer.adapter.save_checkpoint(str(tmp_path), model_only=False)

    assert fake_accelerator.save_calls == 1
    assert fake_accelerator.model_value == 10
    assert fake_accelerator.optimizer_value == 20
    assert (tmp_path / _METADATA_FILENAME).is_file()


def test_canonical_save_rejects_open_phase_before_accelerate_mutation(tmp_path: Path) -> None:
    trainer = _trainer_runtime("full")
    trainer._register_multirole_checkpointing()
    fake_accelerator = MutatingStateAccelerator()
    trainer.accelerator = fake_accelerator  # type: ignore[assignment]
    trainer.adapter.accelerator = fake_accelerator  # type: ignore[assignment]
    trainer.role_optimization._active_role_name = "generator"
    try:
        with pytest.raises(RuntimeError, match="checkpoint.*closed phase.*active_phase.*generator"):
            trainer.adapter.save_checkpoint(str(tmp_path), model_only=False)
    finally:
        trainer.role_optimization._active_role_name = None

    assert fake_accelerator.save_calls == 0
    assert fake_accelerator.model_value == 1
    assert fake_accelerator.optimizer_value == 2
    assert not (tmp_path / _METADATA_FILENAME).exists()


def test_canonical_load_rejects_metadata_before_accelerate_mutation(tmp_path: Path) -> None:
    trainer = _trainer_runtime("full")
    trainer._register_multirole_checkpointing()
    trainer._multirole_checkpoint_state.prepare_save(str(tmp_path))
    metadata_path = tmp_path / _METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["optimizer_group_roles"] = ["fake", "generator"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    fake_accelerator = MutatingStateAccelerator()
    trainer.accelerator = fake_accelerator  # type: ignore[assignment]
    trainer.adapter.accelerator = fake_accelerator  # type: ignore[assignment]

    with pytest.raises(ValueError, match="optimizer_group_roles.*expected.*received"):
        trainer.adapter.load_checkpoint(str(tmp_path), resume_type="state")

    assert fake_accelerator.load_calls == 0
    assert fake_accelerator.model_value == 1
    assert fake_accelerator.optimizer_value == 2


def test_legacy_single_role_does_not_register_multirole_checkpoint_contract(
    tmp_path: Path,
) -> None:
    trainer = _trainer_runtime("full", role_names=("generator",))
    trainer._register_multirole_checkpointing()

    assert trainer.accelerator._custom_objects == []
    assert not trainer.accelerator._save_model_state_pre_hook
    assert not trainer.accelerator._load_model_state_pre_hook
    trainer.accelerator.save_state(str(tmp_path))
    trainer.accelerator.load_state(str(tmp_path))
    assert not (tmp_path / _METADATA_FILENAME).exists()


def _install_role_proxy(adapter: TinyRoleAdapter) -> None:
    registry = adapter.model_role_registry
    bundle = ModelBundle(registry.bundle_members())
    adapter.set_component(
        "transformer",
        RoutedComponentProxy(bundle, "transformer", registry, bundle.members),
    )


def test_generator_export_context_restores_prior_role_even_on_error(tmp_path: Path) -> None:
    accelerator = Accelerator(cpu=True)
    adapter = TinyRoleAdapter(accelerator, "full")
    adapter.configure_model_roles(("generator", "fake"))
    _install_role_proxy(adapter)
    registry = adapter.model_role_registry
    registry.activate("fake")

    def fail_save(*args: Any, **kwargs: Any) -> None:
        assert registry.active_role == "generator"
        raise RuntimeError("export failed")

    adapter._save_full_model = fail_save  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="export failed"):
        adapter.save_checkpoint(str(tmp_path), save_ema=False)

    assert registry.active_role == "fake"


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_export_contains_only_canonical_generator_weights(
    tmp_path: Path,
    finetune_type: str,
) -> None:
    accelerator = Accelerator(cpu=True)
    adapter = TinyRoleAdapter(accelerator, finetune_type)
    adapter.configure_model_roles(("generator", "fake", "surrogate"))
    registry = adapter.model_role_registry
    for parameter in registry.bundle_members()["transformer"].parameters():
        parameter.data.fill_(1)
    for parameter in registry.parameters("generator"):
        parameter.data.fill_(1)
    for parameter in registry.parameters("fake"):
        parameter.data.fill_(2)
    for parameter in registry.parameters("surrogate"):
        parameter.data.fill_(3)
    _install_role_proxy(adapter)
    registry.activate("fake")

    adapter.save_checkpoint(str(tmp_path), save_ema=False)

    exported_paths = tuple(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert not any("fake" in path or "surrogate" in path for path in exported_paths)
    weight_files = tuple(tmp_path.rglob("*.safetensors"))
    assert weight_files
    exported_state = {}
    for weight_file in weight_files:
        exported_state.update(load_file(weight_file))
    assert exported_state
    assert not any("fake" in name or "surrogate" in name for name in exported_state)
    assert all(torch.equal(value, torch.ones_like(value)) for value in exported_state.values())
    assert registry.active_role == "fake"


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_parameter_ema_swaps_surrogate_snapshot_and_restores_live_parameters(
    finetune_type: str,
) -> None:
    accelerator = Accelerator(cpu=True)
    adapter = TinyRoleAdapter(accelerator, finetune_type)
    adapter.configure_model_roles(("generator", "fake", "surrogate"))
    registry = adapter.model_role_registry
    registry.create_parameter_ema("surrogate", "old_surrogate")
    initial = [parameter.detach().clone() for parameter in registry.parameters("surrogate")]
    for parameter in registry.parameters("surrogate"):
        parameter.data.add_(2)
    live = [parameter.detach().clone() for parameter in registry.parameters("surrogate")]

    with registry.use_parameter_ema("old_surrogate"):
        for parameter, expected in zip(registry.parameters("surrogate"), initial):
            torch.testing.assert_close(parameter, expected)

    for parameter, expected in zip(registry.parameters("surrogate"), live):
        torch.testing.assert_close(parameter, expected)
    registry.update_parameter_ema("old_surrogate", decay=0.25)
    with registry.use_parameter_ema("old_surrogate"):
        for parameter, old, current in zip(registry.parameters("surrogate"), initial, live):
            torch.testing.assert_close(parameter, old * 0.25 + current * 0.75)


def test_parameter_ema_state_round_trip_is_exact_and_not_export_metadata() -> None:
    accelerator = Accelerator(cpu=True)
    adapter = TinyRoleAdapter(accelerator, "full")
    adapter.configure_model_roles(("generator", "fake", "surrogate"))
    registry = adapter.model_role_registry
    registry.create_parameter_ema("surrogate", "old_surrogate")
    for parameter in registry.parameters("surrogate"):
        parameter.data.add_(3)
    registry.update_parameter_ema("old_surrogate", decay=0.4)
    expected = registry.parameter_ema_state_dict()

    for parameter in registry.parameters("surrogate"):
        parameter.data.add_(10)
    registry.update_parameter_ema("old_surrogate", decay=0.1)
    registry.load_parameter_ema_state_dict(expected)

    actual = registry.parameter_ema_state_dict()
    assert actual["update_counts"] == {"old_surrogate": 1}
    for name, tensor in expected["snapshots"]["old_surrogate"]["parameters"].items():
        torch.testing.assert_close(
            actual["snapshots"]["old_surrogate"]["parameters"][name],
            tensor,
            rtol=0,
            atol=0,
        )
    assert "old_surrogate" not in str(registry.metadata())


def test_accelerate_round_trip_restores_old_surrogate_parameter_ema(tmp_path: Path) -> None:
    trainer = _trainer_runtime("full", role_names=("generator", "fake", "surrogate"))
    registry = trainer.adapter.model_role_registry
    registry.create_parameter_ema("surrogate", "old_surrogate")
    for parameter in registry.parameters("surrogate"):
        parameter.data.add_(2)
    registry.update_parameter_ema("old_surrogate", decay=0.3)
    expected = registry.parameter_ema_state_dict()
    trainer._register_multirole_checkpointing()
    trainer.accelerator.save_state(str(tmp_path))

    for parameter in registry.parameters("surrogate"):
        parameter.data.add_(10)
    registry.update_parameter_ema("old_surrogate", decay=0.1)
    trainer.accelerator.load_state(str(tmp_path))

    actual = registry.parameter_ema_state_dict()
    assert actual["update_counts"] == expected["update_counts"]
    for name, tensor in expected["snapshots"]["old_surrogate"]["parameters"].items():
        torch.testing.assert_close(
            actual["snapshots"]["old_surrogate"]["parameters"][name],
            tensor,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_official_generator_ema_export_is_ordered_and_restores_live_weights(
    tmp_path: Path,
    finetune_type: str,
) -> None:
    accelerator = Accelerator(cpu=True)
    adapter = TinyRoleAdapter(accelerator, finetune_type)
    adapter.configure_model_roles(("generator", "fake", "surrogate"))
    registry = adapter.model_role_registry
    registry.create_parameter_ema("generator", "generator_ema")
    expected_ema = registry.parameter_ema_tensors("generator_ema")
    for parameter in registry.parameters("generator"):
        parameter.data.add_(5)
    live = [parameter.detach().clone() for parameter in registry.parameters("generator")]
    _install_role_proxy(adapter)
    registry.activate("fake")

    adapter.save_official_generator_ema(
        str(tmp_path),
        emit_ema_parameters=True,
    )

    artifact = torch.load(tmp_path / "ema.ckpt", weights_only=True)
    assert tuple(artifact) == ("ema_parameters",)
    assert len(artifact["ema_parameters"]) == len(expected_ema)
    for actual, expected in zip(artifact["ema_parameters"], expected_ema):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for parameter, expected in zip(registry.parameters("generator"), live):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    assert registry.active_role == "fake"
    exported = str(tuple(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))
    assert "old_surrogate" not in exported
    assert "fake" not in exported
    assert "surrogate" not in exported
