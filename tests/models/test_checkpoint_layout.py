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

"""Prove a checkpoint reads back exactly what it wrote, role by role.

Save and load used to reconstruct the same directory rule independently, in four
places. These tests are round trips rather than file-existence checks because the
failure that costs a training run is not a missing file -- it is a fake score
restored onto the generator, which trains happily and converges to nothing.
"""

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

from flow_factory.models import abc as adapter_module
from flow_factory.models.abc import (
    CHECKPOINT_MANIFEST_NAME,
    BaseAdapter,
)
from flow_factory.models.model_bundle import ModelBundle, RoutedComponentProxy
from flow_factory.models.variants import DEFAULT_BASE_VARIANT as BASE_VARIANT


class TinyModule(torch.nn.Module):
    """Hold one target projection that a checkpoint can round-trip."""

    def __init__(self) -> None:
        super().__init__()
        self.target = torch.nn.Linear(2, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the tiny projection."""
        return self.target(inputs)

    def save_pretrained(
        self,
        save_directory: str,
        max_shard_size: str = "10GB",
        safe_serialization: bool = True,
    ) -> None:
        """Write a minimal Diffusers-compatible full checkpoint."""
        del max_shard_size, safe_serialization
        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), output / "diffusion_pytorch_model.safetensors")

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "TinyModule":
        """Read back what :meth:`save_pretrained` wrote.

        Raises:
            OSError: If the directory holds no weights, which is the signal the
                adapter uses to fall back to a manual state-dict load.
        """
        weights = Path(load_directory) / "diffusion_pytorch_model.safetensors"
        if not weights.is_file():
            raise OSError(f"expected weights at {weights}, received a directory without them")
        module = cls()
        module.load_state_dict(load_file(weights))
        return module


class TinyAdapter(BaseAdapter):
    """Expose the checkpoint surface over tiny components."""

    def __init__(
        self,
        accelerator: Accelerator,
        finetune_type: str,
        component_names: Tuple[str, ...] = ("transformer",),
    ) -> None:
        self.accelerator = accelerator
        self.model_args = SimpleNamespace(
            finetune_type=finetune_type,
            target_components=list(component_names),
            target_modules=["target"],
            lora_alpha=1,
        )
        self.target_module_map = {name: ["target"] for name in component_names}
        self._component_names = component_names
        self._components: dict = {}
        for name in component_names:
            component: torch.nn.Module = TinyModule()
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
            self._components[name] = component
        self.ema_wrapper = None
        self._ref_ema = None

    @property
    def trainable_component_names(self) -> List[str]:
        """Return every component that owns target modules."""
        return list(self._component_names)

    def has_component(self, name: str) -> bool:
        """Report whether a component exists."""
        return name in self._components

    def get_component(self, name: str) -> torch.nn.Module:
        """Return a component."""
        return self._components[name]

    def set_component(self, name: str, module: torch.nn.Module) -> None:
        """Replace a component."""
        self._components[name] = module

    @contextmanager
    def use_ref_parameters(self) -> Iterator[None]:
        """Provide a no-op reference context for registry activation."""
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


def _adapter(
    finetune_type: str,
    roles: Tuple[str, ...] = (BASE_VARIANT,),
    component_names: Tuple[str, ...] = ("transformer",),
) -> TinyAdapter:
    adapter = TinyAdapter(Accelerator(cpu=True), finetune_type, component_names)
    adapter.declare_component_variants(roles)
    # Component access has to route through the registry, or every role would read
    # the base module and the round trip would pass without proving anything.
    registry = adapter.component_variant_registry
    bundle = ModelBundle(registry.bundle_members())
    for name in component_names:
        adapter.set_component(name, RoutedComponentProxy(bundle, name, registry, bundle.members))
    return adapter


def _fill(adapter: TinyAdapter, role: str, value: float) -> None:
    for parameter in adapter.component_variant_registry.parameters(role):
        parameter.data.fill_(value)


def _values(adapter: TinyAdapter, role: str) -> List[torch.Tensor]:
    return [
        parameter.detach().clone()
        for parameter in adapter.component_variant_registry.parameters(role)
    ]


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_a_single_role_checkpoint_keeps_the_flat_layout_and_reads_back(
    tmp_path: Path,
    finetune_type: str,
) -> None:
    source = _adapter(finetune_type)
    _fill(source, BASE_VARIANT, 4.0)
    expected = _values(source, BASE_VARIANT)

    source.save_checkpoint(str(tmp_path), save_ema=False, include_training_roles=True)

    # The shape released checkpoints already use: weights at the root, no nesting.
    assert (tmp_path / CHECKPOINT_MANIFEST_NAME).is_file()
    assert not (tmp_path / "roles").exists()
    assert not (tmp_path / "transformer").exists()

    target = _adapter(finetune_type)
    _fill(target, BASE_VARIANT, 9.0)
    target.load_checkpoint(str(tmp_path))

    for actual, want in zip(_values(target, BASE_VARIANT), expected):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_every_role_returns_to_its_own_variant_not_to_whichever_is_active(
    tmp_path: Path,
    finetune_type: str,
) -> None:
    """The failure this guards against is silent: fake weights landing on base."""
    source = _adapter(finetune_type, roles=(BASE_VARIANT, "fake"))
    _fill(source, BASE_VARIANT, 1.0)
    _fill(source, "fake", 2.0)
    expected_base = _values(source, BASE_VARIANT)
    expected_fake = _values(source, "fake")

    source.save_checkpoint(str(tmp_path), save_ema=False, include_training_roles=True)

    assert (tmp_path / "roles" / "fake").is_dir()
    manifest = json.loads((tmp_path / CHECKPOINT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert {(e["role"], e["path"]) for e in manifest["entries"]} == {
        (BASE_VARIANT, "."),
        ("fake", "roles/fake"),
    }

    target = _adapter(finetune_type, roles=(BASE_VARIANT, "fake"))
    _fill(target, BASE_VARIANT, 9.0)
    _fill(target, "fake", 9.0)
    target.load_checkpoint(str(tmp_path))

    for actual, want in zip(_values(target, BASE_VARIANT), expected_base):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
    for actual, want in zip(_values(target, "fake"), expected_fake):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
    # The two roles must stay distinguishable, or the assertions above would pass
    # even if one role had overwritten the other.
    assert not torch.equal(_values(target, BASE_VARIANT)[0], _values(target, "fake")[0])


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_several_components_nest_by_component_and_still_round_trip(
    tmp_path: Path,
    finetune_type: str,
) -> None:
    components = ("transformer", "transformer_2")
    source = _adapter(finetune_type, roles=(BASE_VARIANT, "fake"), component_names=components)
    _fill(source, BASE_VARIANT, 3.0)
    _fill(source, "fake", 5.0)
    expected_base = _values(source, BASE_VARIANT)
    expected_fake = _values(source, "fake")

    source.save_checkpoint(str(tmp_path), save_ema=False, include_training_roles=True)

    for component in components:
        assert (tmp_path / component).is_dir()
        assert (tmp_path / component / "roles" / "fake").is_dir()

    target = _adapter(finetune_type, roles=(BASE_VARIANT, "fake"), component_names=components)
    _fill(target, BASE_VARIANT, 9.0)
    _fill(target, "fake", 9.0)
    target.load_checkpoint(str(tmp_path))

    for actual, want in zip(_values(target, BASE_VARIANT), expected_base):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
    for actual, want in zip(_values(target, "fake"), expected_fake):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_a_checkpoint_written_before_manifests_still_loads(
    tmp_path: Path,
    finetune_type: str,
) -> None:
    source = _adapter(finetune_type)
    _fill(source, BASE_VARIANT, 6.0)
    expected = _values(source, BASE_VARIANT)
    source.save_checkpoint(str(tmp_path), save_ema=False)
    (tmp_path / CHECKPOINT_MANIFEST_NAME).unlink()

    target = _adapter(finetune_type)
    _fill(target, BASE_VARIANT, 9.0)
    target.load_checkpoint(str(tmp_path))

    for actual, want in zip(_values(target, BASE_VARIANT), expected):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)


def test_loading_an_export_into_a_multi_role_run_names_the_roles_it_cannot_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initializing from an export is legitimate; leaving it unsaid is not."""
    export = _adapter("lora")
    _fill(export, BASE_VARIANT, 7.0)
    export.save_checkpoint(str(tmp_path), save_ema=False)

    warnings: List[str] = []
    monkeypatch.setattr(
        adapter_module.logger, "warning", lambda message, *a, **k: warnings.append(str(message))
    )

    target = _adapter("lora", roles=(BASE_VARIANT, "fake"))
    _fill(target, "fake", 9.0)
    untouched_fake = _values(target, "fake")
    target.load_checkpoint(str(tmp_path))

    assert any("roles ['fake'] keep their initial weights" in message for message in warnings)
    for actual, want in zip(_values(target, "fake"), untouched_fake):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_the_manifest_states_the_format_so_detection_never_has_to_guess(
    tmp_path: Path,
    finetune_type: str,
) -> None:
    adapter = _adapter(finetune_type)
    adapter.save_checkpoint(str(tmp_path), save_ema=False)

    assert adapter._detect_checkpoint_type(str(tmp_path)) == finetune_type


def test_a_manifest_from_a_future_layout_is_rejected_rather_than_misread(
    tmp_path: Path,
) -> None:
    adapter = _adapter("lora")
    adapter.save_checkpoint(str(tmp_path), save_ema=False)
    manifest_path = tmp_path / CHECKPOINT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="format_version 1.*received 99"):
        adapter.load_checkpoint(str(tmp_path))


def test_an_export_ships_the_base_role_only(tmp_path: Path) -> None:
    adapter = _adapter("lora", roles=(BASE_VARIANT, "fake"))
    adapter.save_checkpoint(str(tmp_path), save_ema=False)

    assert not (tmp_path / "roles").exists()
    manifest = json.loads((tmp_path / CHECKPOINT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert [entry["role"] for entry in manifest["entries"]] == [BASE_VARIANT]


def test_naming_a_role_that_owns_nothing_is_refused_with_the_declared_names(
    tmp_path: Path,
) -> None:
    adapter = _adapter("lora", roles=(BASE_VARIANT, "fake"))

    with pytest.raises(ValueError, match="variant 'surrogate'.*declared variants.*fake"):
        adapter.save_checkpoint(str(tmp_path), save_ema=False, variant="surrogate")


def test_each_role_directory_is_directly_loadable_by_peft(tmp_path: Path) -> None:
    """A user pointing PEFT at a role folder should not need to know our layout."""
    adapter = _adapter("lora", roles=(BASE_VARIANT, "fake"))
    adapter.save_checkpoint(str(tmp_path), save_ema=False, include_training_roles=True)

    for directory in (tmp_path, tmp_path / "roles" / "fake"):
        assert (directory / "adapter_config.json").is_file()
        assert (directory / "adapter_model.safetensors").is_file()
        assert load_file(directory / "adapter_model.safetensors")
