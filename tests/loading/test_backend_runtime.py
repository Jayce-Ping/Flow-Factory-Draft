import os
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType

from flow_factory.loading.backend import (
    FSDPBackendLoadRuntime,
    configure_backend_loading,
)
from flow_factory.loading.domain import (
    ComponentDescriptor,
    ComponentRole,
    LoadPlanner,
)


class _AcceleratorFake:
    distributed_type = DistributedType.FSDP
    device = torch.device("cpu")
    num_processes = 2
    process_index = 0

    def __init__(self, *, efficient: bool) -> None:
        self.state = SimpleNamespace(
            fsdp_plugin=SimpleNamespace(
                fsdp_version=2,
                cpu_ram_efficient_loading=efficient,
            )
        )

    def wait_for_everyone(self) -> None:
        return None


def _plan():
    return LoadPlanner().build(
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


def test_backend_capability_disables_global_efficient_loading_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = _AcceleratorFake(efficient=True)
    monkeypatch.setenv("FSDP_CPU_RAM_EFFICIENT_LOADING", "true")
    adapter_class = type(
        "EagerAdapter",
        (),
        {"supports_fsdp2_cpu_efficient_loading": False},
    )

    configure_backend_loading(accelerator, adapter_class)

    assert not accelerator.state.fsdp_plugin.cpu_ram_efficient_loading
    assert os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] == "false"


def test_backend_capability_preserves_supported_efficient_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = _AcceleratorFake(efficient=True)
    monkeypatch.setenv("FSDP_CPU_RAM_EFFICIENT_LOADING", "true")
    adapter_class = type(
        "ModularAdapter",
        (),
        {"supports_fsdp2_cpu_efficient_loading": True},
    )

    configure_backend_loading(accelerator, adapter_class)

    assert accelerator.state.fsdp_plugin.cpu_ram_efficient_loading
    assert os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] == "true"


def test_reward_scope_restores_target_loading_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = _AcceleratorFake(efficient=True)
    adapter = SimpleNamespace()
    runtime = FSDPBackendLoadRuntime(accelerator, _plan(), adapter)
    monkeypatch.setenv("FSDP_CPU_RAM_EFFICIENT_LOADING", "true")

    with runtime.load_scope(ComponentRole.REWARD):
        assert os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] == "false"

    assert os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] == "true"


def test_physical_target_root_is_not_selected_as_auxiliary() -> None:
    adapter = SimpleNamespace(
        _resolve_component_names=lambda components: list(components),
    )
    runtime = FSDPBackendLoadRuntime(
        _AcceleratorFake(efficient=True),
        _plan(),
        adapter,
    )

    assert runtime._logical_names(
        ["bagel", "vae"],
        role=ComponentRole.AUXILIARY,
    ) == ["vae"]


def test_target_bootstrap_broadcasts_buffers_not_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = torch.nn.Linear(2, 2)
    target.register_buffer("rope", torch.ones(2))
    vae = torch.nn.Linear(2, 2)
    broadcasts = []
    monkeypatch.setattr(
        "flow_factory.loading.backend.dist.broadcast",
        lambda tensor, src: broadcasts.append(tensor.clone()),
    )
    monkeypatch.setattr(
        "flow_factory.loading.backend.dist.all_reduce",
        lambda tensor, op: None,
    )
    adapter = SimpleNamespace(
        get_component=lambda name: {"bagel": target, "vae": vae}[name],
    )
    runtime = FSDPBackendLoadRuntime(
        _AcceleratorFake(efficient=True),
        _plan(),
        adapter,
    )

    runtime.bootstrap_targets()

    assert len(broadcasts) == 1
    assert torch.equal(broadcasts[0], target.rope)


def test_target_bootstrap_raises_before_buffer_broadcast_on_meta_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = torch.nn.Module()
    target.register_buffer("rope", torch.empty(2, device="meta"))
    monkeypatch.setattr(
        "flow_factory.loading.backend.dist.all_reduce",
        lambda tensor, op: None,
    )
    monkeypatch.setattr(
        "flow_factory.loading.backend.dist.broadcast",
        lambda tensor, src: pytest.fail("invalid ranks must not enter buffer broadcast"),
    )
    runtime = FSDPBackendLoadRuntime(
        _AcceleratorFake(efficient=True),
        _plan(),
        SimpleNamespace(get_component=lambda name: target),
    )

    with pytest.raises(RuntimeError, match=r"target root='bagel'.*meta buffer"):
        runtime.bootstrap_targets()


def test_replicated_root_fingerprint_is_cached_after_first_stage() -> None:
    adapter = SimpleNamespace(
        _resolve_component_names=lambda components: list(components),
    )
    runtime = FSDPBackendLoadRuntime(
        _AcceleratorFake(efficient=False),
        _plan(),
        adapter,
    )
    calls = []
    runtime._check_replica_fingerprints = lambda roots: calls.append(list(roots))

    runtime.components_loaded(["vae"])
    runtime.components_loaded(["vae"])

    assert calls == [["vae"]]
