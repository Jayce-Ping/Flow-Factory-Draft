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

from pathlib import Path
from types import SimpleNamespace

import pytest
from accelerate.utils import DistributedType

from flow_factory.trainers.abc import (
    configure_deepspeed_micro_batch_size,
    validate_supported_distributed_plan,
)


def _accelerator(distributed_type: DistributedType, zero_stage: object = None) -> SimpleNamespace:
    plugin = None if zero_stage is None else SimpleNamespace(zero_stage=zero_stage)
    return SimpleNamespace(
        distributed_type=distributed_type,
        state=SimpleNamespace(deepspeed_plugin=plugin),
    )


def test_zero_three_is_rejected_before_any_weights_load() -> None:
    """The backend validator rejects parameter-sharded DeepSpeed."""
    accelerator = _accelerator(DistributedType.DEEPSPEED, zero_stage=3)

    with pytest.raises(ValueError, match="ZeRO-3 is not supported"):
        validate_supported_distributed_plan(accelerator)


def test_loader_rejects_zero_three_before_loading_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The trainer factory must reject ZeRO-3 before constructing an adapter."""
    from flow_factory.trainers import loader

    accelerator = _accelerator(DistributedType.DEEPSPEED, zero_stage=3)
    model_load_attempted = False

    class Adapter:
        ddp_find_unused_parameters = False

    def unexpected_model_load(**kwargs: object) -> None:
        del kwargs
        nonlocal model_load_attempted
        model_load_attempted = True
        raise AssertionError("load_model must not run for DeepSpeed ZeRO-3")

    config = SimpleNamespace(
        mixed_precision="bf16",
        model_args=SimpleNamespace(model_type="test"),
        log_args=SimpleNamespace(save_dir="/tmp", run_name="zero3-rejection-test"),
        training_args=SimpleNamespace(
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            seed=42,
        ),
    )
    monkeypatch.setattr(loader, "get_model_adapter_class", lambda model_type: Adapter)
    monkeypatch.setattr(loader, "Accelerator", lambda **kwargs: accelerator)
    monkeypatch.setattr(loader, "load_model", unexpected_model_load)

    with pytest.raises(ValueError, match="ZeRO-3 is not supported"):
        loader.load_trainer(config)

    assert model_load_attempted is False


@pytest.mark.parametrize("zero_stage", [1, 2])
def test_supported_deepspeed_stages_pass(zero_stage: int) -> None:
    """ZeRO-1 and ZeRO-2 are the supported DeepSpeed configurations."""
    validate_supported_distributed_plan(_accelerator(DistributedType.DEEPSPEED, zero_stage))


@pytest.mark.parametrize(
    "distributed_type",
    [DistributedType.NO, DistributedType.MULTI_GPU, DistributedType.FSDP],
)
def test_non_deepspeed_plans_pass(distributed_type: DistributedType) -> None:
    """DDP and FSDP carry no DeepSpeed plugin and are supported unchanged."""
    validate_supported_distributed_plan(_accelerator(distributed_type))


def test_deepspeed_without_a_plugin_is_not_rejected() -> None:
    """A DeepSpeed distributed type with no plugin has no stage to reject."""
    validate_supported_distributed_plan(_accelerator(DistributedType.DEEPSPEED))


def test_deepspeed_micro_batch_size_is_set_for_custom_train_loader() -> None:
    accelerator = _accelerator(DistributedType.DEEPSPEED, zero_stage=2)
    accelerator.state.deepspeed_plugin.deepspeed_config = {}

    configure_deepspeed_micro_batch_size(accelerator, per_device_batch_size=3)

    assert (
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ]
        == 3
    )


def test_muon_with_deepspeed_is_rejected_as_unverified() -> None:
    """Muon runs inside a composite; DeepSpeed rebuilds its own optimizer wrapper."""
    from flow_factory.hparams.optimizer_args import (
        AdamWOptimizerArguments,
        MuonOptimizerArguments,
    )
    from flow_factory.trainers.abc import BaseTrainer

    trainer = SimpleNamespace(accelerator=_accelerator(DistributedType.DEEPSPEED, zero_stage=2))

    with pytest.raises(ValueError, match="Muon with DeepSpeed is not verified"):
        BaseTrainer._validate_optimizer_backend(trainer, (MuonOptimizerArguments(name="base"),))

    # AdamW is unaffected, and Muon is fine on the backends that only read groups.
    BaseTrainer._validate_optimizer_backend(trainer, (AdamWOptimizerArguments(name="base"),))
    fsdp_trainer = SimpleNamespace(accelerator=_accelerator(DistributedType.FSDP))
    BaseTrainer._validate_optimizer_backend(fsdp_trainer, (MuonOptimizerArguments(name="base"),))


def test_no_zero_three_profile_is_shipped() -> None:
    """A shipped profile would invite a configuration the trainer refuses."""
    config_dir = Path(__file__).resolve().parents[2] / "config" / "deepspeed"

    assert config_dir.is_dir()
    assert not (config_dir / "deepspeed_zero3.yaml").exists()


def test_deepspeed_gradient_clipping_is_wired_from_the_configured_norm() -> None:
    """DeepSpeed clips inside its engine and ignores the value passed at the call site.

    accelerate reads the threshold from this environment variable when building the
    plugin, so leaving it unset ships an unresolved "auto" and max_grad_norm never
    takes effect on that backend.
    """
    import inspect

    from flow_factory.trainers import loader

    source = inspect.getsource(loader.load_trainer)
    assert "ACCELERATE_GRADIENT_CLIPPING" in source
    assert source.index("ACCELERATE_GRADIENT_CLIPPING") < source.index("accelerator = Accelerator(")
