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

from flow_factory.trainers.abc import validate_supported_distributed_plan


def _accelerator(distributed_type: DistributedType, zero_stage: object = None) -> SimpleNamespace:
    plugin = None if zero_stage is None else SimpleNamespace(zero_stage=zero_stage)
    return SimpleNamespace(
        distributed_type=distributed_type,
        state=SimpleNamespace(deepspeed_plugin=plugin),
    )


def test_zero_three_is_rejected_before_any_weights_load() -> None:
    """Parameter sharding breaks reward loading, so it must fail at startup."""
    accelerator = _accelerator(DistributedType.DEEPSPEED, zero_stage=3)

    with pytest.raises(ValueError, match="ZeRO-3 is not supported"):
        validate_supported_distributed_plan(accelerator)


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
