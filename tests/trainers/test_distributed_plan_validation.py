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

import pytest
from accelerate.utils import DistributedType

from flow_factory.trainers.abc import validate_supported_distributed_plan


def _accelerator(distributed_type: DistributedType, zero_stage: int | None) -> SimpleNamespace:
    plugin = None if zero_stage is None else SimpleNamespace(zero_stage=zero_stage)
    return SimpleNamespace(
        distributed_type=distributed_type,
        state=SimpleNamespace(deepspeed_plugin=plugin),
    )


def test_deepspeed_zero3_is_rejected_before_training_starts() -> None:
    accelerator = _accelerator(DistributedType.DEEPSPEED, zero_stage=3)

    with pytest.raises(ValueError, match=r"ZeRO-3.*not supported.*ZeRO-1/2.*received.*stage 3"):
        validate_supported_distributed_plan(accelerator)


@pytest.mark.parametrize("zero_stage", [1, 2])
def test_supported_deepspeed_zero_stages_are_accepted(zero_stage: int) -> None:
    accelerator = _accelerator(DistributedType.DEEPSPEED, zero_stage=zero_stage)

    validate_supported_distributed_plan(accelerator)


@pytest.mark.parametrize(
    "distributed_type",
    [DistributedType.NO, DistributedType.MULTI_GPU, DistributedType.FSDP],
)
def test_non_deepspeed_plans_are_accepted(distributed_type: DistributedType) -> None:
    accelerator = _accelerator(distributed_type, zero_stage=None)

    validate_supported_distributed_plan(accelerator)
