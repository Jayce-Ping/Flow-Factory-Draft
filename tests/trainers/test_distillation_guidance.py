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

"""Pin which role classifier-free guidance is allowed to reach.

The real score defines the target distribution, so guiding it is the point. The
generator rolls out CFG-free and the fake score must model what the generator
actually produces -- guiding either of those two makes the fake score fit a
distribution the generator never samples from.
"""

from types import SimpleNamespace

import pytest
from accelerate.utils import DistributedType

from flow_factory.hparams.training_args.dmd2 import DMD2TrainingArguments
from flow_factory.trainers.role_optimization import RoleOptimizationCoordinator
from flow_factory.trainers.distillation.distillation_runtime import (
    reference_forward_kwargs,
    replay_forward_kwargs,
)


def _args(**overrides: object) -> SimpleNamespace:
    values = {"guidance_scale": 1.0, "real_guidance_scale": None, "extra_kwargs": None}
    values.update(overrides)
    return SimpleNamespace(**values)


def _adapter(guidance_key: str = "guidance_scale") -> SimpleNamespace:
    return SimpleNamespace(
        reference_guidance_kwargs=lambda scale: {guidance_key: scale},
    )


def test_only_the_real_score_receives_the_guided_scale() -> None:
    training_args = _args(guidance_scale=1.0, real_guidance_scale=4.0)
    batch: dict = {}

    assert replay_forward_kwargs(training_args, batch)["guidance_scale"] == 1.0
    assert reference_forward_kwargs(_adapter(), training_args, batch)["guidance_scale"] == 4.0


def test_leaving_the_real_scale_unset_keeps_every_role_on_one_scale() -> None:
    training_args = _args(guidance_scale=3.0)
    batch: dict = {}

    assert reference_forward_kwargs(_adapter(), training_args, batch)["guidance_scale"] == 3.0
    assert replay_forward_kwargs(training_args, batch)["guidance_scale"] == 3.0


def test_a_batch_that_carries_guidance_still_wins() -> None:
    training_args = _args(guidance_scale=1.0, real_guidance_scale=4.0)
    batch = {"guidance_scale": 2.0}

    assert "guidance_scale" not in reference_forward_kwargs(_adapter(), training_args, batch)


def test_adapter_maps_reference_guidance_to_its_forward_name() -> None:
    training_args = _args(
        guidance_scale=1.0,
        real_guidance_scale=4.0,
        extra_kwargs={"cfg_text_scale": 1.0},
    )

    assert reference_forward_kwargs(_adapter("cfg_text_scale"), training_args, {}) == {
        "cfg_text_scale": 4.0,
    }


@pytest.mark.parametrize("value", [0.5, 0.0])
def test_a_real_scale_below_one_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match=f"real_guidance_scale >= 1.0, received {value!r}"):
        DMD2TrainingArguments(real_guidance_scale=value)


def test_a_negative_real_scale_is_refused() -> None:
    with pytest.raises(ValueError, match="real_guidance_scale >= 0, received -1.0"):
        DMD2TrainingArguments(real_guidance_scale=-1.0)


def test_preprocessing_encodes_negatives_for_the_guided_real_score() -> None:
    """Otherwise the guided query finds no negatives and quietly runs unguided.

    Preprocessing decides whether to encode negative prompts from one number. The
    configuration this algorithm wants -- a CFG-free generator beside a guided real
    score -- puts 1.0 in `guidance_scale`, which would skip them.
    """
    guided = DMD2TrainingArguments(guidance_scale=1.0, real_guidance_scale=4.0)

    assert guided.get_preprocess_guidance_scale() == 4.0


def test_preprocessing_follows_the_sampling_scale_when_the_real_score_is_unset() -> None:
    plain = DMD2TrainingArguments(guidance_scale=3.5)

    assert plain.get_preprocess_guidance_scale() == 3.5


@pytest.mark.parametrize(
    ("distributed_type", "authoritative"),
    [
        (DistributedType.MULTI_GPU, True),
        (DistributedType.FSDP, False),
        (DistributedType.DEEPSPEED, False),
    ],
)
def test_only_backends_that_leave_gradients_on_parameters_are_asserted_against(
    distributed_type: DistributedType,
    authoritative: bool,
) -> None:
    """DeepSpeed and FSDP1 do not expose authoritative original-parameter grads.

    Reading it there reports "no gradients" for every role of every correct run, so the
    per-microbatch assertions have to know when they can see anything at all.
    """
    coordinator = object.__new__(RoleOptimizationCoordinator)
    coordinator.accelerator = SimpleNamespace(distributed_type=distributed_type)

    assert coordinator._gradients_reach_parameters() is authoritative


def test_fsdp2_dtensor_parameter_gradients_remain_authoritative() -> None:
    coordinator = object.__new__(RoleOptimizationCoordinator)
    coordinator.accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        state=SimpleNamespace(fsdp_plugin=SimpleNamespace(fsdp_version=2)),
    )

    assert coordinator._gradients_reach_parameters() is True
