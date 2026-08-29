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

from collections import namedtuple
from types import MappingProxyType

import pytest
import torch

from flow_factory.contracts import NON_MODEL_CONDITION_KEYS
from flow_factory.models.condition_state import PreparedConditionState
from flow_factory.trainers.common.offline_batch import (
    bind_output_forward_context,
    bind_prepared_condition_output,
    move_condition_to_device,
)


def test_move_condition_copies_containers_and_moves_nested_tensor_leaves() -> None:
    tensor = torch.tensor([1.0])
    nested_list = [tensor]
    condition = MappingProxyType(
        {
            "prompt_embeds": tensor,
            "nested": {"values": nested_list},
            "label": "kept on CPU",
        }
    )

    moved = move_condition_to_device(condition, "cpu")

    assert moved is not condition
    assert moved["nested"] is not condition["nested"]
    assert moved["nested"]["values"] is not nested_list
    assert moved["prompt_embeds"].device.type == "cpu"
    assert moved["nested"]["values"][0].device.type == "cpu"
    assert moved["label"] == "kept on CPU"


def test_move_condition_preserves_namedtuple_shape() -> None:
    Pair = namedtuple("Pair", ("left", "right"))
    condition = {"pair": Pair(torch.ones(1), [torch.zeros(1)])}

    moved = move_condition_to_device(condition, "cpu")

    assert isinstance(moved["pair"], Pair)
    assert moved["pair"] is not condition["pair"]
    assert moved["pair"].right is not condition["pair"].right


@pytest.mark.parametrize("value", [{torch.tensor(1)}, frozenset({"field"})])
def test_move_condition_rejects_unsupported_tree_containers(value: object) -> None:
    with pytest.raises(TypeError, match="unsupported"):
        move_condition_to_device({"nested": value}, "cpu")


def test_move_condition_rejects_non_boolean_non_blocking() -> None:
    with pytest.raises(TypeError, match="non_blocking must be a bool"):
        move_condition_to_device({}, "cpu", non_blocking=1)


def test_move_condition_validates_nested_mapping_keys() -> None:
    with pytest.raises(TypeError, match="string keys"):
        move_condition_to_device({"nested": {1: torch.ones(1)}}, "cpu")


def test_bind_output_context_preserves_input_ownership_without_mutation() -> None:
    condition = MappingProxyType({"prompt_embeds": torch.ones(1, 2)})
    context = MappingProxyType({"img_ids": torch.zeros(4, 3)})

    bound = bind_output_forward_context(condition, context)

    assert tuple(bound) == ("prompt_embeds", "img_ids")
    assert bound["prompt_embeds"] is condition["prompt_embeds"]
    assert bound["img_ids"] is context["img_ids"]
    assert tuple(condition) == ("prompt_embeds",)
    assert tuple(context) == ("img_ids",)


def test_bind_prepared_condition_preserves_input_and_output_ownership() -> None:
    prepared = PreparedConditionState(
        condition={"prompt_embeds": torch.ones(1, 2)},
        forward_context={"condition_prefix": torch.zeros(1, 3)},
        output_context={"codec_only": torch.ones(1)},
    )

    bound = bind_prepared_condition_output(prepared, {"output_ids": torch.zeros(4, 3)})

    assert tuple(bound) == ("prompt_embeds", "condition_prefix", "output_ids")
    assert "codec_only" not in bound


def test_bind_prepared_condition_rejects_output_collision() -> None:
    prepared = PreparedConditionState(
        condition={"prompt_embeds": torch.ones(1, 2)},
        forward_context={"condition_prefix": torch.zeros(1, 3)},
        output_context={},
    )

    with pytest.raises(ValueError, match=r"collides.*condition_prefix"):
        bind_prepared_condition_output(prepared, {"condition_prefix": torch.ones(1, 3)})


def test_bind_output_context_rejects_ambiguous_key_ownership() -> None:
    with pytest.raises(ValueError, match=r"collides.*\('geometry',\)"):
        bind_output_forward_context({"geometry": 1}, {"geometry": 2})


@pytest.mark.parametrize("side", ["condition", "context"])
@pytest.mark.parametrize("key", sorted(NON_MODEL_CONDITION_KEYS))
def test_bind_output_context_rejects_non_model_fields(side: str, key: str) -> None:
    condition = {key: object()} if side == "condition" else {}
    context = {key: object()} if side == "context" else {}

    with pytest.raises(ValueError, match=rf"cannot enter model forward.*{key}"):
        bind_output_forward_context(condition, context)


@pytest.mark.parametrize(
    ("invalid", "expected_error"),
    [
        (["not", "a", "mapping"], TypeError),
        ({1: "not a string key"}, TypeError),
        ({"": "empty key"}, ValueError),
    ],
)
def test_offline_condition_helpers_reject_invalid_mapping_contract(
    invalid: object,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        move_condition_to_device(invalid, "cpu")
    with pytest.raises(expected_error):
        bind_output_forward_context(invalid, {})
    with pytest.raises(expected_error):
        bind_output_forward_context({}, invalid)


def test_bind_output_context_reports_all_collisions_in_stable_order() -> None:
    with pytest.raises(ValueError, match=r"\('alpha', 'zeta'\)"):
        bind_output_forward_context(
            MappingProxyType({"zeta": 1, "alpha": 2}),
            MappingProxyType({"alpha": 3, "zeta": 4}),
        )


@pytest.mark.parametrize(
    "key",
    ["generator", "loss_weight", "noise", "schema_version", "timestep"],
)
def test_algorithm_vocabulary_does_not_close_the_model_condition_namespace(key: str) -> None:
    marker = object()

    bound = bind_output_forward_context({key: marker}, {})

    assert bound[key] is marker
