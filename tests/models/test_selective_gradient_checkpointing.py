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
import torch

from flow_factory.hparams.gradient_checkpointing import GradientCheckpointingSpec
from flow_factory.models.checkpointing import (
    discover_gradient_checkpointing_units,
    select_gradient_checkpointing_units,
    selective_gradient_checkpointing_function,
)
from flow_factory.models.minimax_h3.adapters import MiniMaxH3T2VAAdapter


class RepeatedBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return value.square()


class RepeatedModel(torch.nn.Module):
    _repeated_blocks = ["RepeatedBlock"]

    def __init__(self, count: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([RepeatedBlock() for _ in range(count)])


class MiniMaxCheckpointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_refiner = torch.nn.Module()
        self.token_refiner.refiner_blocks = torch.nn.ModuleList([RepeatedBlock(), RepeatedBlock()])
        self.transformer_blocks = torch.nn.ModuleList(
            [RepeatedBlock(), RepeatedBlock(), RepeatedBlock()]
        )
        self.checkpointing_function = None

    def enable_gradient_checkpointing(self, function=None) -> None:
        self.checkpointing_function = function


class TransformersCheckpointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.kwargs = gradient_checkpointing_kwargs


def test_checkpoint_units_follow_registered_block_order() -> None:
    model = RepeatedModel(4)

    units = discover_gradient_checkpointing_units(model)

    assert [name for name, _ in units] == [
        "blocks.0",
        "blocks.1",
        "blocks.2",
        "blocks.3",
    ]


def test_fraction_selection_is_even_and_includes_endpoints() -> None:
    units = discover_gradient_checkpointing_units(RepeatedModel(10))

    selected = select_gradient_checkpointing_units(
        GradientCheckpointingSpec(mode="fraction", fraction=0.3),
        units,
    )

    assert [name for name, _ in selected] == ["blocks.0", "blocks.4", "blocks.9"]


def test_explicit_layer_selection_rejects_unknown_index() -> None:
    units = discover_gradient_checkpointing_units(RepeatedModel(2))

    with pytest.raises(ValueError, match=r"exceed unit_count=2"):
        select_gradient_checkpointing_units(
            GradientCheckpointingSpec(mode="layers", layers=(2,)),
            units,
        )


def test_selective_callback_recomputes_only_selected_blocks() -> None:
    selected_block = RepeatedBlock()
    direct_block = RepeatedBlock()
    function = selective_gradient_checkpointing_function([("selected", selected_block)])
    selected_input = torch.tensor(2.0, requires_grad=True)
    direct_input = torch.tensor(3.0, requires_grad=True)

    (function(selected_block, selected_input) + function(direct_block, direct_input)).backward()

    assert selected_block.calls == 2
    assert direct_block.calls == 1
    torch.testing.assert_close(selected_input.grad, torch.tensor(4.0))
    torch.testing.assert_close(direct_input.grad, torch.tensor(6.0))


def test_h3_checkpoint_units_match_forward_stack_order() -> None:
    adapter = object.__new__(MiniMaxH3T2VAAdapter)
    component = torch.nn.Module()
    component.token_refiner = torch.nn.Module()
    component.token_refiner.refiner_blocks = torch.nn.ModuleList([RepeatedBlock(), RepeatedBlock()])
    component.transformer_blocks = torch.nn.ModuleList(
        [RepeatedBlock(), RepeatedBlock(), RepeatedBlock()]
    )

    units = adapter._gradient_checkpointing_units("transformer", component)

    assert [name for name, _ in units] == [
        "token_refiner.refiner_blocks.0",
        "token_refiner.refiner_blocks.1",
        "transformer_blocks.0",
        "transformer_blocks.1",
        "transformer_blocks.2",
    ]


def test_h3_adapter_applies_fraction_policy_to_selected_unit_only() -> None:
    adapter = object.__new__(MiniMaxH3T2VAAdapter)
    component = MiniMaxCheckpointModel()
    adapter.accelerator = SimpleNamespace(unwrap_model=lambda module: module)
    adapter.component_runtime = SimpleNamespace(
        get_component=lambda name: component,
    )
    adapter.model_args = SimpleNamespace(target_components=["transformer"])
    adapter.training_args = SimpleNamespace(
        enable_gradient_checkpointing=GradientCheckpointingSpec(
            mode="fraction",
            fraction=0.2,
        )
    )

    adapter.enable_gradient_checkpointing()

    assert component.checkpointing_function is not None
    value = torch.tensor(1.0, requires_grad=True)
    units = adapter._gradient_checkpointing_units("transformer", component)
    for _, block in units:
        value = component.checkpointing_function(block, value)
    value.backward()
    assert [block.calls for _, block in units] == [1, 1, 2, 1, 1]


def test_full_policy_bridges_transformers_checkpointing_api() -> None:
    adapter = object.__new__(MiniMaxH3T2VAAdapter)
    component = TransformersCheckpointModel()
    adapter.accelerator = SimpleNamespace(unwrap_model=lambda module: module)
    adapter.component_runtime = SimpleNamespace(
        get_component=lambda name: component,
    )
    adapter.model_args = SimpleNamespace(target_components=["transformer"])
    adapter.training_args = SimpleNamespace(enable_gradient_checkpointing=True)

    adapter.enable_gradient_checkpointing()

    assert component.kwargs == {"use_reentrant": False}


def test_selective_policy_rejects_component_without_callback_api() -> None:
    adapter = object.__new__(MiniMaxH3T2VAAdapter)
    component = TransformersCheckpointModel()
    adapter.accelerator = SimpleNamespace(unwrap_model=lambda module: module)
    adapter.component_runtime = SimpleNamespace(
        get_component=lambda name: component,
    )
    adapter.model_args = SimpleNamespace(target_components=["transformer"])
    adapter.training_args = SimpleNamespace(
        enable_gradient_checkpointing=GradientCheckpointingSpec(
            mode="every_n",
            every_n=2,
        )
    )

    with pytest.raises(TypeError, match=r"requires enable_gradient_checkpointing"):
        adapter.enable_gradient_checkpointing()
