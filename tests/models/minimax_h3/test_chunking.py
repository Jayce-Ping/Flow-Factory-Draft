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

from copy import deepcopy

import pytest
import torch
from torch import nn

from flow_factory.models.minimax_h3._chunking import (
    _ChunkedFeedForward,
    install_h3_feed_forward_chunking,
)


class SwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(5, 28, bias=False)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.activation(gate)


class FeedForwardFake(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.ModuleList(
            [
                SwiGLU(),
                nn.Dropout(0.0),
                nn.Linear(14, 5, bias=False),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class BlockFake(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ff = FeedForwardFake()


class TransformerFake(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_refiner = nn.Module()
        self.token_refiner.refiner_blocks = nn.ModuleList([BlockFake(), BlockFake()])
        self.transformer_blocks = nn.ModuleList([BlockFake(), BlockFake(), BlockFake()])


def _feed_forwards(transformer: TransformerFake) -> list[nn.Module]:
    return [
        *[block.ff for block in transformer.token_refiner.refiner_blocks],
        *[block.ff for block in transformer.transformer_blocks],
    ]


def test_chunking_preserves_parameter_tree_and_is_idempotent() -> None:
    transformer = TransformerFake()
    state = deepcopy(transformer.state_dict())
    keys_before = tuple(state)
    parameter_ids_before = {
        name: id(parameter) for name, parameter in transformer.named_parameters()
    }

    assert install_h3_feed_forward_chunking(transformer, max_tokens=4) == 5
    assert install_h3_feed_forward_chunking(transformer, max_tokens=4) == 5

    assert all(isinstance(module, _ChunkedFeedForward) for module in _feed_forwards(transformer))
    assert tuple(transformer.state_dict()) == keys_before
    assert {
        name: id(parameter) for name, parameter in transformer.named_parameters()
    } == parameter_ids_before
    transformer.load_state_dict(state, strict=True)
    assert "transformer_blocks.0.ff.net.0.proj.weight" in transformer.state_dict()
    assert not any("inner" in name for name, _ in transformer.named_modules())


def test_chunking_handles_remainder_and_preserves_forward_backward() -> None:
    torch.manual_seed(17)
    direct = TransformerFake().double()
    chunked = deepcopy(direct)
    install_h3_feed_forward_chunking(chunked, max_tokens=4)
    direct_ff = direct.transformer_blocks[0].ff
    chunked_ff = chunked.transformer_blocks[0].ff
    chunk_sizes: list[int] = []
    handle = chunked_ff.net[0].register_forward_pre_hook(
        lambda _module, inputs: chunk_sizes.append(inputs[0].shape[1])
    )
    direct_input = torch.randn(2, 9, 5, dtype=torch.float64, requires_grad=True)
    chunked_input = direct_input.detach().clone().requires_grad_(True)
    output_gradient = torch.randn(2, 9, 5, dtype=torch.float64)

    direct_output = direct_ff(direct_input)
    chunked_output = chunked_ff(chunked_input)
    direct_output.backward(output_gradient)
    chunked_output.backward(output_gradient)
    handle.remove()

    assert chunk_sizes == [4, 4, 1]
    torch.testing.assert_close(chunked_output, direct_output)
    torch.testing.assert_close(chunked_input.grad, direct_input.grad)
    for direct_parameter, chunked_parameter in zip(direct_ff.parameters(), chunked_ff.parameters()):
        torch.testing.assert_close(chunked_parameter.grad, direct_parameter.grad)


def test_short_feed_forward_uses_one_execution() -> None:
    transformer = TransformerFake()
    install_h3_feed_forward_chunking(transformer, max_tokens=4)
    feed_forward = transformer.transformer_blocks[0].ff
    chunk_sizes: list[int] = []
    handle = feed_forward.net[0].register_forward_pre_hook(
        lambda _module, inputs: chunk_sizes.append(inputs[0].shape[1])
    )

    feed_forward(torch.randn(2, 4, 5))
    handle.remove()

    assert chunk_sizes == [4]


@pytest.mark.parametrize("max_tokens", [0, -1, True, 1.5])
def test_chunking_rejects_invalid_max_tokens(max_tokens: object) -> None:
    with pytest.raises(ValueError, match="max_tokens expected a positive int"):
        install_h3_feed_forward_chunking(TransformerFake(), max_tokens=max_tokens)


def test_chunking_rejects_conflicting_reinstallation() -> None:
    transformer = TransformerFake()
    install_h3_feed_forward_chunking(transformer, max_tokens=4)

    with pytest.raises(ValueError, match="already uses max_tokens=4.*conflicting 8"):
        install_h3_feed_forward_chunking(transformer, max_tokens=8)
