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

"""Cover the positional alignment that names an official TDM release's tensors."""

from __future__ import annotations

import importlib.util
import os
from typing import Any

import pytest
import torch

_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
    "convert_tdm_checkpoint.py",
)
_spec = importlib.util.spec_from_file_location("convert_tdm_checkpoint", _SCRIPT)
convert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert)


def _reference(count: int) -> list:
    """Build alternating lora_A/lora_B names with the release's shapes."""
    names = []
    for index in range(count):
        kind = "lora_A" if index % 2 == 0 else "lora_B"
        shape = torch.Size((4, 8)) if index % 2 == 0 else torch.Size((8, 4))
        names.append((f"base_model.model.block{index // 2}.attn.to_q.{kind}.default.weight", shape))
    return names


def _release(count: int) -> list:
    return [
        torch.ones(4, 8) if index % 2 == 0 else torch.ones(8, 4) for index in range(count)
    ]


def test_names_are_assigned_by_position_and_lose_the_adapter_suffix() -> None:
    """The saved adapter drops the adapter name PEFT carries on live parameters."""
    state_dict = convert.align(_release(4), _reference(4))

    assert list(state_dict) == [
        "base_model.model.block0.attn.to_q.lora_A.weight",
        "base_model.model.block0.attn.to_q.lora_B.weight",
        "base_model.model.block1.attn.to_q.lora_A.weight",
        "base_model.model.block1.attn.to_q.lora_B.weight",
    ]


def test_a_wrong_target_module_set_is_caught_by_the_count() -> None:
    """Silently naming 382 tensors with 384 slots would train from shifted weights."""
    with pytest.raises(ValueError, match="target_modules do not match"):
        convert.align(_release(4), _reference(6))


def test_a_reordered_rebuild_is_caught_by_the_shapes() -> None:
    """The list carries no names, so shape is the only evidence of the right order."""
    reference = _reference(4)
    reference[0], reference[1] = reference[1], reference[0]

    with pytest.raises(ValueError, match="module order does not match"):
        convert.align(_release(4), reference)


def test_a_rebuild_without_lora_pairs_is_rejected() -> None:
    """A wrapper that produced only one side of the factorization is not a LoRA."""
    reference = [
        ("base_model.model.block0.attn.to_q.lora_A.default.weight", torch.Size((4, 8))),
        ("base_model.model.block1.attn.to_q.lora_A.default.weight", torch.Size((4, 8))),
    ]

    with pytest.raises(ValueError, match="matching lora_A and lora_B"):
        convert.align([torch.ones(4, 8), torch.ones(4, 8)], reference)


def test_a_checkpoint_without_the_parameter_list_names_what_it_has(tmp_path: Any) -> None:
    """Other .ckpt layouts exist upstream; the message has to say which one arrived."""
    path = tmp_path / "wrong.ckpt"
    torch.save({"state_dict": {}}, path)

    with pytest.raises(ValueError, match="received keys"):
        convert.read_release_parameters(str(path))


def test_a_missing_checkpoint_is_reported_before_any_model_is_built(tmp_path: Any) -> None:
    """Rebuilding the wrapper first would spend minutes to reach the same conclusion."""
    with pytest.raises(FileNotFoundError, match="expected a TDM checkpoint"):
        convert.read_release_parameters(str(tmp_path / "absent.ckpt"))
