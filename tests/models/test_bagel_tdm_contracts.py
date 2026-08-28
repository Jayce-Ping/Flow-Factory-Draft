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

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from types import MethodType
from typing import Any

import pytest
import torch

import flow_factory.utils.imports as import_utils
from flow_factory.samples import BaseSample
from flow_factory.trainers.distillation.distillation_runtime import (
    validate_media_free_rollout,
    without_media_decoding,
)


def _load_bagel_types(monkeypatch: pytest.MonkeyPatch) -> tuple[type, type]:
    """Load Bagel behind the same optional-kernel seam as its adapter tests."""
    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    flash_attn.flash_attn_varlen_func = lambda *args, **kwargs: None
    cv2 = types.ModuleType("cv2")
    cv2.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setattr(import_utils, "is_flash_attn_available", lambda *args: True)
    monkeypatch.setattr(import_utils, "get_flash_attn_version", lambda: "test")

    module = importlib.import_module("flow_factory.models.bagel.bagel")
    return module.BagelAdapter, module.BagelSample


def _adapter(adapter_type: type, decoder: Any) -> Any:
    adapter = object.__new__(adapter_type)
    adapter.decode_latents = MethodType(decoder, adapter)
    return adapter


def _result(batch_size: int = 2) -> dict[str, Any]:
    initial = torch.zeros(batch_size, 4, 8)
    terminal = torch.ones(batch_size, 4, 8)
    return {
        "final_latents": terminal,
        "all_latents": [initial, terminal],
        "all_log_probs": None,
        "timesteps": torch.tensor([1000.0]),
        "latent_index_map": torch.tensor([0, 1]),
        "log_prob_index_map": None,
        "callback_results": {},
        "callback_index_map": None,
    }


def test_bagel_assembles_samples_from_one_batched_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_type, sample_type = _load_bagel_types(monkeypatch)
    calls: list[tuple[tuple[int, ...], tuple[int, int] | None]] = []

    def decode(
        _self: Any,
        latents: torch.Tensor,
        image_shape: tuple[int, int] | None = None,
    ) -> list[torch.Tensor]:
        calls.append((tuple(latents.shape), image_shape))
        return [torch.full((3, 8, 8), index) for index in range(latents.shape[0])]

    adapter = _adapter(adapter_type, decode)
    samples = adapter._assemble_samples(
        _result(),
        prompts=["first", "second"],
        condition_images_list=None,
        height=64,
        width=64,
    )

    assert calls == [((2, 4, 8), (64, 64))]
    assert all(isinstance(sample, sample_type) for sample in samples)
    assert torch.equal(samples[0].image, torch.zeros(3, 8, 8))
    assert torch.equal(samples[1].image, torch.ones(3, 8, 8))


def test_bagel_media_free_samples_keep_replay_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_type, _ = _load_bagel_types(monkeypatch)

    def decode(
        _self: Any,
        latents: torch.Tensor,
        image_shape: tuple[int, int] | None = None,
    ) -> list[torch.Tensor]:
        del latents, image_shape
        raise AssertionError("TDM must not invoke the real Bagel decoder")

    adapter = _adapter(adapter_type, decode)
    validate_media_free_rollout(adapter, algorithm_name="TDM")

    with without_media_decoding(adapter, algorithm_name="TDM"):
        samples = adapter._assemble_samples(
            _result(),
            prompts=["first", "second"],
            condition_images_list=None,
            height=64,
            width=64,
        )

    assert [sample.image for sample in samples] == [None, None]
    batch = BaseSample.stack(samples)
    replay = adapter.get_replay_step(batch, 0)
    assert replay.state.components["latent"].shape == (2, 4, 8)
    assert replay.next_state.components["latent"].shape == (2, 4, 8)

    with pytest.raises(AssertionError, match="real Bagel decoder"):
        adapter.decode_latents(torch.zeros(2, 4, 8), image_shape=(64, 64))


def test_bagel_maps_reference_guidance_to_text_cfg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_type, _ = _load_bagel_types(monkeypatch)
    adapter = object.__new__(adapter_type)

    assert adapter.reference_guidance_kwargs(4.0) == {"cfg_text_scale": 4.0}
