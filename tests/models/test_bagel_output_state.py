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

"""Fake-component coverage for Bagel's offline target-state codec."""

import importlib
import sys
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from PIL import Image

# Import the framework/Transformers boundary before presenting a fake flash-attn
# module. Transformers otherwise interprets the test double as an installed package
# while resolving its own package metadata. Bagel itself is still imported through
# the same fail-fast dependency gate used in production.
import flow_factory.models.abc  # noqa: F401, E402
import flow_factory.utils.imports as import_utils  # noqa: E402
from flow_factory.contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaBinding,
    InputMediaOrder,
    MediaType,
    NegativePromptPolicy,
)
from flow_factory.models.output_state import GeometrySignature, MediaGeometrySignature

_fake_flash_attn = ModuleType("flash_attn")
_fake_flash_attn.__spec__ = ModuleSpec("flash_attn", loader=None)
_fake_flash_attn.flash_attn_varlen_func = lambda *args, **kwargs: None
_fake_cv2 = ModuleType("cv2")
_fake_cv2.__spec__ = ModuleSpec("cv2", loader=None)
with (
    patch.object(import_utils, "is_flash_attn_available", return_value=True),
    patch.dict(
        sys.modules,
        {"flash_attn": _fake_flash_attn, "cv2": _fake_cv2},
    ),
):
    BagelAdapter = importlib.import_module("flow_factory.models.bagel.bagel").BagelAdapter


@dataclass(frozen=True)
class _DecodedMedia:
    type: str
    payload: Any
    fps: Optional[float] = None
    sample_rate: Optional[int] = None


class _FakeBagelTransform:
    """Expose the same resize + tensor-transform seam as Bagel's ImageTransform."""

    def __init__(self, output_shapes: dict[tuple[int, int], tuple[int, int]]) -> None:
        self.output_shapes = output_shapes
        self.calls: list[tuple[str, tuple[int, int], bool]] = []

    def _shape(self, image: Image.Image) -> tuple[int, int]:
        return self.output_shapes[image.size]

    def resize_transform(self, image: Image.Image) -> Image.Image:
        height, width = self._shape(image)
        self.calls.append(("resize", image.size, torch.is_grad_enabled()))
        return image.resize((width, height))

    def __call__(self, image: Image.Image) -> torch.Tensor:
        resized = self.resize_transform(image)
        self.calls.append(("tensor", image.size, torch.is_grad_enabled()))
        height, width = resized.height, resized.width
        return torch.arange(3 * height * width, dtype=torch.float32).reshape(
            3,
            height,
            width,
        )


class _FakeBagelVAE(nn.Module):
    """Custom-VAE double whose encode result is already Bagel-normalized."""

    def __init__(self, *, posterior_mean: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.reg = SimpleNamespace(sample=not posterior_mean)
        self.inputs: list[torch.Tensor] = []
        self.grad_enabled: list[bool] = []
        self.outputs: list[torch.Tensor] = []
        self.decode_inputs: list[torch.Tensor] = []

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.inputs.append(pixel_values)
        self.grad_enabled.append(torch.is_grad_enabled())
        batch_size = pixel_values.shape[0]
        latents = torch.arange(
            batch_size * 2 * 4 * 6,
            dtype=torch.float32,
            device=pixel_values.device,
        ).reshape(batch_size, 2, 4, 6)
        latents = (latents + 0.25).to(pixel_values.dtype)
        self.outputs.append(latents)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(latents)
        return latents


def _install_adapter(
    *,
    transform: _FakeBagelTransform,
    vae: _FakeBagelVAE,
) -> BagelAdapter:
    adapter = object.__new__(BagelAdapter)
    adapter.training_args = SimpleNamespace(latent_storage_dtype=None)
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.vae_transform = transform
    adapter.pipeline = SimpleNamespace(
        bagel=SimpleNamespace(
            latent_patch_size=2,
            latent_channel=2,
            latent_downsample=4,
        ),
        vae=vae,
    )
    adapter.component_runtime = SimpleNamespace(get_component=lambda name: vae)
    adapter._output_state_encoding_modules = ("vae",)
    adapter._output_state_codec = adapter.build_output_state_codec()
    return adapter


def _media_batch(*sizes: tuple[int, int]) -> tuple[tuple[_DecodedMedia, ...], ...]:
    return tuple((_DecodedMedia(type="image", payload=Image.new("RGB", size)),) for size in sizes)


def _manual_patchify(latents: torch.Tensor) -> torch.Tensor:
    """Reference Bagel's h,w,p,q,c token order without using einsum."""
    packed_samples = []
    for sample in latents:
        tokens = []
        for latent_h in range(2):
            for latent_w in range(3):
                token = []
                for patch_h in range(2):
                    for patch_w in range(2):
                        for channel in range(2):
                            token.append(
                                sample[
                                    channel,
                                    latent_h * 2 + patch_h,
                                    latent_w * 2 + patch_w,
                                ]
                            )
                tokens.append(torch.stack(token))
        packed_samples.append(torch.stack(tokens))
    return torch.stack(packed_samples)


def test_bagel_declares_dynamic_one_image_output_and_optional_ordered_inputs() -> None:
    BagelAdapter.validate_offline_output_capability()
    contract = BagelAdapter.pipeline_io_contract

    assert contract.negative_prompt is NegativePromptPolicy.UNSUPPORTED
    assert contract.geometry_source is GeometrySource.OUTPUT_MEDIA
    assert contract.batch_capability is BatchCapability.UNIFORM
    assert contract.input_media.binding is InputMediaBinding.GROUPED_BY_TYPE
    assert contract.input_media.order is InputMediaOrder.WITHIN_TYPE
    assert len(contract.input_media.rules) == 1
    assert contract.input_media.rules[0].format.type is MediaType.IMAGE
    assert contract.input_media.rules[0].min_count == 0
    assert contract.input_media.rules[0].max_count is None
    assert tuple(item.type for item in contract.output_media.items) == (MediaType.IMAGE,)


def test_bagel_target_codec_uses_custom_vae_and_official_patch_order() -> None:
    transform = _FakeBagelTransform({(7, 9): (8, 12), (11, 13): (8, 12)})
    vae = _FakeBagelVAE()
    adapter = _install_adapter(transform=transform, vae=vae)

    encoded = adapter.encode_output_state(
        _media_batch((7, 9), (11, 13)),
        {"prompt": ["first", "second"]},
        torch.Generator().manual_seed(7),
    )

    assert len(vae.inputs) == 1
    assert vae.inputs[0].shape == (2, 3, 8, 12)
    assert vae.inputs[0].dtype is torch.bfloat16
    assert vae.grad_enabled == [False]
    assert all(not grad_enabled for kind, _, grad_enabled in transform.calls if kind == "tensor")
    assert torch.equal(
        encoded.clean_state.components["latent"],
        _manual_patchify(vae.outputs[0]),
    )
    assert dict(encoded.forward_context) == {"image_shape": (8, 12)}
    assert dict(encoded.decode_context) == {"image_shape": (8, 12)}
    signature = GeometrySignature(
        media=(
            MediaGeometrySignature(
                type=MediaType.IMAGE,
                height=8,
                width=12,
            ),
        )
    )
    assert encoded.geometry_signatures == (signature, signature)

    reencoded = adapter.encode_output_state(
        _media_batch((7, 9), (11, 13)),
        {"prompt": ["first", "second"]},
    )
    assert len(vae.inputs) == 2
    assert torch.equal(
        reencoded.clean_state.components["latent"],
        encoded.clean_state.components["latent"],
    )

    decoded = adapter.decode_output_state(encoded, output_type="pt")
    assert isinstance(decoded, list)
    assert torch.equal(torch.cat(vae.decode_inputs), vae.outputs[0])


def test_bagel_target_codec_blocks_ragged_geometry_before_vae_encode() -> None:
    transform = _FakeBagelTransform({(7, 9): (8, 12), (11, 13): (12, 8)})
    vae = _FakeBagelVAE()
    adapter = _install_adapter(transform=transform, vae=vae)

    with pytest.raises(ValueError, match="batch size 1 or batch targets by the geometry"):
        adapter.encode_output_state(_media_batch((7, 9), (11, 13)), {})

    assert vae.inputs == []


def test_bagel_target_codec_requires_deterministic_custom_vae_posterior() -> None:
    transform = _FakeBagelTransform({(7, 9): (8, 12)})
    vae = _FakeBagelVAE(posterior_mean=False)
    adapter = _install_adapter(transform=transform, vae=vae)

    with pytest.raises(RuntimeError, match=r"vae\.reg\.sample=False"):
        adapter.encode_output_state(_media_batch((7, 9)), {})

    assert vae.inputs == []
