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

"""Fake-only coverage for image-family offline output codecs."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch
from PIL import Image

from diffusers import Flux2Pipeline, FluxKontextPipeline, QwenImageEditPlusPipeline
from flow_factory.contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaOrder,
    MediaType,
)
from flow_factory.models.flux._output import (
    encode_flux1_vae_image,
    encode_flux2_vae_image,
    prepare_flux2_condition_latents,
)
from flow_factory.models.flux.flux1 import Flux1Adapter
from flow_factory.models.flux.flux1_kontext import Flux1KontextAdapter
from flow_factory.models.flux.flux2 import Flux2Adapter
from flow_factory.models.flux.flux2_klein import Flux2KleinAdapter
from flow_factory.models.qwen_image._output import encode_qwen_vae_image
from flow_factory.models.qwen_image.qwen_image import QwenImageAdapter
from flow_factory.models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusAdapter
from flow_factory.models.sensenova.sensenova import SenseNovaAdapter
from flow_factory.models.z_image.z_image import ZImageAdapter
from flow_factory.samples import ComponentTimes, LatentState


@dataclass(frozen=True)
class _DecodedMedia:
    type: str
    payload: Any
    fps: Optional[float] = None
    sample_rate: Optional[int] = None


class _Processor:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, bool]] = []

    def preprocess(
        self,
        images: list[Image.Image],
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        self.calls.append((len(images), height, width, torch.is_grad_enabled()))
        return torch.arange(
            len(images) * 3 * height * width,
            dtype=torch.float32,
        ).reshape(len(images), 3, height, width)


class _Posterior:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value
        self.mode_calls = 0

    def mode(self) -> torch.Tensor:
        self.mode_calls += 1
        return self.value

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        del generator
        raise AssertionError("offline target encoding must not sample a VAE posterior")


class _ConvVAE:
    def __init__(self, *, shift: float = 1.5, scale: float = 2.0) -> None:
        self.dtype = torch.float32
        self.config = SimpleNamespace(shift_factor=shift, scaling_factor=scale)
        self.inputs: list[torch.Tensor] = []
        self.posteriors: list[_Posterior] = []

    def encode(self, values: torch.Tensor) -> Any:
        self.inputs.append(values)
        posterior = _Posterior(values[:, :2, ::8, ::8] + 3.0)
        self.posteriors.append(posterior)
        return SimpleNamespace(latent_dist=posterior)


class _QwenVAE:
    def __init__(self) -> None:
        self.dtype = torch.float32
        self.config = SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0])
        self.inputs: list[torch.Tensor] = []
        self.posteriors: list[_Posterior] = []

    def encode(self, values: torch.Tensor) -> Any:
        self.inputs.append(values)
        assert values.ndim == 5
        latent = torch.cat(
            [
                values[:, :1, :, ::8, ::8] + 5.0,
                values[:, 1:2, :, ::8, ::8] + 10.0,
            ],
            dim=1,
        )
        posterior = _Posterior(latent)
        self.posteriors.append(posterior)
        return SimpleNamespace(latent_dist=posterior)


def _media_batch(batch_size: int = 2) -> tuple[tuple[_DecodedMedia, ...], ...]:
    return tuple(
        (
            _DecodedMedia(
                type="image",
                payload=Image.new(
                    "RGB",
                    (11 + index, 13 + index),
                    color=(17 + index, 80, 200),
                ),
            ),
        )
        for index in range(batch_size)
    )


def _install_adapter_runtime(
    adapter: Any,
    *,
    pipeline: Any,
    component: Any,
    height: int = 32,
    width: int = 32,
) -> Any:
    adapter.training_args = SimpleNamespace(
        height=height,
        width=width,
        latent_storage_dtype=None,
    )
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.pipeline = pipeline
    adapter.component_runtime = SimpleNamespace(
        get_component=lambda name: component,
    )
    adapter._output_state_encoding_modules = (
        () if isinstance(adapter, SenseNovaAdapter) else ("vae",)
    )
    adapter._output_state_codec = adapter.build_output_state_codec()
    return adapter


class _Flux1Pipeline:
    vae_scale_factor = 8

    def __init__(self, processor: _Processor) -> None:
        self.image_processor = processor

    @staticmethod
    def _pack_latents(
        latents: torch.Tensor,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        assert tuple(latents.shape) == (batch_size, channels, height, width)
        return (
            latents.reshape(batch_size, channels, height // 2, 2, width // 2, 2)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch_size, height // 2 * (width // 2), channels * 4)
        )

    @staticmethod
    def _prepare_latent_image_ids(
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del batch_size
        return torch.zeros(height * width, 3, device=device, dtype=dtype)


def test_flux1_target_codec_uses_the_same_deterministic_vae_transform_as_kontext() -> None:
    processor = _Processor()
    vae = _ConvVAE()
    pipeline = _Flux1Pipeline(processor)
    plain = _install_adapter_runtime(
        object.__new__(Flux1Adapter),
        pipeline=pipeline,
        component=vae,
    )

    encoded = plain.encode_output_state(_media_batch(), {})

    expected = (vae.posteriors[0].value - 1.5) * 2.0
    assert vae.posteriors[0].mode_calls == 1
    assert torch.equal(
        encoded.clean_state.components["latent"],
        pipeline._pack_latents(expected, 2, 2, 4, 4),
    )
    assert encoded.forward_context["img_ids"].shape == (4, 3)
    assert processor.calls == [(2, 32, 32, False)]

    kontext = _install_adapter_runtime(
        object.__new__(Flux1KontextAdapter),
        pipeline=pipeline,
        component=vae,
    )
    condition_ids = torch.ones(2, 5, 3)
    conditioned = kontext.encode_output_state(
        _media_batch(),
        {"image_ids": condition_ids},
    )
    assert conditioned.forward_context["latent_ids"].shape == (9, 3)
    assert torch.equal(conditioned.forward_context["latent_ids"][-5:], condition_ids[0])


def test_flux1_shared_vae_transform_matches_pinned_diffusers() -> None:
    pixels = torch.randn(2, 3, 32, 32)
    expected_vae = _ConvVAE()
    expected = FluxKontextPipeline._encode_vae_image(
        SimpleNamespace(vae=expected_vae),
        pixels,
        torch.Generator().manual_seed(11),
    )
    actual_vae = _ConvVAE()
    actual = encode_flux1_vae_image(
        SimpleNamespace(vae=actual_vae),
        pixels,
    )

    assert torch.equal(actual, expected)
    assert actual_vae.posteriors[0].mode_calls == 1


@pytest.mark.parametrize("adapter_cls", [Flux2Adapter, Flux2KleinAdapter])
def test_flux2_variants_delegate_target_encoding_to_official_pipeline_primitives(
    adapter_cls: type,
) -> None:
    processor = _Processor()
    generator = torch.Generator().manual_seed(7)
    calls: dict[str, Any] = {}

    class _Flux2VAE:
        dtype = torch.bfloat16
        config = SimpleNamespace(batch_norm_eps=0.0)
        bn = SimpleNamespace(running_mean=torch.zeros(2), running_var=torch.ones(2))

        def __init__(self) -> None:
            self.inputs: list[torch.Tensor] = []
            self.posteriors: list[_Posterior] = []

        def encode(self, image: torch.Tensor) -> Any:
            self.inputs.append(image)
            calls["encode_grad"] = torch.is_grad_enabled()
            posterior = _Posterior(image[:, :2, ::8, ::8])
            self.posteriors.append(posterior)
            return SimpleNamespace(latent_dist=posterior)

    vae = _Flux2VAE()

    def ids(latents: torch.Tensor) -> torch.Tensor:
        calls["ids"] = latents
        return torch.arange(latents.shape[-2] * latents.shape[-1] * 4).reshape(-1, 4)

    def pack(latents: torch.Tensor) -> torch.Tensor:
        calls["pack"] = latents
        return latents.flatten(2).transpose(1, 2)

    def condition_ids(latents: list[torch.Tensor]) -> torch.Tensor:
        length = sum(item.shape[-2] * item.shape[-1] for item in latents)
        return torch.zeros(1, length, 4, dtype=torch.long)

    pipeline = SimpleNamespace(
        image_processor=processor,
        vae_scale_factor=8,
        _patchify_latents=lambda latents: latents,
        _prepare_latent_ids=ids,
        _prepare_image_ids=condition_ids,
        _pack_latents=pack,
    )
    adapter = _install_adapter_runtime(
        object.__new__(adapter_cls),
        pipeline=pipeline,
        component=vae,
    )

    encoded = adapter.encode_output_state(_media_batch(), {}, generator)

    assert vae.inputs[0].dtype is torch.bfloat16
    assert calls["encode_grad"] is False
    assert vae.posteriors[0].mode_calls == 1
    assert calls["ids"] is calls["pack"]
    assert encoded.forward_context["latent_ids"] is encoded.decode_context["latent_ids"]
    assert encoded.clean_state.components["latent"].shape == (2, 16, 2)

    condition_latents, condition_latent_ids = prepare_flux2_condition_latents(
        adapter,
        [torch.zeros(1, 3, 32, 32)],
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert vae.posteriors[1].mode_calls == 1
    assert condition_latents.shape == (1, 16, 2)
    assert condition_latent_ids.shape == (1, 16, 4)


def test_flux2_shared_vae_transform_matches_pinned_diffusers() -> None:
    pixels = torch.randn(2, 3, 32, 32, dtype=torch.bfloat16)

    class _Flux2VAE:
        config = SimpleNamespace(batch_norm_eps=1e-5)
        bn = SimpleNamespace(
            running_mean=torch.tensor([0.25, -0.5]),
            running_var=torch.tensor([0.011197119951248169, 0.75]),
        )

        def encode(self, image: torch.Tensor) -> Any:
            return SimpleNamespace(latent_dist=_Posterior(image[:, :2, ::8, ::8]))

    patchify = lambda latents: latents.contiguous()
    expected_vae = _Flux2VAE()
    expected = Flux2Pipeline._encode_vae_image(
        SimpleNamespace(vae=expected_vae, _patchify_latents=patchify),
        pixels,
        torch.Generator().manual_seed(12),
    )
    actual_vae = _Flux2VAE()
    actual = encode_flux2_vae_image(
        SimpleNamespace(
            vae=actual_vae,
            pipeline=SimpleNamespace(_patchify_latents=patchify),
        ),
        pixels,
    )

    assert torch.equal(actual, expected)


def _qwen_pipeline(processor: _Processor) -> Any:
    def pack(
        latents: torch.Tensor,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        assert tuple(latents.shape) == (batch_size, channels, 1, height, width)
        return latents.squeeze(2).flatten(2).transpose(1, 2)

    return SimpleNamespace(
        image_processor=processor,
        vae_scale_factor=8,
        _pack_latents=pack,
    )


def test_qwen_target_codec_uses_five_dimensional_vae_and_img_shapes() -> None:
    processor = _Processor()
    vae = _QwenVAE()
    adapter = _install_adapter_runtime(
        object.__new__(QwenImageAdapter),
        pipeline=_qwen_pipeline(processor),
        component=vae,
    )

    encoded = adapter.encode_output_state(_media_batch(), {})

    assert vae.inputs[0].shape == (2, 3, 1, 32, 32)
    assert vae.posteriors[0].mode_calls == 1
    assert encoded.clean_state.components["latent"].shape == (2, 16, 2)
    assert encoded.forward_context["img_shapes"] == [[(1, 2, 2)], [(1, 2, 2)]]


def test_qwen_shared_vae_transform_matches_pinned_diffusers() -> None:
    pixels = torch.randn(2, 3, 1, 32, 32)
    expected_vae = _QwenVAE()
    expected = QwenImageEditPlusPipeline._encode_vae_image(
        SimpleNamespace(vae=expected_vae, latent_channels=2),
        pixels,
        torch.Generator().manual_seed(13),
    )
    actual_vae = _QwenVAE()
    actual = encode_qwen_vae_image(
        SimpleNamespace(vae=actual_vae),
        pixels,
    )

    assert torch.equal(actual, expected)
    assert actual_vae.posteriors[0].mode_calls == 1


def test_qwen_edit_target_geometry_and_shape_metadata_are_condition_owned() -> None:
    processor = _Processor()
    vae = _QwenVAE()
    adapter = _install_adapter_runtime(
        object.__new__(QwenImageEditPlusAdapter),
        pipeline=_qwen_pipeline(processor),
        component=vae,
        height=64,
        width=64,
    )
    condition = {
        "condition_image_sizes": [[(32, 32), (64, 32)]],
        "vae_image_sizes": [[(32, 32), (64, 32)]],
    }

    encoded = adapter.encode_output_state(_media_batch(1), condition)

    assert processor.calls == [(1, 32, 96, False)]
    assert dict(encoded.decode_context) == {"height": 32, "width": 96}
    assert encoded.forward_context["img_shapes"] == [[(1, 2, 6), (1, 2, 2), (1, 2, 4)]]


def test_z_image_codec_uses_deterministic_shift_scale_latents() -> None:
    processor = _Processor()
    vae = _ConvVAE(shift=0.25, scale=1.75)
    pipeline = SimpleNamespace(image_processor=processor, vae_scale_factor=8)
    adapter = _install_adapter_runtime(
        object.__new__(ZImageAdapter),
        pipeline=pipeline,
        component=vae,
    )

    encoded = adapter.encode_output_state(_media_batch(), {})

    assert torch.equal(
        encoded.clean_state.components["latent"],
        (vae.posteriors[0].value - 0.25) * 1.75,
    )
    assert vae.posteriors[0].mode_calls == 1


def test_sensenova_codec_keeps_targets_as_vae_free_pixel_flow_state() -> None:
    transformer = SimpleNamespace(dtype=torch.bfloat16)
    adapter = object.__new__(SenseNovaAdapter)
    adapter._offline_output_geometry = lambda: (16, 24)
    adapter = _install_adapter_runtime(
        adapter,
        pipeline=SimpleNamespace(),
        component=transformer,
        height=16,
        width=24,
    )

    encoded = adapter.encode_output_state(_media_batch(), {})

    pixels = encoded.clean_state.components["latent"]
    assert pixels.shape == (2, 3, 16, 24)
    assert pixels.dtype is torch.bfloat16
    assert pixels.min() >= -1
    assert pixels.max() <= 1
    assert dict(encoded.forward_context) == {}
    decoded = adapter.decode_latents(pixels, output_type="pt")
    expected_rgb = torch.tensor([17, 80, 200], dtype=torch.float32) / 255
    assert torch.allclose(decoded[0, :, 0, 0], expected_rgb, atol=5e-3)


def test_sensenova_offline_noise_uses_the_resolution_scale() -> None:
    adapter = object.__new__(SenseNovaAdapter)
    adapter._noise_scale = lambda image_size: 3.0
    clean_tensor = torch.zeros(2, 3, 4, 6)
    clean = LatentState({"latent": clean_tensor})
    sigma = torch.tensor([0.25, 0.75])
    times = ComponentTimes(
        timestep={"latent": sigma * 1000},
        next_timestep={"latent": torch.zeros_like(sigma)},
        sigma={"latent": sigma},
        next_sigma={"latent": torch.zeros_like(sigma)},
    )
    generator = torch.Generator().manual_seed(19)
    expected_generator = torch.Generator().manual_seed(19)
    expected_noise = torch.randn(clean_tensor.shape, generator=expected_generator) * 3.0

    noised = adapter.add_forward_process_noise(clean, times, generator=generator)

    assert torch.equal(noised.noise.components["latent"], expected_noise)
    assert torch.equal(noised.target_velocity.components["latent"], expected_noise)
    assert torch.allclose(
        noised.state.components["latent"],
        sigma.view(2, 1, 1, 1) * expected_noise,
    )


@pytest.mark.parametrize(
    ("adapter_cls", "geometry_source", "batch_capability", "input_order"),
    [
        (
            Flux1Adapter,
            GeometrySource.CONFIGURED,
            BatchCapability.UNIFORM,
            InputMediaOrder.INSENSITIVE,
        ),
        (
            Flux1KontextAdapter,
            GeometrySource.CONFIGURED,
            BatchCapability.UNIFORM,
            InputMediaOrder.INSENSITIVE,
        ),
        (
            Flux2Adapter,
            GeometrySource.CONFIGURED,
            BatchCapability.UNIFORM,
            InputMediaOrder.WITHIN_TYPE,
        ),
        (
            Flux2KleinAdapter,
            GeometrySource.CONFIGURED,
            BatchCapability.UNIFORM,
            InputMediaOrder.WITHIN_TYPE,
        ),
        (
            QwenImageAdapter,
            GeometrySource.CONFIGURED,
            BatchCapability.UNIFORM,
            InputMediaOrder.INSENSITIVE,
        ),
        (
            QwenImageEditPlusAdapter,
            GeometrySource.INPUT_MEDIA,
            BatchCapability.SINGLE_SAMPLE,
            InputMediaOrder.WITHIN_TYPE,
        ),
        (
            ZImageAdapter,
            GeometrySource.CONFIGURED,
            BatchCapability.UNIFORM,
            InputMediaOrder.INSENSITIVE,
        ),
        (
            SenseNovaAdapter,
            GeometrySource.CONFIGURED,
            BatchCapability.UNIFORM,
            InputMediaOrder.WITHIN_TYPE,
        ),
    ],
)
def test_image_adapter_contracts_are_explicit_and_orthogonal(
    adapter_cls: type,
    geometry_source: GeometrySource,
    batch_capability: BatchCapability,
    input_order: InputMediaOrder,
) -> None:
    contract = adapter_cls.pipeline_io_contract

    assert contract is not None
    assert tuple(item.type for item in contract.output_media.items) == (MediaType.IMAGE,)
    assert contract.geometry_source is geometry_source
    assert contract.batch_capability is batch_capability
    assert contract.input_media.order is input_order
