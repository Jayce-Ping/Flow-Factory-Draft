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

"""Fake-only tests for SD3.5 on-the-fly offline target encoding."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch
from PIL import Image

from flow_factory.contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaBinding,
    InputMediaOrder,
    MediaType,
    NegativePromptPolicy,
    RateRequirement,
)
from flow_factory.models.output_state import (
    EncodedOutputState,
    GeometrySignature,
    MediaGeometrySignature,
)
from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter
from flow_factory.samples import LatentState

HEIGHT = 6
WIDTH = 10


@dataclass(frozen=True)
class _DecodedMedia:
    type: str
    payload: Any
    fps: Optional[float] = None
    sample_rate: Optional[int] = None


class _FakeProcessor:
    def __init__(self, output_size: Optional[tuple[int, int]] = None) -> None:
        self.output_size = output_size
        self.calls: list[tuple[Any, int, int]] = []
        self.grad_enabled: list[bool] = []

    def preprocess(
        self,
        images: list[Image.Image],
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        self.calls.append((images, height, width))
        self.grad_enabled.append(torch.is_grad_enabled())
        output_height, output_width = self.output_size or (height, width)
        values = torch.arange(
            len(images) * 3 * output_height * output_width,
            dtype=torch.float32,
        )
        return values.reshape(len(images), 3, output_height, output_width)


class _MethodPosterior:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value
        self.mode_calls = 0
        self.sample_calls = 0
        self.grad_enabled: list[bool] = []

    def mode(self) -> torch.Tensor:
        self.mode_calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.value

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        del generator
        self.sample_calls += 1
        raise AssertionError("SD3.5 target encoding must use posterior mode, not sample")


class _PropertyPosterior:
    def __init__(self, value: torch.Tensor) -> None:
        self.mode = value
        self.sample_calls = 0

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        del generator
        self.sample_calls += 1
        raise AssertionError("SD3.5 target encoding must use posterior mode, not sample")


class _FakeVAE:
    def __init__(
        self,
        *,
        output_style: str = "object",
        posterior_style: str = "method",
    ) -> None:
        self.dtype = torch.bfloat16
        self.config = SimpleNamespace(shift_factor=1.5, scaling_factor=2.0)
        self.output_style = output_style
        self.posterior_style = posterior_style
        self.encode_inputs: list[torch.Tensor] = []
        self.encode_grad_enabled: list[bool] = []
        self.posteriors: list[Any] = []

    def encode(self, pixel_values: torch.Tensor) -> Any:
        self.encode_inputs.append(pixel_values)
        self.encode_grad_enabled.append(torch.is_grad_enabled())
        call_offset = float(len(self.encode_inputs))
        value = pixel_values[:, :1, ::2, ::2] + call_offset
        if self.posterior_style == "property":
            posterior: Any = _PropertyPosterior(value)
        else:
            posterior = _MethodPosterior(value)
        self.posteriors.append(posterior)
        if self.output_style == "tuple":
            return (posterior,)
        if self.output_style == "direct":
            return posterior
        return SimpleNamespace(latent_dist=posterior)


def _adapter(
    *,
    height: Any = HEIGHT,
    width: Any = WIDTH,
    processor: Optional[_FakeProcessor] = None,
    vae: Optional[_FakeVAE] = None,
    build_codec: bool = True,
) -> SD3_5Adapter:
    adapter = object.__new__(SD3_5Adapter)
    processor = processor or _FakeProcessor()
    vae = vae or _FakeVAE()
    adapter.training_args = SimpleNamespace(
        height=height,
        width=width,
        latent_storage_dtype=None,
    )
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.pipeline = SimpleNamespace(image_processor=processor, vae=vae)
    adapter.component_runtime = SimpleNamespace(get_component=lambda name: vae)
    adapter._output_state_encoding_modules = ("vae",)
    adapter._output_state_codec = adapter.build_output_state_codec() if build_codec else None
    return adapter


def _media_batch(batch_size: int = 2) -> tuple[tuple[_DecodedMedia, ...], ...]:
    return tuple(
        (
            _DecodedMedia(
                type="image",
                payload=Image.new("RGB", (3 + index, 4 + index), color=(index, 0, 0)),
            ),
        )
        for index in range(batch_size)
    )


def _encoded_geometry(
    *,
    height: int = HEIGHT,
    width: int = WIDTH,
    decode_context: Optional[dict[str, int]] = None,
) -> EncodedOutputState:
    return EncodedOutputState(
        clean_state=LatentState({"latent": torch.zeros(1, 1, 3, 5)}),
        forward_context={},
        decode_context=(
            {"height": HEIGHT, "width": WIDTH} if decode_context is None else decode_context
        ),
        geometry_signatures=(
            GeometrySignature(
                media=(
                    MediaGeometrySignature(
                        type=MediaType.IMAGE,
                        height=height,
                        width=width,
                    ),
                )
            ),
        ),
    )


def test_sd3_5_declares_text_to_image_pipeline_contract() -> None:
    contract = SD3_5Adapter.pipeline_io_contract

    assert contract is not None
    assert contract.input_media.rules == ()
    assert contract.input_media.binding is InputMediaBinding.GROUPED_BY_TYPE
    assert contract.input_media.order is InputMediaOrder.INSENSITIVE
    assert contract.negative_prompt is NegativePromptPolicy.OPTIONAL
    assert tuple(item.type for item in contract.output_media.items) == (MediaType.IMAGE,)
    assert contract.output_media.items[0].fps is RateRequirement.NOT_APPLICABLE
    assert contract.output_media.items[0].sample_rate is RateRequirement.NOT_APPLICABLE
    assert contract.geometry_source is GeometrySource.CONFIGURED
    assert contract.batch_capability is BatchCapability.UNIFORM


def test_sd3_5_codec_reencodes_every_call_with_mode_scaling_and_no_grad() -> None:
    processor = _FakeProcessor()
    vae = _FakeVAE()
    adapter = _adapter(processor=processor, vae=vae)
    media_batch = _media_batch(2)
    generator = torch.Generator().manual_seed(23)

    first = adapter.encode_output_state(media_batch, {}, generator)
    second = adapter.encode_output_state(media_batch, {}, generator)

    assert adapter.output_state_encoding_modules == ("vae",)
    assert adapter.output_state_codec is not None
    assert adapter.output_state_codec.required_components == ("vae",)
    assert len(processor.calls) == 2
    assert len(vae.encode_inputs) == 2
    assert processor.grad_enabled == [False, False]
    assert vae.encode_grad_enabled == [False, False]
    for images, height, width in processor.calls:
        assert height == HEIGHT
        assert width == WIDTH
        assert all(isinstance(image, Image.Image) for image in images)
    assert all(values.dtype is torch.bfloat16 for values in vae.encode_inputs)
    assert all(tuple(values.shape) == (2, 3, HEIGHT, WIDTH) for values in vae.encode_inputs)

    for posterior in vae.posteriors:
        assert posterior.mode_calls == 1
        assert posterior.sample_calls == 0
        assert posterior.grad_enabled == [False]
    expected_first = (vae.posteriors[0].value - 1.5) * 2.0
    expected_second = (vae.posteriors[1].value - 1.5) * 2.0
    assert torch.equal(first.clean_state.components["latent"], expected_first)
    assert torch.equal(second.clean_state.components["latent"], expected_second)
    assert not torch.equal(expected_first, expected_second)
    assert first.clean_state.components["latent"].requires_grad is False
    assert dict(first.forward_context) == {}
    assert dict(first.decode_context) == {"height": HEIGHT, "width": WIDTH}
    assert all(
        signature.media
        == (
            MediaGeometrySignature(
                type=MediaType.IMAGE,
                height=HEIGHT,
                width=WIDTH,
            ),
        )
        for signature in first.geometry_signatures
    )


@pytest.mark.parametrize(
    ("output_style", "posterior_style"),
    [
        ("tuple", "method"),
        ("direct", "method"),
        ("object", "property"),
    ],
)
def test_sd3_5_codec_accepts_supported_diffusers_latent_distribution_surfaces(
    output_style: str,
    posterior_style: str,
) -> None:
    vae = _FakeVAE(output_style=output_style, posterior_style=posterior_style)
    adapter = _adapter(vae=vae)

    encoded = adapter.encode_output_state(_media_batch(1), {})

    assert encoded.clean_state.components["latent"].shape == (1, 1, 3, 5)
    assert vae.posteriors[0].sample_calls == 0


def test_sd3_5_codec_rejects_processor_geometry_before_vae_encode() -> None:
    processor = _FakeProcessor(output_size=(HEIGHT - 1, WIDTH))
    vae = _FakeVAE()
    adapter = _adapter(processor=processor, vae=vae)

    with pytest.raises(ValueError, match=r"did not produce configured geometry"):
        adapter.encode_output_state(_media_batch(1), {})

    assert vae.encode_inputs == []


def test_sd3_5_codec_requires_decoded_pil_targets() -> None:
    processor = _FakeProcessor()
    vae = _FakeVAE()
    adapter = _adapter(processor=processor, vae=vae)
    media_batch = ((_DecodedMedia(type="image", payload=torch.zeros(3, HEIGHT, WIDTH)),),)

    with pytest.raises(TypeError, match=r"expected decoded PIL.Image targets"):
        adapter.encode_output_state(media_batch, {})

    assert processor.calls == []
    assert vae.encode_inputs == []


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("height", None, TypeError, r"training_args.height to be int"),
        ("height", True, TypeError, r"training_args.height to be int"),
        ("width", 0, ValueError, r"training_args.width > 0"),
        ("width", -1, ValueError, r"training_args.width > 0"),
    ],
)
def test_sd3_5_codec_rejects_malformed_configured_geometry(
    field: str,
    value: Any,
    error: type[Exception],
    message: str,
) -> None:
    dimensions = {"height": HEIGHT, "width": WIDTH, field: value}
    adapter = _adapter(**dimensions, build_codec=False)

    with pytest.raises(error, match=message):
        adapter.build_output_state_codec()


@pytest.mark.parametrize(
    ("encoded", "message"),
    [
        (
            _encoded_geometry(height=HEIGHT + 1),
            r"disagrees with configured height/width",
        ),
        (
            _encoded_geometry(decode_context={"height": HEIGHT, "width": WIDTH + 1}),
            r"decode_context must exactly match configured output geometry",
        ),
        (
            _encoded_geometry(decode_context={"height": HEIGHT, "width": WIDTH, "extra": 1}),
            r"decode_context must exactly match configured output geometry",
        ),
    ],
)
def test_sd3_5_geometry_hook_rejects_signature_or_decode_context_drift(
    encoded: EncodedOutputState,
    message: str,
) -> None:
    adapter = _adapter()

    with pytest.raises(ValueError, match=message):
        adapter._validate_encoded_output_geometry(_media_batch(1), {}, encoded)
