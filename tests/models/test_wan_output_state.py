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

"""Fake-only tests for Wan target-video output-state encoding."""

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
import torch

from flow_factory.contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaBinding,
    InputMediaOrder,
    MediaType,
    NegativePromptPolicy,
    RateRequirement,
)
from flow_factory.models.output_state import MediaGeometrySignature
from flow_factory.models.wan.wan2_i2v import Wan2_I2V_Adapter
from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter
from flow_factory.scheduler import UniPCMultistepSDESchedulerOutput

HEIGHT = 32
WIDTH = 48
NUM_FRAMES = 5
FPS = 24.0
Z_DIM = 4
TEMPORAL_SCALE = 4
SPATIAL_SCALE = 8


@dataclass(frozen=True)
class _DecodedMedia:
    type: str
    payload: Any
    fps: Optional[float] = None
    sample_rate: Optional[int] = None


class _FakeVideoProcessor:
    def __init__(self, output_shape: Optional[tuple[int, ...]] = None) -> None:
        self.output_shape = output_shape
        self.calls: list[tuple[list[Any], int, int]] = []
        self.grad_enabled: list[bool] = []

    def preprocess_video(
        self,
        videos: list[Any],
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        self.calls.append((videos, height, width))
        self.grad_enabled.append(torch.is_grad_enabled())
        shape = self.output_shape or (len(videos), 3, NUM_FRAMES, height, width)
        values = torch.arange(np.prod(shape), dtype=torch.float32)
        return values.reshape(shape)


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
        raise AssertionError("Wan target encoding must use posterior mode, not sample")


class _PropertyPosterior:
    def __init__(self, value: torch.Tensor) -> None:
        self.mode = value
        self.sample_calls = 0

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        del generator
        self.sample_calls += 1
        raise AssertionError("Wan target encoding must use posterior mode, not sample")


class _FakeVAE:
    def __init__(
        self,
        *,
        output_style: str = "object",
        posterior_style: str = "method",
        latent_shape: Optional[tuple[int, ...]] = None,
        latents_std: tuple[float, ...] = (2.0, 4.0, 5.0, 8.0),
    ) -> None:
        self.dtype = torch.bfloat16
        self.config = SimpleNamespace(
            z_dim=Z_DIM,
            latents_mean=(1.0, 2.0, 3.0, 4.0),
            latents_std=latents_std,
        )
        self.output_style = output_style
        self.posterior_style = posterior_style
        self.latent_shape = latent_shape
        self.encode_inputs: list[torch.Tensor] = []
        self.encode_grad_enabled: list[bool] = []
        self.posteriors: list[Any] = []

    def encode(self, pixel_values: torch.Tensor) -> Any:
        self.encode_inputs.append(pixel_values)
        self.encode_grad_enabled.append(torch.is_grad_enabled())
        expected_shape = (
            pixel_values.shape[0],
            Z_DIM,
            (NUM_FRAMES - 1) // TEMPORAL_SCALE + 1,
            HEIGHT // SPATIAL_SCALE,
            WIDTH // SPATIAL_SCALE,
        )
        shape = self.latent_shape or expected_shape
        values = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
        values = values.to(pixel_values.dtype) + len(self.encode_inputs)
        if self.posterior_style == "property":
            posterior: Any = _PropertyPosterior(values)
        else:
            posterior = _MethodPosterior(values)
        self.posteriors.append(posterior)
        if self.output_style == "tuple":
            return (posterior,)
        if self.output_style == "direct":
            return posterior
        return SimpleNamespace(latent_dist=posterior)


class _RecordingTransformer(torch.nn.Module):
    def __init__(self, marker: float) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.tensor(marker))
        self.dtype = torch.float32
        self.calls: list[dict[str, torch.Tensor]] = []
        self.cache_modes: list[str] = []

    @contextmanager
    def cache_context(self, mode: str):
        self.cache_modes.append(mode)
        yield

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_kwargs: Any,
        return_dict: bool,
    ) -> tuple[torch.Tensor]:
        del attention_kwargs
        assert return_dict is False
        self.calls.append(
            {
                "hidden_states": hidden_states.detach().clone(),
                "timestep": timestep.detach().clone(),
                "encoder_hidden_states": encoder_hidden_states.detach().clone(),
            }
        )
        return (torch.ones_like(hidden_states) * self.marker,)


class _RecordingScheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_train_timesteps=1000)
        self.calls: list[dict[str, Any]] = []

    def step(self, **kwargs: Any) -> UniPCMultistepSDESchedulerOutput:
        self.calls.append(kwargs)
        return UniPCMultistepSDESchedulerOutput(velocity=kwargs["velocity"])


def _forward_adapter(
    *,
    expand_timesteps: bool,
) -> tuple[Wan2_T2V_Adapter, _RecordingTransformer, _RecordingTransformer, _RecordingScheduler]:
    high = _RecordingTransformer(10.0)
    low = _RecordingTransformer(20.0)
    scheduler = _RecordingScheduler()
    modules = {"transformer": high, "transformer_2": low}
    adapter = object.__new__(Wan2_T2V_Adapter)
    adapter.pipeline = SimpleNamespace(
        transformer=high,
        transformer_2=low,
        scheduler=scheduler,
        config=SimpleNamespace(boundary_ratio=0.5, expand_timesteps=expand_timesteps),
    )
    adapter.component_runtime = SimpleNamespace(
        get_component=lambda name: modules[name],
    )
    return adapter, high, low, scheduler


def _adapter(
    *,
    height: Any = HEIGHT,
    width: Any = WIDTH,
    num_frames: Any = NUM_FRAMES,
    processor: Optional[_FakeVideoProcessor] = None,
    vae: Optional[_FakeVAE] = None,
    build_codec: bool = True,
) -> Wan2_T2V_Adapter:
    adapter = object.__new__(Wan2_T2V_Adapter)
    processor = processor or _FakeVideoProcessor()
    vae = vae or _FakeVAE()
    adapter.training_args = SimpleNamespace(
        height=height,
        width=width,
        num_frames=num_frames,
        latent_storage_dtype=None,
    )
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.pipeline = SimpleNamespace(
        video_processor=processor,
        vae=vae,
        vae_scale_factor_temporal=TEMPORAL_SCALE,
        vae_scale_factor_spatial=SPATIAL_SCALE,
        transformer=SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2))),
        transformer_2=None,
    )
    adapter.component_runtime = SimpleNamespace(get_component=lambda name: vae)
    adapter._output_state_encoding_modules = ("vae",)
    adapter._output_state_codec = adapter.build_output_state_codec() if build_codec else None
    return adapter


def _media_batch(
    batch_size: int = 2,
    *,
    num_frames: int = NUM_FRAMES,
    fps: float = FPS,
) -> tuple[tuple[_DecodedMedia, ...], ...]:
    return tuple(
        (
            _DecodedMedia(
                type="video",
                payload=np.full(
                    (num_frames, 7 + index, 9 + index, 3),
                    index,
                    dtype=np.uint8,
                ),
                fps=fps,
            ),
        )
        for index in range(batch_size)
    )


def test_wan_t2v_declares_text_to_video_pipeline_contract() -> None:
    contract = Wan2_T2V_Adapter.pipeline_io_contract

    assert contract is not None
    assert contract.input_media.rules == ()
    assert contract.input_media.binding is InputMediaBinding.GROUPED_BY_TYPE
    assert contract.input_media.order is InputMediaOrder.INSENSITIVE
    assert contract.negative_prompt is NegativePromptPolicy.OPTIONAL
    assert tuple(item.type for item in contract.output_media.items) == (MediaType.VIDEO,)
    assert contract.output_media.items[0].fps is RateRequirement.OPTIONAL
    assert contract.output_media.items[0].sample_rate is RateRequirement.NOT_APPLICABLE
    assert contract.geometry_source is GeometrySource.CONFIGURED
    assert contract.batch_capability is BatchCapability.UNIFORM


def test_wan_t2v_input_media_encoders_are_explicit_no_ops() -> None:
    adapter = object.__new__(Wan2_T2V_Adapter)

    assert adapter.encode_image(torch.zeros(3, 2, 2)) is None
    assert adapter.encode_video(torch.zeros(1, 3, 2, 2)) is None


def test_wan_t2v_codec_reencodes_every_call_with_mode_normalization_and_no_grad() -> None:
    processor = _FakeVideoProcessor()
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
    for videos, height, width in processor.calls:
        assert height == HEIGHT
        assert width == WIDTH
        assert all(isinstance(video, np.ndarray) for video in videos)
    assert all(values.dtype is torch.bfloat16 for values in vae.encode_inputs)
    assert all(
        tuple(values.shape) == (2, 3, NUM_FRAMES, HEIGHT, WIDTH) for values in vae.encode_inputs
    )

    for posterior in vae.posteriors:
        assert posterior.mode_calls == 1
        assert posterior.sample_calls == 0
        assert posterior.grad_enabled == [False]
    mean = torch.tensor((1.0, 2.0, 3.0, 4.0), dtype=torch.bfloat16).view(1, -1, 1, 1, 1)
    std = torch.tensor((2.0, 4.0, 5.0, 8.0), dtype=torch.bfloat16).view(1, -1, 1, 1, 1)
    expected_first = (vae.posteriors[0].value - mean) / std
    expected_second = (vae.posteriors[1].value - mean) / std
    assert torch.equal(first.clean_state.components["latent"], expected_first)
    assert torch.equal(second.clean_state.components["latent"], expected_second)
    assert not torch.equal(expected_first, expected_second)
    assert first.clean_state.components["latent"].requires_grad is False
    assert dict(first.forward_context) == {}
    assert dict(first.decode_context) == {
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
    }
    assert all(
        signature.media
        == (
            MediaGeometrySignature(
                type=MediaType.VIDEO,
                height=HEIGHT,
                width=WIDTH,
                frames=NUM_FRAMES,
                fps=FPS,
            ),
        )
        for signature in first.geometry_signatures
    )


def test_wan_t2v_forward_routes_independent_batch_timesteps_and_restores_order() -> None:
    adapter, high, low, scheduler = _forward_adapter(expand_timesteps=True)
    latents = torch.stack([torch.full((2, 2, 3, 5), float(index)) for index in range(3)])
    prompt_embeds = torch.stack([torch.full((2, 4), float(index)) for index in range(3)])
    timesteps = torch.tensor([100.0, 500.0, 200.0])
    next_timesteps = torch.tensor([50.0, 800.0, 150.0])

    output = adapter.forward(
        t=timesteps,
        t_next=next_timesteps,
        latents=latents,
        prompt_embeds=prompt_embeds,
        guidance_scale=1.0,
        compute_log_prob=False,
        return_kwargs=["velocity"],
    )

    assert len(high.calls) == 1
    assert len(low.calls) == 1
    assert high.calls[0]["hidden_states"][:, 0, 0, 0, 0].tolist() == [1.0]
    assert low.calls[0]["hidden_states"][:, 0, 0, 0, 0].tolist() == [0.0, 2.0]
    assert tuple(high.calls[0]["timestep"].shape) == (1, 12)
    assert tuple(low.calls[0]["timestep"].shape) == (2, 12)
    assert torch.all(high.calls[0]["timestep"] == 500.0)
    assert torch.all(low.calls[0]["timestep"][0] == 100.0)
    assert torch.all(low.calls[0]["timestep"][1] == 200.0)
    assert output.velocity is not None
    assert output.velocity[:, 0, 0, 0, 0].tolist() == [20.0, 10.0, 20.0]
    assert len(scheduler.calls) == 1
    assert scheduler.calls[0]["timestep"] is timesteps
    assert scheduler.calls[0]["timestep_next"] is next_timesteps
    assert scheduler.calls[0]["latents"] is latents

    output.velocity.sum().backward()
    assert high.marker.grad is not None
    assert low.marker.grad is not None


def test_wan_t2v_forward_preserves_shared_scalar_rollout_semantics() -> None:
    adapter, high, low, scheduler = _forward_adapter(expand_timesteps=False)
    latents = torch.zeros(2, 2, 2, 3, 5)
    prompt_embeds = torch.zeros(2, 2, 4)
    timestep = torch.tensor(700.0)
    next_timestep = torch.tensor(600.0)

    output = adapter.forward(
        t=timestep,
        t_next=next_timestep,
        latents=latents,
        prompt_embeds=prompt_embeds,
        guidance_scale=1.0,
        compute_log_prob=False,
        return_kwargs=["velocity"],
    )

    assert high.calls[0]["timestep"].tolist() == [700.0, 700.0]
    assert low.calls == []
    assert output.velocity is not None
    assert torch.all(output.velocity == 10.0)
    assert scheduler.calls[0]["timestep"] is timestep
    assert scheduler.calls[0]["timestep_next"] is next_timestep


@pytest.mark.parametrize(
    ("output_style", "posterior_style"),
    [
        ("tuple", "method"),
        ("direct", "method"),
        ("object", "property"),
    ],
)
def test_wan_t2v_codec_accepts_supported_diffusers_posterior_surfaces(
    output_style: str,
    posterior_style: str,
) -> None:
    vae = _FakeVAE(output_style=output_style, posterior_style=posterior_style)
    adapter = _adapter(vae=vae)

    encoded = adapter.encode_output_state(_media_batch(1), {})

    assert encoded.clean_state.components["latent"].shape == (1, Z_DIM, 2, 4, 6)
    assert vae.posteriors[0].sample_calls == 0


def test_wan_t2v_codec_rejects_temporal_mismatch_before_video_preprocessing() -> None:
    processor = _FakeVideoProcessor()
    vae = _FakeVAE()
    adapter = _adapter(processor=processor, vae=vae)

    with pytest.raises(ValueError, match=r"expected 5 frames, received 4"):
        adapter.encode_output_state(_media_batch(1, num_frames=4), {})

    assert processor.calls == []
    assert vae.encode_inputs == []


def test_wan_t2v_codec_rejects_processor_geometry_before_vae_encode() -> None:
    processor = _FakeVideoProcessor(output_shape=(1, 3, NUM_FRAMES, HEIGHT - 1, WIDTH))
    vae = _FakeVAE()
    adapter = _adapter(processor=processor, vae=vae)

    with pytest.raises(ValueError, match=r"changed configured target geometry"):
        adapter.encode_output_state(_media_batch(1), {})

    assert vae.encode_inputs == []


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("num_frames", None, TypeError, r"training_args.num_frames to be int"),
        ("num_frames", 4, ValueError, r"num_frames - 1.*divisible"),
        ("height", 31, ValueError, r"height/width divisible"),
        ("width", 0, ValueError, r"training_args.width > 0"),
    ],
)
def test_wan_t2v_codec_rejects_malformed_configured_geometry(
    field: str,
    value: Any,
    error: type[Exception],
    message: str,
) -> None:
    dimensions = {"height": HEIGHT, "width": WIDTH, "num_frames": NUM_FRAMES, field: value}
    adapter = _adapter(**dimensions, build_codec=False)

    with pytest.raises(error, match=message):
        adapter.build_output_state_codec()


def test_wan_t2v_codec_rejects_invalid_vae_normalization_statistics() -> None:
    vae = _FakeVAE(latents_std=(2.0, 0.0, 5.0, 8.0))
    adapter = _adapter(vae=vae)

    with pytest.raises(ValueError, match=r"latents_std values must all be positive"):
        adapter.encode_output_state(_media_batch(1), {})


def test_wan_i2v_declares_contract_and_static_codec_capability_blocker() -> None:
    contract = Wan2_I2V_Adapter.pipeline_io_contract
    assert contract is not None
    assert tuple(rule.format.type for rule in contract.input_media.rules) == (MediaType.IMAGE,)
    assert contract.input_media.rules[0].min_count == 1
    assert contract.input_media.rules[0].max_count == 1
    assert tuple(item.type for item in contract.output_media.items) == (MediaType.VIDEO,)

    reason = Wan2_I2V_Adapter.output_state_codec_unavailable_reason
    assert reason is not None
    assert "source pixels required to rebuild its VAE condition tensor" in reason
    assert "do not infer" in reason
