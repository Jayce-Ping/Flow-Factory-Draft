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
from typing import Any

import numpy as np
import pytest
import torch

from flow_factory.contracts import GeometrySource, MediaType, RateRequirement
from flow_factory.data_utils.offline_dataset import DecodedMedia
from flow_factory.models.wan._output import WanVideoOutputCodec
from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter


class _Posterior:
    def __init__(self, latents: torch.Tensor) -> None:
        self.latents = latents
        self.generators: list[torch.Generator | None] = []

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        self.generators.append(generator)
        return self.latents


class _VAE:
    dtype = torch.float32

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            z_dim=3,
            latents_mean=[1.0, 2.0, 3.0],
            latents_std=[2.0, 4.0, 5.0],
        )
        raw_channels = torch.tensor([3.0, 6.0, 8.0]).view(1, 3, 1, 1, 1)
        self.posterior = _Posterior(raw_channels.expand(1, 3, 2, 2, 2).clone())
        self.encoded_pixels: list[torch.Tensor] = []

    def encode(self, pixels: torch.Tensor) -> Any:
        self.encoded_pixels.append(pixels)
        return SimpleNamespace(latent_dist=self.posterior)


class _VideoProcessor:
    def __init__(self) -> None:
        self.videos: list[list[np.ndarray]] = []

    def preprocess_video(
        self,
        videos: list[np.ndarray],
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        self.videos.append(videos)
        return torch.zeros(len(videos), 3, videos[0].shape[0], height, width)


class _Adapter:
    _configured_video_output_geometry = Wan2_T2V_Adapter._configured_video_output_geometry
    _resample_output_video = staticmethod(Wan2_T2V_Adapter._resample_output_video)
    _normalize_output_video_latents = Wan2_T2V_Adapter._normalize_output_video_latents

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.training_args = SimpleNamespace(
            height=16,
            width=16,
            num_frames=5,
            frame_rate=4.0,
        )
        self.vae = _VAE()
        self.pipeline = SimpleNamespace(
            vae_scale_factor_temporal=4,
            vae_scale_factor_spatial=8,
            transformer=SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2))),
            transformer_2=None,
            video_processor=_VideoProcessor(),
        )


def _media(video: np.ndarray, fps: float = 8.0):
    return (
        (
            DecodedMedia(
                type="video",
                path="target.mp4",
                payload=video,
                fps=fps,
            ),
        ),
    )


def test_wan_t2v_declares_required_video_output_semantics() -> None:
    contract = Wan2_T2V_Adapter.pipeline_io_contract

    assert contract.geometry_source is GeometrySource.CONFIGURED
    assert contract.output_media.items[0].type is MediaType.VIDEO
    assert contract.output_media.items[0].fps is RateRequirement.REQUIRED
    Wan2_T2V_Adapter.validate_offline_output_capability()

    adapter = object.__new__(Wan2_T2V_Adapter)
    codec = adapter.build_output_state_codec()
    assert isinstance(codec, WanVideoOutputCodec)
    assert codec.required_components == ("vae",)


def test_wan_codec_resamples_preprocesses_and_samples_target_latents() -> None:
    adapter = _Adapter()
    codec = WanVideoOutputCodec(adapter)
    source = np.arange(9 * 4 * 4 * 3, dtype=np.uint8).reshape(9, 4, 4, 3)
    generator = torch.Generator().manual_seed(7)

    encoded = codec.encode_output_state(_media(source), {}, generator)

    selected = adapter.pipeline.video_processor.videos[0][0]
    np.testing.assert_array_equal(selected, source[[0, 2, 4, 6, 8]])
    assert adapter.vae.posterior.generators == [generator]
    torch.testing.assert_close(
        encoded.clean_state.components["latent"],
        torch.ones(1, 3, 2, 2, 2),
    )
    assert encoded.forward_context == {}
    assert dict(encoded.decode_context) == {
        "height": 16,
        "width": 16,
        "num_frames": 5,
        "frame_rate": 4.0,
    }
    geometry = encoded.geometry_signatures[0].media[0]
    assert (geometry.type, geometry.height, geometry.width, geometry.frames, geometry.fps) == (
        MediaType.VIDEO,
        16,
        16,
        5,
        4.0,
    )


def test_wan_codec_rejects_insufficient_duration_and_invalid_latent_grid() -> None:
    adapter = _Adapter()
    codec = WanVideoOutputCodec(adapter)
    short = np.zeros((8, 4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="too short"):
        codec.encode_output_state(_media(short), {})

    adapter.training_args.num_frames = 6
    with pytest.raises(ValueError, match="num_frames must satisfy"):
        adapter._configured_video_output_geometry()


def test_wan_geometry_validator_rejects_output_context_drift() -> None:
    adapter = _Adapter()
    encoded = WanVideoOutputCodec(adapter).encode_output_state(
        _media(np.zeros((9, 4, 4, 3), dtype=np.uint8)),
        {},
    )

    Wan2_T2V_Adapter._validate_encoded_output_geometry(
        adapter, _media(np.zeros((9, 1, 1, 3), dtype=np.uint8)), {}, encoded
    )

    drifted = type(encoded)(
        clean_state=encoded.clean_state,
        forward_context=encoded.forward_context,
        decode_context={**dict(encoded.decode_context), "frame_rate": 8.0},
        geometry_signatures=encoded.geometry_signatures,
    )
    with pytest.raises(ValueError, match="decode_context 'frame_rate'"):
        Wan2_T2V_Adapter._validate_encoded_output_geometry(
            adapter,
            _media(np.zeros((9, 1, 1, 3), dtype=np.uint8)),
            {},
            drifted,
        )
