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

"""On-the-fly clean-video encoding for Wan text-to-video adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Tuple

import numpy as np
import torch

from ...contracts import MediaType
from ...samples import LatentState
from ..configured_image_output import retrieve_vae_latents
from ..output_state import (
    DecodedMediaBatch,
    EncodedOutputState,
    GeometrySignature,
    MediaGeometrySignature,
)


@dataclass(frozen=True, slots=True)
class WanVideoOutputCodec:
    """Encode configured Wan target videos without retaining pixels or latents."""

    adapter: Any
    required_components: ClassVar[Tuple[str, ...]] = ("vae",)

    def encode_output_state(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> EncodedOutputState:
        """Preprocess, VAE-sample, and normalize one video per sample."""
        del condition
        height, width, num_frames, frame_rate = self.adapter._configured_video_output_geometry()
        videos = []
        for sample_index, candidate in enumerate(media_batch):
            if len(candidate) != 1:
                raise ValueError(
                    "Wan output codec expected one video per sample, "
                    f"received {len(candidate)} for sample {sample_index}"
                )
            media = candidate[0]
            payload = media.payload
            if not isinstance(payload, np.ndarray):
                raise TypeError(
                    "Wan output codec expected decoded NumPy video targets, "
                    f"received {type(payload).__name__} for sample {sample_index}"
                )
            videos.append(
                self.adapter._resample_output_video(
                    payload,
                    source_fps=media.fps,
                    target_frames=num_frames,
                    target_fps=frame_rate,
                )
            )

        pixel_values = self.adapter.pipeline.video_processor.preprocess_video(
            videos,
            height=height,
            width=width,
        )
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(
                "Wan video_processor.preprocess_video must return torch.Tensor, "
                f"received {type(pixel_values).__name__}"
            )
        expected_shape = (len(videos), 3, num_frames, height, width)
        if tuple(pixel_values.shape) != expected_shape:
            raise ValueError(
                "Wan video preprocessing changed configured output geometry: "
                f"expected {expected_shape}, received {tuple(pixel_values.shape)}"
            )

        vae = self.adapter.vae
        vae_dtype = getattr(vae, "dtype", None)
        if not isinstance(vae_dtype, torch.dtype) or not vae_dtype.is_floating_point:
            raise TypeError(
                "Wan output codec expected VAE to expose a floating dtype, "
                f"received {vae_dtype!r}"
            )
        pixel_values = pixel_values.to(device=self.adapter.device, dtype=vae_dtype)
        encoded = vae.encode(pixel_values)
        latents = retrieve_vae_latents(
            encoded,
            sample_mode="sample",
            generator=generator,
            source="Wan target video",
        )
        latents = self.adapter._normalize_output_video_latents(latents)

        temporal_scale = self.adapter.pipeline.vae_scale_factor_temporal
        spatial_scale = self.adapter.pipeline.vae_scale_factor_spatial
        expected_latent_shape = (
            len(videos),
            getattr(vae.config, "z_dim", latents.shape[1]),
            (num_frames - 1) // temporal_scale + 1,
            height // spatial_scale,
            width // spatial_scale,
        )
        if tuple(latents.shape) != expected_latent_shape:
            raise ValueError(
                "Wan VAE target latent geometry mismatch: "
                f"expected {expected_latent_shape}, received {tuple(latents.shape)}"
            )

        signature = GeometrySignature(
            media=(
                MediaGeometrySignature(
                    type=MediaType.VIDEO,
                    height=height,
                    width=width,
                    frames=num_frames,
                    fps=frame_rate,
                ),
            )
        )
        return EncodedOutputState(
            clean_state=LatentState({"latent": latents}),
            forward_context={},
            decode_context={
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
            },
            geometry_signatures=tuple(signature for _ in videos),
        )


__all__ = ["WanVideoOutputCodec"]
