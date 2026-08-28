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

"""Role-neutral VAE encoding shared by Qwen-Image pipeline variants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch

from ..configured_image_output import (
    EncodedImageTensor,
    VAELatentSampleMode,
    retrieve_vae_latents,
)


def encode_qwen_vae_image(
    adapter: Any,
    video_values: torch.Tensor,
    *,
    sample_mode: VAELatentSampleMode,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply explicit posterior selection shared by Qwen image roles.

    Args:
        adapter: Qwen-Image adapter exposing the canonical VAE.
        video_values: Preprocessed BCFHW image-as-video tensor.
        sample_mode: Explicit posterior selection for the calling pipeline role.
        generator: Generator forwarded unchanged for posterior sampling.

    Returns:
        Channel-normalized five-dimensional clean latents.
    """
    latents = retrieve_vae_latents(
        adapter.vae.encode(video_values),
        sample_mode=sample_mode,
        generator=generator,
        source=f"{type(adapter).__name__} VAE encode",
    )
    latent_channels = latents.shape[1]
    means = _channel_statistics(
        adapter.vae.config.latents_mean,
        latent_channels,
        latents,
        "latents_mean",
    )
    stds = _channel_statistics(
        adapter.vae.config.latents_std,
        latent_channels,
        latents,
        "latents_std",
    )
    if torch.any(stds <= 0):
        raise ValueError(f"{type(adapter).__name__} VAE latents_std must be positive")
    return (latents - means) / stds


def encode_qwen_output_images(
    adapter: Any,
    pixel_values: torch.Tensor,
    condition: Mapping[str, Any],
    generator: Optional[torch.Generator],
    *,
    condition_sizes_key: Optional[str] = None,
) -> EncodedImageTensor:
    """Apply official 5D Qwen VAE normalization, packing, and shape metadata.

    Args:
        adapter: Qwen-Image adapter exposing VAE and packing primitives.
        pixel_values: Preprocessed BCHW target images.
        condition: Cached model condition for the same batch.
        generator: Generator forwarded unchanged to target posterior sampling.
        condition_sizes_key: Optional condition field containing per-image VAE geometry.

    Returns:
        Packed clean target latents and target-first image-shape metadata.
    """
    video_values = pixel_values.unsqueeze(2)
    latents = encode_qwen_vae_image(
        adapter,
        video_values,
        sample_mode="sample",
        generator=generator,
    )
    latent_channels = latents.shape[1]

    batch_size = latents.shape[0]
    latent_height, latent_width = latents.shape[-2:]
    packed = adapter.pipeline._pack_latents(
        latents,
        batch_size,
        latent_channels,
        latent_height,
        latent_width,
    )
    target_shape = (1, latent_height // 2, latent_width // 2)
    img_shapes = [[target_shape] for _ in range(batch_size)]
    if condition_sizes_key is not None:
        condition_sizes = condition.get(condition_sizes_key)
        parsed_sizes = _parse_condition_sizes(
            condition_sizes,
            batch_size=batch_size,
            source=f"{type(adapter).__name__} condition[{condition_sizes_key!r}]",
        )
        scale = adapter.pipeline.vae_scale_factor * 2
        for sample_index, sizes in enumerate(parsed_sizes):
            for width, height in sizes:
                if width % scale or height % scale:
                    raise ValueError(
                        f"{type(adapter).__name__} condition VAE geometry {(height, width)} "
                        f"must be divisible by {scale}"
                    )
                img_shapes[sample_index].append((1, height // scale, width // scale))

    return EncodedImageTensor(
        latents=packed,
        forward_context={"img_shapes": img_shapes},
        decode_context={},
    )


def _channel_statistics(
    values: Any,
    channels: int,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Materialize one finite Qwen latent statistic per channel."""
    statistics = torch.as_tensor(values, device=reference.device, dtype=reference.dtype)
    if statistics.ndim != 1 or statistics.numel() != channels:
        raise ValueError(
            f"Qwen VAE {name} expected {channels} values, received shape "
            f"{tuple(statistics.shape)}"
        )
    if not torch.isfinite(statistics).all():
        raise ValueError(f"Qwen VAE {name} must contain only finite values")
    return statistics.view(1, channels, 1, 1, 1)


def _parse_condition_sizes(
    value: Any,
    *,
    batch_size: int,
    source: str,
) -> list[list[tuple[int, int]]]:
    """Validate collated per-sample ``(width, height)`` condition geometry."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{source} must be a per-sample sequence, received {type(value).__name__}")
    if len(value) != batch_size:
        raise ValueError(
            f"{source} expected batch size {batch_size}, received sequence length {len(value)}"
        )
    result: list[list[tuple[int, int]]] = []
    for sample_index, sizes in enumerate(value):
        if not isinstance(sizes, Sequence) or isinstance(sizes, (str, bytes)):
            raise TypeError(f"{source}[{sample_index}] must be a sequence of (width, height) pairs")
        parsed: list[tuple[int, int]] = []
        for size_index, size in enumerate(sizes):
            if not isinstance(size, Sequence) or isinstance(size, (str, bytes)) or len(size) != 2:
                raise TypeError(
                    f"{source}[{sample_index}][{size_index}] must be a (width, height) pair"
                )
            width, height = size
            if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
                raise ValueError(
                    f"{source}[{sample_index}][{size_index}] expected positive integer "
                    f"geometry, received {tuple(size)!r}"
                )
            parsed.append((width, height))
        result.append(parsed)
    return result


__all__ = ["encode_qwen_output_images", "encode_qwen_vae_image"]
