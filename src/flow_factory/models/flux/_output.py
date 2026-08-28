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

"""Role-neutral VAE encoding shared by the FLUX pipeline variants."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, List, Optional, Tuple

import torch

from ..configured_image_output import (
    EncodedImageTensor,
    VAELatentSampleMode,
    encode_shift_scale_vae_image,
    retrieve_vae_latents,
)


def encode_flux1_vae_image(
    adapter: Any,
    pixel_values: torch.Tensor,
    *,
    sample_mode: VAELatentSampleMode,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply explicit posterior selection shared by FLUX.1 roles.

    Args:
        adapter: FLUX.1 adapter exposing the canonical VAE.
        pixel_values: Preprocessed BCHW image tensor.
        sample_mode: ``sample`` for targets or the official condition selection.
        generator: Generator forwarded unchanged for posterior sampling.

    Returns:
        Shifted and scaled convolutional VAE latents.
    """
    return encode_shift_scale_vae_image(
        adapter,
        pixel_values,
        sample_mode=sample_mode,
        generator=generator,
    )


def encode_flux2_output_images(
    adapter: Any,
    pixel_values: torch.Tensor,
    generator: Optional[torch.Generator],
) -> EncodedImageTensor:
    """Sample the FLUX.2 target posterior, normalize, and pack tokens.

    Args:
        adapter: FLUX.2 adapter exposing pipeline packing primitives.
        pixel_values: Preprocessed BCHW target images.
        generator: Generator forwarded unchanged to target posterior sampling.

    Returns:
        Packed clean latents with their position identifiers.
    """
    latents = encode_flux2_vae_image(
        adapter,
        pixel_values,
        sample_mode="sample",
        generator=generator,
    )
    latent_ids = adapter.pipeline._prepare_latent_ids(latents).to(adapter.device)
    packed = adapter.pipeline._pack_latents(latents)
    return EncodedImageTensor(
        latents=packed,
        forward_context={"latent_ids": latent_ids},
        decode_context={"latent_ids": latent_ids},
    )


def encode_flux2_vae_image(
    adapter: Any,
    pixel_values: torch.Tensor,
    *,
    sample_mode: VAELatentSampleMode,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply the exact FLUX.2 transform shared by condition and target roles.

    Args:
        adapter: FLUX.2 adapter exposing the canonical VAE and patchify primitive.
        pixel_values: Preprocessed BCHW image tensor.
        sample_mode: Explicit posterior selection for the calling pipeline role.
        generator: Generator forwarded unchanged for posterior sampling.

    Returns:
        Patchified and BatchNorm-normalized convolutional latents.
    """
    if pixel_values.ndim != 4:
        raise ValueError(
            f"{type(adapter).__name__} FLUX.2 VAE expected BCHW input, "
            f"received shape {tuple(pixel_values.shape)}"
        )
    vae = adapter.vae
    latents = retrieve_vae_latents(
        vae.encode(pixel_values),
        sample_mode=sample_mode,
        generator=generator,
        source=f"{type(adapter).__name__} FLUX.2 VAE encode",
    )
    latents = adapter.pipeline._patchify_latents(latents)
    if latents.ndim != 4:
        raise ValueError(
            f"{type(adapter).__name__} FLUX.2 patchify expected BCHW output, "
            f"received shape {tuple(latents.shape)}"
        )

    batch_norm = getattr(vae, "bn", None)
    running_mean = getattr(batch_norm, "running_mean", None)
    running_var = getattr(batch_norm, "running_var", None)
    if not isinstance(running_mean, torch.Tensor) or not isinstance(running_var, torch.Tensor):
        raise TypeError(
            f"{type(adapter).__name__} FLUX.2 VAE must expose BatchNorm running_mean/running_var"
        )
    channels = latents.shape[1]
    if running_mean.ndim != 1 or running_var.ndim != 1:
        raise ValueError(
            f"{type(adapter).__name__} FLUX.2 BatchNorm statistics must be rank 1, "
            f"received mean={tuple(running_mean.shape)}, var={tuple(running_var.shape)}"
        )
    if running_mean.numel() != channels or running_var.numel() != channels:
        raise ValueError(
            f"{type(adapter).__name__} FLUX.2 BatchNorm expected {channels} values, "
            f"received mean={running_mean.numel()}, var={running_var.numel()}"
        )
    eps = getattr(getattr(vae, "config", None), "batch_norm_eps", None)
    if isinstance(eps, bool) or not isinstance(eps, Real):
        raise TypeError(
            f"{type(adapter).__name__} FLUX.2 VAE config expected numeric batch_norm_eps, "
            f"received {type(eps).__name__}: {eps!r}"
        )
    if not math.isfinite(float(eps)) or eps < 0:
        raise ValueError(
            f"{type(adapter).__name__} FLUX.2 VAE batch_norm_eps must be finite and >= 0, "
            f"received {eps!r}"
        )
    mean_values = running_mean.view(1, -1, 1, 1)
    variance_values = running_var.view(1, -1, 1, 1)
    if not torch.isfinite(mean_values).all() or not torch.isfinite(variance_values).all():
        raise ValueError(f"{type(adapter).__name__} FLUX.2 BatchNorm statistics must be finite")
    if torch.any(variance_values + float(eps) <= 0):
        raise ValueError(
            f"{type(adapter).__name__} FLUX.2 BatchNorm variance plus epsilon must be positive"
        )
    mean = mean_values.to(device=latents.device, dtype=latents.dtype)
    std = torch.sqrt(variance_values + float(eps)).to(
        device=latents.device,
        dtype=latents.dtype,
    )
    return (latents - mean) / std


def prepare_flux2_condition_latents(
    adapter: Any,
    images: List[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose FLUX.2 condition tokens around the shared VAE transform.

    Args:
        adapter: FLUX.2 adapter exposing condition packing primitives.
        images: One or more preprocessed BCHW condition tensors for one sample.
        batch_size: Number of prompt rows that reuse the condition sequence.
        device: Target device for condition tensors and identifiers.
        dtype: VAE input dtype.

    Returns:
        Packed condition latents and their repeated position identifiers.
    """
    if not images:
        raise ValueError(f"{type(adapter).__name__} requires at least one condition image")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError(
            f"{type(adapter).__name__} condition batch_size must be a positive int, "
            f"received {batch_size!r}"
        )
    image_latents = [
        encode_flux2_vae_image(
            adapter,
            image.to(device=device, dtype=dtype),
            sample_mode="argmax",
        )
        for image in images
    ]
    image_latent_ids = adapter.pipeline._prepare_image_ids(image_latents)
    packed_latents = [adapter.pipeline._pack_latents(latent).squeeze(0) for latent in image_latents]
    packed = torch.cat(packed_latents, dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
    image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1).to(device)
    return packed, image_latent_ids


__all__ = [
    "encode_flux1_vae_image",
    "encode_flux2_output_images",
    "encode_flux2_vae_image",
    "prepare_flux2_condition_latents",
]
