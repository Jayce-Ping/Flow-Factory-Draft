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

# src/flow_factory/models/latent_geometry.py
"""Model-agnostic latent geometry.

Describes the *axis roles* of an adapter's latent tensor (:class:`LatentAxes`) so
model-agnostic consumers can locate the batch / channel / spatial / temporal /
sequence axes across every adapter layout:

- ``PACKED`` ``(B, Seq, C)``     -- FLUX*, Qwen-Image*, LTX2*, Bagel
- ``CONV``   ``(B, C, H, W)``    -- SD3.5, Z-Image
- ``VIDEO``  ``(B, C, T, H, W)`` -- Wan2 T2V/I2V/V2V

The geometry records *which axis plays which role*, never concrete sizes, so it
stays valid as resolution, frame count, or reference-image count change at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class LatentLayout(str, Enum):
    """Canonical latent tensor layouts across adapters."""

    PACKED = "packed"  # (B, Seq, C)
    CONV = "conv"  # (B, C, H, W)
    VIDEO = "video"  # (B, C, T, H, W)


@dataclass(frozen=True)
class LatentAxes:
    """Resolution-invariant axis roles for a latent tensor.

    Records only *which axis plays which role*, never concrete sizes. Dynamic
    dims (sequence length, height, width, frames) change with resolution / frame
    count and are intentionally not stored.

    Attributes:
        layout: The canonical :class:`LatentLayout`.
        batch: Index of the batch axis (always 0 in this codebase).
        channel: Index of the latent-channel axis (``-1`` packed, ``1`` conv/video).
        spatial: Indices of spatial axes (``()`` for packed -- H/W are folded into
            the sequence dim by patchify; ``(2, 3)`` conv; ``(3, 4)`` video).
        temporal: Index of the temporal axis if present (``2`` for video) else ``None``.
        sequence: Index of the packed-token axis if present (``1`` for packed) else ``None``.
    """

    layout: LatentLayout
    batch: int = 0
    channel: int = -1
    spatial: Tuple[int, ...] = ()
    temporal: Optional[int] = None
    sequence: Optional[int] = None


# Canonical axis descriptors per supported ndim.
_PACKED_AXES = LatentAxes(layout=LatentLayout.PACKED, batch=0, channel=-1, sequence=1)
_CONV_AXES = LatentAxes(layout=LatentLayout.CONV, batch=0, channel=1, spatial=(2, 3))
_VIDEO_AXES = LatentAxes(layout=LatentLayout.VIDEO, batch=0, channel=1, temporal=2, spatial=(3, 4))

_NDIM_TO_AXES = {3: _PACKED_AXES, 4: _CONV_AXES, 5: _VIDEO_AXES}


def infer_latent_axes(ndim: int) -> LatentAxes:
    """Infer :class:`LatentAxes` from a (batched) latent tensor's ndim.

    Args:
        ndim: Number of dimensions of the batched latent tensor.

    Returns:
        The canonical :class:`LatentAxes` for ``ndim``.

    Raises:
        ValueError: If ``ndim`` is not one of the supported ranks (3 / 4 / 5).
    """
    axes = _NDIM_TO_AXES.get(ndim)
    if axes is None:
        raise ValueError(
            f"Cannot infer LatentAxes for latents with ndim={ndim}; supported "
            f"ranks are {sorted(_NDIM_TO_AXES)} (3=packed (B,Seq,C), "
            f"4=conv (B,C,H,W), 5=video (B,C,T,H,W)). Override `LATENT_AXES` on "
            f"the adapter for non-standard layouts."
        )
    return axes
