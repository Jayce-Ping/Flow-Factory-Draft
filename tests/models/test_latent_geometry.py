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

"""Unit tests for the model-agnostic latent geometry abstraction.

Covers axis-role inference (rank 3/4/5 + fail-fast) and the ``BaseAdapter``
resolution hook (ndim inference + ``LATENT_AXES`` override) via a lightweight
stub adapter that never loads a real pipeline (constructed with ``object.__new__``
to bypass ``__init__``).
"""

from __future__ import annotations

import pytest
import torch

from flow_factory.models.abc import BaseAdapter
from flow_factory.models.latent_geometry import (
    LatentAxes,
    LatentLayout,
    infer_latent_axes,
)

# ============================== infer_latent_axes ==============================


def test_infer_latent_axes_packed():
    axes = infer_latent_axes(3)
    assert axes.layout is LatentLayout.PACKED
    assert axes.batch == 0
    assert axes.channel == -1
    assert axes.sequence == 1
    assert axes.spatial == ()
    assert axes.temporal is None


def test_infer_latent_axes_conv():
    axes = infer_latent_axes(4)
    assert axes.layout is LatentLayout.CONV
    assert axes.channel == 1
    assert axes.spatial == (2, 3)
    assert axes.temporal is None
    assert axes.sequence is None


def test_infer_latent_axes_video():
    axes = infer_latent_axes(5)
    assert axes.layout is LatentLayout.VIDEO
    assert axes.channel == 1
    assert axes.temporal == 2
    assert axes.spatial == (3, 4)
    assert axes.sequence is None


@pytest.mark.parametrize("ndim", [2, 6])
def test_infer_latent_axes_rejects_unsupported(ndim):
    with pytest.raises(ValueError, match="ndim"):
        infer_latent_axes(ndim)


# ============================== BaseAdapter.resolve_latent_axes ==============================


class _StubAdapter(BaseAdapter):
    """Minimal concrete adapter implementing abstractmethods (never ``__init__``'d).

    Only ``resolve_latent_axes`` (inherited from ``BaseAdapter``) is exercised; it
    touches no instance state beyond the ``LATENT_AXES`` class var, so a
    pipeline-free instance built via ``object.__new__`` is sufficient.
    """

    def load_pipeline(self):  # pragma: no cover - never called
        raise NotImplementedError

    def decode_latents(self, latents, **kwargs):  # pragma: no cover - never called
        raise NotImplementedError

    def forward(self, *args, **kwargs):  # pragma: no cover - never called
        raise NotImplementedError

    def inference(self, *args, **kwargs):  # pragma: no cover - never called
        raise NotImplementedError


def _make_stub(cls=_StubAdapter):
    """Build a pipeline-free adapter instance (bypass ``__init__``)."""
    return object.__new__(cls)


def test_resolve_latent_axes_infers_from_ndim():
    adapter = _make_stub()
    assert adapter.resolve_latent_axes(torch.randn(2, 7, 4)).layout is LatentLayout.PACKED
    assert adapter.resolve_latent_axes(torch.randn(2, 4, 8, 8)).layout is LatentLayout.CONV
    assert adapter.resolve_latent_axes(torch.randn(2, 4, 3, 8, 8)).layout is LatentLayout.VIDEO


def test_latent_axes_override_takes_precedence():
    class _OverrideAdapter(_StubAdapter):
        LATENT_AXES = LatentAxes(layout=LatentLayout.CONV, channel=1, spatial=(2, 3))

    adapter = _make_stub(_OverrideAdapter)
    # Even a rank-3 latent resolves to the explicit override (CONV), not inferred PACKED.
    assert adapter.resolve_latent_axes(torch.randn(2, 7, 4)).layout is LatentLayout.CONV
