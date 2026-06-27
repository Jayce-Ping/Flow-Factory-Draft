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

# src/flow_factory/acceleration/attention_backend.py
"""Lossless attention-backend accelerator (unified config knob).

Selects an *exact* attention backend (FlashAttention 2/3, xformers, native SDPA)
for every transformer through the same ``acceleration`` block that drives compile
and caching, complementing the lower-level ``model_args.attn_backend`` field.

Approximate backends (e.g. ``sage``, which quantizes attention to int8) are
**lossy** and intentionally rejected here — they belong to a rollout-only lossy
accelerator so the paradigm validator can gate them.
"""

from typing import TYPE_CHECKING

from .abc import BaseAccelerator
from ..utils.logger_utils import setup_logger

if TYPE_CHECKING:
    from ..models.abc import BaseAdapter

logger = setup_logger(__name__)

# Exact, numerically-faithful diffusers attention backends. Kept conservative on
# purpose; lossy/approximate backends are excluded so this stays a lossless knob.
_EXACT_BACKENDS = frozenset(
    {"native", "flash", "flash_hub", "_flash_3", "_flash_3_hub", "xformers"}
)


class AttentionBackendAccelerator(BaseAccelerator):
    """Set an exact attention backend on every transformer component.

    Lossless and stage-``both``. Reuses diffusers' ``set_attention_backend`` (the
    same mechanism behind ``model_args.attn_backend``) but routes the choice
    through the unified ``acceleration`` config.

    Parameters (from ``acceleration.shared_params``):
        backend: One of ``native`` / ``flash`` / ``flash_hub`` / ``_flash_3`` /
            ``_flash_3_hub`` / ``xformers``.
    """

    safety = "lossless"
    stage = "both"

    def setup(self, adapter: "BaseAdapter") -> None:
        backend = self.params.get("backend")
        if backend is None:
            raise ValueError(
                "AttentionBackendAccelerator requires a `backend` parameter "
                f"(one of {sorted(_EXACT_BACKENDS)})."
            )
        if backend not in _EXACT_BACKENDS:
            raise ValueError(
                f"AttentionBackendAccelerator: backend={backend!r} is not an exact (lossless) "
                f"backend. Allowed: {sorted(_EXACT_BACKENDS)}. Approximate backends such as "
                "'sage' are lossy and must be configured as a rollout-only accelerator."
            )

        applied = False
        for name in adapter.transformer_names:
            transformer = adapter.get_component(name)
            if hasattr(transformer, "set_attention_backend"):
                transformer.set_attention_backend(backend)
                applied = True
                logger.info(
                    "AttentionBackendAccelerator: set backend '%s' for '%s'.", backend, name
                )
        if not applied:
            raise ValueError(
                "AttentionBackendAccelerator: no transformer component supports "
                "`set_attention_backend`; check the diffusers version or adapter."
            )
