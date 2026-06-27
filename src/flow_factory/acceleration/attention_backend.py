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
"""Attention-backend accelerator — the single code path that selects the
diffusers attention backend for every transformer.

This replaces the old ``BaseAdapter._set_attention_backend`` call: the backend is
applied here (after ``accelerator.prepare`` / ``post_init`` and before compile)
instead of in the adapter constructor, so all transformer-level acceleration flows
through the same plugin mechanism.

The backend is read from ``model.attn_backend`` by default (so existing configs
keep working) and may be overridden by an explicit ``backend`` param. Whatever
string is given is forwarded to diffusers' ``set_attention_backend`` verbatim —
including approximate backends like ``sage`` — matching the previous behavior.

Marked ``stage='both'`` / ``safety='lossless'``: a backend is applied to the
transformer shared by rollout ``inference()`` and training ``forward()``, so the
two stay consistent (the property the validator's lossless category guarantees)
even for approximate kernels and coupled algorithms.
"""

from typing import TYPE_CHECKING

from .abc import BaseAccelerator
from ..utils.logger_utils import setup_logger

if TYPE_CHECKING:
    from ..models.abc import BaseAdapter

logger = setup_logger(__name__)


class AttentionBackendAccelerator(BaseAccelerator):
    """Set the diffusers attention backend on every transformer component.

    Parameters (from ``acceleration.shared_params``, all optional):
        backend: Backend name forwarded to ``transformer.set_attention_backend``
            (e.g. ``native`` / ``flash`` / ``_flash_3`` / ``_flash_3_hub`` /
            ``sage`` / ``xformers``). Defaults to ``model.attn_backend``.

    See https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends
    for the full list of supported backends.
    """

    safety = "lossless"
    stage = "both"

    def setup(self, adapter: "BaseAdapter") -> None:
        backend = self.params.get("backend") or adapter.model_args.attn_backend
        if backend is None:
            # Nothing requested (no `backend` param and `model.attn_backend` unset).
            return

        applied = False
        for name in adapter.transformer_names:
            transformer = adapter.get_component(name)
            if hasattr(transformer, "set_attention_backend"):
                transformer.set_attention_backend(backend)
                applied = True
                if adapter.accelerator.is_main_process:
                    logger.info(
                        "AttentionBackendAccelerator: set backend '%s' for '%s'.", backend, name
                    )
        if not applied:
            logger.warning(
                "AttentionBackendAccelerator: backend '%s' requested but no transformer component "
                "supports `set_attention_backend`; leaving the diffusers default.",
                backend,
            )
