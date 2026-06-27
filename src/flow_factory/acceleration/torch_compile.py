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

# src/flow_factory/acceleration/torch_compile.py
"""Lossless ``torch.compile`` accelerator for the shared transformer(s)."""

from typing import TYPE_CHECKING, Any, Dict

from .abc import BaseAccelerator
from ..utils.logger_utils import setup_logger

if TYPE_CHECKING:
    from ..models.abc import BaseAdapter

logger = setup_logger(__name__)


class CompileAccelerator(BaseAccelerator):
    """Apply ``torch.compile`` to every transformer the adapter exposes.

    Lossless and stage-``both``: the compiled module backs both rollout
    ``inference()`` and the training ``forward()`` (they share
    ``adapter.transformer``), so numerical behavior stays consistent across the
    two — safe even for coupled algorithms.

    Parameters (from ``acceleration.shared_params``):
        mode: ``"regional"`` (default) compiles only the repeated transformer
            blocks via diffusers' ``compile_repeated_blocks`` — fast warmup and
            robust to the variable image/sequence lengths set per resolution.
            ``"full"`` compiles the whole module in place.
        compile_kwargs: Extra kwargs forwarded to the underlying compile call
            (e.g. ``{"mode": "max-autotune", "dynamic": true}``).
    """

    safety = "lossless"
    stage = "both"

    def setup(self, adapter: "BaseAdapter") -> None:
        mode = self.params.get("mode", "regional")
        compile_kwargs: Dict[str, Any] = self.params.get("compile_kwargs", {})

        if mode not in ("regional", "full"):
            raise ValueError(
                f"CompileAccelerator: unknown mode={mode!r}; expected 'regional' or 'full'."
            )

        transformer_names = adapter.transformer_names
        if not transformer_names:
            raise ValueError(
                "CompileAccelerator: adapter exposes no transformer components to compile."
            )

        for name in transformer_names:
            module = adapter.get_component(name)
            if mode == "regional":
                if not hasattr(module, "compile_repeated_blocks"):
                    raise ValueError(
                        f"CompileAccelerator: component '{name}' "
                        f"({type(adapter._unwrap(module)).__name__}) has no "
                        "`compile_repeated_blocks`; use `mode: full` for whole-module compilation."
                    )
                module.compile_repeated_blocks(**compile_kwargs)
            else:
                # nn.Module.compile compiles the module's forward in place, so the
                # routed bundle call hits the compiled path on subsequent forwards.
                module.compile(**compile_kwargs)
            logger.info("CompileAccelerator: compiled '%s' (mode=%s).", name, mode)
