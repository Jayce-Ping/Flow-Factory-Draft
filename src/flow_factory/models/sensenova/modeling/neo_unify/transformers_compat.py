# Copyright 2026 Jayce-Ping
#
# Vendored from OpenSenseNova/SenseNova-U1 (Apache-2.0). See the accompanying
# third-party notice in ``src/flow_factory/models/sensenova``.
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

# isort: skip_file

"""Small compatibility seams for the supported Transformers 4/5 window."""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any, Callable

import torch
import transformers
from packaging.version import Version

try:
    from transformers.utils.generic import merge_with_config_defaults
    from transformers.utils.output_capturing import capture_outputs

    def model_input_compat(func: Callable[..., Any]) -> Callable[..., Any]:
        return merge_with_config_defaults(capture_outputs(func))

except ImportError:  # Transformers 4.x
    from transformers.utils.generic import check_model_inputs as model_input_compat


@lru_cache(maxsize=None)
def _parameter_names(callable_: Callable[..., Any]) -> frozenset[str]:
    return frozenset(inspect.signature(callable_).parameters)


def causal_mask_kwargs(
    mask_factory: Callable[..., Any],
    *,
    config: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_key_values: Any,
    position_ids: torch.Tensor,
) -> dict[str, Any]:
    """Build arguments accepted by the installed ``create_causal_mask``.

    Transformers 4.57 uses ``input_embeds`` and ``cache_position`` while newer
    5.x releases use ``inputs_embeds`` and derive the cache position internally.
    """
    parameters = _parameter_names(mask_factory)
    embedding_parameter = "inputs_embeds" if "inputs_embeds" in parameters else "input_embeds"
    candidates = {
        "config": config,
        embedding_parameter: inputs_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    return {name: value for name, value in candidates.items() if name in parameters}


def pretrained_dtype_kwargs(dtype: torch.dtype) -> dict[str, torch.dtype]:
    """Use the public dtype keyword supported throughout Transformers 4.57+."""
    return {"dtype": dtype}


def tied_weights_keys(output_key: str, input_key: str) -> list[str] | dict[str, str]:
    """Return the `_tied_weights_keys` shape expected by Transformers 4 or 5."""
    if Version(transformers.__version__).major >= 5:
        return {output_key: input_key}
    return [output_key]
