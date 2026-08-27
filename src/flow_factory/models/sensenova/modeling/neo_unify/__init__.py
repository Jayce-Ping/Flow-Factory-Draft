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

from __future__ import annotations

from transformers import AutoConfig, AutoModel

from .configuration_neo_chat import NEOChatConfig, NEOLLMConfig, NEOMoELLMConfig
from .configuration_neo_vit import NEOVisionConfig
from .modeling_neo_chat import NEOChatModel
from .modeling_neo_vit import NEOVisionModel
from .modeling_qwen3 import _HAS_FLASH_ATTN as has_flash_attn
from .modeling_qwen3 import (
    Qwen3ForCausalLM,
    effective_attn_backend,
    get_attn_backend,
    set_attn_backend,
)
from .modeling_qwen3_moe import Qwen3MoeForCausalLM

__all__ = [
    "NEOChatConfig",
    "NEOLLMConfig",
    "NEOMoELLMConfig",
    "NEOVisionConfig",
    "NEOChatModel",
    "NEOVisionModel",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "register",
    "set_attn_backend",
    "get_attn_backend",
    "effective_attn_backend",
    "has_flash_attn",
]


_REGISTERED = False


def register() -> None:
    """Register NEO-Unify types with ``transformers.Auto*``.

    Importing ``flow_factory.models.sensenova.modeling.neo_unify`` calls this
    registration hook, after which SenseNova-U1 1.0/1.5 checkpoints can be loaded
    through ``AutoConfig.from_pretrained`` / ``AutoModel.from_pretrained``.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    AutoConfig.register("neo_vision", NEOVisionConfig, exist_ok=True)
    AutoConfig.register("neo_chat", NEOChatConfig, exist_ok=True)

    AutoModel.register(NEOVisionConfig, NEOVisionModel, exist_ok=True)
    AutoModel.register(NEOChatConfig, NEOChatModel, exist_ok=True)

    _REGISTERED = True
