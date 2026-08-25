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

import pytest
import torch

from flow_factory.models.registry import get_model_adapter_class
from flow_factory.models.sensenova.modeling.neo_unify.configuration_neo_chat import NEOChatConfig
from flow_factory.models.sensenova.modeling.neo_unify.modeling_neo_chat import (
    NEOChatModel,
    clear_flash_kv_cache,
    create_block_causal_mask,
    prepare_flash_kv_cache,
)
from flow_factory.models.sensenova.pipeline import SenseNovaDenoiser
from flow_factory.models.sensenova.sensenova import SenseNovaAdapter


def _tiny_config(use_pixel_head: bool) -> NEOChatConfig:
    """Build a CPU-sized NEO config while preserving the production tensor shapes."""
    vision_config = {
        "architectures": ["NEOVisionModel"],
        "patch_size": 16,
        "hidden_size": 16,
        "llm_hidden_size": 32,
        "downsample_ratio": 0.5,
        "max_position_embeddings_vision": 128,
        "num_channels": 3,
        "rope_theta_vision": 10000.0,
    }
    llm_config = {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "rope_theta_hw": 10000.0,
        "max_position_embeddings_hw": 128,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rms_norm_eps": 1e-6,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "tie_word_embeddings": False,
    }
    return NEOChatConfig(
        vision_config=vision_config,
        llm_config=llm_config,
        downsample_ratio=0.5,
        template="neo1_0",
        patch_size=16,
        fm_head_layers=2,
        fm_head_dim=16,
        fm_head_mlp_ratio=1,
        use_pixel_head=use_pixel_head,
        t_eps=0.05,
        add_noise_scale_embedding=True,
        noise_scale=1.0,
        noise_scale_mode="resolution",
        noise_scale_base_image_seq_len=64,
        noise_scale_max_value=16.0,
        concat_time_token_num=0,
        time_schedule="standard",
        time_shift_type="exponential",
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=64,
        max_image_seq_len=4096,
    )


def _prefix_cache(model: NEOChatModel):
    """Build a short text prefix cache without requiring a tokenizer package."""
    indexes = torch.stack(
        [torch.arange(3), torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
    )
    attention_mask = {"full_attention": create_block_causal_mask(indexes[0])}
    with torch.inference_mode():
        cache, _ = model._t2i_prefix_forward(torch.tensor([[2, 3, 4]]), indexes, attention_mask)
    prepare_flash_kv_cache(cache, current_len=1, batch_size=1)
    return cache, model._build_t2i_image_indexes(1, 1, 3, device="cpu")


def test_sensenova_registry_entry():
    """The public key resolves to the new adapter."""
    assert get_model_adapter_class("sensenova") is SenseNovaAdapter


@pytest.mark.parametrize("use_pixel_head", [False, True])
def test_sensenova_denoiser_supports_u1_heads(use_pixel_head: bool):
    """U1.0 and U1.5 head variants produce a valid native patch velocity."""
    model = NEOChatModel(_tiny_config(use_pixel_head)).eval()
    cache, indexes_image = _prefix_cache(model)
    try:
        with torch.inference_mode():
            velocity = SenseNovaDenoiser(model)(
                latents=torch.randn(1, 3, 32, 32),
                timestep=torch.tensor(0.2),
                past_key_values=cache,
                indexes_image=indexes_image,
                attention_mask={"full_attention": None},
                image_size=(32, 32),
                noise_scale=1.0,
            )
        assert velocity.shape == (1, 1, 3072)
        assert torch.isfinite(velocity).all()
    finally:
        clear_flash_kv_cache(cache)
