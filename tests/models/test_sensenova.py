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

from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from flow_factory.hparams.args import Arguments
from flow_factory.models.registry import get_model_adapter_class
from flow_factory.models.sensenova import sensenova as sensenova_module
from flow_factory.models.sensenova.modeling.neo_unify.configuration_neo_chat import NEOChatConfig
from flow_factory.models.sensenova.modeling.neo_unify.modeling_neo_chat import (
    NEOChatModel,
    clear_flash_kv_cache,
    create_block_causal_mask,
    prepare_flash_kv_cache,
)
from flow_factory.models.sensenova.pipeline import SenseNovaDenoiser
from flow_factory.models.sensenova.sensenova import SenseNovaAdapter, SenseNovaI2ISample
from flow_factory.scheduler import FlowMatchEulerDiscreteSDEScheduler


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


def test_sensenova_example_uses_adapter_generation_parameters():
    """The example uses the adapter's canonical guidance and I2I parameters."""
    config_path = Path(__file__).parents[2] / "examples/grpo/lora/sensenova/default.yaml"
    config = Arguments.load_from_yaml(str(config_path))
    accepted = set(signature(SenseNovaAdapter.inference).parameters)
    model_specific = {"img_cfg_scale", "cfg_norm", "cfg_interval", "timestep_shift"}

    assert config.training_args.guidance_scale == 1.0
    assert config.eval_args.guidance_scale == 4.0
    assert {"guidance_scale", *model_specific} <= accepted
    assert model_specific <= set(config.training_args.extra_kwargs)
    assert model_specific <= set(config.eval_args.extra_kwargs)
    assert "cfg_scale" not in config.training_args.extra_kwargs
    assert "cfg_scale" not in config.eval_args.extra_kwargs


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


class _FakeTokenizer:
    """Minimal tokenizer preserving the visual-token counts used by NEO-Unify."""

    def convert_tokens_to_ids(self, token):
        return {"<IMG_CONTEXT>": 5, "<img>": 6, "</img>": 7}.get(token, 2)

    def __call__(self, text, return_tensors="pt"):
        ids = [2]
        ids.extend([6] * text.count("<img>"))
        ids.extend([5] * text.count("<IMG_CONTEXT>"))
        ids.extend([7] * text.count("</img>"))
        ids.append(2)
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _TinySenseNovaAdapter(SenseNovaAdapter):
    """Adapter shell for CPU tests without constructing the full training runtime."""

    @property
    def tokenizer(self):
        return self._test_tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._test_tokenizer = value

    @property
    def transformer(self):
        return self._test_transformer

    @transformer.setter
    def transformer(self, value):
        self._test_transformer = value

    @property
    def device(self):
        return torch.device("cpu")

    def _unwrap(self, model):
        return model


def test_sensenova_multi_reference_prefill(monkeypatch):
    """Official image-prefill supports ordered multi-reference images and CFG branches."""
    model = NEOChatModel(_tiny_config(use_pixel_head=True)).eval()
    adapter = _TinySenseNovaAdapter.__new__(_TinySenseNovaAdapter)
    adapter.transformer = SenseNovaDenoiser(model)
    adapter.tokenizer = _FakeTokenizer()

    def fake_load_image_native(image, **kwargs):
        del image, kwargs
        return torch.zeros(4, 3 * 16 * 16), torch.tensor([[2, 2]], dtype=torch.long)

    monkeypatch.setattr(sensenova_module, "load_image_native", fake_load_image_native)
    references = [Image.new("RGB", (32, 32), color=(index * 40, 0, 0)) for index in range(2)]
    context = adapter._build_context(
        "Combine the references into one image.",
        (32, 32),
        guidance_scale=3.0,
        img_cfg_scale=2.0,
        condition_images=references,
    )
    try:
        assert context["is_i2i"] is True
        assert context["img_past_key_values"] is not None
        assert context["uncond_past_key_values"] is not None
        velocity = adapter.transformer(
            latents=torch.randn(1, 3, 32, 32),
            timestep=torch.tensor(0.2),
            past_key_values=context["past_key_values"],
            indexes_image=context["indexes_image"],
            attention_mask=context["attention_mask"],
            image_size=(32, 32),
            noise_scale=1.0,
        )
        image_velocity = adapter.transformer(
            latents=torch.randn(1, 3, 32, 32),
            timestep=torch.tensor(0.2),
            past_key_values=context["img_past_key_values"],
            indexes_image=context["img_indexes_image"],
            attention_mask=context["img_attention_mask"],
            image_size=(32, 32),
            noise_scale=1.0,
        )
        assert velocity.shape == image_velocity.shape == (1, 1, 3072)
        assert torch.isfinite(velocity).all()
        assert torch.isfinite(image_velocity).all()
        uncond_velocity = adapter.transformer(
            latents=torch.randn(1, 3, 32, 32),
            timestep=torch.tensor(0.2),
            past_key_values=context["uncond_past_key_values"],
            indexes_image=context["uncond_indexes_image"],
            attention_mask=context["uncond_attention_mask"],
            image_size=(32, 32),
            noise_scale=1.0,
        )
        assert uncond_velocity.shape == velocity.shape
        assert torch.isfinite(uncond_velocity).all()

        adapter.pipeline = SimpleNamespace(
            scheduler=FlowMatchEulerDiscreteSDEScheduler(dynamics_type="ODE")
        )
        adapter.scheduler.set_timesteps(num_inference_steps=2, device="cpu")
        output = adapter.forward(
            t=torch.tensor(1000.0),
            t_next=torch.tensor(500.0),
            latents=torch.randn(1, 3, 32, 32),
            prompt="Combine the references into one image.",
            height=32,
            width=32,
            cfg_scale=3.0,
            img_cfg_scale=2.0,
            past_key_values=context["past_key_values"],
            indexes_image=context["indexes_image"],
            attention_mask=context["attention_mask"],
            img_past_key_values=context["img_past_key_values"],
            img_indexes_image=context["img_indexes_image"],
            img_attention_mask=context["img_attention_mask"],
            uncond_past_key_values=context["uncond_past_key_values"],
            uncond_indexes_image=context["uncond_indexes_image"],
            uncond_attention_mask=context["uncond_attention_mask"],
            compute_log_prob=False,
        )
        assert output.next_latents.shape == (1, 3, 32, 32)
    finally:
        adapter._clear_context(context)

    sample = SenseNovaI2ISample(condition_images=references)
    assert sample.condition_images_as_pil is True
    assert len(sample.condition_images) == 2
    encoded = adapter.encode_image([references, [references[0]]])
    assert [len(images) for images in encoded["condition_images"]] == [2, 1]
    stacked = SenseNovaI2ISample.stack(
        [sample, SenseNovaI2ISample(condition_images=[references[0]])]
    )
    assert [len(images) for images in stacked["condition_images"]] == [2, 1]
