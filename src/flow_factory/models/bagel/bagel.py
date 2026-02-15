# src/flow_factory/models/bagel/bagel.py
"""
Bagel Model Adapter for Flow-Factory (Refactored)

Integrates ByteDance's Bagel (unified multimodal model) into the
Flow-Factory RL fine-tuning framework.

Architecture (flat pipeline — no Bagel wrapper)::

    BagelAdapter
      └── self.pipeline: BagelPseudoPipeline
            ├── .transformer        Qwen2ForCausalLM
            ├── .vit                SiglipVisionModel
            ├── .vae                AutoEncoder
            ├── .vae2llm / .llm2vae / .time_embedder / ...
            └── prepare_* / forward_cache_update_* / forward_denoise_step

Training-mode Caveats:
    Qwen2Model.forward() dispatches to ``forward_inference()`` based on
    ``self.training``. During RL training we always call
    ``forward_inference`` directly, keeping the model in eval mode.
    Gradients still flow (autograd is orthogonal to train/eval mode).
"""
from __future__ import annotations

import os
import random
from contextlib import contextmanager
from copy import deepcopy
from typing import Union, List, Dict, Any, Optional, Tuple, Literal, ClassVar
from dataclasses import dataclass, field
from collections import defaultdict
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

from accelerate import Accelerator

from ...samples import T2ISample, I2ISample
from ..abc import BaseAdapter
from ...hparams import Arguments
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    SDESchedulerOutput,
)
from ...utils.base import filter_kwargs
from ...utils.trajectory_collector import (
    TrajectoryCollector,
    CallbackCollector,
    TrajectoryIndicesType,
    create_trajectory_collector,
    create_callback_collector,
)
from ...utils.image import (
    ImageSingle,
    ImageBatch,
    MultiImageBatch,
    is_image,
    is_image_batch,
    is_multi_image_batch,
    standardize_image_batch,
    pil_image_to_tensor,
)
from ...utils.logger_utils import setup_logger

from .pipeline import BagelPseudoPipeline

logger = setup_logger(__name__)

CONDITION_IMAGE_SIZE = (1024, 1024)

# ============================================================================
# Sample Dataclasses
# ============================================================================

@dataclass
class BagelSample(T2ISample):
    """Sample for Bagel T2I: trajectory + packed tensor info for KV-cache."""

    _shared_fields: ClassVar[frozenset[str]] = frozenset({"image_shape"})

    packed_text_ids: Optional[torch.LongTensor] = None
    packed_text_indexes: Optional[torch.LongTensor] = None
    packed_vae_position_ids: Optional[torch.LongTensor] = None
    packed_vae_token_indexes: Optional[torch.LongTensor] = None
    packed_seqlens: Optional[torch.IntTensor] = None
    packed_position_ids: Optional[torch.LongTensor] = None
    packed_indexes: Optional[torch.LongTensor] = None
    packed_key_value_indexes: Optional[torch.LongTensor] = None
    key_values_lens: Optional[torch.IntTensor] = None
    cfg_text_generation_input: Optional[Dict[str, torch.Tensor]] = None
    cfg_img_generation_input: Optional[Dict[str, torch.Tensor]] = None
    image_shape: Optional[Tuple[int, int]] = None


@dataclass
class BagelI2ISample(I2ISample):
    """Sample for Bagel I2I: trajectory + packed tensor info for KV-cache."""

    _shared_fields: ClassVar[frozenset[str]] = frozenset({"image_shape"})

    packed_text_ids: Optional[torch.LongTensor] = None
    packed_text_indexes: Optional[torch.LongTensor] = None
    packed_vae_position_ids: Optional[torch.LongTensor] = None
    packed_vae_token_indexes: Optional[torch.LongTensor] = None
    packed_seqlens: Optional[torch.IntTensor] = None
    packed_position_ids: Optional[torch.LongTensor] = None
    packed_indexes: Optional[torch.LongTensor] = None
    packed_key_value_indexes: Optional[torch.LongTensor] = None
    key_values_lens: Optional[torch.IntTensor] = None
    cfg_text_generation_input: Optional[Dict[str, torch.Tensor]] = None
    cfg_img_generation_input: Optional[Dict[str, torch.Tensor]] = None
    image_shape: Optional[Tuple[int, int]] = None


def calculate_dimensions(target_area, ratio):
    """Calculate (width, height) from target pixel area and aspect ratio (h/w)."""
    height = math.sqrt(target_area * ratio)
    width = height / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height


# ============================================================================
# BagelAdapter
# ============================================================================

class BagelAdapter(BaseAdapter):
    """
    Flow-Factory adapter for Bagel multimodal models.

    Uses flat BagelPseudoPipeline — all components accessed via
    ``self.pipeline.<name>`` with no wrapper indirection.
    """

    def __init__(self, config: Arguments, accelerator: Accelerator):
        self._model_path = config.model_args.model_name_or_path
        self._init_tokenizer_and_transforms()
        super().__init__(config, accelerator)
        self.pipeline: BagelPseudoPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

    # ─────────────────── Tokenizer & Transforms ───────────────────

    def _init_tokenizer_and_transforms(self):
        """Initialize tokenizer, special tokens, and image transforms."""
        from .modeling.qwen2 import Qwen2Tokenizer
        from .data_utils import add_special_tokens
        from .data.transforms import ImageTransform

        self._tokenizer = Qwen2Tokenizer.from_pretrained(self._model_path)
        self._tokenizer, self.new_token_ids, _ = add_special_tokens(self._tokenizer)

        # VAE transform: max_size=1024, min_size=512, patch=16
        self.vae_transform = ImageTransform(1024, 512, 16)
        # ViT transform: max_size=980, min_size=224, patch=14
        self.vit_transform = ImageTransform(980, 224, 14)

    # ======================== Pipeline & Scheduler ========================

    def load_pipeline(self) -> BagelPseudoPipeline:
        """Load all Bagel components into a flat pseudo-pipeline."""
        return BagelPseudoPipeline.from_pretrained(
            self._model_path,
            low_cpu_mem_usage=False,
            **self.model_args.extra_kwargs,
        )

    def load_scheduler(self) -> FlowMatchEulerDiscreteSDEScheduler:
        """Create scheduler for Bagel's shifted flow matching schedule."""
        scheduler_kwargs = {
            "num_train_timesteps": 1000,
            "shift": self.model_args.extra_kwargs.get("timestep_shift", 3.0),
        }
        if hasattr(self.config, "scheduler_args") and self.config.scheduler_args:
            scheduler_kwargs.update(self.config.scheduler_args.to_dict())
        return FlowMatchEulerDiscreteSDEScheduler(**scheduler_kwargs)

    # ======================== Module Management ========================

    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Bagel's Qwen2 decoder layers."""
        return [
            # Attention
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            # MLP / MoE
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def text_encoder_names(self) -> List[str]:
        return []

    @property
    def text_encoders(self) -> List[nn.Module]:
        return []

    @property
    def text_encoder(self) -> Optional[nn.Module]:
        return None

    @property
    def preprocessing_modules(self) -> List[str]:
        """Modules needed for preprocessing (ViT + VAE for condition images)."""
        return ["vae", "vit", "connector", "vit_pos_embed"]

    @property
    def inference_modules(self) -> List[str]:
        """All modules needed for inference — clean, no duplicates."""
        return [
            "transformer", "vit", "vae",
            "vae2llm", "llm2vae",
            "time_embedder", "latent_pos_embed",
            "connector", "vit_pos_embed",
        ]

    @property
    def bagel_model(self) -> nn.Module:
        """The underlying transformer (Qwen2ForCausalLM)."""
        return self.get_component("transformer")

    @property
    def bagel_config(self):
        """The BagelConfig from the loaded model."""
        return self.pipeline.config

    # ======================== Mode Management ========================

    # @property
    # def mode(self) -> str:
    #     return self._mode

    # def eval(self):
    #     """Set all components to evaluation mode."""
    #     super().eval()
    #     for comp in self.pipeline.named_components().values():
    #         comp.eval()

    # def rollout(self, *args, **kwargs):
    #     """Set model to rollout mode (eval for all components)."""
    #     self.eval()
    #     if hasattr(self.scheduler, 'rollout'):
    #         self.scheduler.rollout(*args, **kwargs)

    # def train(self, mode: bool = True):
    #     """Set trainable components to training mode."""
    #     super().train(mode)
    #     if mode:
    #         # Only transformer is trainable; other components stay frozen
    #         self.pipeline.transformer.train()

    # ======================== Encoding ========================

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
    ) -> Dict[str, Any]:
        """Return raw prompts (encoding is deferred to KV-cache building)."""
        if isinstance(prompt, str):
            prompt = [prompt]
        return {"prompt": prompt}

    def standardize_images(
        self,
        images: Union[ImageSingle, ImageBatch],
        output_type: Literal['pil', 'pt', 'np'] = 'pil',
    ) -> ImageBatch:
        """Standardize input images to a consistent format."""
        if is_image(images):
            images = [images]
        return standardize_image_batch(images, output_type=output_type)

    def encode_image(
        self,
        images: Union[ImageSingle, ImageBatch, MultiImageBatch],
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        device: Optional[torch.device] = None,
    ) -> Optional[Dict[str, Any]]:
        """Encode condition images to tensors for I2I generation."""
        if images is None:
            return None
        device = device or self.device

        if is_image(images):
            images = [[images]]
        elif is_image_batch(images):
            images = [images]

        condition_images = []
        for batch in images:
            batch_tensors = []
            for img in batch:
                t = pil_image_to_tensor(img).to(device)
                batch_tensors.append(t)
            condition_images.append(batch_tensors)

        return {"condition_images": condition_images}

    def encode_video(
        self,
        videos: Any
    ):
        """No need fot Bagel to encode video."""
        pass

    # ======================== Decoding ========================

    def decode_latents(
        self,
        latents: torch.Tensor,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        """Decode packed latent tokens back into PIL images."""
        vae = self.pipeline.vae
        ds = self.pipeline.latent_downsample
        p = self.pipeline.latent_patch_size
        ch = self.pipeline.latent_channel

        single = latents.dim() == 2
        if single:
            latents = latents.unsqueeze(0)

        images = []
        for lat in latents:
            H, W = image_shape
            h, w = H // ds, W // ds
            lat = lat.reshape(1, h, w, p, p, ch)
            lat = torch.einsum("nhwpqc->nchpwq", lat)
            lat = lat.reshape(1, ch, h * p, w * p)
            decoded = vae.decode(lat.to(vae.dtype if hasattr(vae, 'dtype') else torch.bfloat16))
            decoded = (decoded * 0.5 + 0.5).clamp(0, 1)[0].float()
            images.append(decoded)

        return images[0] if single else images

    # ======================== Context Building ========================

    def _build_gen_context(
        self,
        prompt: str,
        condition_images: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        think: bool = False,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Build KV-cache contexts for generation (gen / cfg_text / cfg_img).

        All calls go through ``self.pipeline.prepare_*`` and
        ``self.pipeline.forward_cache_update_*`` — no Bagel wrapper.
        """
        from .modeling.bagel.qwen2_navit import NaiveCache

        num_layers = self.pipeline.config.llm_config.num_hidden_layers

        def _init_ctx():
            return {
                "kv_lens": [0],
                "ropes": [0],
                "past_key_values": NaiveCache(num_layers),
            }

        gen_context = _init_ctx()
        cfg_text_context = _init_ctx()
        cfg_img_context = _init_ctx()

        # Optional thinking prompt
        if think:
            system_prompt = (
                "You should first think about the planning process in the mind "
                "and then generate the image.\nThe planning process is enclosed "
                "within <think> </think> tags."
            )
            gen_context = self._update_context_text(system_prompt, gen_context)
            cfg_img_context = self._update_context_text(system_prompt, cfg_img_context)

        # I2I: condition images first
        if condition_images is not None:
            for img_tensor in condition_images:
                gen_context = self._update_context_image(img_tensor, gen_context)
                cfg_text_context = deepcopy(gen_context)

        # Text always last
        cfg_text_context = deepcopy(gen_context)
        gen_context = self._update_context_text(prompt, gen_context)
        cfg_img_context = self._update_context_text(prompt, cfg_img_context)

        return gen_context, cfg_text_context, cfg_img_context

    @torch.no_grad()
    def _update_context_text(self, text: str, gen_context: Dict) -> Dict:
        """Add text tokens to KV-cache context via pipeline methods."""
        device = self.device
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        gen_input, kv_lens, ropes = self.pipeline.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self._tokenizer,
            new_token_ids=self.new_token_ids,
        )
        gen_input = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in gen_input.items()
        }
        past_key_values = self.pipeline.forward_cache_update_text(
            gen_context["past_key_values"], **gen_input
        )
        return {"kv_lens": kv_lens, "ropes": ropes, "past_key_values": past_key_values}

    @torch.no_grad()
    def _update_context_image(
        self, image, gen_context: Dict,
        vae: bool = True, vit: bool = True,
    ) -> Dict:
        """Add image tokens (ViT + VAE) to KV-cache context via pipeline methods."""
        device = self.device
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        past_key_values = gen_context["past_key_values"]
        image = self.standardize_images(image, output_type='pil')[0]

        if vae:
            gen_input, kv_lens, ropes = self.pipeline.prepare_vae_images(
                curr_kvlens=kv_lens, curr_rope=ropes,
                images=[image], transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            gen_input = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in gen_input.items()
            }
            past_key_values = self.pipeline.forward_cache_update_vae(
                past_key_values, **gen_input
            )

        if vit:
            gen_input, kv_lens, ropes = self.pipeline.prepare_vit_images(
                curr_kvlens=kv_lens, curr_rope=ropes,
                images=[image], transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            gen_input = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in gen_input.items()
            }
            past_key_values = self.pipeline.forward_cache_update_vit(
                past_key_values, **gen_input
            )

        return {"kv_lens": kv_lens, "ropes": ropes, "past_key_values": past_key_values}

    # ======================== Flow Forward (grad-safe) ========================

    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids=None,
        cfg_text_packed_query_indexes=None,
        cfg_text_key_values_lens=None,
        cfg_text_past_key_values=None,
        cfg_text_packed_key_value_indexes=None,
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids=None,
        cfg_img_packed_query_indexes=None,
        cfg_img_key_values_lens=None,
        cfg_img_past_key_values=None,
        cfg_img_packed_key_value_indexes=None,
        cfg_type: str = "parallel",
    ):
        """
        Flow velocity prediction with CFG — delegates to pipeline.

        This is the grad-safe version (no @torch.no_grad) used during
        RL training. Gradients flow through the LLM forward pass.
        """
        return self.pipeline.forward_denoise_step(
            x_t=x_t,
            timestep=timestep,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_vae_position_ids=packed_vae_position_ids,
            packed_text_ids=packed_text_ids,
            packed_text_indexes=packed_text_indexes,
            packed_indexes=packed_indexes,
            packed_position_ids=packed_position_ids,
            packed_seqlens=packed_seqlens,
            key_values_lens=key_values_lens,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            cfg_text_scale=cfg_text_scale,
            cfg_text_packed_position_ids=cfg_text_packed_position_ids,
            cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
            cfg_text_key_values_lens=cfg_text_key_values_lens,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
            cfg_img_scale=cfg_img_scale,
            cfg_img_packed_position_ids=cfg_img_packed_position_ids,
            cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
            cfg_img_key_values_lens=cfg_img_key_values_lens,
            cfg_img_past_key_values=cfg_img_past_key_values,
            cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
            cfg_type=cfg_type,
        )

    # ======================== Denoising Loop ========================

    def _denoise_loop(
        self,
        gen_input: Dict[str, torch.Tensor],
        past_key_values,
        cfg_text_past_kv,
        cfg_img_past_kv,
        cfg_text_generation_input: Dict[str, torch.Tensor],
        cfg_img_generation_input: Dict[str, torch.Tensor],
        image_shape: Tuple[int, int],
        num_inference_steps: int,
        timestep_shift: float,
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_interval: Tuple[float, float],
        cfg_renorm_min: float,
        cfg_renorm_type: str,
        noise_level: float,
        compute_log_prob: bool,
        trajectory_indices: TrajectoryIndicesType,
        extra_call_back_kwargs: List[str],
        device: torch.device,
    ) -> Dict[str, Any]:
        """Core denoising loop using Bagel's flow matching."""
        # 1. Build shifted sigma schedule
        linear_sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        shifted_sigmas = (
            timestep_shift * linear_sigmas
            / (1 + (timestep_shift - 1) * linear_sigmas)
        )
        self.scheduler.set_timesteps(sigmas=shifted_sigmas.tolist(), device=device)
        timesteps = self.scheduler.timesteps

        # 2. Initial noise
        x_t = gen_input["packed_init_noises"].to(device)
        gen_input = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in gen_input.items()
        }

        # 3. Collectors
        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latent_collector.collect(x_t, step_idx=0)
        log_prob_collector = (
            create_trajectory_collector(trajectory_indices, num_inference_steps)
            if compute_log_prob else None
        )
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        # 4. Denoising loop
        for i, t in enumerate(timesteps):
            t_next = (
                timesteps[i + 1] if i + 1 < len(timesteps)
                else torch.tensor(0.0, device=device)
            )
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            current_compute_log_prob = compute_log_prob and current_noise_level > 0
            return_kwargs = list(set(
                ['next_latents', 'log_prob', 'noise_pred'] + extra_call_back_kwargs
            ))

            output = self.forward(
                t=t.unsqueeze(0),
                latents=x_t,
                past_key_values=past_key_values,
                cfg_text_past_kv=cfg_text_past_kv,
                cfg_img_past_kv=cfg_img_past_kv,
                packed_text_ids=gen_input["packed_text_ids"],
                packed_text_indexes=gen_input["packed_text_indexes"],
                packed_vae_position_ids=gen_input["packed_vae_position_ids"],
                packed_vae_token_indexes=gen_input["packed_vae_token_indexes"],
                packed_seqlens=gen_input["packed_seqlens"],
                packed_position_ids=gen_input["packed_position_ids"],
                packed_indexes=gen_input["packed_indexes"],
                packed_key_value_indexes=gen_input["packed_key_value_indexes"],
                key_values_lens=gen_input["key_values_lens"],
                cfg_text_generation_input=cfg_text_generation_input,
                cfg_img_generation_input=cfg_img_generation_input,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                t_next=t_next.unsqueeze(0),
                noise_level=current_noise_level,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=return_kwargs,
            )

            x_t = output.next_latents
            latent_collector.collect(x_t, step_idx=i + 1)
            if current_compute_log_prob and log_prob_collector is not None:
                log_prob_collector.collect(output.log_prob, step_idx=i)
            callback_collector.collect_step(
                step_idx=i, output=output,
                keys=extra_call_back_kwargs,
                capturable={"noise_level": current_noise_level},
            )

        # 5. Unpack final latent
        # H, W = image_shape
        # ds = self.pipeline.latent_downsample
        # p = self.pipeline.latent_patch_size
        # ch = self.pipeline.latent_channel
        # h, w = H // ds, W // ds
        # unpacked = x_t.reshape(1, h, w, p, p, ch)
        # unpacked = torch.einsum("nhwpqc->nchpwq", unpacked).reshape(1, ch, h * p, w * p)

        return {
            "final_packed_latent": x_t,
            "all_latents": latent_collector.get_result(),
            "all_log_probs": (
                log_prob_collector.get_result() if log_prob_collector else None
            ),
            "timesteps": timesteps,
            "latent_index_map": latent_collector.get_index_map(),
            "log_prob_index_map": (
                log_prob_collector.get_index_map() if log_prob_collector else None
            ),
            "callback_results": callback_collector.get_result(),
            "callback_index_map": callback_collector.get_index_map(),
        }

    # ======================== Forward (Training & Inference) ========================

    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        packed_text_ids: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
        packed_vae_position_ids: Optional[torch.Tensor] = None,
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_seqlens: Optional[torch.Tensor] = None,
        packed_position_ids: Optional[torch.Tensor] = None,
        packed_indexes: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        past_key_values=None,
        cfg_text_past_kv=None,
        cfg_img_past_kv=None,
        cfg_text_generation_input: Optional[Dict[str, torch.Tensor]] = None,
        cfg_img_generation_input: Optional[Dict[str, torch.Tensor]] = None,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: Tuple[float, float] = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        noise_level: Optional[float] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = [
            "noise_pred", "next_latents", "next_latents_mean",
            "std_dev_t", "dt", "log_prob",
        ],
        prompt: Optional[Union[str, List[str]]] = None,
        condition_images: Optional[List[torch.Tensor]] = None,
        image_shape: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> SDESchedulerOutput:
        """Single denoising step: flow prediction → scheduler step."""
        device = latents.device

        # 1. Rebuild KV-cache contexts if not provided (training path)
        if past_key_values is None:
            if prompt is None:
                raise ValueError(
                    "BagelAdapter.forward() requires either `past_key_values` "
                    "(inference) or `prompt` (training) to build KV caches."
                )
            if isinstance(prompt, str):
                prompt = [prompt]

            _image_shape = image_shape or (
                kwargs.get("height", 1024), kwargs.get("width", 1024)
            )

            with torch.no_grad():
                gen_ctx, cfg_text_ctx, cfg_img_ctx = self._build_gen_context(
                    prompt=prompt[0],
                    condition_images=condition_images,
                )
                gen_input = self.pipeline.prepare_vae_latent(
                    curr_kvlens=gen_ctx["kv_lens"],
                    curr_rope=gen_ctx["ropes"],
                    image_sizes=[_image_shape],
                    new_token_ids=self.new_token_ids,
                    device=device,
                )

            past_key_values = gen_ctx["past_key_values"]
            cfg_text_past_kv = cfg_text_ctx["past_key_values"]
            cfg_img_past_kv = cfg_img_ctx["past_key_values"]

            packed_text_ids = gen_input["packed_text_ids"]
            packed_text_indexes = gen_input["packed_text_indexes"]
            packed_vae_position_ids = gen_input["packed_vae_position_ids"]
            packed_vae_token_indexes = gen_input["packed_vae_token_indexes"]
            packed_seqlens = gen_input["packed_seqlens"]
            packed_position_ids = gen_input["packed_position_ids"]
            packed_indexes = gen_input["packed_indexes"]
            packed_key_value_indexes = gen_input["packed_key_value_indexes"]
            key_values_lens = gen_input["key_values_lens"]

            with torch.no_grad():
                cfg_text_generation_input = self.pipeline.prepare_vae_latent_cfg(
                    curr_kvlens=cfg_text_ctx["kv_lens"],
                    curr_rope=cfg_text_ctx["ropes"],
                    image_sizes=[_image_shape],
                    device=device,
                )
                cfg_img_generation_input = self.pipeline.prepare_vae_latent_cfg(
                    curr_kvlens=cfg_img_ctx["kv_lens"],
                    curr_rope=cfg_img_ctx["ropes"],
                    image_sizes=[_image_shape],
                    device=device,
                )

        # 2. Convert [0, 1000] → [0, 1] sigma for Bagel
        sigma = t.float() / 1000.0
        timestep_for_bagel = sigma.expand(latents.shape[0])

        # 3. CFG gating
        sigma_val = sigma.flatten()[0].item()
        if sigma_val > cfg_interval[0] and sigma_val <= cfg_interval[1]:
            cfg_text_s = cfg_text_scale
            cfg_img_s = cfg_img_scale
        else:
            cfg_text_s = 1.0
            cfg_img_s = 1.0

        def _cfg(d: Optional[Dict], key: str) -> Optional[torch.Tensor]:
            if d is None:
                return None
            v = d.get(key)
            return v.to(device) if isinstance(v, torch.Tensor) else None

        # 4. Flow velocity prediction (gradient-safe)
        v_t = self._forward_flow(
            x_t=latents,
            timestep=timestep_for_bagel,
            packed_vae_token_indexes=packed_vae_token_indexes.to(device),
            packed_vae_position_ids=packed_vae_position_ids.to(device),
            packed_text_ids=packed_text_ids.to(device),
            packed_text_indexes=packed_text_indexes.to(device),
            packed_position_ids=packed_position_ids.to(device),
            packed_indexes=packed_indexes.to(device),
            packed_seqlens=packed_seqlens.to(device),
            key_values_lens=key_values_lens.to(device),
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes.to(device),
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            cfg_text_scale=cfg_text_s,
            cfg_text_packed_position_ids=_cfg(cfg_text_generation_input, "cfg_packed_position_ids"),
            cfg_text_packed_query_indexes=_cfg(cfg_text_generation_input, "cfg_packed_query_indexes"),
            cfg_text_key_values_lens=_cfg(cfg_text_generation_input, "cfg_key_values_lens"),
            cfg_text_past_key_values=cfg_text_past_kv,
            cfg_text_packed_key_value_indexes=_cfg(cfg_text_generation_input, "cfg_packed_key_value_indexes"),
            cfg_img_scale=cfg_img_s,
            cfg_img_packed_position_ids=_cfg(cfg_img_generation_input, "cfg_packed_position_ids"),
            cfg_img_packed_query_indexes=_cfg(cfg_img_generation_input, "cfg_packed_query_indexes"),
            cfg_img_key_values_lens=_cfg(cfg_img_generation_input, "cfg_key_values_lens"),
            cfg_img_past_key_values=cfg_img_past_kv,
            cfg_img_packed_key_value_indexes=_cfg(cfg_img_generation_input, "cfg_packed_key_value_indexes"),
            cfg_type="parallel",
        )

        # 5. Scheduler step
        scheduler_output = self.scheduler.step(
            noise_pred=v_t,
            timestep=t,
            latents=latents,
            timestep_next=t_next,
            next_latents=next_latents,
            noise_level=noise_level,
            return_dict=True,
            compute_log_prob=compute_log_prob,
            return_kwargs=return_kwargs,
        )
        return scheduler_output

    # ======================== Inference ========================

    @torch.no_grad()
    def inference(
        self,
        num_inference_steps: int = 50,
        height: int = 1024,
        width: int = 1024,
        prompt: Union[str, List[str]] = None,
        images: Optional[Union[ImageSingle, ImageBatch, MultiImageBatch]] = None,
        condition_images: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: Tuple[float, float] = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        timestep_shift: float = 3.0,
        noise_level: float = 0.7,
        compute_log_prob: bool = True,
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
        generator: Optional[torch.Generator] = None,
        think: bool = False,
    ) -> List[BagelSample]:
        """Full generation: build context → denoise → decode → return samples."""
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        device = self.device
        image_shape = (height, width)
        is_i2i = (condition_images is not None or images is not None)
        if is_i2i:
            if condition_images is None:
                encoded = self.encode_image(images, condition_image_size, device)
                condition_images = encoded["condition_images"] if encoded else None
            else:
                condition_images = [
                    [t.to(device) for t in imgs] for imgs in condition_images
                ]

        samples = []
        for b in range(batch_size):
            cur_prompt = prompt[b]
            cur_cond_images = (
                condition_images[b] if condition_images is not None else None
            )

            # 1. Build KV-cache contexts
            gen_ctx, cfg_text_ctx, cfg_img_ctx = self._build_gen_context(
                prompt=cur_prompt,
                condition_images=cur_cond_images,
                think=think,
            )

            # 2. Prepare latent generation inputs
            gen_input = self.pipeline.prepare_vae_latent(
                curr_kvlens=gen_ctx["kv_lens"],
                curr_rope=gen_ctx["ropes"],
                image_sizes=[image_shape],
                new_token_ids=self.new_token_ids,
                device=device,
            )
            cfg_text_gen_input = self.pipeline.prepare_vae_latent_cfg(
                curr_kvlens=cfg_text_ctx["kv_lens"],
                curr_rope=cfg_text_ctx["ropes"],
                image_sizes=[image_shape],
                device=device,
            )
            cfg_img_gen_input = self.pipeline.prepare_vae_latent_cfg(
                curr_kvlens=cfg_img_ctx["kv_lens"],
                curr_rope=cfg_img_ctx["ropes"],
                image_sizes=[image_shape],
                device=device,
            )

            # 3. Denoise
            result = self._denoise_loop(
                gen_input=gen_input,
                past_key_values=gen_ctx["past_key_values"],
                cfg_text_past_kv=cfg_text_ctx["past_key_values"],
                cfg_img_past_kv=cfg_img_ctx["past_key_values"],
                cfg_text_generation_input=cfg_text_gen_input,
                cfg_img_generation_input=cfg_img_gen_input,
                image_shape=image_shape,
                num_inference_steps=num_inference_steps,
                timestep_shift=timestep_shift,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                noise_level=noise_level,
                compute_log_prob=compute_log_prob,
                trajectory_indices=trajectory_indices,
                extra_call_back_kwargs=extra_call_back_kwargs,
                device=device,
            )

            # 4. Decode
            final_latent = result["final_packed_latent"]
            image = self.decode_latents(final_latent, image_shape=image_shape)

            # 5. Build sample
            SampleCls = BagelI2ISample if is_i2i else BagelSample
            sample = SampleCls(
                # Denoising trajectory
                timesteps=result["timesteps"],
                all_latents=result["all_latents"],
                log_probs=result["all_log_probs"],
                latent_index_map=result["latent_index_map"],
                log_prob_index_map=result["log_prob_index_map"],
                # Generated image & metadata
                image=image,
                height=height,
                width=width,
                # Prompt
                prompt=cur_prompt,
                packed_text_ids=gen_input["packed_text_ids"],
                packed_text_indexes=gen_input["packed_text_indexes"],
                packed_vae_position_ids=gen_input["packed_vae_position_ids"],
                packed_vae_token_indexes=gen_input["packed_vae_token_indexes"],
                packed_seqlens=gen_input["packed_seqlens"],
                packed_position_ids=gen_input["packed_position_ids"],
                packed_indexes=gen_input["packed_indexes"],
                packed_key_value_indexes=gen_input["packed_key_value_indexes"],
                key_values_lens=gen_input["key_values_lens"],
                cfg_text_generation_input=cfg_text_gen_input,
                cfg_img_generation_input=cfg_img_gen_input,
                image_shape=image_shape,
                extra_kwargs={
                    **result.get("callback_results", {}),
                    "callback_index_map": result.get("callback_index_map"),
                },
            )
            samples.append(sample)

        return samples