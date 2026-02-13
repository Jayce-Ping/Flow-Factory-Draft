# src/flow_factory/models/bagel/bagel.py
"""
Bagel Model Adapter for Flow-Factory

Integrates ByteDance's Bagel (unified multimodal model) into the
Flow-Factory RL fine-tuning framework.

Architecture Mapping:
    ┌─────────────────────────────────────────────────────┐
    │ Flow-Factory Interface     │  Bagel Component        │
    ├───────────────────────────┼─────────────────────────┤
    │ self.transformer          │  Bagel (LLM + gen heads) │
    │ self.vae                  │  Custom Autoencoder      │
    │ self.tokenizer            │  Qwen2Tokenizer          │
    │ encode_prompt()           │  Build KV-cache context  │
    │ encode_image()            │  ViT + VAE transforms    │
    │ forward()                 │  _forward_flow + sched   │
    │ inference()               │  Full denoising loop     │
    │ decode_latents()          │  VAE decode              │
    └───────────────────────────┴─────────────────────────┘

Supported Tasks:
    - Text-to-Image (T2I): prompt → image
    - Image(s)-to-Image (I2I): images + prompt → image
"""
from __future__ import annotations

import os
import random
from copy import deepcopy
from typing import Union, List, Dict, Any, Optional, Tuple, Literal, ClassVar
from dataclasses import dataclass, field
from collections import defaultdict

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
from ...utils.logger_utils import setup_logger

from .pipeline import BagelPseudoPipeline
from .modeling.bagel import Bagel

logger = setup_logger(__name__)


# ============================================================================
# Sample Dataclasses
# ============================================================================

@dataclass
class BagelSample(T2ISample):
    """
    Sample class for Bagel T2I generation.

    Stores denoising trajectory plus Bagel-specific packed tensor info
    needed to reconstruct the KV-cache context during training.
    """

    _shared_fields: ClassVar[frozenset[str]] = frozenset({
        "image_shape",
    })

    # Bagel-specific: packed generation inputs (per-sample, from context building)
    packed_text_ids: Optional[torch.LongTensor] = None
    packed_text_indexes: Optional[torch.LongTensor] = None
    packed_vae_position_ids: Optional[torch.LongTensor] = None
    packed_vae_token_indexes: Optional[torch.LongTensor] = None
    packed_seqlens: Optional[torch.IntTensor] = None
    packed_position_ids: Optional[torch.LongTensor] = None
    packed_indexes: Optional[torch.LongTensor] = None
    packed_key_value_indexes: Optional[torch.LongTensor] = None
    key_values_lens: Optional[torch.IntTensor] = None

    # CFG inputs (text-cfg and image-cfg packed tensors)
    cfg_text_generation_input: Optional[Dict[str, torch.Tensor]] = None
    cfg_img_generation_input: Optional[Dict[str, torch.Tensor]] = None

    # Image shape for latent unpacking
    image_shape: Optional[Tuple[int, int]] = None


@dataclass
class BagelI2ISample(I2ISample):
    """Sample class for Bagel Image(s)-to-Image generation."""

    _shared_fields: ClassVar[frozenset[str]] = frozenset({
        "image_shape",
    })

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


# ============================================================================
# BagelAdapter
# ============================================================================

class BagelAdapter(BaseAdapter):
    """
    Flow-Factory adapter for Bagel multimodal models.

    Key differences from diffusers-based adapters:
      1. No separate text_encoder; text encoding is internal to the Bagel model
         via its language_model.embed_tokens + KV-cache prefill.
      2. Image understanding uses ViT (SiglipVisionModel) inside the Bagel model.
      3. Denoising operates on packed latent sequences with position-aware indexing.
      4. CFG uses separate pre-computed KV caches for text-only and image-only conditions.
    """

    def __init__(self, config: Arguments, accelerator: Accelerator):
        # Load tokenizer and transforms before super().__init__
        # because load_pipeline may need them, and base __init__ calls load_pipeline
        self._model_path = config.model_args.model_name_or_path
        self._init_tokenizer_and_transforms()

        super().__init__(config, accelerator)
        self.pipeline: BagelPseudoPipeline

    # ─────────────────── Tokenizer & Transforms ───────────────────

    def _init_tokenizer_and_transforms(self):
        """Initialize tokenizer, special tokens, and image transforms."""
        from .modeling.qwen2 import Qwen2Tokenizer
        from .data.data_utils import add_special_tokens
        from .data.transforms import ImageTransform

        self._tokenizer = Qwen2Tokenizer.from_pretrained(self._model_path)
        self._tokenizer, self.new_token_ids, _ = add_special_tokens(self._tokenizer)

        # VAE transform: max_size=1024, min_size=512, patch=16
        self.vae_transform = ImageTransform(1024, 512, 16)
        # ViT transform: max_size=980, min_size=224, patch=14
        self.vit_transform = ImageTransform(980, 224, 14)

    # ======================== Pipeline & Scheduler ========================

    def load_pipeline(self) -> BagelPseudoPipeline:
        """Load the Bagel model and VAE into a pseudo-pipeline."""
        pipeline = BagelPseudoPipeline.from_pretrained(
            self._model_path,
            low_cpu_mem_usage=False,
            **self.model_args.extra_kwargs,
        )
        return pipeline

    def load_scheduler(self) -> FlowMatchEulerDiscreteSDEScheduler:
        """
        Create a FlowMatchEulerDiscreteSDEScheduler for Bagel.

        Bagel uses flow matching with a shifted timestep schedule:
            t_shifted = shift * t / (1 + (shift - 1) * t)
        This is handled by the scheduler's sigma configuration.
        """
        scheduler_kwargs = {
            "num_train_timesteps": 1000,
            "shift": self.model_args.extra_kwargs.get("timestep_shift", 3.0),
        }
        if hasattr(self.config, "scheduler_args") and self.config.scheduler_args:
            scheduler_kwargs.update(self.config.scheduler_args.to_dict())

        scheduler = FlowMatchEulerDiscreteSDEScheduler(**scheduler_kwargs)
        # Apply SDE config from training args
        if hasattr(self.training_args, "noise_level"):
            scheduler.noise_level = self.training_args.noise_level
        if hasattr(self.training_args, "dynamics_type"):
            scheduler.dynamics_type = self.training_args.dynamics_type

        return scheduler

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
        """Bagel has no separate text encoder; encoding is inside the transformer."""
        return []

    @property
    def text_encoders(self) -> List[nn.Module]:
        return []

    @property
    def text_encoder(self) -> Optional[nn.Module]:
        return None

    @property
    def preprocessing_modules(self) -> List[str]:
        """Modules needed for preprocessing (tokenization uses CPU, VAE for decode)."""
        return ["vae"]

    @property
    def inference_modules(self) -> List[str]:
        """Modules needed for inference: the full Bagel model + VAE."""
        return ["transformer", "vae"]

    # ─────────────── Convenience accessors ───────────────

    @property
    def bagel_model(self) -> Bagel:
        """The underlying Bagel nn.Module (alias for transformer)."""
        return self.get_component("transformer")

    @property
    def bagel_config(self):
        """The BagelConfig from the loaded model."""
        return self.pipeline._bagel_config

    # ======================== Encoding ========================

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
    ) -> Dict[str, Any]:
        """
        Tokenize text prompts for Bagel.

        Unlike diffusers adapters, Bagel's prompt encoding is deferred to
        ``inference()`` / ``forward()`` where it becomes part of KV-cache
        context building. Here we just return the raw prompt strings.

        Returns:
            Dict with ``prompt`` key mapping to the list of prompts.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        return {"prompt": prompt}

    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image], List[List[Image.Image]]],
    ) -> Optional[Dict[str, Any]]:
        """
        Pre-process condition images for Bagel I2I tasks.

        Converts PIL images to RGB and stores them for later context building.
        The actual ViT/VAE encoding happens in ``inference()`` / ``forward()``.

        Returns:
            Dict with ``condition_images`` key, or None if no images.
        """
        from .data.data_utils import pil_img2rgb

        if images is None:
            return None

        # Normalize to List[List[Image.Image]]
        if isinstance(images, Image.Image):
            images = [[images]]
        elif isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
            images = [[img] for img in images]

        # Convert to RGB
        processed = [
            [pil_img2rgb(img) for img in img_list]
            for img_list in images
        ]
        return {"condition_images": processed}

    def encode_video(self, videos, **kwargs):
        """Bagel does not support video generation. No-op."""
        return None

    # ======================== Decoding ========================

    def decode_latents(
        self,
        latents: torch.Tensor,
        image_shape: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Decode packed latent tokens back into PIL images.

        Args:
            latents: Packed latent tensor of shape ``(seq_len, patch_dim)``
                     or ``(B, seq_len, patch_dim)`` for a batch.
            image_shape: ``(H, W)`` of the target image (pre-downsampling).

        Returns:
            Single PIL Image or list of PIL Images.
        """
        bagel = self._unwrap(self.bagel_model)
        vae = self.get_component("vae")

        p = bagel.latent_patch_size
        ch = bagel.latent_channel
        ds = bagel.latent_downsample

        single = latents.dim() == 2
        if single:
            latents = latents.unsqueeze(0)

        images = []
        for lat in latents:
            H, W = image_shape
            h, w = H // ds, W // ds
            # (seq, patch_dim) → (1, C, H_lat, W_lat)
            lat = lat.reshape(1, h, w, p, p, ch)
            lat = torch.einsum("nhwpqc->nchpwq", lat)
            lat = lat.reshape(1, ch, h * p, w * p)
            decoded = vae.decode(lat.to(vae.dtype if hasattr(vae, 'dtype') else torch.bfloat16))
            decoded = (decoded * 0.5 + 0.5).clamp(0, 1)[0].float()
            images.append(decoded)

        if single:
            return images[0]
        return images

    # ======================== Context Building ========================

    def _build_gen_context(
        self,
        prompt: str,
        condition_images: Optional[List[Image.Image]] = None,
        think: bool = False,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Build KV-cache contexts for generation.

        Constructs three contexts:
          - gen_context: full context (text + images)
          - cfg_text_context: context without text (for text-CFG)
          - cfg_img_context: context without images (for image-CFG)

        Args:
            prompt: Text prompt string.
            condition_images: Optional list of condition images for I2I.
            think: Whether to prepend the thinking system prompt.

        Returns:
            Tuple of (gen_context, cfg_text_context, cfg_img_context)
        """
        from .modeling.bagel.qwen2_navit import NaiveCache
        from .data.data_utils import pil_img2rgb

        bagel = self._unwrap(self.bagel_model)
        num_layers = bagel.config.llm_config.num_hidden_layers

        def _init_ctx():
            return {
                "kv_lens": [0],
                "ropes": [0],
                "past_key_values": NaiveCache(num_layers),
            }

        gen_context = _init_ctx()
        cfg_text_context = _init_ctx()
        cfg_img_context = _init_ctx()

        # --- Optional thinking prompt ---
        if think:
            system_prompt = (
                "You should first think about the planning process in the mind "
                "and then generate the image.\nThe planning process is enclosed "
                "within <think> </think> tags."
            )
            gen_context = self._update_context_text(system_prompt, gen_context)
            cfg_img_context = self._update_context_text(system_prompt, cfg_img_context)

        # --- Process interleaved inputs ---
        # For I2I: images go first, then text
        if condition_images:
            for img in condition_images:
                img_tensor = self.vae_transform.resize_transform(pil_img2rgb(img))
                gen_context = self._update_context_image(img_tensor, gen_context)
                cfg_text_context = deepcopy(gen_context)

        # Text always comes last (before generation)
        cfg_text_context = deepcopy(gen_context)
        gen_context = self._update_context_text(prompt, gen_context)
        cfg_img_context = self._update_context_text(prompt, cfg_img_context)

        return gen_context, cfg_text_context, cfg_img_context

    # ─── _update_context_text ───
    @torch.no_grad()
    def _update_context_text(self, text: str, gen_context: Dict) -> Dict:
        """Add text tokens to the KV-cache context."""
        bagel = self._unwrap(self.bagel_model)
        device = self.device
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        generation_input, kv_lens, ropes = bagel.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self._tokenizer,
            new_token_ids=self.new_token_ids,
        )
        # ★ Move all tensors to model device before forward
        generation_input = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in generation_input.items()
        }
        past_key_values = bagel.forward_cache_update_text(
            gen_context["past_key_values"], **generation_input
        )
        return {"kv_lens": kv_lens, "ropes": ropes, "past_key_values": past_key_values}


    # ─── _update_context_image ───
    @torch.no_grad()
    def _update_context_image(
        self,
        image_tensor,
        gen_context: Dict,
        vae: bool = True,
        vit: bool = True,
    ) -> Dict:
        """Add image tokens (ViT + VAE) to the KV-cache context."""
        bagel = self._unwrap(self.bagel_model)
        vae_model = self.get_component("vae")
        device = self.device
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        past_key_values = gen_context["past_key_values"]

        if vae:
            gen_input, kv_lens, ropes = bagel.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image_tensor],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            # ★ Move all tensors to model device
            gen_input = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in gen_input.items()
            }
            past_key_values = bagel.forward_cache_update_vae(
                vae_model, past_key_values, **gen_input
            )

        if vit:
            gen_input, kv_lens, ropes = bagel.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image_tensor],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            # ★ Move all tensors to model device
            gen_input = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in gen_input.items()
            }
            past_key_values = bagel.forward_cache_update_vit(
                past_key_values, **gen_input
            )

        return {"kv_lens": kv_lens, "ropes": ropes, "past_key_values": past_key_values}

    # ======================== Inference ========================

    @torch.no_grad()
    def inference(
        self,
        # Generation params
        num_inference_steps: int = 50,
        height: int = 1024,
        width: int = 1024,
        # Prompt
        prompt: Union[str, List[str]] = None,
        # Condition images for I2I
        condition_images: Optional[List[List[Image.Image]]] = None,
        # CFG params
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: Tuple[float, float] = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        timestep_shift: float = 3.0,
        # SDE params
        noise_level: float = 0.7,
        compute_log_prob: bool = True,
        # Trajectory
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
        # Other
        generator: Optional[torch.Generator] = None,
        think: bool = False,
        **kwargs,
    ) -> List[BagelSample]:
        """
        Full generation loop: build context → denoise → decode → return samples.

        Runs one sample at a time (batch_size=1 per call) due to Bagel's
        KV-cache architecture. The trainer handles outer batching.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        device = self.device
        bagel = self._unwrap(self.bagel_model)
        image_shape = (height, width)

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
            gen_input = bagel.prepare_vae_latent(
                curr_kvlens=gen_ctx["kv_lens"],
                curr_rope=gen_ctx["ropes"],
                image_sizes=[image_shape],
                new_token_ids=self.new_token_ids,
                device=device
            )

            cfg_text_gen_input = bagel.prepare_vae_latent_cfg(
                curr_kvlens=cfg_text_ctx["kv_lens"],
                curr_rope=cfg_text_ctx["ropes"],
                image_sizes=[image_shape],
                device=device,
            )
            cfg_img_gen_input = bagel.prepare_vae_latent_cfg(
                curr_kvlens=cfg_img_ctx["kv_lens"],
                curr_rope=cfg_img_ctx["ropes"],
                image_sizes=[image_shape],
                device=device
            )

            # 3. Run denoising loop
            result = self._denoise_loop(
                bagel=bagel,
                gen_input=gen_input,
                past_key_values=gen_ctx["past_key_values"],
                cfg_text_past_kv=cfg_text_ctx["past_key_values"],
                cfg_img_past_kv=cfg_img_ctx["past_key_values"],
                cfg_text_gen_input=cfg_text_gen_input,
                cfg_img_gen_input=cfg_img_gen_input,
                image_shape=image_shape,
                num_timesteps=num_inference_steps,
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

            # 4. Decode final latent
            final_latent = result["unpacked_latent"]
            image = self.decode_latents(final_latent, image_shape=image_shape)

            # 5. Build sample
            is_i2i = cur_cond_images is not None and len(cur_cond_images) > 0
            SampleCls = BagelI2ISample if is_i2i else BagelSample

            sample = SampleCls(
                # Trajectory
                timesteps=result.get("timesteps"),
                all_latents=(
                    torch.stack(result["all_latents"]) if result.get("all_latents") else None
                ),
                log_probs=(
                    torch.stack(result["all_log_probs"]) if result.get("all_log_probs") else None
                ),
                latent_index_map=result.get("latent_index_map"),
                log_prob_index_map=result.get("log_prob_index_map"),
                # Prompt
                prompt=cur_prompt,
                # Image
                height=height,
                width=width,
                image=image,
                image_shape=image_shape,
                # Packed inputs (for forward() during training)
                packed_text_ids=gen_input.get("packed_text_ids"),
                packed_text_indexes=gen_input.get("packed_text_indexes"),
                packed_vae_position_ids=gen_input.get("packed_vae_position_ids"),
                packed_vae_token_indexes=gen_input.get("packed_vae_token_indexes"),
                packed_seqlens=gen_input.get("packed_seqlens"),
                packed_position_ids=gen_input.get("packed_position_ids"),
                packed_indexes=gen_input.get("packed_indexes"),
                packed_key_value_indexes=gen_input.get("packed_key_value_indexes"),
                key_values_lens=gen_input.get("key_values_lens"),
                cfg_text_generation_input=cfg_text_gen_input,
                cfg_img_generation_input=cfg_img_gen_input,
                # Condition images (for I2I)
                **(
                    {"condition_images": cur_cond_images}
                    if is_i2i and hasattr(SampleCls, "condition_images")
                    else {}
                ),
                extra_kwargs={
                    **{
                        k: v
                        for k, v in result.get("callback_results", {}).items()
                    },
                    "callback_index_map": result.get("callback_index_map"),
                },
            )
            samples.append(sample)

        return samples

    def _denoise_loop(
        self,
        bagel: Bagel,
        gen_input: Dict[str, torch.Tensor],
        past_key_values,
        cfg_text_past_kv,
        cfg_img_past_kv,
        cfg_text_gen_input: Dict[str, torch.Tensor],
        cfg_img_gen_input: Dict[str, torch.Tensor],
        image_shape: Tuple[int, int],
        num_timesteps: int,
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
        """
        Core denoising loop using Bagel's flow matching.

        Mirrors ``Bagel.generate_image`` but integrated with Flow-Factory's
        trajectory collection and SDE scheduler.
        """
        # Build timestep schedule
        x_t = gen_input["packed_init_noises"].to(device)
        timesteps = torch.linspace(1, 0, num_timesteps + 1, device=device)
        timesteps = (
            timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        )
        dts = timesteps[1:] - timesteps[:-1]
        timesteps_step = timesteps[:-1]  # T steps

        # SDE window
        sde_window_size = getattr(self.training_args, "sde_window_size", 1)
        sde_window_range = getattr(
            self.training_args, "sde_window_range", (0, max(num_timesteps // 3, 1))
        )
        sde_begin = random.randint(sde_window_range[0], sde_window_range[1])

        # Collectors
        latent_collector = create_trajectory_collector(trajectory_indices, num_timesteps)
        latent_collector.collect(x_t, step_idx=0)
        log_prob_collector = (
            create_trajectory_collector(trajectory_indices, num_timesteps)
            if compute_log_prob
            else None
        )
        callback_collector = create_callback_collector(
            trajectory_indices, num_timesteps
        )

        all_latents_raw = []
        all_log_probs_raw = []
        all_timesteps_raw = []

        # Move packed tensors to device
        gen_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in gen_input.items()}

        for i, t in enumerate(timesteps_step):
            # Determine noise level for SDE
            if i < sde_begin:
                cur_noise_level = 0.0
            elif i == sde_begin:
                cur_noise_level = noise_level
                all_latents_raw.append(x_t)
            elif i < sde_begin + sde_window_size:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0

            # CFG gating
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0

            timestep_tensor = torch.tensor([t] * x_t.shape[0], device=device)

            # Forward flow prediction
            v_t = bagel._forward_flow(
                x_t=x_t,
                timestep=timestep_tensor,
                packed_vae_token_indexes=gen_input["packed_vae_token_indexes"],
                packed_vae_position_ids=gen_input["packed_vae_position_ids"],
                packed_text_ids=gen_input["packed_text_ids"],
                packed_text_indexes=gen_input["packed_text_indexes"],
                packed_position_ids=gen_input["packed_position_ids"],
                packed_indexes=gen_input["packed_indexes"],
                packed_seqlens=gen_input["packed_seqlens"],
                key_values_lens=gen_input["key_values_lens"],
                past_key_values=past_key_values,
                packed_key_value_indexes=gen_input["packed_key_value_indexes"],
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # Text CFG
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_gen_input.get("cfg_packed_position_ids"),
                cfg_text_packed_query_indexes=cfg_text_gen_input.get("cfg_packed_query_indexes"),
                cfg_text_key_values_lens=cfg_text_gen_input.get("cfg_key_values_lens"),
                cfg_text_past_key_values=cfg_text_past_kv,
                cfg_text_packed_key_value_indexes=cfg_text_gen_input.get("cfg_packed_key_value_indexes"),
                # Image CFG
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_gen_input.get("cfg_packed_position_ids"),
                cfg_img_packed_query_indexes=cfg_img_gen_input.get("cfg_packed_query_indexes"),
                cfg_img_key_values_lens=cfg_img_gen_input.get("cfg_key_values_lens"),
                cfg_img_past_key_values=cfg_img_past_kv,
                cfg_img_packed_key_value_indexes=cfg_img_gen_input.get("cfg_packed_key_value_indexes"),
                cfg_type="parallel",
            )

            # Scheduler step
            # TODO: use output = self.scheduler.step(....)
            t_next = timesteps_step[i + 1] if i + 1 < len(timesteps_step) else t * 0
            x_t, log_prob, _, _ = bagel._sde_step_with_logprob(
                v_t,
                timesteps_step[i],
                t_next,
                dts[i],
                x_t,
                sigma_max=timesteps_step[1] if len(timesteps_step) > 1 else timesteps_step[0],
                noise_level=cur_noise_level,
            )

            # Collect trajectory
            latent_collector.collect(x_t, i + 1)
            if compute_log_prob and cur_noise_level > 0:
                log_prob_collector.collect(log_prob, i)
                if i >= sde_begin and i < sde_begin + sde_window_size:
                    all_latents_raw.append(x_t)
                    all_log_probs_raw.append(log_prob)
                    all_timesteps_raw.append(t.item())

            callback_collector.collect_step(
                step_idx=i,
                output=SDESchedulerOutput(
                    next_latents=x_t,
                    log_prob=log_prob if cur_noise_level > 0 else None,
                    noise_pred=v_t,
                ),
                keys=extra_call_back_kwargs,
                capturable={"noise_level": cur_noise_level},
            )

        # Unpack final latent
        packed_seqlens = gen_input["packed_seqlens"]
        unpacked = x_t.split((packed_seqlens - 2).tolist())

        return {
            "unpacked_latent": unpacked[0].float(),
            "all_latents": all_latents_raw or None,
            "all_log_probs": all_log_probs_raw or None,
            "timesteps": (
                torch.tensor(all_timesteps_raw, device=device)
                if all_timesteps_raw
                else timesteps_step
            ),
            "latent_index_map": latent_collector.get_index_map(),
            "log_prob_index_map": (
                log_prob_collector.get_index_map() if log_prob_collector else None
            ),
            "callback_results": callback_collector.get_result(),
            "callback_index_map": callback_collector.get_index_map(),
        }

    # ======================== Forward (Training) ========================

    def forward(
        self,
        # Timestep
        t: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
        # Latents
        latents: torch.Tensor = None,
        next_latents: Optional[torch.Tensor] = None,
        # Prompt (for rebuilding context)
        prompt: Optional[Union[str, List[str]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        # Packed inputs (from sample, avoids rebuilding context)
        packed_text_ids: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
        packed_vae_position_ids: Optional[torch.Tensor] = None,
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_seqlens: Optional[torch.Tensor] = None,
        packed_position_ids: Optional[torch.Tensor] = None,
        packed_indexes: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        # CFG (from sample)
        cfg_text_generation_input: Optional[Dict] = None,
        cfg_img_generation_input: Optional[Dict] = None,
        # CFG params
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: Tuple[float, float] = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # SDE
        noise_level: Optional[float] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = [
            "noise_pred", "next_latents", "next_latents_mean",
            "std_dev_t", "dt", "log_prob",
        ],
        **kwargs,
    ) -> SDESchedulerOutput:
        """
        Single denoising step for training.

        Rebuilds KV-cache context from prompt/images, runs one forward flow
        prediction, and delegates to the scheduler for SDE step + log_prob.

        This method is called by the GRPO trainer at each timestep in the
        recorded trajectory.
        """
        bagel = self._unwrap(self.bagel_model)
        device = latents.device

        # 1. Rebuild KV-cache contexts
        if isinstance(prompt, str):
            prompt = [prompt]
        if prompt is not None:
            # Rebuild from scratch (expensive but necessary for training)
            gen_ctx, cfg_text_ctx, cfg_img_ctx = self._build_gen_context(
                prompt=prompt[0],
                condition_images=(
                    condition_images[0] if condition_images else None
                ),
            )
            # Get packed generation inputs
            gen_input = bagel.prepare_vae_latent(
                curr_kvlens=gen_ctx["kv_lens"],
                curr_rope=gen_ctx["ropes"],
                image_sizes=[(kwargs.get("height", 1024), kwargs.get("width", 1024))],
                new_token_ids=self.new_token_ids,
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

            cfg_text_gen_input = bagel.prepare_vae_latent_cfg(
                curr_kvlens=cfg_text_ctx["kv_lens"],
                curr_rope=cfg_text_ctx["ropes"],
                image_sizes=[(kwargs.get("height", 1024), kwargs.get("width", 1024))],
            )
            cfg_img_gen_input = bagel.prepare_vae_latent_cfg(
                curr_kvlens=cfg_img_ctx["kv_lens"],
                curr_rope=cfg_img_ctx["ropes"],
                image_sizes=[(kwargs.get("height", 1024), kwargs.get("width", 1024))],
            )
            cfg_text_generation_input = cfg_text_gen_input
            cfg_img_generation_input = cfg_img_gen_input
        else:
            # Context must be provided via packed inputs + stored KV caches
            raise ValueError(
                "BagelAdapter.forward() requires `prompt` to rebuild KV caches. "
                "Pass `prompt` from the stored sample."
            )

        # 2. CFG gating
        if t.item() > cfg_interval[0] and t.item() <= cfg_interval[1]:
            cfg_text_s = cfg_text_scale
            cfg_img_s = cfg_img_scale
        else:
            cfg_text_s = 1.0
            cfg_img_s = 1.0

        timestep_tensor = t.expand(latents.shape[0])

        # 3. Forward flow prediction
        v_t = bagel._forward_flow(
            x_t=latents,
            timestep=timestep_tensor,
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
            cfg_text_packed_position_ids=cfg_text_generation_input.get("cfg_packed_position_ids", torch.tensor([], device=device)).to(device),
            cfg_text_packed_query_indexes=cfg_text_generation_input.get("cfg_packed_query_indexes", torch.tensor([], device=device)).to(device),
            cfg_text_key_values_lens=cfg_text_generation_input.get("cfg_key_values_lens", torch.tensor([], device=device)).to(device),
            cfg_text_past_key_values=cfg_text_past_kv,
            cfg_text_packed_key_value_indexes=cfg_text_generation_input.get("cfg_packed_key_value_indexes", torch.tensor([], device=device)).to(device),
            cfg_img_scale=cfg_img_s,
            cfg_img_packed_position_ids=cfg_img_generation_input.get("cfg_packed_position_ids", torch.tensor([], device=device)).to(device),
            cfg_img_packed_query_indexes=cfg_img_generation_input.get("cfg_packed_query_indexes", torch.tensor([], device=device)).to(device),
            cfg_img_key_values_lens=cfg_img_generation_input.get("cfg_key_values_lens", torch.tensor([], device=device)).to(device),
            cfg_img_past_key_values=cfg_img_past_kv,
            cfg_img_packed_key_value_indexes=cfg_img_generation_input.get("cfg_packed_key_value_indexes", torch.tensor([], device=device)).to(device),
            cfg_type="parallel",
        )

        # 4. Scheduler step (compute log_prob, next_latents, etc.)
        output = self.scheduler.step(
            noise_pred=v_t,
            timestep=t,
            latents=latents,
            timestep_next=t_next,
            next_latents=next_latents,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            return_kwargs=return_kwargs,
            noise_level=noise_level,
        )
        return output