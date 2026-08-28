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

# src/flow_factory/models/wan/wan2_t2v.py
from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator
from diffusers.pipelines.wan.pipeline_wan import WanPipeline, prompt_clean
from peft import PeftModel
from PIL import Image

from ...contracts import GeometrySource, NegativePromptPolicy, RateRequirement
from ...hparams import *
from ...samples import T2VSample
from ...scheduler import UniPCMultistepSDEScheduler, UniPCMultistepSDESchedulerOutput
from ...utils.base import filter_kwargs
from ...utils.logger_utils import setup_logger
from ...utils.trajectory_collector import (
    CallbackCollector,
    TrajectoryCollector,
    TrajectoryIndicesType,
    create_callback_collector,
    create_trajectory_collector,
)
from ..abc import BaseAdapter
from ..output_state import DecodedMediaBatch, EncodedOutputState, OutputStateCodec
from ..pipeline_contracts import video_output_contract
from ._output import WanVideoOutputCodec

logger = setup_logger(__name__)


@dataclass
class WanT2VSample(T2VSample):
    #  Class var
    _shared_fields: ClassVar[frozenset[str]] = frozenset({})


class Wan2_T2V_Adapter(BaseAdapter):
    # Wan2.2 trains both transformer and transformer_2 but uses only one per
    # timestep (boundary_ratio), so under DDP the other's trainable params get no
    # gradient in a given step. Ignored under DeepSpeed/FSDP.
    ddp_find_unused_parameters = True
    supports_diffusers_cache = True
    component_load_dtype_defaults = {
        "transformers": torch.bfloat16,
        "text_encoders": torch.bfloat16,
        "vae": torch.float32,
    }
    pipeline_io_contract = video_output_contract(
        negative_prompt=NegativePromptPolicy.OPTIONAL,
        output_fps=RateRequirement.REQUIRED,
        geometry_source=GeometrySource.CONFIGURED,
    )

    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: WanPipeline
        self.scheduler: UniPCMultistepSDEScheduler

    def load_pipeline(self) -> WanPipeline:
        return self._load_diffusers_pipeline(
            WanPipeline,
            self.model_args.model_name_or_path,
        )

    def apply_lora(
        self,
        target_modules: Union[str, List[str]],
        components: Union[str, List[str]] = ["transformer", "transformer_2"],
        **kwargs,
    ) -> Union[PeftModel, Dict[str, PeftModel]]:
        return super().apply_lora(target_modules=target_modules, components=components, **kwargs)

    # ============================ Module Management ============================
    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Wan transformer."""
        return [
            # --- Self Attention ---
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            # --- Cross Attention ---
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
            # --- Feed Forward Network ---
            "ffn.net.0.proj",
            "ffn.net.2",
        ]

    @property
    def inference_modules(self) -> List[str]:
        """Modules that are required for inference and forward"""
        if self.pipeline.config.boundary_ratio is None or self.pipeline.config.boundary_ratio <= 0:
            return ["transformer", "vae"]

        if self.pipeline.config.boundary_ratio >= 1:
            return ["transformer_2", "vae"]

        return ["transformer", "transformer_2", "vae"]

    # ======================== Component Getters & Setters ========================
    @property
    def transformer_2(self) -> torch.nn.Module:
        return self.get_component("transformer_2")

    @transformer_2.setter
    def transformer_2(self, module: torch.nn.Module):
        self.set_component("transformer_2", module)

    # ======================== Encoding & Decoding ========================
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.pipeline.text_encoder(
            text_input_ids.to(device), mask.to(device)
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )

        return text_input_ids, prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 3.5,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            guidance_scale (`float`, *optional*, defaults to `3.5`):
                Guidance scale for classifier-free guidance. CFG is enabled when `guidance_scale > 1.0`.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_ids, prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        results = {
            "prompt_ids": prompt_ids,
            "prompt_embeds": prompt_embeds,
        }

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            )
            negative_prompt = negative_prompt * (
                len(prompt) // len(negative_prompt)
            )  # Expand to match batch size
            assert len(negative_prompt) == len(
                prompt
            ), "The number of negative prompts must match the number of prompts."

            negative_prompt_ids, negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
            results.update(
                {
                    "negative_prompt_ids": negative_prompt_ids,
                    "negative_prompt_embeds": negative_prompt_embeds,
                }
            )

        return results

    def encode_image(self, images: Union[Image.Image, torch.Tensor, List[torch.Tensor]]):
        """Not needed for Wan text-to-video models."""
        return None

    def encode_video(self, videos: Union[torch.Tensor, List[torch.Tensor]]):
        """Not needed for Wan text-to-video models."""
        return None

    def build_output_state_codec(self) -> OutputStateCodec:
        """Declare the on-the-fly target-video codec without loading components."""
        return WanVideoOutputCodec(self)

    def _configured_video_output_geometry(self) -> Tuple[int, int, int, float]:
        """Return configured Wan geometry after exact latent-grid validation."""
        geometry = []
        for name in ("height", "width", "num_frames"):
            value = getattr(self.training_args, name, None)
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"Wan output geometry requires positive integer train.{name}, "
                    f"received {value!r}"
                )
            geometry.append(value)
        frame_rate = getattr(self.training_args, "frame_rate", None)
        if isinstance(frame_rate, bool) or not isinstance(frame_rate, Real):
            raise TypeError(
                "Wan output geometry requires finite positive train.frame_rate, "
                f"received {type(frame_rate).__name__}: {frame_rate!r}"
            )
        frame_rate = float(frame_rate)
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError(
                "Wan output geometry requires finite positive train.frame_rate, "
                f"received {frame_rate!r}"
            )

        height, width, num_frames = geometry
        temporal_scale = self.pipeline.vae_scale_factor_temporal
        spatial_scale = self.pipeline.vae_scale_factor_spatial
        if (num_frames - 1) % temporal_scale:
            raise ValueError(
                "Wan output num_frames must satisfy "
                f"(num_frames - 1) % {temporal_scale} == 0, received {num_frames}"
            )
        transformer = (
            self.pipeline.transformer
            if self.pipeline.transformer is not None
            else self.pipeline.transformer_2
        )
        if transformer is None:
            raise RuntimeError("Wan output geometry requires one materialized transformer")
        patch_size = transformer.config.patch_size
        height_multiple = spatial_scale * patch_size[1]
        width_multiple = spatial_scale * patch_size[2]
        if height % height_multiple or width % width_multiple:
            raise ValueError(
                "Wan output height/width must be divisible by transformer latent-grid "
                f"multiples {(height_multiple, width_multiple)}, received {(height, width)}"
            )
        return height, width, num_frames, frame_rate

    @staticmethod
    def _resample_output_video(
        video: np.ndarray,
        *,
        source_fps: Optional[float],
        target_frames: int,
        target_fps: float,
    ) -> np.ndarray:
        """Select deterministic nearest-time frames for configured target cadence."""
        if video.dtype != np.uint8 or video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(
                "Wan decoded target video must be uint8 RGB shaped (F,H,W,3), "
                f"received dtype={video.dtype}, shape={tuple(video.shape)}"
            )
        if video.shape[0] < 1:
            raise ValueError("Wan decoded target video must contain at least one frame")
        if isinstance(source_fps, bool) or not isinstance(source_fps, Real):
            raise TypeError(
                "Wan target video requires source fps metadata, "
                f"received {type(source_fps).__name__}: {source_fps!r}"
            )
        source_fps = float(source_fps)
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError(f"Wan target video requires positive finite fps, got {source_fps!r}")
        indices = np.rint(
            np.arange(target_frames, dtype=np.float64) * source_fps / target_fps
        ).astype(np.int64)
        if indices[-1] >= video.shape[0]:
            required_duration = (target_frames - 1) / target_fps
            available_duration = (video.shape[0] - 1) / source_fps
            raise ValueError(
                "Wan target video is too short for configured temporal geometry: "
                f"requires {required_duration:.6f}s, has {available_duration:.6f}s"
            )
        return np.ascontiguousarray(video[indices])

    def _normalize_output_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply the exact inverse of Wan's existing decode normalization."""
        if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
            raise ValueError(
                "Wan VAE target latents must be rank-5 BCFHW, "
                f"received {type(latents).__name__} with shape "
                f"{getattr(latents, 'shape', None)}"
            )
        config = self.vae.config
        z_dim = config.z_dim
        if latents.shape[1] != z_dim:
            raise ValueError(
                f"Wan VAE target latent channels must equal z_dim={z_dim}, "
                f"received {latents.shape[1]}"
            )
        latents_mean = torch.as_tensor(
            config.latents_mean,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, z_dim, 1, 1, 1)
        inverse_std = (
            torch.as_tensor(
                config.latents_std,
                device=latents.device,
                dtype=latents.dtype,
            )
            .reciprocal()
            .view(1, z_dim, 1, 1, 1)
        )
        return (latents - latents_mean) * inverse_std

    def _validate_encoded_output_geometry(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        encoded: EncodedOutputState,
    ) -> None:
        """Require encoded signatures and decode metadata to match train geometry."""
        del condition
        height, width, num_frames, frame_rate = self._configured_video_output_geometry()
        if len(encoded.geometry_signatures) != len(media_batch):
            raise ValueError(
                "Wan output codec must return one geometry signature per sample, "
                f"received {len(encoded.geometry_signatures)} for {len(media_batch)}"
            )
        for sample_index, signature in enumerate(encoded.geometry_signatures):
            geometry = signature.media[0]
            received = (
                geometry.height,
                geometry.width,
                geometry.frames,
                geometry.fps,
            )
            expected = (height, width, num_frames, frame_rate)
            if received != expected:
                raise ValueError(
                    "Wan encoded output geometry disagrees with configured geometry for "
                    f"sample {sample_index}: expected {expected}, received {received}"
                )
        expected_context = {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
        }
        for name, expected in expected_context.items():
            if encoded.decode_context.get(name) != expected:
                raise ValueError(
                    f"Wan decode_context {name!r} must equal {expected!r}, "
                    f"received {encoded.decode_context.get(name)!r}"
                )

    def decode_latents(
        self, latents: torch.Tensor, output_type: Literal["pt", "pil", "np"] = "pil"
    ) -> torch.Tensor:
        """Decode the latents using the VAE decoder."""
        latents = latents.float()
        latents_mean = (
            torch.tensor(self.pipeline.vae.config.latents_mean)
            .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.pipeline.vae.config.latents_std).view(
            1, self.pipeline.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        video = self.pipeline.vae.decode(latents, return_dict=False)[0]

        video = self.pipeline.video_processor.postprocess_video(video, output_type=output_type)
        return video

    # ======================== Inference ========================

    @torch.no_grad()
    def inference(
        self,
        # Ordinary args
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # Prompt encoding args
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # Other args
        compute_log_prob: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        # Extra callback arguments
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
    ) -> List[WanT2VSample]:
        # 1. Setup args
        device = self.device
        do_classifier_free_guidance = guidance_scale > 1.0

        if self.pipeline.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale
        # Check `num_frames`
        if (num_frames - 1) % self.pipeline.vae_scale_factor_temporal != 0:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.pipeline.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = (
                num_frames
                // self.pipeline.vae_scale_factor_temporal
                * self.pipeline.vae_scale_factor_temporal
                + 1
            )
        num_frames = max(num_frames, 1)
        # Check `height` and `width`
        # Check `height` and `width`
        patch_size = (
            self.pipeline.transformer.config.patch_size
            if self.pipeline.transformer is not None
            else self.pipeline.transformer_2.config.patch_size
        )
        h_multiple_of = self.pipeline.vae_scale_factor_spatial * patch_size[1]
        w_multiple_of = self.pipeline.vae_scale_factor_spatial * patch_size[2]
        calc_height = height // h_multiple_of * h_multiple_of
        calc_width = width // w_multiple_of * w_multiple_of
        if height != calc_height or width != calc_width:
            logger.warning(
                f"`height` and `width` must be multiples of ({h_multiple_of}, {w_multiple_of}) for proper patchification. "
                f"Adjusting ({height}, {width}) -> ({calc_height}, {calc_width})."
            )
            height, width = calc_height, calc_width

        # 2. Encode prompt
        if prompt_embeds is None:
            encoded = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            prompt_ids = encoded["prompt_ids"]
            prompt_embeds = encoded["prompt_embeds"]
            negative_prompt_ids = encoded.get("negative_prompt_ids", None)
            negative_prompt_embeds = encoded.get("negative_prompt_embeds", None)
        else:
            prompt_embeds = prompt_embeds.to(device)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device)

        batch_size = prompt_embeds.shape[0]
        transformer_dtype = (
            self.pipeline.transformer.dtype
            if self.pipeline.transformer is not None
            else self.pipeline.transformer_2.dtype
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 3. Set scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = (
            self.pipeline.transformer.config.in_channels
            if self.pipeline.transformer is not None
            else self.pipeline.transformer_2.config.in_channels
        )
        latents = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self.pipeline._num_timesteps = len(timesteps)

        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latents = self.cast_latents(latents)
        latent_collector.collect(latents, step_idx=0)
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(
                trajectory_indices, num_inference_steps
            )
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        for i, t in enumerate(timesteps):
            self.pipeline._current_timestep = t
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            return_kwargs = list(
                set(["next_latents", "log_prob", "velocity"] + extra_call_back_kwargs)
            )
            current_compute_log_prob = compute_log_prob and current_noise_level > 0

            output = self.forward(
                t=t,
                t_next=t_next,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2,
                attention_kwargs=attention_kwargs,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=return_kwargs,
                noise_level=current_noise_level,
            )

            latents = self.cast_latents(output.next_latents)
            latent_collector.collect(latents, i + 1)
            if current_compute_log_prob:
                log_prob_collector.collect(output.log_prob, i)

            callback_collector.collect_step(
                step_idx=i,
                output=output,
                keys=extra_call_back_kwargs,
                capturable={"noise_level": current_noise_level},
            )

        self.pipeline._current_timestep = None

        # 7. Decode latents to videos (list of pil images)
        decoded_videos = self.decode_latents(latents, output_type="pt")

        # 8. Prepare output samples
        extra_call_back_res = callback_collector.get_result()  # (B, len(trajectory_indices), ...)
        callback_index_map = callback_collector.get_index_map()  # (T,) LongTensor
        all_latents = latent_collector.get_result()  # List[torch.Tensor(B, ...)]
        latent_index_map = latent_collector.get_index_map()  # (T+1,) LongTensor
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        log_prob_index_map = log_prob_collector.get_index_map() if compute_log_prob else None
        samples = [
            WanT2VSample(
                # Denoising trajectory
                timesteps=timesteps,
                all_latents=(
                    torch.stack([lat[b] for lat in all_latents], dim=0)
                    if all_latents is not None
                    else None
                ),
                log_probs=(
                    torch.stack([lp[b] for lp in all_log_probs], dim=0)
                    if all_log_probs is not None
                    else None
                ),
                latent_index_map=latent_index_map,
                log_prob_index_map=log_prob_index_map,
                # Generated video & metadata
                video=decoded_videos[b],
                height=height,
                width=width,
                # Prompt info
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b],
                prompt_embeds=prompt_embeds[b],
                # Negative prompt info
                negative_prompt=(
                    negative_prompt[b] if isinstance(negative_prompt, list) else negative_prompt
                ),
                negative_prompt_ids=(
                    negative_prompt_ids[b] if negative_prompt_ids is not None else None
                ),
                negative_prompt_embeds=(
                    negative_prompt_embeds[b] if negative_prompt_embeds is not None else None
                ),
                # Extra kwargs
                extra_kwargs={
                    **{k: v[b] for k, v in extra_call_back_res.items()},
                    "callback_index_map": callback_index_map,
                },
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()

        return samples

    # ======================== Forward (Training) ========================

    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        # Optional for CFG
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        # Next timestep info
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Other
        noise_level: Optional[float] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = [
            "velocity",
            "next_latents",
            "next_latents_mean",
            "std_dev_t",
            "dt",
            "log_prob",
        ],
        boundary_timestep: Optional[float] = None,
    ) -> UniPCMultistepSDESchedulerOutput:
        """
        Core forward pass for T2V generation.

        Args:
            t: Current timestep tensor.
            latents: Current latent representations (B, C, T, H, W).
            prompt_embeds: Text prompt embeddings.
            negative_prompt_embeds: Optional negative prompt embeddings (for CFG).
            guidance_scale: CFG scale factor.
            transformer: Transformer module to use (defaults to self.transformer).
            pipeline_model: Pipeline model for cache_context (defaults to self.pipeline.transformer).
            next_latents: Optional target latents for log-prob computation.
            attention_kwargs: Optional kwargs for attention layers.
            compute_log_prob: Whether to compute log probabilities.
            return_kwargs: List of outputs to return.
            noise_level: Current noise level for SDE sampling.

        Returns:
            UniPCMultistepSDESchedulerOutput containing requested outputs.
        """
        # 1. Prepare variables
        t = t[0] if t.ndim == 1 else t  # A scalar
        if t_next is not None:
            t_next = t_next[0] if t_next.ndim == 1 else t_next

        batch_size = latents.shape[0]
        device = latents.device
        dtype = (
            self.pipeline.transformer.dtype
            if self.pipeline.transformer is not None
            else self.pipeline.transformer_2.dtype
        )

        # Determine boundary timestep
        if boundary_timestep is None and self.pipeline.config.boundary_ratio is not None:
            boundary_timestep = (
                self.pipeline.config.boundary_ratio * self.scheduler.config.num_train_timesteps
            )
        # Determine which transformer to use
        if boundary_timestep is None or t >= boundary_timestep:
            pipeline_transformer = self.pipeline.transformer
            transformer = self.transformer
            current_guidance_scale = guidance_scale
        else:
            pipeline_transformer = self.pipeline.transformer_2
            transformer = self.transformer_2
            current_guidance_scale = (
                guidance_scale_2 if guidance_scale_2 is not None else guidance_scale
            )

        # Auto-detect CFG
        if current_guidance_scale > 1.0 and negative_prompt_embeds is None:
            logger.warning(
                "Passed `guidance_scale` > 1.0, but no `negative_prompt_embeds` provided. "
                "Classifier-free guidance will be disabled."
            )
        do_classifier_free_guidance = (
            negative_prompt_embeds is not None and current_guidance_scale > 1.0
        )

        # 2. Prepare timestep
        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)
        latent_model_input = latents.to(dtype)

        if self.pipeline.config.expand_timesteps:
            temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
            timestep = temp_ts.unsqueeze(0).expand(batch_size, -1)
        else:
            timestep = t.expand(batch_size)

        # 3. Transformer forward pass
        with pipeline_transformer.cache_context("cond"):
            velocity = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

        # 4. Apply CFG
        if do_classifier_free_guidance:
            with pipeline_transformer.cache_context("uncond"):
                velocity_uncond = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            velocity = velocity_uncond + current_guidance_scale * (velocity - velocity_uncond)

        # 5. Scheduler step
        output = self.scheduler.step(
            velocity=velocity,
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
