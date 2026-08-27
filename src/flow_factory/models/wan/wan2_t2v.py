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
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator
from diffusers.pipelines.wan.pipeline_wan import WanPipeline, prompt_clean
from peft import PeftModel
from PIL import Image

from ...contracts import GeometrySource, MediaType, NegativePromptPolicy
from ...hparams import *
from ...samples import LatentState, T2VSample
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
from ..output_state import (
    DecodedMediaBatch,
    EncodedOutputState,
    GeometrySignature,
    MediaGeometrySignature,
    OutputStateCodec,
)
from ..pipeline_contracts import video_output_contract

logger = setup_logger(__name__)


@dataclass
class WanT2VSample(T2VSample):
    #  Class var
    _shared_fields: ClassVar[frozenset[str]] = frozenset({})


@dataclass(frozen=True, slots=True)
class _WanT2VOutputStateCodec:
    """Encode decoded videos into deterministic normalized Wan clean latents."""

    adapter: "Wan2_T2V_Adapter"
    required_components: ClassVar[Tuple[str, ...]] = ("vae",)

    def encode_output_state(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> EncodedOutputState:
        """Preprocess and VAE-encode one exact configured-geometry target batch."""
        del condition, generator  # Wan target encoding uses deterministic posterior mode.
        height, width, num_frames = self.adapter._configured_output_geometry()
        videos, frame_rates = self._extract_videos(media_batch, num_frames)
        pixel_values = self.adapter.pipeline.video_processor.preprocess_video(
            videos,
            height=height,
            width=width,
        )
        self._validate_pixel_values(
            pixel_values,
            batch_size=len(videos),
            height=height,
            width=width,
            num_frames=num_frames,
        )

        vae = self.adapter.vae
        vae_dtype = getattr(vae, "dtype", None)
        if not isinstance(vae_dtype, torch.dtype) or not vae_dtype.is_floating_point:
            raise TypeError(
                "Wan2 T2V output codec expected VAE to expose a floating dtype, "
                f"received {vae_dtype!r}"
            )
        pixel_values = pixel_values.to(device=self.adapter.device, dtype=vae_dtype)
        encoder_output = vae.encode(pixel_values)
        latents = self._posterior_mode(encoder_output)

        z_dim, latents_mean, latents_std = self._normalization_statistics(
            vae,
            device=latents.device,
            dtype=latents.dtype,
        )
        temporal_scale, spatial_scale = self.adapter._vae_scale_factors()
        expected_latent_shape = (
            len(videos),
            z_dim,
            (num_frames - 1) // temporal_scale + 1,
            height // spatial_scale,
            width // spatial_scale,
        )
        if tuple(latents.shape) != expected_latent_shape:
            raise ValueError(
                "Wan2 T2V VAE encode returned incompatible target geometry: "
                f"expected {expected_latent_shape}, received {tuple(latents.shape)}"
            )
        latents = (latents - latents_mean) / latents_std

        signatures = tuple(
            GeometrySignature(
                media=(
                    MediaGeometrySignature(
                        type=MediaType.VIDEO,
                        height=height,
                        width=width,
                        frames=num_frames,
                        fps=fps,
                    ),
                )
            )
            for fps in frame_rates
        )
        return EncodedOutputState(
            clean_state=LatentState({"latent": latents}),
            forward_context={},
            decode_context={"height": height, "width": width, "num_frames": num_frames},
            geometry_signatures=signatures,
        )

    @staticmethod
    def _extract_videos(
        media_batch: DecodedMediaBatch,
        num_frames: int,
    ) -> Tuple[List[Any], List[Optional[float]]]:
        """Extract one decoded pixel video per sample and require exact frame count."""
        videos: List[Any] = []
        frame_rates: List[Optional[float]] = []
        for sample_index, candidate in enumerate(media_batch):
            if len(candidate) != 1:
                raise ValueError(
                    "Wan2 T2V output codec expected one video per sample, "
                    f"received {len(candidate)} for sample {sample_index}"
                )
            media = candidate[0]
            payload = media.payload
            if isinstance(payload, np.ndarray):
                if payload.ndim != 4 or payload.shape[-1] != 3:
                    raise ValueError(
                        "Wan2 T2V output codec expected NumPy video shaped (F,H,W,3), "
                        f"received {payload.shape} for sample {sample_index}"
                    )
                frame_count = payload.shape[0]
            elif isinstance(payload, torch.Tensor):
                if payload.ndim != 4 or payload.shape[1] != 3:
                    raise ValueError(
                        "Wan2 T2V output codec expected tensor video shaped (F,3,H,W), "
                        f"received {tuple(payload.shape)} for sample {sample_index}"
                    )
                frame_count = payload.shape[0]
            elif isinstance(payload, (list, tuple)):
                if not payload:
                    raise ValueError(
                        f"Wan2 T2V output codec received an empty video for sample {sample_index}"
                    )
                frame_count = len(payload)
                payload = list(payload)
            else:
                raise TypeError(
                    "Wan2 T2V output codec expected decoded video as NumPy, tensor, or "
                    f"frame list, received {type(payload).__name__} for sample {sample_index}"
                )
            if frame_count != num_frames:
                raise ValueError(
                    "Wan2 T2V target video must already match configured temporal geometry: "
                    f"expected {num_frames} frames, received {frame_count} for sample "
                    f"{sample_index}"
                )
            videos.append(payload)
            frame_rates.append(media.fps)
        return videos, frame_rates

    @staticmethod
    def _validate_pixel_values(
        pixel_values: object,
        *,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
    ) -> None:
        """Require Diffusers video preprocessing to preserve exact B/F/H/W."""
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(
                "Wan VideoProcessor.preprocess_video expected torch.Tensor output, "
                f"received {type(pixel_values).__name__}"
            )
        expected_shape = (batch_size, 3, num_frames, height, width)
        if tuple(pixel_values.shape) != expected_shape:
            raise ValueError(
                "Wan VideoProcessor.preprocess_video changed configured target geometry: "
                f"expected {expected_shape}, received {tuple(pixel_values.shape)}"
            )

    @staticmethod
    def _posterior_mode(encoder_output: Any) -> torch.Tensor:
        """Resolve current Diffusers Wan VAE output surfaces without sampling."""
        posterior = getattr(encoder_output, "latent_dist", None)
        if posterior is None and isinstance(encoder_output, (tuple, list)):
            if len(encoder_output) != 1:
                raise TypeError(
                    "Wan2 T2V VAE encode expected a single latent distribution tuple, "
                    f"received length {len(encoder_output)}"
                )
            posterior = getattr(encoder_output[0], "latent_dist", encoder_output[0])
        if posterior is None and getattr(encoder_output, "mode", None) is not None:
            posterior = encoder_output
        mode = getattr(posterior, "mode", None)
        latents = mode() if callable(mode) else mode
        if not isinstance(latents, torch.Tensor):
            raise TypeError(
                "Wan2 T2V VAE latent distribution must expose tensor mode, "
                f"received {type(latents).__name__}"
            )
        if latents.ndim != 5:
            raise ValueError(
                "Wan2 T2V VAE posterior mode expected rank-5 BCFHW tensor, "
                f"received shape {tuple(latents.shape)}"
            )
        return latents

    @staticmethod
    def _normalization_statistics(
        vae: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Return validated per-channel Wan latent mean and standard deviation."""
        config = getattr(vae, "config", None)
        z_dim = getattr(config, "z_dim", None)
        if type(z_dim) is not int or z_dim <= 0:
            raise TypeError(
                "Wan2 T2V VAE config expected positive int z_dim, " f"received {z_dim!r}"
            )
        try:
            mean = torch.as_tensor(getattr(config, "latents_mean", None), dtype=torch.float32)
            std = torch.as_tensor(getattr(config, "latents_std", None), dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Wan2 T2V VAE config expected numeric latents_mean and latents_std"
            ) from exc
        if mean.ndim != 1 or std.ndim != 1 or mean.numel() != z_dim or std.numel() != z_dim:
            raise ValueError(
                "Wan2 T2V VAE config expected one latent mean/std value per channel: "
                f"z_dim={z_dim}, mean_shape={tuple(mean.shape)}, std_shape={tuple(std.shape)}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Wan2 T2V VAE latent normalization statistics must be finite")
        if not torch.all(std > 0):
            raise ValueError("Wan2 T2V VAE latents_std values must all be positive")
        view_shape = (1, z_dim, 1, 1, 1)
        return (
            z_dim,
            mean.view(view_shape).to(device=device, dtype=dtype),
            std.view(view_shape).to(device=device, dtype=dtype),
        )


class Wan2_T2V_Adapter(BaseAdapter):
    # Wan2.2 trains both transformer and transformer_2 but uses only one per
    # timestep (boundary_ratio), so under DDP the other's trainable params get no
    # gradient in a given step. Ignored under DeepSpeed/FSDP.
    ddp_find_unused_parameters = True
    supports_diffusers_cache = True
    pipeline_io_contract = video_output_contract(
        negative_prompt=NegativePromptPolicy.OPTIONAL,
        geometry_source=GeometrySource.CONFIGURED,
    )

    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: WanPipeline
        self.scheduler: UniPCMultistepSDEScheduler

    def load_pipeline(self) -> WanPipeline:
        return WanPipeline.from_pretrained(
            self.model_args.model_name_or_path,
        )

    def build_output_state_codec(self) -> OutputStateCodec:
        """Build the deterministic on-the-fly Wan target-video encoder."""
        self._configured_output_geometry()
        return _WanT2VOutputStateCodec(self)

    def _vae_scale_factors(self) -> Tuple[int, int]:
        """Return strictly validated temporal and spatial VAE compression factors."""
        temporal = getattr(self.pipeline, "vae_scale_factor_temporal", None)
        spatial = getattr(self.pipeline, "vae_scale_factor_spatial", None)
        for name, value in (("temporal", temporal), ("spatial", spatial)):
            if type(value) is not int or value <= 0:
                raise TypeError(
                    f"Wan2 T2V expected positive int VAE {name} scale factor, "
                    f"received {value!r}"
                )
        return temporal, spatial

    def _transformer_patch_size(self) -> Tuple[int, int, int]:
        """Return the active Wan transformer's three-axis patch size."""
        transformer = getattr(self.pipeline, "transformer", None)
        if transformer is None:
            transformer = getattr(self.pipeline, "transformer_2", None)
        patch_size = getattr(getattr(transformer, "config", None), "patch_size", None)
        if (
            not isinstance(patch_size, (tuple, list))
            or len(patch_size) != 3
            or any(type(value) is not int or value <= 0 for value in patch_size)
        ):
            raise TypeError(
                "Wan2 T2V expected transformer patch_size as three positive ints, "
                f"received {patch_size!r}"
            )
        return tuple(patch_size)

    def _configured_output_geometry(self) -> Tuple[int, int, int]:
        """Return configured H/W/F after fail-fast Wan geometry validation."""
        geometry = []
        for name in ("height", "width", "num_frames"):
            value = getattr(self.training_args, name, None)
            if type(value) is not int:
                raise TypeError(
                    f"Wan2 T2V output geometry expected training_args.{name} to be int, "
                    f"received {type(value).__name__}: {value!r}"
                )
            if value <= 0:
                raise ValueError(
                    f"Wan2 T2V output geometry expected training_args.{name} > 0, "
                    f"received {value}"
                )
            geometry.append(value)
        height, width, num_frames = geometry
        temporal_scale, spatial_scale = self._vae_scale_factors()
        if (num_frames - 1) % temporal_scale:
            raise ValueError(
                "Wan2 T2V output geometry requires (num_frames - 1) divisible by "
                f"{temporal_scale}, received num_frames={num_frames}"
            )
        patch_size = self._transformer_patch_size()
        height_multiple = spatial_scale * patch_size[1]
        width_multiple = spatial_scale * patch_size[2]
        if height % height_multiple or width % width_multiple:
            raise ValueError(
                "Wan2 T2V output geometry requires height/width divisible by "
                f"({height_multiple}, {width_multiple}), received ({height}, {width})"
            )
        return height, width, num_frames

    def _validate_encoded_output_geometry(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        encoded: EncodedOutputState,
    ) -> None:
        """Verify codec signatures and decode fields against configured H/W/F."""
        del condition
        height, width, num_frames = self._configured_output_geometry()
        if len(encoded.geometry_signatures) != len(media_batch):
            raise ValueError(
                "Wan2 T2V output geometry expected one signature per target sample, "
                f"received {len(encoded.geometry_signatures)} for batch size {len(media_batch)}"
            )
        for sample_index, (candidate, signature) in enumerate(
            zip(media_batch, encoded.geometry_signatures)
        ):
            expected = GeometrySignature(
                media=(
                    MediaGeometrySignature(
                        type=MediaType.VIDEO,
                        height=height,
                        width=width,
                        frames=num_frames,
                        fps=candidate[0].fps,
                    ),
                )
            )
            if signature != expected:
                raise ValueError(
                    "Wan2 T2V encoded output geometry disagrees with configured H/W/F "
                    f"for sample {sample_index}: expected {expected!r}, received {signature!r}"
                )
        expected_decode_context = {
            "height": height,
            "width": width,
            "num_frames": num_frames,
        }
        if dict(encoded.decode_context) != expected_decode_context:
            raise ValueError(
                "Wan2 T2V decode_context must exactly match configured output geometry "
                f"{expected_decode_context}, received {dict(encoded.decode_context)!r}"
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

    def encode_image(
        self,
        images: Union[Image.Image, torch.Tensor, List[torch.Tensor]],
    ) -> None:
        """Return no condition because Wan T2V accepts no input images."""
        del images
        return None

    def encode_video(self, videos: Union[torch.Tensor, List[torch.Tensor]]) -> None:
        """Return no condition because Wan T2V accepts no input videos."""
        del videos
        return None

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
            t: Shared scalar or one current timestep per batch sample.
            latents: Current latent representations (B, C, T, H, W).
            prompt_embeds: Text prompt embeddings.
            negative_prompt_embeds: Optional negative prompt embeddings (for CFG).
            guidance_scale: CFG scale for the primary/high-noise transformer.
            guidance_scale_2: Optional CFG scale for the low-noise transformer.
            t_next: Shared scalar or one next timestep per batch sample.
            next_latents: Optional target latents for log-prob computation.
            noise_level: Current noise level for SDE sampling.
            attention_kwargs: Optional kwargs for attention layers.
            compute_log_prob: Whether to compute log probabilities.
            return_kwargs: List of outputs to return.
            boundary_timestep: Optional explicit Wan2.2 transformer boundary.

        Returns:
            UniPCMultistepSDESchedulerOutput containing requested outputs.
        """
        # 1. Prepare variables. Online rollout supplies one shared scalar while
        # offline objectives supply one independently sampled timestep per sample.
        batch_size = latents.shape[0]
        device = latents.device
        model_timesteps, scheduler_timestep = self._normalize_forward_timesteps(
            t,
            batch_size=batch_size,
            device=device,
            identifier="t",
        )
        scheduler_timestep_next = None
        if t_next is not None:
            _, scheduler_timestep_next = self._normalize_forward_timesteps(
                t_next,
                batch_size=batch_size,
                device=device,
                identifier="t_next",
            )

        # Determine boundary timestep
        if boundary_timestep is None and self.pipeline.config.boundary_ratio is not None:
            boundary_timestep = (
                self.pipeline.config.boundary_ratio * self.scheduler.config.num_train_timesteps
            )
        if boundary_timestep is None:
            high_noise_indices = torch.arange(batch_size, device=device)
            low_noise_indices = torch.empty(0, dtype=torch.long, device=device)
        else:
            high_noise_indices = torch.nonzero(
                model_timesteps >= boundary_timestep,
                as_tuple=False,
            ).flatten()
            low_noise_indices = torch.nonzero(
                model_timesteps < boundary_timestep,
                as_tuple=False,
            ).flatten()
        low_noise_guidance_scale = (
            guidance_scale_2 if guidance_scale_2 is not None else guidance_scale
        )

        # Auto-detect CFG
        if max(guidance_scale, low_noise_guidance_scale) > 1.0 and negative_prompt_embeds is None:
            logger.warning(
                "Passed `guidance_scale` > 1.0, but no `negative_prompt_embeds` provided. "
                "Classifier-free guidance will be disabled."
            )

        # 2-4. Wan2.2 may route different offline samples through different
        # transformers. Execute homogeneous partitions and restore dataset order.
        velocity_parts: List[torch.Tensor] = []
        index_parts: List[torch.Tensor] = []
        partitions = (
            (
                high_noise_indices,
                self.pipeline.transformer,
                self.transformer if high_noise_indices.numel() else None,
                guidance_scale,
                "transformer",
            ),
            (
                low_noise_indices,
                self.pipeline.transformer_2,
                self.transformer_2 if low_noise_indices.numel() else None,
                low_noise_guidance_scale,
                "transformer_2",
            ),
        )
        for (
            indices,
            pipeline_transformer,
            transformer,
            current_guidance_scale,
            component_name,
        ) in partitions:
            if indices.numel() == 0:
                continue
            if pipeline_transformer is None or transformer is None:
                raise RuntimeError(f"Wan2 forward routed samples to unavailable {component_name}")
            latent_model_input = latents.index_select(0, indices).to(pipeline_transformer.dtype)
            partition_timesteps = model_timesteps.index_select(0, indices)
            if self.pipeline.config.expand_timesteps:
                spatial_mask = torch.ones_like(
                    latent_model_input[:, 0, :, ::2, ::2],
                    dtype=torch.float32,
                )
                transformer_timestep = (
                    spatial_mask * partition_timesteps[:, None, None, None]
                ).flatten(1)
            else:
                transformer_timestep = partition_timesteps
            partition_prompt_embeds = prompt_embeds.index_select(0, indices)
            with pipeline_transformer.cache_context("cond"):
                partition_velocity = transformer(
                    hidden_states=latent_model_input,
                    timestep=transformer_timestep,
                    encoder_hidden_states=partition_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

            do_classifier_free_guidance = (
                negative_prompt_embeds is not None and current_guidance_scale > 1.0
            )
            if do_classifier_free_guidance:
                partition_negative_prompt_embeds = negative_prompt_embeds.index_select(
                    0,
                    indices,
                )
                with pipeline_transformer.cache_context("uncond"):
                    velocity_uncond = transformer(
                        hidden_states=latent_model_input,
                        timestep=transformer_timestep,
                        encoder_hidden_states=partition_negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                partition_velocity = velocity_uncond + current_guidance_scale * (
                    partition_velocity - velocity_uncond
                )
            index_parts.append(indices)
            velocity_parts.append(partition_velocity)

        velocity_dtypes = {part.dtype for part in velocity_parts}
        if len(velocity_dtypes) != 1:
            raise TypeError(
                "Wan2 transformer partitions returned different velocity dtypes: "
                f"{tuple(sorted(str(dtype) for dtype in velocity_dtypes))}"
            )
        partition_order = torch.cat(index_parts)
        restore_order = torch.argsort(partition_order)
        velocity = torch.cat(velocity_parts, dim=0).index_select(0, restore_order)

        # 5. Scheduler step
        output = self.scheduler.step(
            velocity=velocity,
            timestep=scheduler_timestep,
            latents=latents,
            timestep_next=scheduler_timestep_next,
            next_latents=next_latents,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            return_kwargs=return_kwargs,
            noise_level=noise_level,
        )

        return output

    @staticmethod
    def _normalize_forward_timesteps(
        value: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        identifier: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-sample model times and scheduler-compatible original rank.

        Args:
            value: Shared scalar, singleton vector, or per-sample timestep vector.
            batch_size: Latent batch size.
            device: Latent device required by transformer and scheduler operations.
            identifier: Argument name used in validation errors.

        Returns:
            A per-sample vector plus a scalar for shared input or the original
            vector for independently sampled input.

        Raises:
            TypeError: If the timestep is not a tensor.
            ValueError: If its rank, length, or device is incompatible.
        """
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Wan2 forward expected {identifier} to be torch.Tensor, "
                f"received {type(value).__name__}"
            )
        if value.device != device:
            raise ValueError(
                f"Wan2 forward expected {identifier} on {device}, received {value.device}"
            )
        if value.ndim == 0:
            return value.expand(batch_size), value
        if value.ndim != 1:
            raise ValueError(
                f"Wan2 forward expected {identifier} rank 0 or 1, "
                f"received shape {tuple(value.shape)}"
            )
        if value.numel() == 1:
            return value.expand(batch_size), value[0]
        if value.numel() != batch_size:
            raise ValueError(
                f"Wan2 forward expected {identifier} length 1 or batch size {batch_size}, "
                f"received {value.numel()}"
            )
        return value, value
