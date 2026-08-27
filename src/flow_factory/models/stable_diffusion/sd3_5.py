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

# src/flow_factory/models/stable_diffusion/sd3_5.py
from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from PIL import Image

from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline,
)

from ...contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaBinding,
    InputMediaOrder,
    InputMediaSpec,
    MediaFormat,
    MediaType,
    NegativePromptPolicy,
    OutputMediaSequence,
    PipelineIOContract,
    RateRequirement,
)
from ...hparams import *
from ...samples import BaseSample, LatentState
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
    SDESchedulerOutput,
    set_scheduler_timesteps,
)
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

logger = setup_logger(__name__)


_SD3_5_PIPELINE_IO_CONTRACT = PipelineIOContract(
    input_media=InputMediaSpec(
        rules=(),
        binding=InputMediaBinding.GROUPED_BY_TYPE,
        order=InputMediaOrder.INSENSITIVE,
    ),
    negative_prompt=NegativePromptPolicy.OPTIONAL,
    output_media=OutputMediaSequence(
        items=(
            MediaFormat(
                type=MediaType.IMAGE,
                fps=RateRequirement.NOT_APPLICABLE,
                sample_rate=RateRequirement.NOT_APPLICABLE,
            ),
        )
    ),
    geometry_source=GeometrySource.CONFIGURED,
    batch_capability=BatchCapability.UNIFORM,
)


@dataclass
class SD3_5Sample(BaseSample):
    # Class var
    _shared_fields: ClassVar[frozenset[str]] = frozenset({})
    # Obj var
    pooled_prompt_embeds: Optional[torch.Tensor] = None
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None
    pooled_prompt_embeds: Optional[torch.Tensor] = None
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None


@dataclass(frozen=True, slots=True)
class _SD3_5OutputStateCodec:
    """Encode decoded target images into deterministic SD3.5 clean latents."""

    adapter: "SD3_5Adapter"
    required_components: ClassVar[Tuple[str, ...]] = ("vae",)

    def encode_output_state(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> EncodedOutputState:
        """Preprocess and VAE-encode one target batch without caching latents."""
        del condition, generator  # Posterior mode is deterministic and consumes no RNG.
        height, width = self.adapter._configured_output_geometry()
        images = []
        for sample_index, candidate in enumerate(media_batch):
            if len(candidate) != 1:
                raise ValueError(
                    "SD3.5 output codec expected one image per sample, "
                    f"received {len(candidate)} for sample {sample_index}"
                )
            image = candidate[0].payload
            if not isinstance(image, Image.Image):
                raise TypeError(
                    "SD3.5 output codec expected decoded PIL.Image targets, "
                    f"received {type(image).__name__} for sample {sample_index}"
                )
            images.append(image)

        pixel_values = self.adapter.pipeline.image_processor.preprocess(
            images,
            height=height,
            width=width,
        )
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(
                "SD3.5 image_processor.preprocess expected torch.Tensor output, "
                f"received {type(pixel_values).__name__}"
            )
        if pixel_values.ndim != 4:
            raise ValueError(
                "SD3.5 image_processor.preprocess expected rank-4 BCHW output, "
                f"received shape {tuple(pixel_values.shape)}"
            )
        if pixel_values.shape[0] != len(images):
            raise ValueError(
                "SD3.5 image_processor.preprocess changed target batch size: "
                f"expected {len(images)}, received shape {tuple(pixel_values.shape)}"
            )
        if tuple(pixel_values.shape[-2:]) != (height, width):
            raise ValueError(
                "SD3.5 image_processor.preprocess did not produce configured geometry "
                f"{(height, width)}, received shape {tuple(pixel_values.shape)}"
            )

        vae = self.adapter.vae
        vae_dtype = getattr(vae, "dtype", None)
        if not isinstance(vae_dtype, torch.dtype) or not vae_dtype.is_floating_point:
            raise TypeError(
                "SD3.5 output codec expected VAE to expose a floating dtype, "
                f"received {vae_dtype!r}"
            )
        pixel_values = pixel_values.to(device=self.adapter.device, dtype=vae_dtype)
        encoder_output = vae.encode(pixel_values)
        latent_dist = self._resolve_latent_distribution(encoder_output)
        mode = getattr(latent_dist, "mode", None)
        latents = mode() if callable(mode) else mode
        if not isinstance(latents, torch.Tensor):
            raise TypeError(
                "SD3.5 VAE latent_dist.mode expected torch.Tensor, "
                f"received {type(latents).__name__}"
            )

        shift_factor, scaling_factor = self._normalization_factors(vae)
        latents = (latents - shift_factor) * scaling_factor
        signature = GeometrySignature(
            media=(
                MediaGeometrySignature(
                    type=MediaType.IMAGE,
                    height=height,
                    width=width,
                ),
            )
        )
        return EncodedOutputState(
            clean_state=LatentState({"latent": latents}),
            forward_context={},
            decode_context={"height": height, "width": width},
            geometry_signatures=tuple(signature for _ in images),
        )

    @staticmethod
    def _resolve_latent_distribution(encoder_output: Any) -> Any:
        """Resolve current and tuple-style diffusers VAE encoder outputs."""
        latent_dist = getattr(encoder_output, "latent_dist", None)
        if latent_dist is not None:
            return latent_dist
        if isinstance(encoder_output, (tuple, list)) and len(encoder_output) == 1:
            candidate = encoder_output[0]
            return getattr(candidate, "latent_dist", candidate)
        if getattr(encoder_output, "mode", None) is not None:
            return encoder_output
        raise TypeError(
            "SD3.5 VAE encode expected an output exposing latent_dist or a single "
            f"latent-distribution tuple, received {type(encoder_output).__name__}"
        )

    @staticmethod
    def _normalization_factors(vae: Any) -> Tuple[float, float]:
        """Return finite SD3 VAE shift and positive scale factors."""
        config = getattr(vae, "config", None)
        shift_factor = getattr(config, "shift_factor", None)
        scaling_factor = getattr(config, "scaling_factor", None)
        for name, value in (
            ("shift_factor", shift_factor),
            ("scaling_factor", scaling_factor),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"SD3.5 VAE config expected numeric {name}, "
                    f"received {type(value).__name__}: {value!r}"
                )
            if not math.isfinite(float(value)):
                raise ValueError(f"SD3.5 VAE config expected finite {name}, received {value!r}")
        if scaling_factor <= 0:
            raise ValueError(
                f"SD3.5 VAE config expected scaling_factor > 0, received {scaling_factor!r}"
            )
        return float(shift_factor), float(scaling_factor)


class SD3_5Adapter(BaseAdapter):
    """Concrete implementation for Stable Diffusion 3 medium."""

    pipeline_io_contract = _SD3_5_PIPELINE_IO_CONTRACT

    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: StableDiffusion3Pipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

    def load_pipeline(self) -> StableDiffusion3Pipeline:
        return StableDiffusion3Pipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False,
        )

    # ============================ Modules & Components ============================
    @property
    def default_target_modules(self) -> List[str]:
        return [
            # Attention modules
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
        ]

    @property
    def tokenizer(self) -> Any:
        return self.pipeline.tokenizer_3

    def build_output_state_codec(self) -> OutputStateCodec:
        """Build the deterministic on-the-fly SD3.5 target-image encoder."""
        self._configured_output_geometry()
        return _SD3_5OutputStateCodec(self)

    def _configured_output_geometry(self) -> Tuple[int, int]:
        """Return strictly validated configured training image geometry."""
        geometry = []
        for name in ("height", "width"):
            value = getattr(self.training_args, name, None)
            if type(value) is not int:
                raise TypeError(
                    f"SD3.5 output geometry expected training_args.{name} to be int, "
                    f"received {type(value).__name__}: {value!r}"
                )
            if value <= 0:
                raise ValueError(
                    f"SD3.5 output geometry expected training_args.{name} > 0, received {value}"
                )
            geometry.append(value)
        return geometry[0], geometry[1]

    def _validate_encoded_output_geometry(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        encoded: EncodedOutputState,
    ) -> None:
        """Verify every codec signature and decode field against configured H/W."""
        del condition
        height, width = self._configured_output_geometry()
        expected_signature = GeometrySignature(
            media=(
                MediaGeometrySignature(
                    type=MediaType.IMAGE,
                    height=height,
                    width=width,
                ),
            )
        )
        if len(encoded.geometry_signatures) != len(media_batch):
            raise ValueError(
                "SD3.5 output geometry expected one signature per target sample, "
                f"received {len(encoded.geometry_signatures)} for batch size {len(media_batch)}"
            )
        for sample_index, signature in enumerate(encoded.geometry_signatures):
            if signature != expected_signature:
                raise ValueError(
                    "SD3.5 encoded output geometry disagrees with configured "
                    f"height/width {(height, width)} for sample {sample_index}: {signature!r}"
                )
        decode_context = encoded.decode_context
        expected_decode_context = {"height": height, "width": width}
        has_exact_decode_geometry = (
            set(decode_context) == set(expected_decode_context)
            and type(decode_context["height"]) is int
            and type(decode_context["width"]) is int
            and decode_context["height"] == height
            and decode_context["width"] == width
        )
        if not has_exact_decode_geometry:
            raise ValueError(
                "SD3.5 decode_context must exactly match configured output geometry "
                f"{expected_decode_context}, received {dict(decode_context)!r}"
            )

    # ============================ Encoding & Decoding ============================
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 3.5,
        max_sequence_length: Optional[int] = 512,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode the prompt(s) into embeddings using the pipeline's text encoder."""
        device = device if device is not None else self.device
        do_classifier_free_guidance = guidance_scale > 1.0
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
        )
        result = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # Token ids for downstream bookkeeping (used as `prompt_ids` in samples)
        result["prompt_ids"] = text_inputs.input_ids.to(device)

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = (
                    [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
                )
                negative_prompt = negative_prompt * (
                    len(prompt) // len(negative_prompt)
                )  # Expand to match batch size
            assert len(prompt) == len(
                negative_prompt
            ), "The number of negative prompts must match the number of prompts."
            result["negative_prompt_embeds"] = negative_prompt_embeds
            result["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

            negative_text_inputs = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            result["negative_prompt_ids"] = negative_text_inputs.input_ids.to(device)

        return result

    def encode_image(self, images: Union[Image.Image, torch.Tensor, List[torch.Tensor]]):
        """Not needed for SD3 text-to-image models."""
        pass

    def encode_video(self, videos: Union[torch.Tensor, List[torch.Tensor]]):
        """Not needed for SD3 text-to-image models."""
        pass

    def decode_latents(
        self,
        latents: torch.Tensor,
        output_type: Literal["pil", "pt", "np"] = "pil",
    ) -> torch.Tensor:
        latents = latents.to(self.pipeline.vae.dtype)
        latents = (
            latents / self.pipeline.vae.config.scaling_factor
        ) + self.pipeline.vae.config.shift_factor

        images = self.pipeline.vae.decode(latents, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(images, output_type=output_type)

        return images

    # ============================ Inference ============================
    @torch.no_grad()
    def inference(
        self,
        # Oridinary args
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        # Encoded Prompt
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # Encoded Negative Prompt
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # Other args
        compute_log_prob: bool = True,
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
    ) -> List[SD3_5Sample]:
        # 1. Setup
        device = self.device
        dtype = self.pipeline.transformer.dtype

        do_classifier_free_guidance = guidance_scale > 1.0
        has_negative_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        if do_classifier_free_guidance and not has_negative_prompt:
            logger.warning(
                "No negative prompt/embeds provided, classifier-free-guidance will be disabled."
            )
            do_classifier_free_guidance = False

        # 2. Encode prompt
        if prompt_embeds is None or pooled_prompt_embeds is None:
            encoded = self.encode_prompt(
                prompt,
                negative_prompt,
                guidance_scale=guidance_scale,
                device=device,
            )
            prompt_embeds = encoded["prompt_embeds"]
            pooled_prompt_embeds = encoded["pooled_prompt_embeds"]
            prompt_ids = encoded["prompt_ids"]
            if do_classifier_free_guidance:
                negative_prompt_embeds = encoded["negative_prompt_embeds"]
                negative_prompt_ids = encoded["negative_prompt_ids"]
                negative_pooled_prompt_embeds = encoded["negative_pooled_prompt_embeds"]
        else:
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
            if do_classifier_free_guidance:
                negative_prompt_embeds = negative_prompt_embeds.to(device)
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device)

        batch_size = len(prompt_embeds)
        num_channels_latents = self.pipeline.transformer.config.in_channels

        # 3. Prepare latent variables
        latents = self.pipeline.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
        )
        # latents : torch.Tensor of shape (B, C, H/8, W/8), not packed

        # 5. Prepare noise schedule
        image_seq_len = (latents.shape[2] // self.pipeline.transformer.config.patch_size) * (
            latents.shape[3] // self.pipeline.transformer.config.patch_size
        )
        timesteps = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=image_seq_len,
            device=device,
        )

        # 6. Denosing loop
        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latents = self.cast_latents(latents, default_dtype=dtype)
        latent_collector.collect(latents, step_idx=0)
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(
                trajectory_indices, num_inference_steps
            )
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        for i, t in enumerate(timesteps):
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
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                guidance_scale=guidance_scale,
                joint_attention_kwargs=joint_attention_kwargs,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=return_kwargs,
                noise_level=current_noise_level,
            )

            latents = self.cast_latents(output.next_latents, default_dtype=dtype)
            latent_collector.collect(latents, i + 1)
            if current_compute_log_prob:
                log_prob_collector.collect(output.log_prob, i)

            callback_collector.collect_step(
                step_idx=i,
                output=output,
                keys=extra_call_back_kwargs,
                capturable={"noise_level": current_noise_level},
            )

        # 7. Decode latents
        images = self.decode_latents(latents=latents, output_type="pt")

        # 8. Create samples
        extra_call_back_res = callback_collector.get_result()  # (B, len(trajectory_indices), ...)
        callback_index_map = callback_collector.get_index_map()  # (T,) LongTensor
        all_latents = latent_collector.get_result()  # List[torch.Tensor(B, ...)]
        latent_index_map = latent_collector.get_index_map()  # (T+1,) LongTensor
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        log_prob_index_map = log_prob_collector.get_index_map() if compute_log_prob else None
        samples = [
            SD3_5Sample(
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
                # Prompt
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b] if prompt_embeds is not None else None,
                pooled_prompt_embeds=(
                    pooled_prompt_embeds[b] if pooled_prompt_embeds is not None else None
                ),
                # Negative Prompt
                negative_prompt=negative_prompt[b] if negative_prompt is not None else None,
                negative_prompt_ids=(
                    negative_prompt_ids[b] if negative_prompt_ids is not None else None
                ),
                negative_prompt_embeds=(
                    negative_prompt_embeds[b] if negative_prompt_embeds is not None else None
                ),
                negative_pooled_prompt_embeds=(
                    negative_pooled_prompt_embeds[b]
                    if negative_pooled_prompt_embeds is not None
                    else None
                ),
                # Image & metadata
                height=height,
                width=width,
                image=images[b],
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

    # ============================ Training Forward ============================
    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        # Optional for CFG
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 7.5,
        # Next timestep info
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Other
        noise_level: Optional[float] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = [
            "velocity",
            "next_latents",
            "next_latents_mean",
            "std_dev_t",
            "dt",
            "log_prob",
        ],
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """
        Core forward pass for T2I generation.

        Args:
            t: Current timestep tensor.
            t_next: Next timestep tensor.
            latents: Current latent representations (B, C, H, W).
            prompt_embeds: Text prompt embeddings.
            pooled_prompt_embeds: Pooled text prompt embeddings.
            negative_prompt_embeds: Optional negative prompt embeddings (for CFG).
            negative_pooled_prompt_embeds: Optional negative pooled prompt embeddings.
            guidance_scale: CFG scale factor.
            next_latents: Optional target latents for log-prob computation.
            joint_attention_kwargs: Optional kwargs for attention layers.
            compute_log_prob: Whether to compute log probabilities.
            return_kwargs: List of outputs to return.
            noise_level: Current noise level for SDE sampling.

        Returns:
            FlowMatchEulerDiscreteSDESchedulerOutput containing requested outputs.
        """
        # 1. Prepare variables
        batch_size = latents.shape[0]
        timestep = t.expand(batch_size).to(latents.dtype)

        # Auto-detect CFG
        if guidance_scale > 1.0 and (
            negative_prompt_embeds is None or negative_pooled_prompt_embeds is None
        ):
            logger.warning(
                "Passed `guidance_scale` > 1.0, but no `negative_prompt_embeds` or "
                "`negative_pooled_prompt_embeds` provided. Classifier-free guidance will be disabled."
            )
        do_classifier_free_guidance = (
            negative_prompt_embeds is not None
            and negative_pooled_prompt_embeds is not None
            and guidance_scale > 1.0
        )

        # 2. Prepare inputs for CFG
        if do_classifier_free_guidance:
            prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds_input = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )
            latents_input = torch.cat([latents, latents], dim=0)
            timestep_input = timestep.repeat(2)
        else:
            prompt_embeds_input = prompt_embeds
            pooled_prompt_embeds_input = pooled_prompt_embeds
            latents_input = latents
            timestep_input = timestep

        # 3. Transformer forward pass
        velocity = self.transformer(
            hidden_states=latents_input,
            timestep=timestep_input,
            encoder_hidden_states=prompt_embeds_input,
            pooled_projections=pooled_prompt_embeds_input,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]

        # 4. Apply CFG
        if do_classifier_free_guidance:
            velocity_uncond, velocity_text = velocity.chunk(2)
            velocity = velocity_uncond + guidance_scale * (velocity_text - velocity_uncond)

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
