# src/flow_factory/models/sd3.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from PIL import Image
import logging

from ...hparams import *
from ..adapter import BaseAdapter, BaseSample
from ...scheduler.flow_matching import FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteSDESchedulerOutput, set_scheduler_timesteps
from ...utils.base import filter_kwargs


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SD3_5Sample(BaseSample):
    pooled_prompt_embeds : Optional[torch.Tensor] = None
    negative_pooled_prompt_embeds : Optional[torch.Tensor] = None

class SD3_5Adapter(BaseAdapter):
    """Concrete implementation for Stable Diffusion 3 medium."""
    def __init__(self, config: Arguments):
        super().__init__(config)
        self.tokenizer_3 = self.pipeline.tokenizer_3 

    def load_pipeline(self) -> StableDiffusion3Pipeline:
        return StableDiffusion3Pipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False
        )

    @property
    def default_target_modules(self) -> List[str]:
        return [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = False,
        **kwargs
    ):
        device = self.device
        (
            prompt_embeds, 
            negative_prompt_embeds, 
            pooled_prompt_embeds, 
            negative_pooled_prompt_embeds
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt
        )
        result = {
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
        }

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # Token ids for downstream bookkeeping (used as `prompt_ids` in samples)
        result['prompt_ids'] = text_inputs.input_ids

        if do_classifier_free_guidance:
            result["negative_prompt_embeds"] = negative_prompt_embeds
            result["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

            negative_text_inputs = self.tokenizer_3(
                negative_prompt,
                padding="max_length",
                max_length=256,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            result['negative_prompt_ids'] = negative_text_inputs.input_ids

        return result

    def encode_image(self, image: Union[Image.Image, torch.Tensor, List[torch.Tensor]], **kwargs):
        return self.pipeline.encode_image(image=image, device=self.device,**kwargs)

    def encode_video(self, video: Union[torch.Tensor, List[torch.Tensor]], **kwargs) -> torch.Tensor:
        """Not needed for FLUX text-to-image models."""
        pass

    def decode_latents(
        self,
        latents: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        latents = latents.to(self.pipeline.vae.dtype)
        latents = (latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor

        images = self.pipeline.vae.decode(latents, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(images, output_type="pil")

        return images


    @torch.no_grad()
    def inference(
        self,
        prompt: Union[str, List[str]],
        prompt_ids : Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        do_classifier_free_guidance: Optional[bool] = False,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        num_images_per_prompt: Optional[int] = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_log_probs: bool = True,
    ):
        # Setup
        height = height or (self.training_args.resolution[0] if self.training else self.training_args.eval_args.resolution[0])
        width = width or (self.training_args.resolution[1] if self.training else self.training_args.eval_args.resolution[1])
        num_inference_steps = num_inference_steps or (self.training_args.num_inference_steps if self.training else self.training_args.eval_args.num_inference_steps)
        guidance_scale = guidance_scale or (self.training_args.guidance_scale if self.training else self.training_args.eval_args.guidance_scale)
        device = self.device
        dtype = self.transformer.dtype

        # encode prompt
        if prompt_embeds is None:
            encoded = self.encode_prompt(prompt, negative_prompt, do_classifier_free_guidance)
            prompt_embeds = encoded['prompt_embeds']
            pooled_prompt_embeds = encoded['pooled_prompt_embeds']
            # Always capture token ids for bookkeeping (may be needed for advantage grouping)
            prompt_ids = encoded.get('prompt_ids')
            if do_classifier_free_guidance:
                negative_prompt_embeds = encoded['negative_prompt_embeds']
                negative_prompt_ids = encoded.get('negative_prompt_ids')
                negative_pooled_prompt_embeds = encoded['negative_pooled_prompt_embeds']
        else:
            if do_classifier_free_guidance:
                if negative_prompt_embeds is None:
                    raise ValueError(
                        "When using CFG with provided prompt_embeds, "
                        "you must also provide negative_prompt_embeds"
                    )
                negative_prompt_embeds = negative_prompt_embeds.to(device)
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device)
            else:
                negative_prompt_embeds = None
                negative_pooled_prompt_embeds = None
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)

        # If token ids were not provided (e.g., caller passed prompt_embeds directly),
        # fall back to tokenizing `prompt` so `prompt_ids` are available for grouping.
        if prompt_ids is None and prompt is not None:
            text_inputs = self.tokenizer_3(
                prompt,
                padding="max_length",
                max_length=256,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            prompt_ids = text_inputs.input_ids

        batch_size = len(prompt_embeds)
        num_channels_latents = self.transformer.config.in_channels

        if do_classifier_free_guidance:
            prompt_embeds = torch.concat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.concat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # prepare latent variables
        latents = self.pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
        )

        # prepare noise schedule
        timesteps = set_scheduler_timesteps(
            scheduler=self.pipeline.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latents.shape[1],
            device=device,
        )
        # denosing loop
        all_latents = [latents]
        all_log_probs = [] if compute_log_probs else None

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # x_t -> x_t-1
            latents_dtype = latents.dtype
            output = self.scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=latents,
                compute_log_prob=compute_log_probs and current_noise_level > 0,
            )

            latents = output.prev_sample.to(dtype)
            all_latents.append(latents)

            if compute_log_probs:
                all_log_probs.append(output.log_prob)

        # decode latents
        images = self.decode_latents(latents=latents)

        # create samples
        samples = [
            SD3_5Sample(
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0),
                timesteps=timesteps,
                height=height,
                width=width,
                image=images[b],
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b] if prompt_embeds is not None else None,
                pooled_prompt_embeds=pooled_prompt_embeds[b] if pooled_prompt_embeds is not None else None,
                negative_prompt=negative_prompt[b] if negative_prompt is not None else None,
                negative_prompt_ids=negative_prompt_ids[b] if negative_prompt_ids is not None else None,
                negative_prompt_embeds=negative_prompt_embeds[b] if negative_prompt_embeds is not None else None,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds[b] if negative_pooled_prompt_embeds is not None else None,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if compute_log_probs else None,
                extra_kwargs={
                    'guidance_scale': guidance_scale,
                    'do_classifier_free_guidance': do_classifier_free_guidance,
                    },                
            )
            for b in range(batch_size)
        ]
        # print(samples)
        return samples
    
    def forward(
        self,
        samples : List[SD3_5Sample],
        timestep_index : int,
        compute_log_prob : bool = True,
        **kwargs
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """Compute log-probabilities for training."""

        batch_size = len(samples)
        device = self.device
        guidance_scale = [
            s.extra_kwargs.get('guidance_scale', self.training_args.guidance_scale)
            for s in samples
        ]        

        latents = torch.stack([s.all_latents[timestep_index] for s in samples], dim=0).to(device)
        next_latents = torch.stack([s.all_latents[timestep_index + 1] for s in samples], dim=0).to(device)
        timestep = torch.stack([s.timesteps[timestep_index] for s in samples], dim=0).to(device)  
        do_classifier_free_guidance = all(
            s.extra_kwargs.get('do_classifier_free_guidance', False)
            for s in samples
        )

        prompt_embeds = torch.stack([s.prompt_embeds for s in samples], dim=0).to(device)
        pooled_prompt_embeds = torch.stack([s.pooled_prompt_embeds for s in samples], dim=0).to(device)    
        negative_prompt_embeds = torch.stack([s.negative_prompt_embeds for s in samples], dim=0).to(device) if do_classifier_free_guidance else []
        negative_pooled_prompt_embeds = torch.stack([s.negative_pooled_prompt_embeds for s in samples], dim=0).to(device) if do_classifier_free_guidance else []

        _ = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=self.training_args.num_inference_steps,
            seq_len=latents.shape[1],
            device=device
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.concat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.concat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            latents_input = torch.concat([latents, latents], dim=0)

        else:
            latents_input = latents
        
        guidance = torch.as_tensor(guidance_scale, device=device, dtype=torch.float32)
        # Forward pass
        noise_pred = self.transformer(
            hidden_states=latents_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale[0] * (noise_pred_text - noise_pred_uncond)

        # Perform scheduler step
        step_kwargs = filter_kwargs(self.scheduler.step, **kwargs)
        output = self.scheduler.step(
            model_output=noise_pred,
            timestep=timestep,
            sample=latents,
            prev_sample=next_latents,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            **step_kwargs,
        )
        return output