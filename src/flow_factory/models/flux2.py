# src/flow_factory/models/flux2.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline, format_input
from diffusers.pipelines.flux2.system_messages import SYSTEM_MESSAGE, SYSTEM_MESSAGE_UPSAMPLING_T2I, SYSTEM_MESSAGE_UPSAMPLING_I2I
from PIL import Image
import logging

from .adapter import BaseAdapter, BaseSample
from ..hparams import *
from ..scheduler import FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteSDESchedulerOutput, set_scheduler_timesteps
from ..utils.base import filter_kwargs, pil_image_to_tensor

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Flux2Sample(BaseSample):
    """Output class for Flux2Adapter models."""
    condition_images : Optional[Union[List[Image.Image], Image.Image]] = None
    text_ids : Optional[torch.Tensor] = None


class Flux2Adapter(BaseAdapter):
    """Concrete implementation for Flow Matching models (FLUX.2)."""
    
    def __init__(self, config: Arguments):
        super().__init__(config)
    
    def load_pipeline(self) -> Flux2Pipeline:
        return Flux2Pipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False
        )

    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Flux.2 DiT."""
        return [
            # --- Double Stream Block Targets ---
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.linear_in", "ff.linear_out", 
            "ff_context.linear_in", "ff_context.linear_out",
            
            # --- Single Stream Block Targets ---
            "attn.to_qkv_mlp_proj", 
            "attn.to_out"
        ]

    # ======================== Encoding & Decoding ========================

    # ------------------------- Text Encoding ------------------------
    def _get_mistral_3_small_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        system_message: str = SYSTEM_MESSAGE,
        hidden_states_layers: List[int] = (10, 20, 30),
    ):
        dtype = self.pipeline.text_encoder.dtype if dtype is None else dtype
        device = self.pipeline.text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Format input messages
        messages_batch = format_input(prompts=prompt, system_message=system_message)

        # Process all messages at once
        inputs = self.pipeline.tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        # Move to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass through the model
        output = self.pipeline.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Only use outputs from intermediate layers and stack them
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return input_ids, prompt_embeds
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        text_encoder_out_layers: List[int] = (10, 20, 30),
    ) -> Dict[str, torch.Tensor]:
        """Encode prompt(s) into embeddings using the Flux.2 text encoder."""
        device = device or self.pipeline.text_encoder.device

        if prompt is None:
            prompt = ""

        prompt = [prompt] if isinstance(prompt, str) else prompt

        prompt_ids, prompt_embeds = self._get_mistral_3_small_prompt_embeds(
            prompt=prompt,
            device=device,
            max_sequence_length=max_sequence_length,
            system_message=self.pipeline.system_message,
            hidden_states_layers=text_encoder_out_layers,
        )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self.pipeline._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return {
            'prompt_ids': prompt_ids,
            'prompt_embeds': prompt_embeds,
            'text_ids': text_ids,
        }

    # ------------------------- Image Encoding ------------------------
    def encode_image(self, image: Union[Image.Image, List[Image.Image]], **kwargs) -> Dict[str, torch.Tensor]:
        """Encode input condition_image(s) into latent representations using the Flux.2 image encoder."""
        device = kwargs.get('device', self.device)
        dtype = kwargs.get('dtype', self.pipeline.vae.dtype)

        image_tensor = pil_image_to_tensor(image=image)
        image_latents, image_latent_ids =  self.pipeline.prepare_image_latents(
            images=image_tensor,
            batch_size=1,
            device=device,
            dtype=dtype
        )
        return {
            'image_latents': image_latents,
            'image_latent_ids': image_latent_ids,
        }
    
    # ------------------------- Video Encoding ------------------------
    def encode_video(self, video: Any, **kwargs) -> None:
        """Flux.2 does not support video encoding."""
        pass

    # ------------------------- Latent Decoding ------------------------
    def decode_latents(self, latents: torch.Tensor, latent_ids, **kwargs) -> List[Image.Image]:
        latents = self.pipeline._unpack_latents_with_ids(latents, latent_ids)

        latents_bn_mean = self.pipeline.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.pipeline.vae.bn.running_var.view(1, -1, 1, 1) + self.pipeline.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self.pipeline._unpatchify_latents(latents)

        images = self.pipeline.vae.decode(latents, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(images, output_type='pil')

        return images
    
    # ========================Preprocessing ========================
    def preprocess_func(
            self,
            prompt: List[str],
            images: Optional[Union[List[Image.Image], List[List[Image.Image]]]] = None,
            caption_upsample_temperature: Optional[float] = None,
            **kwargs
        ) -> Dict[str, Any]:
        """Preprocess inputs for Flux.2 model."""
        assert isinstance(prompt, list), "Prompt must be a batch of strings."
        assert images is None or isinstance(images, list), "Images must be a batch of condition image lists (can be empty)."

        batch_size = len(prompt)
        batch = []
        for p, imgs in zip(
            prompt,
            [None] * len(prompt) if images is None else images
        ):
            if caption_upsample_temperature:
                final_prompt = self.pipeline.upsample_prompt(
                    prompt=p,
                    images=imgs,
                    temperature=caption_upsample_temperature,
                    device=self.pipeline.text_encoder.device
                )
            else:
                final_prompt = p
            batch.append(
                {
                    'prompt': final_prompt,
                    'condition_images': imgs
                }
            )

        

        
    
    # ======================== Sampling / Inference ========================
    def inference(
        self,
        images: Optional[Union[List[Image.Image], Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (10, 20, 30),
        caption_upsample_temperature: float = None,
    ):
        # 1. Setup
        height = height or (self.training_args.resolution[0] if self.training else self.training_args.eval_args.resolution[0])
        width = width or (self.training_args.resolution[1] if self.training else self.training_args.eval_args.resolution[1])
        num_inference_steps = num_inference_steps or (self.training_args.num_inference_steps if self.training else self.training_args.eval_args.num_inference_steps)
        guidance_scale = guidance_scale or (self.training_args.guidance_scale if self.training else self.training_args.eval_args.guidance_scale)
        device = self.device
        dtype = self.transformer.dtype

        # 2. Prepare prompt embeddings
        if caption_upsample_temperature:
            prompt = self.pipeline.upsample_prompt(
                prompt, images=condition_image, temperature=caption_upsample_temperature, device=device
            )

        if prompt_embeds is None:
            encoded = self.encode_prompt(prompt)
            prompt_ids = encoded['prompt_ids'] # Token ids
            prompt_embeds = encoded['prompt_embeds'] # Embeddings
            text_ids = encoded['text_ids'] # Positional ids
        else:
            prompt_embeds = prompt_embeds.to(device)
            text_ids = text_ids.to(device)

        batch_size = len(prompt_embeds)