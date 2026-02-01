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

# src/flow_factory/models/z_image/z_image.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple, ClassVar, Literal
from dataclasses import dataclass
from PIL import Image
from collections import defaultdict
import logging

import torch
from accelerate import Accelerator
from diffusers.pipelines.z_image.pipeline_z_image_omni import ZImageOmniPipeline

from ..abc import BaseAdapter
from ..samples import T2ISample
from ...hparams import *
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
    SDESchedulerOutput,
    set_scheduler_timesteps
)
from ...utils.image import (
    ImageSingle,
    ImageBatch,
    MultiImageBatch,
    is_image,
    is_image_batch,
    is_multi_image_batch,
    standardize_image_batch,
)
from ...utils.trajectory_collector import (
    TrajectoryCollector, 
    TrajectoryIndicesType, 
    create_trajectory_collector,
)
from ...utils.base import filter_kwargs
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)

CONDITION_IMAGE_SIZE = (1024, 1024)

@dataclass
class ZImageOmniSample(T2ISample):
    # Class var
    _shared_fields: ClassVar[frozenset[str]] = frozenset({})
    # Obj var
    condition_latents: Optional[torch.Tensor] = None
    negative_condition_latents: Optional[torch.Tensor] = None
    condition_siglip_embeds: Optional[torch.Tensor] = None
    negative_condition_siglip_embeds: Optional[torch.Tensor] = None



class ZImageOmniAdapter(BaseAdapter):
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: ZImageOmniPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

        self._has_warned_inference_fallback = False
        self._has_warned_forward_fallback = False

    def load_pipeline(self) -> ZImageOmniPipeline:
        return ZImageOmniPipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False
        )
    
    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Z-Image-Omni transformer."""
        return [
            # TODO
        ]

    # ======================== Encoding / Decoding ======================== 
    # ----------------------- Prompt Encoding -----------------------   
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        num_condition_images: Union[int, List[int]] = 0,
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        device = device or self.pipeline.text_encoder.device

        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = len(prompt)

        # Normalize num_condition_images to per-sample list
        if isinstance(num_condition_images, int):
            num_condition_images = [num_condition_images] * batch_size

        # Format prompts based on per-sample condition image count
        formatted_prompts = []
        for prompt_item, n_cond in zip(prompt, num_condition_images):
            if n_cond == 0:
                formatted_prompts.append(
                    ["<|im_start|>user\n" + prompt_item + "<|im_end|>\n<|im_start|>assistant\n"]
                )
            else:
                prompt_list = ["<|im_start|>user\n<|vision_start|>"]
                prompt_list += ["<|vision_end|><|vision_start|>"] * (n_cond - 1)
                prompt_list += ["<|vision_end|>" + prompt_item + "<|im_end|>\n<|im_start|>assistant\n<|vision_start|>"]
                prompt_list += ["<|vision_end|><|im_end|>"]
                formatted_prompts.append(prompt_list)

        # Flatten for batch tokenization
        flattened_prompt = []
        prompt_list_lengths = []
        for p in formatted_prompts:
            prompt_list_lengths.append(len(p))
            flattened_prompt.extend(p)

        text_inputs = self.tokenizer(
            flattened_prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        # Reconstruct nested structure
        embeddings_list = []
        start_idx = 0
        for length in prompt_list_lengths:
            batch_embeddings = []
            for j in range(start_idx, start_idx + length):
                batch_embeddings.append(prompt_embeds[j][prompt_masks[j]])
            embeddings_list.append(batch_embeddings)
            start_idx += length

        return text_input_ids, embeddings_list

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        max_sequence_length: int = 512,
        num_condition_images: Union[int, List[int]] = 0,
    ) -> Dict[str, Any]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_ids, prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            max_sequence_length=max_sequence_length,
            num_condition_images=num_condition_images,
        )

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = [""] * len(prompt)
            else:
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            assert len(prompt) == len(negative_prompt), "The length of `prompt` and `negative_prompt` must be the same for classifier free guidance."
            negative_prompt_ids, negative_prompt_embeds = self._encode_prompt(
                prompt=negative_prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                num_condition_images=num_condition_images,
            )
        else:
            negative_prompt_embeds = []
        
        return {
            'prompt_ids': prompt_ids,
            'prompt_embeds': prompt_embeds,
            'negative_prompt_ids': negative_prompt_ids if do_classifier_free_guidance else None,
            'negative_prompt_embeds': negative_prompt_embeds if do_classifier_free_guidance else None
        }
    
    # ----------------------- Image Encoding -----------------------
    @staticmethod
    def _is_multi_images_batch(images: Union[ImageBatch, MultiImageBatch]) -> bool:
        return is_multi_image_batch(images)

    def _standardize_image_input(
        self,
        images: Union[ImageSingle, ImageBatch],
        output_type: Literal['pil', 'pt', 'np'] = 'pil',
    ) -> ImageBatch:
        """Standardize image input to PIL format."""
        if isinstance(images, Image.Image):
            images = [images]
        return standardize_image_batch(images, output_type=output_type)

    def _preprocess_condition_images(
        self,
        images: Union[ImageSingle, ImageBatch],
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
    ) -> Tuple[List[Image.Image], List[torch.Tensor]]:
        """
        Preprocess a batch of condition images.
        
        Args:
            images: Single image or list of images
            condition_image_size: Max size constraint.
                - int: 
                - Tuple[int, int]: (max_height, max_width)
        
        Returns:
            resized_images: List[PIL.Image] for siglip encoding
            image_tensors: List[torch.Tensor(1, C, H, W)] for VAE encoding
        """
        if isinstance(condition_image_size, int):
            condition_image_size = (condition_image_size, condition_image_size)

        if isinstance(images, Image.Image):
            images = [images]

        images = self._standardize_image_input(
            images,
            output_type='pil',
        )

        max_area = condition_image_size[0] * condition_image_size[1]

        condition_images_resized = []
        condition_image_tensors = []
        for img in images:
            image_width, image_height = img.size
            if image_width * image_height > max_area:
                img = self.pipeline.image_processor._resize_to_target_area(img, max_area)
                image_width, image_height = img.size

            condition_images_resized.append(img)
            multiple_of = self.pipeline.vae_scale_factor * 2
            image_width = (image_width // multiple_of) * multiple_of
            image_height = (image_height // multiple_of) * multiple_of
            img = self.pipeline.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
            condition_image_tensors.append(img)

        return condition_images_resized, condition_image_tensors
    
    def _prepare_image_latents(
        self,
        images: List[List[torch.Tensor]],
        device : torch.device,
        dtype : torch.dtype,
    ) -> List[List[torch.Tensor]]:
        """
        Encode condition images into latent space using VAE.
        Args:
            images: List of List[torch.Tensor(1, C, H, W)] or torch.Tensor(B, C, H, W)
            device: Target device
            dtype: Target data type
        Returns:
            image_latents: List of List[torch.Tensor(latent_channels, 1, H//vae_scale, W//vae_scale)]
        """

        image_latents = []
        for cond_images in images:
            img_latents = []
            for img in cond_images:
                img = img.to(device=device, dtype=dtype)
                image_latent = (
                    self.pipeline.vae.encode(img.bfloat16()).latent_dist.mode()[0] - self.pipeline.vae.config.shift_factor
                ) * self.pipeline.vae.config.scaling_factor # (latent_channels, H//vae_scale, W//vae_scale)
                image_latent = image_latent.unsqueeze(1).to(dtype) # (latent_channels, 1, H//vae_scale, W//vae_scale)
                img_latents.append(image_latent)

            image_latents.append(img_latents)

        return image_latents # List[List[torch.Tensor(latent_channels, 1, H//vae_scale, W//vae_scale)]]
    
    def _prepare_siglip_embeds(
        self,
        images: List[List[Image.Image]],
        device : torch.device,
        dtype : torch.dtype,
    ) -> List[List[torch.Tensor]]:
        """
        Encode condition images into SigLiP embeddings.
        Args:
            images: List of List[PIL.Image]
            device: Target device
            dtype: Target data type
        Returns:
            siglip_embeds: List of List[torch.Tensor(H_patches, W_patches, hidden_dim)]
        """
        siglip_embeds = []
        for cond_images in images:
            embeds = []
            for img in cond_images:
                siglip_inputs = self.pipeline.siglip_processor(images=[img], return_tensors="pt").to(device)
                shape = siglip_inputs.spatial_shapes[0]
                hidden_state = self.pipeline.siglip(**siglip_inputs).last_hidden_state
                B, N, C = hidden_state.shape # (1, num_patches, hidden_dim)
                hidden_state = hidden_state[:, : shape[0] * shape[1]]
                hidden_state = hidden_state.view(shape[0], shape[1], C) # (H_patches, W_patches, hidden_dim)
                embeds.append(hidden_state.to(dtype))

            siglip_embeds.append(embeds)

        return siglip_embeds # List[List[torch.Tensor(H_patches, W_patches, hidden_dim)]]

    def encode_image(
        self,
        images: Union[ImageBatch, MultiImageBatch],
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        device : Optional[torch.device] = None,
        dtype : Optional[torch.dtype] = None,
        do_classifier_free_guidance: bool = True,
    ) -> Dict[str, Any]:
        """Encode condition images for Z-Image-Omni model."""
        device = device or self.pipeline.device
        dtype = dtype or self.pipeline.text_encoder.dtype
        images = [images] if not self._is_multi_images_batch(images) else images
        condition_image_resized = []
        condition_image_tensors = []
        for img_batch in images:
            condition_image_resized_batch, condition_image_tensors_batch = self._preprocess_condition_images(
                images=img_batch,
                condition_image_size=condition_image_size,
            )
            condition_image_resized.append(condition_image_resized_batch)
            condition_image_tensors.append(condition_image_tensors_batch)

        image_latents = self._prepare_image_latents(
            images=condition_image_tensors,
            device=device,
            dtype=dtype,
        )
        siglip_embeds = self._prepare_siglip_embeds(
            images=condition_image_resized,
            device=device,
            dtype=dtype,
        )

        # Convert back to [0, 1] range tensors for storage
        condition_image_tensors: List[List[torch.Tensor]] = [
            [
                self.pipeline.image_processor.postprocess(img, output_type='pt')[0]
                for img in cond_img_tensors
            ]
            for cond_img_tensors in condition_image_tensors
        ]

        res = {
            'condition_images': condition_image_tensors, # List[List[torch.Tensor(C, H, W)]]
            'condition_latents': image_latents, # List[List[torch.Tensor(latent_channels, 1, H//vae_scale, W//vae_scale)]]
            'condition_siglip_embeds': siglip_embeds, # List[List[torch.Tensor(H_patches, W_patches, hidden_dim)]]
        }

        if do_classifier_free_guidance:
            # Duplicate for negative guidance
            res.update({
                'negative_condition_latents': [[lat.clone() for lat in batch] for batch in res['condition_latents']],
                'negative_condition_siglip_embeds': [[se.clone() for se in batch] for batch in res['condition_siglip_embeds']]
            })
        
        return res
    
    # ----------------------- Video Encoding -----------------------
    def encode_video(
        self,
        videos: Union[torch.Tensor, List[torch.Tensor]],
    ):
        """Not needed for Z-Image-Omni models."""
        pass

    # ----------------------- Decoding -----------------------
    def decode_latents(
        self,
        latents: torch.Tensor,
        output_type: Literal['pil', 'pt', 'np'] = 'pil',
    ) -> torch.Tensor:
        latents = latents.to(self.pipeline.vae.dtype)
        latents = (latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor

        images = self.pipeline.vae.decode(latents, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(images, output_type=output_type)

        return images
    
    # ======================== Inference ========================
    @torch.no_grad()
    def _inference(
        self,
        # Generation parameters
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        height: int = 1024,
        width: int = 1024,
        # Conditioning inputs (raw)
        images: Optional[MultiImageBatch] = None,
        # Prompt
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[List[List[torch.Tensor]]] = None,
        # Negative prompt
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[List[List[torch.Tensor]]] = None,
        # Encoded condition images
        condition_images: Optional[List[List[torch.Tensor]]] = None,
        condition_latents: Optional[List[List[torch.Tensor]]] = None,
        condition_siglip_embeds: Optional[List[List[torch.Tensor]]] = None,
        negative_condition_latents: Optional[List[List[torch.Tensor]]] = None,
        negative_condition_siglip_embeds: Optional[List[List[torch.Tensor]]] = None,
        # CFG options
        cfg_normalization: bool = False,
        cfg_truncation: Optional[float] = 1.0,
        # Other parameters
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        max_sequence_length: int = 512,
        compute_log_prob: bool = True,
        # Extra callback arguments
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = 'all',
    ) -> List[ZImageOmniSample]:
        """Generate images using Z-Image-Omni model."""
        # TODO: Implement inference loop
        pass
    
    # ======================== Forward (Training) ========================
    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: List[torch.FloatTensor],
        # Optional for CFG
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: Optional[float] = 1.0,
        # Next timestep info
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Other
        noise_level: Optional[float] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = ['noise_pred', 'next_latents', 'next_latents_mean', 'std_dev_t', 'dt', 'log_prob'],
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """Perform a forward pass through the Z-Image-Omni model for training."""
        pass