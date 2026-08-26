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

"""SenseNova's explicit pseudo-pipeline and differentiable denoiser wrapper.

SenseNova-U1 is distributed as a custom Transformers composite model rather than
as a diffusers pipeline.  The wrapper below keeps the official model intact while
making the image-generation path an explicit Flow-Factory component.  In
particular, the wrapper owns the gradient-bearing denoising forward so the
component runtime can route it through DDP/FSDP/DeepSpeed and PEFT.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .modeling.neo_unify.modeling_neo_chat import NEOChatModel


class SenseNovaDenoiser(nn.Module):
    """Expose SenseNova's image flow-matching prediction as one trainable module."""

    def __init__(self, model: NEOChatModel):
        super().__init__()
        self.model = model

    @property
    def config(self):
        """Delegate the checkpoint config for adapter/runtime introspection."""
        return self.model.config

    @property
    def device(self) -> torch.device:
        """Expose the underlying model device like a diffusers component."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Expose the underlying model dtype like a diffusers component."""
        return next(self.parameters()).dtype

    def forward(
        self,
        *,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        past_key_values: Any,
        indexes_image: torch.Tensor,
        attention_mask: Dict[str, Any],
        image_size: tuple[int, int],
        noise_scale: float,
    ) -> torch.Tensor:
        """Predict the native SenseNova clean-minus-noise velocity.

        The Flow-Factory adapter calls this with a single image at a time.  The
        single-image contract keeps the official variable-length text cache
        semantics exact and avoids padding-dependent changes to the RoPE indexes.
        """
        if latents.ndim != 4 or latents.shape[0] != 1:
            raise ValueError(
                "SenseNovaDenoiser expects latents with shape (1, 3, H, W); "
                f"received {tuple(latents.shape)}."
            )

        model = self.model
        merge_size = int(1 / model.downsample_ratio)
        patch_size = model.patch_size
        token_h = image_size[1] // (patch_size * merge_size)
        token_w = image_size[0] // (patch_size * merge_size)
        grid_h = image_size[1] // patch_size
        grid_w = image_size[0] // patch_size
        image_token_num = token_h * token_w

        z = model.patchify(latents, patch_size * merge_size)
        image_input = model.patchify(latents, patch_size, channel_first=True)
        grid_hw = torch.tensor([[grid_h, grid_w]], device=latents.device, dtype=torch.long)
        image_embeds = model.extract_feature(
            image_input.view(grid_h * grid_w, -1),
            gen_model=True,
            grid_hw=grid_hw,
        ).view(1, image_token_num, -1)

        native_t = timestep.reshape(-1)
        if native_t.numel() != 1:
            raise ValueError(
                "SenseNovaDenoiser expects one native timestep per call; "
                f"received shape {tuple(timestep.shape)}."
            )
        t_expanded = native_t.expand(image_token_num)
        timestep_embeddings = model.fm_modules["timestep_embedder"](t_expanded).view(
            1, image_token_num, -1
        )
        if model.add_noise_scale_embedding:
            noise_scale_tensor = torch.full_like(
                t_expanded, noise_scale / model.noise_scale_max_value
            )
            timestep_embeddings = timestep_embeddings + model.fm_modules["noise_scale_embedder"](
                noise_scale_tensor
            ).view(1, image_token_num, -1)
        image_embeds = image_embeds + timestep_embeddings

        outputs = model.language_model.model(
            inputs_embeds=image_embeds,
            image_gen_indicators=torch.ones(
                (1, image_token_num), dtype=torch.bool, device=latents.device
            ),
            indexes=indexes_image,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            update_cache=False,
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state[:, -image_token_num:]

        if model.use_pixel_head:
            hidden_2d = hidden_states.view(1, token_h, token_w, -1)
            hidden_2d = torch.einsum("b h w c -> b c h w", hidden_2d).contiguous()
            smoothed = model.fm_modules["fm_head"](hidden_2d)
            smoothed = smoothed.view(
                1,
                3,
                token_h,
                patch_size * merge_size,
                token_w,
                patch_size * merge_size,
            )
            smoothed = torch.einsum("b c h p w q -> b h w p q c", smoothed)
            x_pred = smoothed.contiguous().view(1, image_token_num, -1)
        elif model.use_deep_fm_head:
            x_pred = model.fm_modules["fm_head"](
                hidden_states.reshape(image_token_num, -1),
                native_t.expand(image_token_num),
            ).view(1, image_token_num, -1)
        else:
            x_pred = model.fm_modules["fm_head"](hidden_states).view(1, image_token_num, -1)

        return (x_pred - z) / (1 - native_t).clamp_min(model.config.t_eps)


class SenseNovaPseudoPipeline:
    """Diffusers-like container for the custom SenseNova checkpoint."""

    def __init__(
        self,
        model: NEOChatModel,
        tokenizer: Any,
        scheduler: Optional[Any] = None,
    ):
        self.model = model
        self.transformer = SenseNovaDenoiser(model)
        self.tokenizer = tokenizer
        self.scheduler = scheduler

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype

    @property
    def components(self) -> Dict[str, nn.Module]:
        """Return the one canonical trainable component."""
        return {"transformer": self.transformer}

    def maybe_free_model_hooks(self) -> None:
        """Compatibility no-op; SenseNova has no diffusers hooks."""

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        low_cpu_mem_usage: bool = False,
        **kwargs: Any,
    ) -> "SenseNovaPseudoPipeline":
        """Load a SenseNova-U1 checkpoint from the Hub or a local directory."""
        tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {}) or {}
        model = NEOChatModel.from_pretrained(
            model_path,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=True,
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, **tokenizer_kwargs
        )
        return cls(model=model, tokenizer=tokenizer)
