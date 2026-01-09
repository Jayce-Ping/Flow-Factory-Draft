# src/flow_factory/models/wan/wan2_i2v.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from PIL import Image
import torch
from accelerate import Accelerator
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from peft import PeftModel

from ..adapter import BaseAdapter
from ..samples import ImageConditionSample
from ...hparams import *
from ...scheduler import SDESchedulerOutput, set_scheduler_timesteps
from ...utils.base import filter_kwargs
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class WanSample(ImageConditionSample):
    pass


class Wan2_I2V_Adapter(BaseAdapter):
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__(config, accelerator)
    
    def load_pipeline(self) -> WanImageToVideoPipeline:
        return WanImageToVideoPipeline.from_pretrained(
            self.model_args.model_name_or_path,
        )
    
    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Wan transformer."""
        return [
            # --- Self Attention ---
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            
            # --- Cross Attention ---
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",

            # --- Feed Forward Network ---
            "ffn.net.0.proj", "ffn.net.2"
        ]
    
    def apply_lora(
        self,
        target_modules: Union[str, List[str]],
        components: Union[str, List[str]] = ['transformer', 'transformer_2'],
    ) -> Union[PeftModel, Dict[str, PeftModel]]:
        return super().apply_lora(target_modules=target_modules, components=components)