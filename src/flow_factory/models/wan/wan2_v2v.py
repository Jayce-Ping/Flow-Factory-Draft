# src/flow_factory/models/wan/wan22_i2v.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from diffusers.pipelines.wan.pipeline_wan_video2video import WanVideoToVideoPipeline
from PIL import Image
import logging
from accelerate import Accelerator
from peft import PeftModel

from ..adapter import BaseAdapter, BaseSample
from ...hparams import *
from ...scheduler import FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteSDESchedulerOutput, set_scheduler_timesteps
from ...utils.base import filter_kwargs
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)



@dataclass
class WanSample(BaseSample):
    pass


class Wan2_I2V_Adapter(BaseAdapter):
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__(config, accelerator)
    
    def load_pipeline(self) -> WanVideoToVideoPipeline:
        return WanVideoToVideoPipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False
        )
    
    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Wan transformer."""
        return []        