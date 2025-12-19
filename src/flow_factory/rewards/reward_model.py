# src/flow_factory/rewards/reward_model.py
"""
Base class for reward models.
Provides common interface for all reward models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from PIL import Image

from accelerate import Accelerator
from diffusers.utils.outputs import BaseOutput
from ..hparams import *
from deepspeed.runtime.zero.partition_parameters import GatheredParameters

@dataclass
class RewardModelOutput(BaseOutput):
    """
    Output class for Reward models.
    
    Args:
        rewards: Reward values (can be tensor, numpy array, or list)
        extra_info: Optional additional information
    """
    rewards: Union[torch.FloatTensor, np.ndarray, List[float]]
    extra_info: Optional[Dict[str, Any]] = None


class BaseRewardModel(ABC):
    """
    Abstract Base Class for reward models.
    
    All reward models should inherit from this class and implement
    the __call__ method to compute rewards.
    """
    model : nn.Module = None
    def __init__(self, config: Arguments, accelerator : Accelerator):
        """
        Initialize reward model.
        
        Args:
            reward_args: Reward model configuration
        """
        self.accelerator = accelerator
        reward_args = config.reward_args
        self.reward_args = reward_args
        self.device = reward_args.device
        self.dtype = reward_args.dtype
        print("The deepspeed stage is:", self.accelerator.state.deepspeed_plugin.zero_stage if self.accelerator.state.deepspeed_plugin else "No Deepspeed")

    @abstractmethod
    def __call__(self, **inputs) -> Union[RewardModelOutput, torch.Tensor, np.ndarray, List[float]]:
        """
        Compute reward given inputs.
        
        Args:
            **inputs: Model-specific inputs (e.g., prompts, images)
        
        Returns:
            Rewards (RewardModelOutput, tensor, array, or list)
        """
        pass