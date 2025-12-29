# src/flow_factory/rewards/my_reward.py
from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel
from typing import Optional, List
from PIL import Image
from contextlib import nullcontext
import torch

from .abc import BaseRewardModel, RewardModelOutput
from ..hparams import *

class MyRewardModel(BaseRewardModel):
    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        # `super().__init__` gives you:
        # self.accelerator = accelerator
        # reward_args = config.reward_args
        # self.reward_args = reward_args
        # self.device = self.accelerator.device if reward_args.device == torch.device('cuda') else reward_args.device
        # self.dtype = reward_args.dtype


        # Implement your custom reward model initialization here
        pass

    @torch.no_grad()
    def compute_rewards(
        self,
        prompt : list[str],
        image : list[Image.Image],
        condition_images: Optional[List[List[Image.Image]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """
        Compute rewards for given prompts and images.
        Args:
            prompt (list[str]): List of text prompts.
            image (list[Image.Image]): List of generated images corresponding to the prompts.
            condition_images (Optional[List[List[Image.Image]]]): Optional list of condition images
                (each element is a list of images. If only one condition image per prompt, this will be a list of single-element lists).
        Returns:
        """

        # Ensure inputs are lists, each of length reward_args.batch_size
        # Implement your custom reward computation here
        rewards = torch.zeros(len(prompt), device=self.device)


        # Wrap rewards in RewardModelOutput
        return RewardModelOutput(
            rewards=rewards,
            extra_info={}, # Add any extra info if needed
        )