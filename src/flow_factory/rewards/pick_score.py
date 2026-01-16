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

# src/flow_factory/rewards/pick_score.py
from typing import Optional
from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from contextlib import nullcontext
import torch

from .abc import PointwiseRewardModel, GroupwiseRewardModel, RewardModelOutput
from ..hparams import *


class PickScoreRewardModel(PointwiseRewardModel):
    required_fields = ("prompt", "image", "video")
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device)

    @torch.no_grad()
    def __call__(
        self,
        prompt : list[str],
        image : Optional[list[Image.Image]] = None,
        video : Optional[list[list[Image.Image]]] = None,
    ):
        if not isinstance(prompt, list):
            prompt = [prompt]

        # Image and Video can not be provided at the same time
        if image is not None and video is not None:
            raise ValueError("Only one of image or video can be provided.")
        # If video is provided, take the middle frame
        if video is not None:
            mid_index = len(video[0]) // 2
            image = [clip[mid_index] for clip in video]
            
        # Preprocess images
        image_inputs = self.processor(
            images=image,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
        # Preprocess text
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}
        
        # Get embeddings
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate scores
        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = scores.diag()
        # norm to 0-1
        scores = scores/26
        return RewardModelOutput(
            rewards=scores,
            extra_info={},
        )

class PickScoreRankRewardModel(GroupwiseRewardModel):
    """
    Ranking-based reward model using PickScore.
    
    Computes PickScore for each sample in the group, then assigns
    linearly spaced rewards based on rank: (0, 1, ..., n-1) / n.
    """
    required_fields = ("prompt", "image", "video")
    
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device)

    def _compute_scores_batch(
        self,
        prompt: list[str],
        image: list[Image.Image],
    ) -> torch.Tensor:
        """Compute PickScore for a single batch."""
        image_inputs = self.processor(
            images=image,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
        
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}
        
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        
        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs * image_embs).sum(dim=-1)
        return scores

    @torch.no_grad()
    def __call__(
        self,
        prompt: list[str],
        image: Optional[list[Image.Image]] = None,
        video: Optional[list[list[Image.Image]]] = None,
    ) -> RewardModelOutput:
        """
        Compute ranking-based rewards for a group of samples.
        
        Args:
            prompt: List of prompts (one per sample in group)
            image: List of images
            video: List of video frames (will extract middle frame)
            
        Returns:
            RewardModelOutput with rewards in range [0, (n-1)/n]
        """
        if video is not None:
            mid_index = len(video[0]) // 2
            image = [clip[mid_index] for clip in video]
        
        group_size = len(prompt)
        batch_size = self.config.batch_size
        
        # Compute PickScore in batches
        all_scores = []
        for i in range(0, group_size, batch_size):
            batch_scores = self._compute_scores_batch(
                prompt[i:i + batch_size],
                image[i:i + batch_size],
            )
            all_scores.append(batch_scores)
        
        raw_scores = torch.cat(all_scores, dim=0)
        
        # Convert scores to ranks, then to linear rewards: (0, 1, ..., n-1) / n
        ranks = raw_scores.argsort().argsort()
        rewards = ranks.float() / group_size
        
        return RewardModelOutput(
            rewards=rewards,
            extra_info={},
        )

def download_model():
    scorer = PickScoreRewardModel(RewardArguments(device='cpu'), accelerator=None)

if __name__ == "__main__":
    download_model()