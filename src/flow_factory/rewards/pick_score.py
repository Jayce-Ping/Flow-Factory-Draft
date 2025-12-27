# src/flow_factory/rewards/pick_score.py
from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from contextlib import nullcontext
import torch

from .abc import BaseRewardModel, RewardModelOutput
from ..hparams import *


class PickScoreRewardModel(BaseRewardModel):
    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device)
        self.model = self.model.to(dtype=self.dtype)

    @torch.no_grad()
    def __call__(self, prompt : list[str], image : list[Image.Image]):
        if not isinstance(prompt, list):
            prompt = [prompt]

        if not isinstance(image, list):
            image = [image]
            
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
        return scores

def download_model():
    scorer = PickScoreRewardModel(RewardArguments(device='cpu'))

if __name__ == "__main__":
    download_model()