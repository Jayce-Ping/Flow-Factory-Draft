# src/flow_factory/rewards/sudoku.py
from accelerate import Accelerator
from typing import Optional, List, Union
from PIL import Image
import torch
import copy
import numpy as np

from .abc import BaseRewardModel, RewardModelOutput
from ..hparams import *


class SudokuRewardModel(BaseRewardModel):
    def __init__(self, reward_args: RewardArguments, accelerator: Accelerator):
        super().__init__(reward_args, accelerator)
        self.size = 9
        self.img_size = 512
        self.cell_size = self.img_size / self.size
        self.ocr = None
        self._init_ocr()

    def _init_ocr(self):
        if self.ocr is None:
            from easyocr import Reader
            if self.accelerator.is_main_process:
                self.ocr = Reader(['en'], gpu=torch.cuda.is_available())
            self.accelerator.wait_for_everyone()
            if self.ocr is None:
                self.ocr = Reader(['en'], gpu=torch.cuda.is_available())
            self.accelerator.wait_for_everyone()

    def _to_pil(self, img: Union[Image.Image, torch.Tensor, np.ndarray, List]) -> List[Image.Image]:
        """Convert tensor/ndarray/PIL (or list of them) to list of PIL Images."""
        if isinstance(img, list):
            return sum([self._to_pil(x) for x in img], [])
        if isinstance(img, Image.Image):
            return [img]
        
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        
        img = img.astype(np.float32)
        
        # Ensure (B, H, W, C)
        if img.ndim == 2:
            img = img[None, ..., None]
        elif img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[-1]:
                img = np.transpose(img, (1, 2, 0))
            img = img[None]
        elif img.ndim == 4 and img.shape[1] in (1, 3, 4) and img.shape[1] < img.shape[-1]:
            img = np.transpose(img, (0, 2, 3, 1))
        
        # Normalize to [0, 255]
        vmin, vmax = img.min(), img.max()
        if vmin >= -1.0 and vmax <= 1.0 and vmin < 0:
            img = (img + 1) * 127.5
        elif vmax <= 1.0:
            img = img * 255
        
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return [Image.fromarray(x.squeeze(-1) if x.shape[-1] == 1 else x) 
                for x in img]

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[Union[List[Image.Image], torch.Tensor]]] = None,
    ) -> RewardModelOutput:
        """
        Compute rewards for given prompts and images.
        Args:
            prompt (list[str]): List of text prompts.
            image (list[Image.Image]): List of generated images corresponding to the prompts.
            video (list[list[Image.Image]]): List of generated videos (each video is a list of frames) corresponding to the prompts.
            condition_images (Optional[List[List[Image.Image] | torch.Tensor]]): Optional list of condition images
                - each element is a list of images. If only one condition image per prompt, this will be a list of single-element lists.
                - each element is a tensor with batch dimension, scaled in [0, 1].
        Returns:
            RewardModelOutput: Contains rewards tensor and any extra information.
        """
        self._init_ocr()

        condition_images = [self._to_pil(cond_imgs) for cond_imgs in condition_images]
        
        batch_size = len(prompt)
        rewards = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            puzzle_img = condition_images[i][0]  # Original puzzle
            solution_img = image[i]              # Generated solution
            
            puzzle_grid = self._parse_image(puzzle_img)
            solution_grid = self._parse_image(solution_img)
            
            a1, a2, reward = self._compute_single_reward(puzzle_grid, solution_grid)
            rewards[i] = reward
        
        return RewardModelOutput(rewards=rewards, extra_info={})
    
    def _parse_image(self, img: Image.Image) -> List[List[int]]:
        """Parse sudoku image to 9x9 grid."""
        img = img.convert('RGB')
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        
        grid = [[0] * 9 for _ in range(9)]
        cs = self.cell_size
        
        for idx in range(81):
            row, col = divmod(idx, 9)
            # Crop cell with padding
            x1, y1 = int(col * cs) + 2, int(row * cs) + 2
            x2, y2 = int((col + 1) * cs) - 2, int((row + 1) * cs) - 2
            cell = np.array(img.crop((x1, y1, x2, y2)))
            
            results = self.ocr.readtext(cell, detail=0, paragraph=False, allowlist='0123456789')
            if results:
                for text in results:
                    if text.isdigit() and text != '0':
                        grid[row][col] = int(text)
                        break
        return grid

    def _compute_single_reward(self, puzzle: List[List[int]], solution: List[List[int]]) -> tuple:
        """
        Compute reward for a single puzzle-solution pair.
        Returns: (a1, a2, reward) where a1=new digit accuracy, a2=old digit modification rate
        reward = a1 - a2
        """
        new_correct, new_total = 0, 0
        old_modified, old_total = 0, 0
        
        # Get ground truth solution
        gt_solutions = self._find_solutions(puzzle, limit=1)
        gt = gt_solutions[0] if gt_solutions else None
        
        for i in range(9):
            for j in range(9):
                if puzzle[i][j] == 0:  # New cell (was empty)
                    new_total += 1
                    if gt and solution[i][j] == gt[i][j]:
                        new_correct += 1
                else:  # Original cell (had value)
                    old_total += 1
                    if solution[i][j] != puzzle[i][j]:
                        old_modified += 1
        
        a1 = new_correct / new_total if new_total > 0 else 1.0
        a2 = old_modified / old_total if old_total > 0 else 0.0
        return a1, a2, a1 - a2

    def _find_solutions(self, puzzle: List[List[int]], limit: int = 1) -> List[List[List[int]]]:
        """Find solutions using backtracking."""
        solutions = []
        grid = copy.deepcopy(puzzle)
        
        def is_valid(r, c, num):
            if num in grid[r]: return False
            if num in [grid[i][c] for i in range(9)]: return False
            br, bc = 3 * (r // 3), 3 * (c // 3)
            for i in range(3):
                for j in range(3):
                    if grid[br+i][bc+j] == num: return False
            return True
        
        def backtrack():
            if len(solutions) >= limit: return
            for i in range(9):
                for j in range(9):
                    if grid[i][j] == 0:
                        for num in range(1, 10):
                            if is_valid(i, j, num):
                                grid[i][j] = num
                                backtrack()
                                grid[i][j] = 0
                        return
            solutions.append(copy.deepcopy(grid))
        
        backtrack()
        return solutions