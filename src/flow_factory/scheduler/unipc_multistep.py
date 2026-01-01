# src/flow_factory/scheduler/unipc_multistep.py
import math
from dataclasses import dataclass, fields, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import numpy as np
from diffusers.utils.outputs import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from ..utils.base import to_broadcast_tensor
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class UniPCMultistepSDESchedulerOutput(BaseOutput):
    """Output for SDE step with log probability support."""
    prev_sample: torch.FloatTensor
    prev_sample_mean: Optional[torch.FloatTensor] = None
    std_dev_t: Optional[torch.FloatTensor] = None
    log_prob: Optional[torch.FloatTensor] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniPCMultistepSDESchedulerOutput":
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in field_names})

class UniPCMultistepSDEScheduler(UniPCMultistepScheduler):
    """
    UniPC scheduler with SDE sampling support for RL fine-tuning.
    
    Extends UniPCMultistepScheduler with:
    - Stochastic sampling via configurable noise injection
    - Log probability computation for policy gradient methods
    - Train/eval mode switching
    
    Args (additional to UniPCMultistepScheduler):
        noise_level: Noise scaling factor for SDE sampling. Default 0.7.
        train_steps: Indices of steps to apply SDE noise. Default all steps.
        num_train_steps: Number of train steps to sample per rollout.
        seed: Random seed for selecting train steps.
        dynamics_type: "SDE" or "ODE". SDE adds stochastic noise.
    """

    def __init__(
        self,
        noise_level : float = 0.7,
        train_steps : Optional[Union[int, list, torch.Tensor]] = None,
        num_train_steps : Optional[int] = None,
        seed : int = 42,
        dynamics_type : Literal["Flow-SDE", 'Dance-SDE', 'CPS', 'ODE'] = "Flow-SDE",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if train_steps is None:
            # Default to all noise steps
            train_steps = list(range(len(self.timesteps)))

        self.noise_level = noise_level

        assert self.noise_level >= 0, "Noise level must be non-negative."

        self.train_steps = torch.tensor(train_steps, dtype=torch.int64)
        self.num_train_steps = num_train_steps if num_train_steps is not None else len(train_steps) # Default to all noise steps
        self.seed = seed
        self.dynamics_type = dynamics_type
        self._is_eval = False

    @property
    def is_eval(self):
        return self._is_eval

    def eval(self):
        """Apply ODE Sampling with noise_level = 0"""
        self._is_eval = True

    def train(self, *args, **kwargs):
        """Apply SDE Sampling"""
        self._is_eval = False

    def rollout(self, *args, **kwargs):
        """Apply SDE rollout sampling"""
        self.train(*args, **kwargs)

    @property
    def current_sde_steps(self) -> torch.Tensor:
        """
            Returns the current SDE step indices under the self.seed.
            Randomly select self.num_train_steps from self.train_steps.
        """
        if self.num_train_steps >= len(self.train_steps):
            return self.train_steps
        generator = torch.Generator().manual_seed(self.seed)
        selected_indices = torch.randperm(len(self.train_steps), generator=generator)[:self.num_train_steps]
        return self.train_steps[selected_indices]

    @property
    def train_timesteps(self) -> torch.Tensor:
        """
            Returns timesteps that to train on.
        """
        return self.current_sde_steps

    def get_train_timesteps(self) -> torch.Tensor:
        """
            Returns timesteps within the current window.
        """
        return self.timesteps[self.train_steps]

    def get_train_sigmas(self) -> torch.Tensor:
        """
            Returns sigmas within the current window.
        """
        return self.sigmas[self.train_steps]

    def get_noise_levels(self) -> torch.Tensor:
        """ Returns noise levels on all timesteps, where noise level is non-zero only within the current window. """
        noise_levels = torch.zeros_like(self.timesteps, dtype=torch.float32)
        noise_levels[self.current_sde_steps] = self.noise_level
        return noise_levels

    def get_noise_level_for_timestep(self, timestep : Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
            Return the noise level for a specific timestep.
        """
        if not isinstance(timestep, torch.Tensor) or timestep.ndim == 0:
            t = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
            timestep_index = self.index_for_timestep(t)
            return self.noise_level if timestep_index in self.train_steps else 0.0

        indices = torch.tensor([self.index_for_timestep(t.item()) for t in timestep])
        mask = torch.isin(indices, self.train_steps)
        return torch.where(mask, self.noise_level, 0.0).to(timestep.dtype)


    def get_noise_level_for_sigma(self, sigma) -> float:
        """
            Return the noise level for a specific sigma.
        """
        sigma_index = (self.sigmas - sigma).abs().argmin().item()
        if sigma_index in self.train_steps:
            return self.noise_level

        return 0.0
    
    def set_seed(self, seed: int):
        """
            Set the random seed for noise steps.
        """
        self.seed = seed

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        noise_level: Optional[float] = None,
        compute_log_prob: bool = True,
        return_dict: bool = True,
    ) -> Union[UniPCMultistepSDESchedulerOutput, Tuple]:
        """
        SDE step for UniPC scheduler.
        
        Args:
            model_output: Direct output from the diffusion model.
            timestep: Current timestep.
            sample: Current sample.
            prev_sample: If provided, compute log_prob for this sample instead of sampling.
            generator: Random generator for noise sampling.
            noise_level: Override default noise level.
            compute_log_prob: Whether to compute log probability.
            return_dict: Return output as dataclass or tuple.
        """
        # TODO
        pass