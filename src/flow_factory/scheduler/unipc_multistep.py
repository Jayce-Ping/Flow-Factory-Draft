# src/flow_factory/scheduler/unipc_multistep.py
import math
from dataclasses import dataclass, fields, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import numpy as np
from diffusers.utils.outputs import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


from ..utils.base import to_broadcast_tensor
from ..utils.logger_utils import setup_logger


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
    
"""
UniPCMultistepSDEScheduler: SDE extension for UniPC scheduler.

Architecture:
- SDESchedulerMixin: Reusable infrastructure for noise level management, train steps, etc.
- UniPCMultistepSDEScheduler: Inherits from UniPCMultistepScheduler, uses mixin for SDE logic.
"""

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
        noise_level: float = 0.7,
        train_steps: Optional[Union[int, List[int], torch.Tensor]] = None,
        num_train_steps: Optional[int] = None,
        seed: int = 42,
        dynamics_type: Literal["SDE", "ODE"] = "SDE",
        **kwargs
    ):
        UniPCMultistepScheduler.__init__(self, **kwargs)
        self._init_sde_mixin(noise_level, train_steps, num_train_steps, seed, dynamics_type)

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
        if self.num_inference_steps is None:
            raise ValueError("Run 'set_timesteps' before calling step()")

        if self.step_index is None:
            self._init_step_index(timestep)

        # --- Original UniPC logic: corrector + model output conversion ---
        use_corrector = (
            self.step_index > 0
            and self.step_index - 1 not in self.disable_corrector
            and self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(model_output, sample=sample)
        
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        # Update history
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        # Compute order
        if self.config.lower_order_final:
            this_order = min(self.config.solver_order, len(self.timesteps) - self.step_index)
        else:
            this_order = self.config.solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)

        self.last_sample = sample

        # --- UniP predictor step (deterministic mean) ---
        prev_sample_mean = self.multistep_uni_p_bh_update(
            model_output=model_output,
            sample=sample,
            order=self.this_order,
        )

        # --- SDE noise injection ---
        effective_noise_level = noise_level if noise_level is not None else (
            0.0 if self.is_eval else self.get_noise_level_for_step_index(self.step_index)
        )

        sigma_t = self.sigmas[self.step_index]
        sigma_prev = self.sigmas[self.step_index + 1]
        
        # Noise std proportional to step size and noise level
        # Using DDPM-style variance: std = noise_level * sqrt(|sigma_t - sigma_prev|)
        dt = (sigma_prev - sigma_t).abs()
        std_dev_t = effective_noise_level * torch.sqrt(dt)
        std_dev_t = to_broadcast_tensor(std_dev_t, prev_sample_mean)

        if self.dynamics_type == "ODE" or effective_noise_level == 0.0:
            # Deterministic
            if prev_sample is None:
                prev_sample = prev_sample_mean
            log_prob = torch.zeros(sample.shape[0], device=sample.device) if compute_log_prob else None
        else:
            # SDE sampling #TODO: Add 3 diff type
            if prev_sample is None:
                noise = randn_tensor(
                    prev_sample_mean.shape,
                    generator=generator,
                    device=prev_sample_mean.device,
                    dtype=prev_sample_mean.dtype,
                )
                prev_sample = prev_sample_mean + std_dev_t * noise

            if compute_log_prob:
                log_prob = (
                    -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std_dev_t ** 2 + 1e-8)
                    - torch.log(std_dev_t + 1e-8)
                    - 0.5 * math.log(2 * math.pi)
                )
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            else:
                log_prob = None

        # Update state
        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1
        self._step_index += 1

        if not return_dict:
            return (prev_sample, prev_sample_mean, std_dev_t, log_prob)

        return UniPCMultistepSDESchedulerOutput(
            prev_sample=prev_sample,
            prev_sample_mean=prev_sample_mean,
            std_dev_t=std_dev_t,
            log_prob=log_prob,
        )