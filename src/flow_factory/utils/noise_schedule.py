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

# src/flow_factory/utils/noise_schedule.py
"""
Utility functions for noise schedule and time sampling.
"""
import torch
from typing import Union

# ============================ Time Samplers ============================
class TimeSampler:
    """Continuous and discrete time sampler for flow matching training."""
    
    @staticmethod
    def logit_normal_shifted(
        batch_size: int,
        num_timesteps: int,
        m: float = 0.0,
        s: float = 1.0,
        shift: float = 3.0,
        device: torch.device = torch.device('cpu'),
        stratified: bool = True,
    ) -> torch.Tensor:
        """
        Logit-normal shifted time sampling.
        
        Returns:
            Tensor of shape (num_timesteps, batch_size) with t in (0, 1).
        """
        if stratified:
            base = (torch.arange(num_timesteps, device=device) + torch.rand(num_timesteps, device=device)) / num_timesteps
            normal_dist = torch.distributions.Normal(loc=0.0, scale=1.0)
            u_standard = normal_dist.icdf(torch.clamp(base, 1e-7, 1 - 1e-7))
            u_standard = u_standard[torch.randperm(num_timesteps, device=device)]
        else:
            u_standard = torch.randn(num_timesteps, device=device)
        
        u = u_standard * s + m
        t = torch.sigmoid(u)
        t = shift * t / (1 + (shift - 1) * t)
        t = torch.clamp(t, min=0.01)
        
        return t.unsqueeze(1).expand(num_timesteps, batch_size)
    
    @staticmethod
    def uniform(
        batch_size: int,
        num_timesteps: int,
        lower: float = 0.2,
        upper: float = 1.0,
        shift: float = 1.0,
        device: torch.device = torch.device('cpu'),
    ) -> torch.Tensor:
        """
        Uniform time sampling with optional shift.
        
        Returns:
            Tensor of shape (num_timesteps, batch_size).
        """
        rand_u = torch.rand(num_timesteps, batch_size, device=device)
        normalized = (torch.arange(num_timesteps, device=device).unsqueeze(1) + rand_u) / num_timesteps
        matrix = lower + normalized * (upper - lower)
        t = torch.gather(matrix, 0, torch.rand_like(matrix).argsort(dim=0))
        t = shift * t / (1 + (shift - 1) * t)
        return t
    
    # ======================== Discrete Time Samplers ========================
    @staticmethod
    def discrete(
        batch_size: int,
        num_train_timesteps: int,
        scheduler_timesteps: torch.Tensor,
        timestep_fraction: float = 1.0,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Discrete stratified time sampling from scheduler timesteps.
        
        Divides the valid timestep range into `num_train_timesteps` strata and
        samples one index uniformly from each stratum.
        
        Args:
            batch_size: Number of samples per timestep.
            num_train_timesteps: Number of training timesteps to sample.
            scheduler_timesteps: Actual timesteps from scheduler, shape (num_inference_steps,).
            timestep_fraction: Fraction of trajectory to use (0, 1].
            normalize: If True, normalize timesteps to (0, 1) by dividing by 1000.
        
        Returns:
            Tensor of shape (num_train_timesteps, batch_size) with sampled timesteps.
        """
        device = scheduler_timesteps.device
        max_idx = int(len(scheduler_timesteps) * timestep_fraction)
        
        boundaries = torch.linspace(0, max_idx, steps=num_train_timesteps + 1, device=device)
        lower_bounds = boundaries[:-1].long()
        upper_bounds = boundaries[1:].long()
        
        rand_u = torch.rand(num_train_timesteps, device=device)
        t_indices = lower_bounds + (rand_u * (upper_bounds - lower_bounds)).long()
        t_indices = t_indices.clamp(max=len(scheduler_timesteps) - 1)
        
        timesteps = scheduler_timesteps[t_indices]  # (num_train_timesteps,)
        timesteps = timesteps.unsqueeze(1).expand(-1, batch_size)  # (T, B)
        
        if normalize:
            timesteps = timesteps.float() / 1000.0
        
        return timesteps
    
    @staticmethod
    def discrete_with_init(
        batch_size: int,
        num_train_timesteps: int,
        scheduler_timesteps: torch.Tensor,
        timestep_fraction: float = 1.0,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Discrete time sampling that always includes t=0 (initial timestep).
        
        The first sampled timestep is always index 0, remaining timesteps are
        stratified-sampled from indices [1, max_idx].
        
        Args:
            batch_size: Number of samples per timestep.
            num_train_timesteps: Total number of training timesteps (including init).
            scheduler_timesteps: Actual timesteps from scheduler, shape (num_inference_steps,).
            timestep_fraction: Fraction of trajectory to use (0, 1].
            normalize: If True, normalize timesteps to (0, 1) by dividing by 1000.
        
        Returns:
            Tensor of shape (num_train_timesteps, batch_size) with sampled timesteps.
        """
        device = scheduler_timesteps.device
        max_idx = int(len(scheduler_timesteps) * timestep_fraction)
        
        init_index = torch.tensor([0], device=device, dtype=torch.long)
        num_remaining = num_train_timesteps - 1
        
        if num_remaining > 0:
            boundaries = torch.linspace(1, max_idx, steps=num_remaining + 1, device=device)
            lower_bounds = boundaries[:-1].long()
            upper_bounds = boundaries[1:].long()
            
            rand_u = torch.rand(num_remaining, device=device)
            remaining_indices = lower_bounds + (rand_u * (upper_bounds - lower_bounds)).long()
            remaining_indices = remaining_indices.clamp(max=len(scheduler_timesteps) - 1)
            t_indices = torch.cat([init_index, remaining_indices])
        else:
            t_indices = init_index
        
        timesteps = scheduler_timesteps[t_indices]  # (num_train_timesteps,)
        timesteps = timesteps.unsqueeze(1).expand(-1, batch_size)  # (T, B)
        
        if normalize:
            timesteps = timesteps.float() / 1000.0
        
        return timesteps
    
    @staticmethod
    def discrete_wo_init(
        batch_size: int,
        num_train_timesteps: int,
        scheduler_timesteps: torch.Tensor,
        timestep_fraction: float = 1.0,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Discrete time sampling excluding t=0 (initial timestep).
        
        Stratified sampling from indices [1, max_idx], useful when the initial
        timestep (pure noise) provides little training signal.
        
        Args:
            batch_size: Number of samples per timestep.
            num_train_timesteps: Number of training timesteps to sample.
            scheduler_timesteps: Actual timesteps from scheduler, shape (num_inference_steps,).
            timestep_fraction: Fraction of trajectory to use (0, 1].
            normalize: If True, normalize timesteps to (0, 1) by dividing by 1000.
        
        Returns:
            Tensor of shape (num_train_timesteps, batch_size) with sampled timesteps.
        """
        device = scheduler_timesteps.device
        max_idx = int(len(scheduler_timesteps) * timestep_fraction)
        
        boundaries = torch.linspace(1, max_idx, steps=num_train_timesteps + 1, device=device)
        lower_bounds = boundaries[:-1].long()
        upper_bounds = boundaries[1:].long()
        
        rand_u = torch.rand(num_train_timesteps, device=device)
        t_indices = lower_bounds + (rand_u * (upper_bounds - lower_bounds)).long()
        t_indices = t_indices.clamp(1, len(scheduler_timesteps) - 1)
        
        timesteps = scheduler_timesteps[t_indices]  # (num_train_timesteps,)
        timesteps = timesteps.unsqueeze(1).expand(-1, batch_size)  # (T, B)
        
        if normalize:
            timesteps = timesteps.float() / 1000.0
        
        return timesteps