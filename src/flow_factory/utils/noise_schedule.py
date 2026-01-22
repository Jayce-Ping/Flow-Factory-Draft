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

# ============================ Time Samplers ============================
class TimeSampler:
    """Continuous time sampler for flow matching training."""
    
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