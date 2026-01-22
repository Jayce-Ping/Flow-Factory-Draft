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

# src/flow_factory/trainers/nft.py
"""
DiffusionNFT Trainer with off-policy and continuous timestep support.
Reference: https://arxiv.org/abs/2509.16117
"""
import os
from typing import List, Dict, Any, Union, Optional
from functools import partial
from collections import defaultdict
from contextlib import nullcontext, contextmanager
import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
import tqdm as tqdm_
tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .abc import BaseTrainer
from ..models.abc import BaseSample
from ..utils.base import filter_kwargs, create_generator, to_broadcast_tensor
from ..utils.logger_utils import setup_logger
from ..utils.noise_schedule import TimeSampler

logger = setup_logger(__name__)



class DiffusionNFTTrainer(BaseTrainer):
    """
    DiffusionNFT Trainer with off-policy and continuous timestep support.
    Reference: https://arxiv.org/abs/2509.16117
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # NFT-specific config
        self.nft_beta = getattr(self.training_args, 'nft_beta', 1.0)
        self.off_policy = getattr(self.training_args, 'off_policy', False)
        
        # Timestep sampling config
        self.time_type = getattr(self.training_args, 'time_type', 'logit_normal')
        self.time_shift = getattr(self.training_args, 'time_shift', 3.0)
        self.num_train_timesteps = getattr(self.training_args, 'num_train_timesteps', self.training_args.num_inference_steps)
        
        # Check args
        self.kl_type = getattr(self.training_args, 'kl_type', 'v-based')
        if self.kl_type != 'v-based':
            logger.warning(f"DiffusionNFT-Trainer only supports 'v-based' KL loss, got {self.kl_type}, switching to 'v-based'.")
            self.kl_type = 'v-based'

    @property
    def enable_kl_loss(self) -> bool:
        """Check if KL penalty is enabled."""
        return self.training_args.kl_beta > 0.0
    
    @contextmanager
    def sampling_context(self):
        """Context manager for sampling with or without EMA parameters."""
        if self.off_policy:
            with self.adapter.use_ema_parameters():
                yield
        else:
            yield

    def _sample_timesteps(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Sample continuous timesteps.
        
        Returns:
            Tensor of shape (num_train_timesteps, batch_size) with t in (0, 1).
        """
        device = device or self.accelerator.device
        available_types = ['logit_normal', 'uniform']
        
        assert self.time_type.lower() in available_types, \
            f"Unknown time_type: {self.time_type}, available: {available_types}"

        if self.time_type == 'logit_normal':
            timesteps = TimeSampler.logit_normal_shifted(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                shift=self.time_shift,
                device=device,
                stratified=True,
            )
        elif self.time_type == 'uniform':
            timesteps = TimeSampler.uniform(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                shift=self.time_shift,
                device=device,
            )

        return timesteps  # (num_train_timesteps, batch_size)

    def start(self):
        """Main training loop."""
        while True:
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)
            
            # Save checkpoint
            if (
                self.log_args.save_freq > 0 and 
                self.epoch % self.log_args.save_freq == 0 and 
                self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.config.run_name),
                    'checkpoints',
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            # Evaluation
            if (
                self.eval_args.eval_freq > 0 and
                self.epoch % self.eval_args.eval_freq == 0
            ):
                self.evaluate()

            # Sampling: use EMA if off_policy
            with self.sampling_context():
                samples = self.sample()

            self.optimize(samples)
            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    def sample(self) -> List[BaseSample]:
        """Generate rollouts for DiffusionNFT."""
        self.adapter.rollout()
        samples = []
        data_iter = iter(self.dataloader)
        
        for batch_index in tqdm(
            range(self.training_args.num_batches_per_epoch),
            desc=f'Epoch {self.epoch} Sampling',
            disable=not self.accelerator.is_local_main_process,
        ):
            batch = next(data_iter)
            
            with torch.no_grad(), self.autocast():
                sample_kwargs = {
                    **self.training_args,
                    'compute_log_prob': False,
                    **batch
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
            
            samples.extend(sample_batch)

        return samples

    # Copied from src.flow_factory.trainers.grpo.GRPOTrainer.compute_advantages.
    def compute_advantages(self, samples: List[BaseSample], rewards: Dict[str, torch.Tensor], store_to_samples: bool = True) -> torch.Tensor:
        """
        Compute advantages for DiffusionNFT.
        Args:
            samples: List of BaseSample instances
            rewards: Dict of reward_name to reward tensors - these should be aligned with samples
        Returns:
            advantages: Tensor of shape (num_samples, ) with computed advantages

        Notes:
            - If you want to customize advantage computation (e.g., different normalization),
            you can override this method in a subclass, e.g., for GDPO.
        """
        # 1. Get rewards
        rewards = {key: torch.as_tensor(value).to(self.accelerator.device) for key, value in rewards.items()}
        gathered_rewards = {
            key: self.accelerator.gather(value).cpu().numpy()
            for key, value in rewards.items()
        }

        # 2. Aggregate rewards if multiple reward models
        aggregated_rewards = np.zeros_like(next(iter(gathered_rewards.values())), dtype=np.float64)
        for key, reward_array in gathered_rewards.items():
            # Simple weighted sum
            aggregated_rewards += reward_array * self.reward_models[key].config.weight

        # 3. Group rewards by unique_ids - each sample has its `unique_id` hashed from its prompt, conditioning, etc.
        unique_ids = torch.tensor([s.unique_id for s in samples], dtype=torch.int64, device=self.accelerator.device)
        gathered_ids = self.accelerator.gather(unique_ids).cpu().numpy()
        _unique_ids, group_indices, _counts = np.unique(gathered_ids, return_inverse=True, return_counts=True)
        
        # 4. Compute advantages within each group
        advantages = np.zeros_like(aggregated_rewards, dtype=np.float64)

        if self.training_args.global_std:
            std = max(np.std(aggregated_rewards, axis=0, keepdims=True), 1e-6)

        for group_id in np.unique(group_indices):
            mask = (group_indices == group_id)
            group_rewards = aggregated_rewards[mask]
            assert len(group_rewards) == self.training_args.group_size, \
                f"Group size mismatch: expected {self.training_args.group_size}, got {len(group_rewards)}"

            mean = np.mean(group_rewards, axis=0, keepdims=True)
            if not self.training_args.global_std:
                std = max(np.std(group_rewards, axis=0, keepdims=True), 1e-6)
            
            advantages[mask] = (group_rewards - mean) / std

        # 5. Log statistics
        # Log per-reward mean
        _log_data = {
            f'train/reward_{key}_mean': np.mean(value)
            for key, value in gathered_rewards.items()
        }
        # Log per-reward std
        _log_data.update({
            f'train/reward_{key}_std': np.std(value)
            for key, value in gathered_rewards.items()
        })
        # Log aggregated reward zero std ratio
        zero_std_ratio = self.reward_processor.compute_group_zero_std_ratio(aggregated_rewards, group_indices)
        _log_data['train/reward_zero_std_ratio'] = zero_std_ratio
        # Log other stats
        _log_data.update({
            'train/reward_mean': np.mean(aggregated_rewards),
            'train/reward_std': np.std(aggregated_rewards),
            'train/adv_max': np.max(advantages),
            'train/adv_min': np.min(advantages),
            'train/adv_abs_mean': np.mean(np.abs(advantages)),
        })
        _log_data['train_samples'] = samples[:30]

        self.log_data(_log_data, step=self.step)

        # 6. Scatter advantages back to align with samples
        advantages = torch.as_tensor(advantages).reshape(
            self.accelerator.num_processes, -1, *advantages.shape[1:]
        )[self.accelerator.process_index].to(self.accelerator.device)

        # Store advantages to samples' extra_kwargs
        if store_to_samples:
            for sample, adv in zip(samples, advantages):
                sample.extra_kwargs['advantage'] = adv

        return advantages

    def _compute_nft_output(
        self,
        batch: Dict[str, Any],
        timestep: torch.Tensor,
        noised_latents: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute NFT forward pass for a single timestep.
        
        Args:
            batch: Batch containing prompt embeddings and other inputs.
            timestep: Timestep tensor of shape (B,) in [0, 1].
            noised_latents: Interpolated latents x_t = (1-t)*x_1 + t*noise.
        
        Returns:
            Dict with noise_pred and std_dev_t.
        """
        t_scaled = (timestep * 1000).view(-1)  # Scale to [0, 1000], ensure (B,)
        
        forward_kwargs = {
            **self.training_args,
            't': t_scaled,
            't_next': torch.zeros_like(t_scaled),
            'latents': noised_latents,
            'compute_log_prob': False,
            'return_kwargs': ['noise_pred'],
            **{k: v for k, v in batch.items() if k not in ['all_latents', 'timesteps', 'advantage']},
        }
        forward_kwargs = filter_kwargs(self.adapter.forward, **forward_kwargs)
        
        with self.autocast():
            output = self.adapter.forward(**forward_kwargs)
        
        return {
            'noise_pred': output.noise_pred,
        }

    def optimize(self, samples: List[BaseSample]) -> None:
        """Main optimization loop with continuous timestep sampling."""
        self.adapter.train()
        # Compute rewards and advantages for samples
        rewards = self.reward_processor.compute_rewards(samples, store_to_samples=True, epoch=self.epoch)
        advantages = self.compute_advantages(samples, rewards, store_to_samples=True)
    
        sample_batches: List[Dict[str, Union[torch.Tensor, Any, List[Any]]]] = [
            BaseSample.stack(samples[i:i + self.training_args.per_device_batch_size])
            for i in range(0, len(samples), self.training_args.per_device_batch_size)
        ]

        loss_info = defaultdict(list)

        for batch_idx, batch in enumerate(tqdm(
            sample_batches,
            total=len(sample_batches),
            desc=f'Epoch {self.epoch} Training',
            position=0,
            disable=not self.accelerator.is_local_main_process,
        )):
            batch_size = batch['all_latents'].shape[0]
            clean_latents = batch['all_latents'][:, -1]  # x0
            
            with self.accelerator.accumulate(self.adapter.transformer):
                # Sample timesteps: (T, B)
                all_timesteps = self._sample_timesteps(batch_size)
                
                for t_idx in tqdm(
                    range(self.num_train_timesteps),
                    desc=f'Epoch {self.epoch} Timestep',
                    position=1,
                    leave=False,
                    disable=not self.accelerator.is_local_main_process,
                ):
                    # 1. Prepare inputs
                    t_flat = all_timesteps[t_idx]  # (B,)
                    t_broadcast = to_broadcast_tensor(t_flat, clean_latents)
                    noise = randn_tensor(clean_latents.shape, device=self.accelerator.device, dtype=clean_latents.dtype)
                    noised_latents = (1 - t_broadcast) * clean_latents + t_broadcast * noise
                    
                    # 2. Forward pass
                    output = self._compute_nft_output(batch, t_flat, noised_latents)
                    new_v_pred = output['noise_pred']
                    
                    # Target velocity: v = noise - x0
                    old_v_pred = noise - clean_latents
                    
                    # 3. Compute NFT loss
                    # Get advantages and clip,
                    adv = batch['advantage']
                    adv_clip_range = self.training_args.adv_clip_range
                    adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                    # Normalize advantage to [0, 1]
                    normalized_adv = (adv / adv_clip_range[1]) / 2.0 + 0.5
                    r = torch.clamp(normalized_adv, 0, 1).view(-1, *([1] * (new_v_pred.dim() - 1)))
                    
                    # Positive/negative predictions
                    positive_pred = self.nft_beta * new_v_pred + (1 - self.nft_beta) * old_v_pred.detach()
                    negative_pred = (1.0 + self.nft_beta) * old_v_pred.detach() - self.nft_beta * new_v_pred
                    
                    # Positive loss
                    x0_pred = noised_latents - t_broadcast * positive_pred
                    with torch.no_grad():
                        weight = torch.abs(x0_pred.double() - clean_latents.double()).mean(
                            dim=tuple(range(1, clean_latents.ndim)), keepdim=True
                        ).clip(min=1e-5)
                    positive_loss = ((x0_pred - clean_latents) ** 2 / weight).mean(dim=tuple(range(1, clean_latents.ndim)))
                    
                    # Negative loss
                    neg_x0_pred = noised_latents - t_broadcast * negative_pred
                    with torch.no_grad():
                        neg_weight = torch.abs(neg_x0_pred.double() - clean_latents.double()).mean(
                            dim=tuple(range(1, clean_latents.ndim)), keepdim=True
                        ).clip(min=1e-5)
                    negative_loss = ((neg_x0_pred - clean_latents) ** 2 / neg_weight).mean(dim=tuple(range(1, clean_latents.ndim)))
                    
                    # Combined loss
                    ori_policy_loss = (r.squeeze() * positive_loss + (1.0 - r.squeeze()) * negative_loss) / self.nft_beta
                    policy_loss = (ori_policy_loss * adv_clip_range[1]).mean()
                    loss = policy_loss
                    
                    # 4. KL penalty
                    if self.enable_kl_loss:
                        with torch.no_grad(), self.adapter.use_ref_parameters():
                            ref_output = self._compute_nft_output(batch, t_flat, noised_latents)
                        # KL-loss in v-space
                        kl_div = torch.mean(
                            (new_v_pred - ref_output['noise_pred']) ** 2,
                            dim=tuple(range(1, new_v_pred.ndim))
                        )
                        
                        kl_div = kl_div.mean()
                        kl_loss = self.training_args.kl_beta * kl_div
                        loss = loss + kl_loss
                        loss_info['kl_div'].append(kl_div.detach())
                        loss_info['kl_loss'].append(kl_loss.detach())
                    
                    loss_info['policy_loss'].append(policy_loss.detach())
                    loss_info['unweighted_policy_loss'].append(ori_policy_loss.mean().detach())
                    loss_info['loss'].append(loss.detach())
                    
                    self.accelerator.backward(loss)
                
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.adapter.get_trainable_parameters(),
                        self.training_args.max_grad_norm,
                    )
                    loss_info = {k: torch.stack(v).mean() for k, v in loss_info.items()}
                    loss_info = self.accelerator.reduce(loss_info, reduction="mean")
                    self.log_data({f'train/{k}': v for k, v in loss_info.items()}, step=self.step)
                    self.step += 1
                    loss_info = defaultdict(list)
                
                self.optimizer.step()
                self.optimizer.zero_grad()

    def evaluate(self) -> None:
        """Evaluation loop."""
        if self.test_dataloader is None:
            return
        
        self.adapter.eval()
        with self.adapter.use_ema_parameters():
            all_samples: List[BaseSample] = []
            
            for batch in tqdm(
                self.test_dataloader,
                desc='Evaluating', 
                disable=not self.accelerator.is_local_main_process
            ):
                generator = create_generator(batch['prompt'], self.training_args.seed)
                inference_kwargs = {
                    'compute_log_prob': False,
                    'generator': generator,
                    **self.eval_args,
                    **batch,
                }
                inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
                with torch.no_grad(), self.autocast():
                    samples = self.adapter.inference(**inference_kwargs)
                all_samples.extend(samples)
            
            rewards = self.eval_reward_processor.compute_rewards(
                samples=all_samples,
                store_to_samples=False,
                epoch=self.epoch,
                split='pointwise',  # Only `pointwise` reward can be compute when evaluation, since there is no `group` here.
            )
            # Gather and log rewards
            rewards = {key: torch.as_tensor(value).to(self.accelerator.device) for key, value in rewards.items()}
            gathered_rewards = {
                key: self.accelerator.gather(value).cpu().numpy()
                for key, value in rewards.items()
            }

            # Log statistics
            if self.accelerator.is_main_process:
                _log_data = {f'eval/reward_{key}_mean': np.mean(value) for key, value in gathered_rewards.items()}
                _log_data.update({f'eval/reward_{key}_std': np.std(value) for key, value in gathered_rewards.items()})
                _log_data['eval_samples'] = all_samples
                self.log_data(_log_data, step=self.step)
            
            self.accelerator.wait_for_everyone()