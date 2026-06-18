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

# src/flow_factory/trainers/dppo.py
"""
Flow-DPPO Trainer.

DPPO is a strict Flow-GRPO variant: it keeps GRPO's group advantages and the
optional KL-vs-reference penalty, but replaces the PPO ratio-clip with a KL
trust-region mask. A sample's gradient is zeroed when its per-step
KL(current || rollout-old) exceeds ``kl_mask_threshold`` and the update would
push the action further in the wrong direction.
"""

from collections import defaultdict
from functools import partial
from typing import List

import torch
import tqdm as tqdm_

from ..hparams import DPPOTrainingArguments
from ..samples import BaseSample
from ..utils.base import filter_kwargs
from ..utils.dist import reduce_loss_info
from ..utils.logger_utils import setup_logger
from ..utils.trajectory_collector import compute_trajectory_indices
from .grpo import GRPOTrainer
from .registry import register_trainer

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)
logger = setup_logger(__name__)


def gaussian_kl_div(p: torch.Tensor, q: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """KL-style squared error between Gaussian means scaled by variance (x-space)."""
    return (p - q) ** 2 / (2 * sigma**2)


# ============================ Flow-DPPO Trainer ============================
@register_trainer("dppo")
class DPPOTrainer(GRPOTrainer):
    """Flow-DPPO Trainer: GRPO with a KL trust-region mask instead of PPO clipping.

    References:
    [1] Flow-GRPO: Training Flow Matching Models via Online RL
        - https://arxiv.org/abs/2505.05470
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: DPPOTrainingArguments

    def _effective_sigma(self, std_dev_t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Per-step Gaussian std for the x-space KL, by sampling dynamics.

        Args:
            std_dev_t: Per-step diffusion std from the adapter forward.
            dt: Per-step time delta from the adapter forward (negative).

        Returns:
            Effective sigma tensor broadcastable to the latent shape.
        """
        dynamics_type = self.adapter.scheduler.dynamics_type
        if dynamics_type in ("Flow-SDE", "Dance-SDE"):
            return std_dev_t * torch.sqrt(-dt)
        if dynamics_type == "CPS":
            return std_dev_t
        raise ValueError(
            f"DPPO x-based KL requires an SDE dynamics_type in "
            f"('Flow-SDE', 'Dance-SDE', 'CPS'), got {dynamics_type!r}. "
            "Coupled algorithms must not use ODE dynamics (see constraints #7)."
        )

    # =========================== Sampling Loop ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts and store the rollout-old quantity the KL mask needs."""
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.scheduler.train_timesteps,
            num_inference_steps=self.training_args.num_inference_steps,
        )
        # The trust-region mask compares the current policy against the rollout-old
        # policy in `kl_mask_type` space, so only that per-step quantity is stored
        # (the ref-KL penalty compares current vs reference, never the old policy).
        mask_field = (
            "noise_pred" if self.training_args.kl_mask_type == "v-based" else "next_latents_mean"
        )
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=True,
            trajectory_indices=trajectory_indices,
            extra_call_back_kwargs=[mask_field],
        )

    # =========================== Optimization Loop ============================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization (Stage 6): KL trust-region masked loss and optional KL-vs-ref."""
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        kl_type = self.training_args.kl_type
        kl_mask_type = self.training_args.kl_mask_type
        kl_guidance_scale = self.training_args.kl_guidance_scale
        kl_mask_threshold = self.training_args.kl_mask_threshold
        # Mask space picks the single rollout-old tensor stored by sample().
        mask_field = "noise_pred" if kl_mask_type == "v-based" else "next_latents_mean"
        for inner_epoch in range(self.training_args.num_inner_epochs):
            shuffled_samples = self._order_samples_for_optimize(samples, inner_epoch)

            self.adapter.train()
            loss_info = defaultdict(list)

            for batch_idx in tqdm(
                range(num_batches),
                total=num_batches,
                desc=f"Epoch {self.epoch} Training",
                position=0,
                disable=not self.show_progress_bar,
            ):
                start = batch_idx * per_device_batch_size
                batch_samples = [
                    sample.to(device)
                    for sample in shuffled_samples[start : start + per_device_batch_size]
                ]
                batch = BaseSample.stack(batch_samples)
                latents_index_map = batch["latent_index_map"]  # (T+1,) LongTensor
                log_probs_index_map = batch["log_prob_index_map"]  # (T,) LongTensor
                callback_index_map = batch["callback_index_map"][0]  # (T,) LongTensor, shared
                for timestep_index in tqdm(
                    self.adapter.scheduler.train_timesteps,
                    desc=f"Epoch {self.epoch} Timestep",
                    position=1,
                    leave=False,
                    disable=not self.show_progress_bar,
                ):
                    with self.accumulate_gradients():
                        # 1. Prepare inputs
                        old_log_prob = batch["log_probs"][:, log_probs_index_map[timestep_index]]
                        # Rollout-old policy in mask space (the only stored callback tensor).
                        old_mask_tensor = batch[mask_field][:, callback_index_map[timestep_index]]
                        num_timesteps = batch["timesteps"].shape[1]
                        t = batch["timesteps"][:, timestep_index]
                        t_next = (
                            batch["timesteps"][:, timestep_index + 1]
                            if timestep_index + 1 < num_timesteps
                            else torch.tensor(0, device=self.accelerator.device)
                        )
                        latents = batch["all_latents"][:, latents_index_map[timestep_index]]
                        next_latents = batch["all_latents"][
                            :, latents_index_map[timestep_index + 1]
                        ]
                        forward_inputs = {
                            **self.training_args,
                            "t": t,
                            "t_next": t_next,
                            "latents": latents,
                            "next_latents": next_latents,
                            "compute_log_prob": True,
                            "noise_level": self.adapter.scheduler.noise_level,
                            **batch,
                        }
                        forward_inputs = filter_kwargs(self.adapter.forward, **forward_inputs)
                        # 2. Forward pass — request only what the mask (and optional ref KL) need.
                        # The mask uses kl_mask_type; the ref penalty uses kl_type. std_dev_t/dt
                        # feed the x-based mask's variance scaling only.
                        return_kwargs = {"log_prob"}
                        if kl_mask_type == "v-based":
                            return_kwargs.add("noise_pred")
                        else:
                            return_kwargs.update(("next_latents_mean", "std_dev_t", "dt"))
                        if self.enable_kl_loss:
                            return_kwargs.add(
                                "noise_pred" if kl_type == "v-based" else "next_latents_mean"
                            )
                        forward_inputs["return_kwargs"] = list(return_kwargs)
                        with self.autocast():
                            output = self.adapter.forward(**forward_inputs)

                        # 3. Compute loss
                        adv = batch["advantage"]
                        adv_clip_range = self.training_args.adv_clip_range
                        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                        ratio = torch.exp(output.log_prob - old_log_prob)

                        # Per-step KL(current || old) for the trust-region mask. The x-based mask
                        # uses the exact variance-scaled Gaussian KL; the v-based mask and the
                        # ref-KL penalty below use unscaled squared error (GRPO convention).
                        if kl_mask_type == "v-based":
                            sq = (output.noise_pred - old_mask_tensor) ** 2
                            kl_new_old = sq.mean(dim=tuple(range(1, sq.ndim)))
                        else:
                            sigma_t = self._effective_sigma(output.std_dev_t, output.dt)
                            kl_elem = gaussian_kl_div(
                                output.next_latents_mean, old_mask_tensor, sigma_t
                            )
                            kl_new_old = kl_elem.mean(dim=tuple(range(1, kl_elem.ndim)))

                        # DPPO mask: zero gradient for trust-region violators that
                        # push the wrong way (ratio>1 & adv>0, or ratio<1 & adv<0).
                        unclipped_loss = -adv * ratio
                        violate = kl_new_old >= kl_mask_threshold
                        pos_rm = violate & (ratio > 1.0) & (adv > 0)
                        neg_rm = violate & (ratio < 1.0) & (adv < 0)
                        keep_mask = (
                            torch.logical_not(pos_rm | neg_rm)
                            .to(dtype=unclipped_loss.dtype)
                            .detach()
                        )
                        policy_loss = torch.mean(unclipped_loss * keep_mask)

                        loss = policy_loss

                        # 4. Optional KL-vs-reference penalty (run at kl_guidance_scale CFG).
                        # negative_* embeds already rode `sample.to(device)` into `batch`, so the
                        # ref forward reuses them on-device without an extra move.
                        if self.enable_kl_loss:
                            with self.autocast():
                                with torch.no_grad(), self.adapter.use_ref_parameters():
                                    ref_forward_inputs = forward_inputs.copy()
                                    ref_forward_inputs["compute_log_prob"] = False
                                    if kl_guidance_scale is not None:
                                        ref_forward_inputs["guidance_scale"] = kl_guidance_scale
                                    if kl_type == "v-based":
                                        ref_forward_inputs["return_kwargs"] = ["noise_pred"]
                                    else:
                                        ref_forward_inputs["return_kwargs"] = ["next_latents_mean"]
                                    ref_output = self.adapter.forward(**ref_forward_inputs)

                                # kl_div must be computed outside `torch.no_grad()` for correct gradients.
                                if kl_type == "v-based":
                                    kl_div = torch.mean(
                                        ((output.noise_pred - ref_output.noise_pred) ** 2),
                                        dim=tuple(range(1, output.noise_pred.ndim)),
                                        keepdim=True,
                                    )
                                else:
                                    kl_div = torch.mean(
                                        (
                                            (
                                                output.next_latents_mean
                                                - ref_output.next_latents_mean
                                            )
                                            ** 2
                                        ),
                                        dim=tuple(range(1, output.next_latents_mean.ndim)),
                                        keepdim=True,
                                    )
                                kl_div = torch.mean(kl_div)
                                kl_loss = self.training_args.kl_beta * kl_div
                                loss += kl_loss
                                loss_info["kl_div"].append(kl_div.detach())
                                loss_info["kl_loss"].append(kl_loss.detach())

                        # 5. Log per-timestep info
                        keep_frac = keep_mask.mean().detach()
                        loss_info["ratio"].append(ratio.detach())
                        loss_info["kl_new_old"].append(kl_new_old.detach())
                        loss_info["unclipped_loss"].append(unclipped_loss.detach())
                        loss_info["policy_loss"].append(policy_loss.detach())
                        loss_info["loss"].append(loss.detach())
                        loss_info["keep_ratio"].append(keep_frac)
                        loss_info["masked_ratio"].append(1.0 - keep_frac)

                        # 6. Backward and optimizer step
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.adapter.get_trainable_parameters(),
                                self.training_args.max_grad_norm,
                            )
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            loss_info = reduce_loss_info(self.accelerator, loss_info)
                            loss_info["grad_norm"] = grad_norm
                            self.log_data(
                                {f"train/{k}": v for k, v in loss_info.items()},
                                step=self.step,
                            )
                            self.step += 1
                            loss_info = defaultdict(list)
