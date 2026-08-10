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

# src/flow_factory/trainers/grpo.py
"""
Group Relative Policy Optimization (GRPO) Trainer.
Implements GRPO algorithm for flow matching models.
"""
import os
from typing import List, Dict, Mapping, Optional, Any, Tuple, Union, Literal, Callable
from functools import partial
from collections import defaultdict
import torch
import numpy as np
import tqdm as tqdm_
tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .abc import BaseTrainer
from ..hparams import GRPOTrainingArguments
from ..samples import (
    BaseSample,
    LatentState,
    MultiModalStepOutput,
    ReplayStep,
    StackedSampleBatch,
)
from ..utils.base import create_generator_by_prompt
from ..utils.logger_utils import setup_logger
from ..utils.trajectory_collector import TrajectoryCollector, compute_trajectory_indices
from ..utils.dist import reduce_loss_info

logger = setup_logger(__name__)

# Reference-KL space -> (MultiModalStepOutput attribute, legacy `return_kwargs` request).
KL_SPACE_FIELDS: Dict[str, Tuple[str, str]] = {
    "v-based": ("velocity", "velocity"),
    "x-based": ("next_state_mean", "next_latents_mean"),
}


# ============================ GRPO Trainer ============================
class GRPOTrainer(BaseTrainer):
    """
    GRPO Trainer for Flow Matching models.
    Implements group-based advantage computation and PPO-style clipping.
    References:
    [1] Flow-GRPO: Training Flow Matching Models via Online RL
        - https://arxiv.org/abs/2505.05470
    """

    # Coupled paradigm: rollout log-probs feed the PPO ratio, so lossy rollout
    # acceleration is disallowed (constraints.md #7). Inherited by GRPOGuard/DPPO.
    paradigm = "coupled"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args : GRPOTrainingArguments
        self.num_train_timesteps = self.adapter.scheduler.num_sde_steps

    @property
    def enable_kl_loss(self) -> bool:
        """Check if KL penalty is enabled."""
        return self.training_args.kl_beta > 0.0

    # =========================== Structured replay helpers ============================
    def _replay_forward_kwargs(self, batch: StackedSampleBatch) -> Dict[str, Any]:
        """Training arguments the batch does not already carry.

        Legacy replay unpacked ``batch`` after ``training_args``, so batch-level
        values win on shared keys.
        """
        return {key: value for key, value in {**self.training_args}.items() if key not in batch}

    def _replay_forward(
        self,
        batch: StackedSampleBatch,
        replay: ReplayStep,
        return_fields: Tuple[str, ...],
    ) -> MultiModalStepOutput:
        """Replay one stored transition through the current policy."""
        return self.adapter.forward_state(
            batch=batch,
            state=replay.state,
            times=replay.times,
            next_state=replay.next_state,
            compute_log_prob=True,
            return_fields=return_fields,
            noise_level=self.adapter.scheduler.noise_level,
            **self._replay_forward_kwargs(batch),
        )

    def _reference_forward(
        self,
        batch: StackedSampleBatch,
        replay: ReplayStep,
        return_fields: Tuple[str, ...],
        **overrides: Any,
    ) -> MultiModalStepOutput:
        """Replay the same transition through the frozen reference parameters.

        Only the forward itself is wrapped in ``no_grad``/``use_ref_parameters`` so
        the caller keeps the KL arithmetic on the policy graph.
        """
        forward_kwargs = self._replay_forward_kwargs(batch)
        forward_kwargs.update(overrides)
        with torch.no_grad(), self.adapter.use_ref_parameters():
            return self.adapter.forward_state(
                batch=batch,
                state=replay.state,
                times=replay.times,
                next_state=replay.next_state,
                compute_log_prob=False,
                return_fields=return_fields,
                noise_level=self.adapter.scheduler.noise_level,
                **forward_kwargs,
            )

    def _require_replay_log_prob(self, replay: ReplayStep, step_index: int) -> torch.Tensor:
        """Return the stored rollout joint log probability for one transition."""
        if replay.log_prob is None:
            raise ValueError(
                f"expected a stored joint log probability at step_index={step_index} for "
                f"{type(self).__name__} replay, received None; rerun sampling with "
                "compute_log_prob=True"
            )
        return replay.log_prob

    def _kl_space_fields(self, kl_space: str, argument_name: str) -> Tuple[str, str]:
        """Resolve the output attribute and legacy request name for a KL space."""
        if kl_space not in KL_SPACE_FIELDS:
            raise ValueError(
                f"expected {argument_name} in {tuple(KL_SPACE_FIELDS)}, received {kl_space!r}"
            )
        return KL_SPACE_FIELDS[kl_space]

    def _require_output_state(
        self, output: MultiModalStepOutput, field: str, source: str
    ) -> LatentState:
        """Return a required latent-state field in authoritative component order."""
        state = getattr(output, field)
        expected_names = self.adapter.trajectory_component_order
        if state is None:
            raise ValueError(
                f"expected {source} forward output field {field!r} in component order "
                f"{expected_names}, received None; request it through return_fields"
            )
        if state.component_names != expected_names:
            raise ValueError(
                f"expected {source} forward output field {field!r} in component order "
                f"{expected_names}, received {state.component_names}"
            )
        return state

    def _require_component_mapping(
        self,
        values: Optional[Mapping[str, torch.Tensor]],
        field: str,
        source: str,
    ) -> Mapping[str, torch.Tensor]:
        """Return a required per-component mapping in authoritative component order."""
        expected_names = self.adapter.trajectory_component_order
        if values is None:
            raise ValueError(
                f"expected {source} {field} for {type(self).__name__} replay in component "
                f"order {expected_names}, received None"
            )
        if tuple(values) != expected_names:
            raise ValueError(
                f"expected {source} {field} for {type(self).__name__} replay in component "
                f"order {expected_names}, received {tuple(values)}"
            )
        return values

    def _component_squared_errors(
        self, new_state: LatentState, old_state: LatentState, source: str
    ) -> Dict[str, torch.Tensor]:
        """Per-component mean squared error over each component's active elements."""
        expected_names = self.adapter.trajectory_component_order
        if old_state.component_names != expected_names:
            raise ValueError(
                f"expected {source} state in component order {expected_names}, "
                f"received {old_state.component_names}"
            )
        return {
            name: (new_state.components[name] - old_state.components[name])
            .flatten(1)
            .pow(2)
            .mean(dim=1)
            for name in expected_names
        }

    def _reference_kl_divergence(
        self,
        output: MultiModalStepOutput,
        ref_output: MultiModalStepOutput,
        replay: ReplayStep,
    ) -> torch.Tensor:
        """Scalar policy-vs-reference squared error in the configured KL space."""
        output_field, _ = self._kl_space_fields(self.training_args.kl_type, "kl_type")
        errors = self._component_squared_errors(
            self._require_output_state(output, output_field, "policy"),
            self._require_output_state(ref_output, output_field, "reference"),
            "reference",
        )
        return torch.mean(
            self.adapter.reduce_latent_values(
                errors,
                active_numel=self.adapter.get_state_active_numel(replay.state),
            )
        )

    def start(self):
        """Main training loop."""
        while self.should_continue_training():
            self.adapter.set_trajectory_seed(self.epoch + self.training_args.seed)
            
            # Save checkpoint
            if (
                self.log_args.save_freq > 0 and 
                self.epoch % self.log_args.save_freq == 0 and 
                self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    'checkpoints',
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            # Evaluation
            if (
                self.eval_args.eval_freq > 0 and
                self.epoch % self.eval_args.eval_freq == 0
            ):
                self.evaluate()

            samples = self.sample()
            self.prepare_feedback(samples)
            self.optimize(samples)

            self.adapter.ema_step(step=self.epoch)

            self.epoch += 1

    # =========================== Sampling Loop ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts for GRPO (stores full trajectory + log-probs)."""
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.get_train_step_indices(),
            num_inference_steps=self.training_args.num_inference_steps,
        )
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=True,
            trajectory_indices=trajectory_indices,
        )

    # =========================== Reward / advantage (Stages 4--5) ============================
    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Finalize rewards from the buffer, compute advantages, and log advantage metrics."""
        rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')
        self.compute_advantages(samples, rewards, store_to_samples=True)
        adv_metrics = self.advantage_processor.pop_advantage_metrics()
        if adv_metrics:
            self.log_data(adv_metrics, step=self.step)

    # =========================== Optimization Loop ============================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization (Stage 6): PPO-style clipped loss and optional KL."""
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        train_step_indices = self.adapter.get_train_step_indices()
        # Forward fields: the ratio needs `log_prob`, and `dt` keeps the legacy request
        # shape; the reference KL space adds its own comparison field.
        if self.enable_kl_loss:
            _, ref_return_field = self._kl_space_fields(self.training_args.kl_type, 'kl_type')
            return_fields = (
                ('log_prob', 'velocity', 'dt')
                if self.training_args.kl_type == 'v-based'
                else ('log_prob', 'next_latents', 'next_latents_mean', 'dt')
            )
        else:
            ref_return_field = None
            return_fields = ('log_prob', 'dt')
        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle unless disabled for pack-composition-dependent adapters.
            shuffled_samples = self._order_samples_for_optimize(samples, inner_epoch)

            self.adapter.train()
            loss_info = defaultdict(list)

            # Reload each micro-batch onto the device (H2D when samples are
            # CPU-offloaded; a no-op when GPU-resident). _iter_prefetched_batches
            # overlaps the next batch's H2D with compute when offload is enabled.
            for batch in tqdm(
                self._iter_prefetched_batches(shuffled_samples, per_device_batch_size),
                total=num_batches,
                desc=f'Epoch {self.epoch} Training',
                position=0,
                disable=not self.show_progress_bar,
            ):
                # Iterate through timesteps
                for timestep_index in tqdm(
                    train_step_indices,
                    desc=f'Epoch {self.epoch} Timestep',
                    position=1,
                    leave=False,
                    disable=not self.show_progress_bar,
                ):
                    step_index = int(timestep_index)
                    with self.accumulate_gradients():
                        # 1. Prepare inputs
                        replay = self.adapter.get_replay_step(batch, step_index)
                        old_log_prob = self._require_replay_log_prob(replay, step_index)
                        # 2. Forward pass
                        with self.autocast():
                            output = self._replay_forward(batch, replay, return_fields)

                        # 3. Compute loss
                        # Clip advantages
                        adv = batch['advantage']
                        adv_clip_range = self.training_args.adv_clip_range
                        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                        # PPO-style clipped loss
                        ratio = torch.exp(output.log_prob - old_log_prob)
                        ratio_clip_range = self.training_args.clip_range

                        unclipped_loss = -adv * ratio
                        clipped_loss = -adv * torch.clamp(ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1])
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        loss = policy_loss

                        # 4. Compute KL-div
                        if self.enable_kl_loss:
                            with self.autocast():
                                ref_output = self._reference_forward(
                                    batch, replay, (ref_return_field,)
                                )
                                # kl_div must be computed outside `torch.no_grad()` for correct gradient behavior.
                                # See: issue #122, PR #123 (https://github.com/X-GenGroup/Flow-Factory/pull/123)
                                kl_div = self._reference_kl_divergence(output, ref_output, replay)
                                kl_loss = self.training_args.kl_beta * kl_div
                                loss += kl_loss
                                loss_info['kl_div'].append(kl_div.detach())
                                loss_info['kl_loss'].append(kl_loss.detach())

                        # 5. Log per-timestep info
                        loss_info['ratio'].append(ratio.detach())
                        loss_info['unclipped_loss'].append(unclipped_loss.detach())
                        loss_info['clipped_loss'].append(clipped_loss.detach())
                        loss_info['policy_loss'].append(policy_loss.detach())
                        loss_info['loss'].append(loss.detach())
                        clip_frac_high = torch.mean((ratio > 1.0 + ratio_clip_range[1]).float())
                        clip_frac_low = torch.mean((ratio < 1.0 + ratio_clip_range[0]).float())
                        loss_info["clip_frac_high"].append(clip_frac_high.detach())
                        loss_info["clip_frac_low"].append(clip_frac_low.detach())
                        loss_info['clip_frac_total'].append((clip_frac_high + clip_frac_low).detach())

                        # 6. Backward and optimizer step
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.adapter.get_trainable_parameters(),
                                self.training_args.max_grad_norm,
                            )
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            # Communicate and log losses
                            loss_info = reduce_loss_info(self.accelerator, loss_info)
                            loss_info['grad_norm'] = grad_norm
                            self.log_data(
                                {f'train/{k}': v for k, v in loss_info.items()},
                                step=self.step,
                            )
                            self.step += 1
                            loss_info = defaultdict(list)

    # =========================== Advantage Computation ============================
    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func: Optional[Union[Literal['sum', 'gdpo'], Callable]] = None,
    ) -> torch.Tensor:
        """Compute advantages — delegates to AdvantageProcessor.

        Args:
            samples: List of BaseSample instances
            rewards: Dict of reward_name to reward tensors aligned with samples
            store_to_samples: Whether to store computed advantages back to samples' extra_kwargs
            aggregation_func: Method to aggregate advantages within each group.
                Options: 'sum' (default GRPO), 'gdpo' (GDPO-style), or a custom callable.
        Returns:
            advantages: Tensor of shape (num_samples, ) with computed advantages
        """
        aggregation_func = aggregation_func or self.training_args.advantage_aggregation
        return self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
        )


# ============================ GRPO-Guard Trainer ============================
class GRPOGuardTrainer(GRPOTrainer):
    """
    GRPOGuard Trainer with reweighted loss.
    References:
    [1] GRPO-Guard: https://arxiv.org/abs/2510.22319
    [2] Temp-FlowGRPO: https://arxiv.org/abs/2508.04324
    """

    # =========================== Sampling Loop ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts for GRPO-Guard with transition means."""
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.get_train_step_indices(),
            num_inference_steps=self.training_args.num_inference_steps,
        )
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=True,
            trajectory_indices=trajectory_indices,
            extra_call_back_kwargs=["next_latents_mean"],
        )

    def _guard_ratio(
        self,
        output: MultiModalStepOutput,
        replay: ReplayStep,
        old_state_mean: LatentState,
    ) -> torch.Tensor:
        """Variance-reweighted importance ratio, per component then DOF-reduced.

        Each component contributes ``(delta_log_prob * scale + mse / (2 * scale))``
        with ``scale = sqrt(-dt) * std_dev_t``; the terms are combined by active
        stochastic degrees of freedom before a single exponentiation.
        """
        expected_names = self.adapter.trajectory_component_order
        new_state_mean = self._require_output_state(output, "next_state_mean", "policy")
        std_dev_t = self._require_component_mapping(output.std_dev_t, "std_dev_t", "policy output")
        dt = self._require_component_mapping(output.dt, "dt", "policy output")
        new_log_probs = self._require_component_mapping(
            output.component_log_probs, "component_log_probs", "policy output"
        )
        old_log_probs = self._require_component_mapping(
            replay.component_log_probs, "component_log_probs", "stored rollout"
        )
        if old_state_mean.component_names != expected_names:
            raise ValueError(
                f"expected stored rollout next_latents_mean in component order "
                f"{expected_names}, received {old_state_mean.component_names}"
            )
        guard_terms = {}
        for name in expected_names:
            scale_factor = torch.sqrt(-dt[name]) * std_dev_t[name]
            mse = (
                (new_state_mean.components[name] - old_state_mean.components[name])
                .flatten(1)
                .pow(2)
                .mean(dim=1)
            )
            guard_terms[name] = (new_log_probs[name] - old_log_probs[name]) * scale_factor + mse / (
                2 * scale_factor
            )
        return torch.exp(
            self.adapter.reduce_latent_values(
                guard_terms,
                active_numel=self.adapter.get_state_active_numel(replay.state),
            )
        )

    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization (Stage 6): GRPO-Guard reweighted loss and optional KL."""
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        train_step_indices = self.adapter.get_train_step_indices()
        # The reweighted ratio always needs the transition mean, its std, and dt.
        return_fields = {'log_prob', 'next_latents_mean', 'std_dev_t', 'dt'}
        if self.enable_kl_loss:
            _, ref_return_field = self._kl_space_fields(self.training_args.kl_type, 'kl_type')
            return_fields.add(ref_return_field)
        else:
            ref_return_field = None
        return_fields = tuple(return_fields)
        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle unless disabled for pack-composition-dependent adapters.
            shuffled_samples = self._order_samples_for_optimize(samples, inner_epoch)

            self.adapter.train()
            loss_info = defaultdict(list)

            # Reload each micro-batch onto the device (H2D when offloaded);
            # _iter_prefetched_batches overlaps the next batch's H2D with compute.
            for batch in tqdm(
                self._iter_prefetched_batches(shuffled_samples, per_device_batch_size),
                total=num_batches,
                desc=f'Epoch {self.epoch} Training',
                position=0,
                disable=not self.show_progress_bar,
            ):
                # Iterate through timesteps
                for timestep_index in tqdm(
                    train_step_indices,
                    desc=f'Epoch {self.epoch} Timestep',
                    position=1,
                    leave=False,
                    disable=not self.show_progress_bar,
                ):
                    step_index = int(timestep_index)
                    with self.accumulate_gradients():
                        # 1. Prepare inputs
                        replay = self.adapter.get_replay_step(batch, step_index)
                        self._require_replay_log_prob(replay, step_index)
                        old_state_mean = self.adapter.get_replay_callback(
                            batch, step_index, 'next_latents_mean'
                        )
                        # 2. Forward pass
                        with self.autocast():
                            output = self._replay_forward(batch, replay, return_fields)

                        # 3. Compute loss
                        # Clip advantages
                        adv = batch['advantage']
                        adv_clip_range = self.training_args.adv_clip_range
                        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                        # Reweighted ratio
                        ratio = self._guard_ratio(output, replay, old_state_mean)
                        # PPO-style clipped loss
                        ratio_clip_range = self.training_args.clip_range

                        unclipped_loss = -adv * ratio
                        clipped_loss = -adv * torch.clamp(ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1])
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        loss = policy_loss

                        # 4. Compute KL-div
                        if self.enable_kl_loss:
                            with self.autocast():
                                ref_output = self._reference_forward(
                                    batch, replay, (ref_return_field,)
                                )
                                # kl_div must be computed outside `torch.no_grad()` for correct gradient behavior.
                                # See: issue #122, PR #123 (https://github.com/X-GenGroup/Flow-Factory/pull/123)
                                kl_div = self._reference_kl_divergence(output, ref_output, replay)
                                kl_loss = self.training_args.kl_beta * kl_div
                                loss += kl_loss
                                loss_info['kl_div'].append(kl_div.detach())
                                loss_info['kl_loss'].append(kl_loss.detach())

                        # 5. Log per-timestep info
                        loss_info['ratio'].append(ratio.detach())
                        loss_info['unclipped_loss'].append(unclipped_loss.detach())
                        loss_info['clipped_loss'].append(clipped_loss.detach())
                        loss_info['policy_loss'].append(policy_loss.detach())
                        loss_info['loss'].append(loss.detach())
                        clip_frac_high = torch.mean((ratio > 1.0 + ratio_clip_range[1]).float())
                        clip_frac_low = torch.mean((ratio < 1.0 + ratio_clip_range[0]).float())
                        loss_info["clip_frac_high"].append(clip_frac_high.detach())
                        loss_info["clip_frac_low"].append(clip_frac_low.detach())
                        loss_info['clip_frac_total'].append((clip_frac_high + clip_frac_low).detach())

                        # 6. Backward and optimizer step
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.adapter.get_trainable_parameters(),
                                self.training_args.max_grad_norm,
                            )
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            # Communicate and log losses
                            loss_info = reduce_loss_info(self.accelerator, loss_info)
                            loss_info['grad_norm'] = grad_norm
                            self.log_data(
                                {f'train/{k}': v for k, v in loss_info.items()},
                                step=self.step,
                            )
                            self.step += 1
                            loss_info = defaultdict(list)