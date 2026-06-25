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

# src/flow_factory/trainers/opd/trainer.py
"""DiffusionOPD on-policy distillation trainer.

Distills several task-specialized LoRA teachers into a single student along
the student's own rollout trajectories using a pathwise mean-matching loss.
Supports both ODE and SDE dynamics — the per-step transition variance is
supplied by ``scheduler.get_kl_divergence_denominator`` so the loss is
dynamics-agnostic.

Reference:
[1] On-Policy Distillation of Diffusion Models — https://github.com/ali-vilab/DiffusionOPD

The distilled denoising steps are selected by ``train.timestep_range`` (a
fraction band of the trajectory step indices; default 0.99 = upstream
``timestep_fraction``), NOT the SDE-only ``scheduler.train_timesteps`` (which is
empty under ODE). See ``_select_train_step_indices``.

Design (2-pass, per epoch):
  sample()    -> student rolls out on-policy trajectories (tagged by source),
                 reusing the standard ``generate_samples`` pipeline.
  optimize()  -> PASS 1 (no_grad): for each teacher (ONE weight swap), forward
                 over its routed samples' stored states x_j and cache the teacher
                 mean mu_T_j on each sample.
              -> PASS 2 (student params only): standard gradient loop that
                 forwards the student at the same x_j and matches mu_S to the
                 cached mu_T_j.

This keeps teacher swaps to M-per-epoch, runs the gradient loop with student
params only (no autocast-cache disable, no DDP bypass), and reuses proven FF
trajectory-replay primitives shared with GRPO.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ...hparams import DiffusionOPDTrainingArguments
from ...hparams.training_args.opd import resolve_distill_step_band
from ...samples import BaseSample
from ...utils.base import filter_kwargs
from ...utils.logger_utils import setup_logger
from ...utils.trajectory_collector import compute_trajectory_indices
from ..abc import BaseTrainer
from .common import load_teachers

logger = setup_logger(__name__)


class DiffusionOPDTrainer(BaseTrainer):
    """Multi-teacher on-policy distillation trainer (ODE + SDE)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.training_args: DiffusionOPDTrainingArguments

        scheduler = self.adapter.scheduler
        self._is_sde = scheduler.dynamics_type != "ODE"
        # Teacher mu_T and student mu_S are computed at the SAME stored state x_j
        # with the SAME noise_level so the transition means are comparable.
        self._student_noise_level = float(scheduler.noise_level) if self._is_sde else 0.0

        # --- Teachers: load each LoRA checkpoint into a named snapshot ---
        teachers = self.training_args.teachers
        self._teacher_names: List[str] = load_teachers(
            self.adapter,
            [teacher.path for teacher in teachers],
            self.training_args.teacher_param_device,
            [teacher.name for teacher in teachers],
        )
        student_gs = float(self.training_args.guidance_scale)
        self._teacher_gs: List[float] = [
            float(teacher.guidance_scale) if teacher.guidance_scale is not None else student_gs
            for teacher in teachers
        ]

        # --- Dataset -> teacher routing ---
        # The config schema permits several teachers to share a dataset (so a
        # future multi-teacher/ensemble trainer can reuse it), but the current
        # DiffusionOPDTrainer distills exactly one teacher per dataset and
        # rejects any overlap below. Routing is keyed on ``BaseSample.source``,
        # which is exactly the dataset name.
        self._source_to_teacher: Dict[str, int] = {}
        for teacher_idx, teacher in enumerate(teachers):
            for dataset in teacher.applicable_datasets:
                if dataset in self._source_to_teacher:
                    raise ValueError(
                        f"Dataset {dataset!r} is claimed by multiple teachers "
                        f"({self._teacher_names[self._source_to_teacher[dataset]]!r} and "
                        f"{self._teacher_names[teacher_idx]!r}). The DiffusionOPD config schema "
                        "permits this for a future multi-teacher/ensemble trainer, but the current "
                        "DiffusionOPDTrainer distills exactly one teacher per dataset."
                    )
                self._source_to_teacher[dataset] = teacher_idx

        # Runtime cross-check against the actually-built per-dataset dataloaders.
        self._available_sources = set(self.train_dataloaders_by_source.keys())
        for teacher_idx, teacher in enumerate(teachers):
            for dataset in teacher.applicable_datasets:
                if dataset not in self._available_sources:
                    raise ValueError(
                        f"Teacher {self._teacher_names[teacher_idx]!r} references dataset {dataset!r} "
                        f"that has no training dataloader. Available datasets: "
                        f"{sorted(self._available_sources)}. Check that `data.datasets` has an entry "
                        "with this `name` and `train.enabled: true`."
                    )

        self._mu_store_device = (
            "cpu" if self.training_args.offload_samples_to_cpu else self.accelerator.device
        )

        logger.info(
            f"DiffusionOPDTrainer initialized: {len(self._teacher_names)} teacher(s) "
            f"{self._teacher_names}, dynamics={scheduler.dynamics_type!r} "
            f"(is_sde={self._is_sde}, student_noise_level={self._student_noise_level}), "
            f"datasets={sorted(self._available_sources)}, "
            f"student_gs={student_gs}, teacher_gs={self._teacher_gs}."
        )

    # =============================== Lifecycle ===============================
    def start(self) -> None:
        """Main training loop (mirrors GRPO/NFT: save -> eval -> sample -> optimize)."""
        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)

            if (
                self.log_args.save_freq > 0
                and self.epoch % self.log_args.save_freq == 0
                and self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    "checkpoints",
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            if self.eval_args.eval_freq > 0 and self.epoch % self.eval_args.eval_freq == 0:
                self.evaluate()

            samples = self.sample()
            self.prepare_feedback(samples)
            self.optimize(samples)

            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    def sample(self) -> List[BaseSample]:
        """Roll out on-policy student trajectories over the multi-source dataloader.

        Stores only the trajectory positions needed for the distilled step band
        (``timestep_range``): current + next latents for each step in the band.
        Rewards are not used by the distillation loss, so no reward buffer is
        attached; reward monitoring is left to :meth:`evaluate`.
        """
        train_step_indices = self._select_train_step_indices(
            self.training_args.num_inference_steps, self.training_args.timestep_range
        )
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=train_step_indices,
            num_inference_steps=self.training_args.num_inference_steps,
        )
        return self.generate_samples(
            reward_buffer=None,
            compute_log_prob=False,
            trajectory_indices=trajectory_indices,
        )

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """No-op: DiffusionOPD has no reward/advantage stage."""
        return

    # =============================== Optimization ===============================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Two-pass distillation: cache teacher means, then student gradient loop."""
        if not samples:
            logger.warning("DiffusionOPD optimize() received no samples; skipping epoch.")
            return

        # Train-mode dynamics for BOTH passes (scheduler.is_eval=False) so the SDE
        # transition means are computed consistently for teacher and student.
        self.adapter.train()
        # Distilled denoising-step band from `timestep_range` (see `_select_train_step_indices`).
        # Same indices the rollout stored in `sample()`, so the replay aligns.
        train_timesteps = self._select_train_step_indices(
            self.training_args.num_inference_steps, self.training_args.timestep_range
        )

        self._precompute_teacher_targets(samples, train_timesteps)
        self._distill(samples, train_timesteps)

    @torch.no_grad()
    def _precompute_teacher_targets(
        self,
        samples: List[BaseSample],
        train_timesteps: torch.Tensor,
    ) -> None:
        """PASS 1: cache each teacher's per-step mean mu_T on its routed samples.

        One ``use_named_parameters`` swap per teacher (performed OUTSIDE the
        autocast block); a per-teacher ``autocast`` scope gives each teacher a
        fresh weight cache (so no stale-cast across swaps), with an explicit
        ``clear_autocast_cache`` as a belt-and-suspenders guard.
        """
        per_device_batch_size = self.training_args.per_device_batch_size

        samples_by_teacher: Dict[int, List[BaseSample]] = defaultdict(list)
        for sample in samples:
            samples_by_teacher[self._teacher_index_for_sample(sample)].append(sample)

        for teacher_idx, teacher_samples in samples_by_teacher.items():
            teacher_name = self._teacher_names[teacher_idx]
            teacher_gs = self._teacher_gs[teacher_idx]
            num_batches = math.ceil(len(teacher_samples) / per_device_batch_size)

            # Swap teacher weights in OUTSIDE the autocast context.
            with self.adapter.use_named_parameters(teacher_name):
                with self.autocast():
                    for batch in tqdm(
                        self._iter_prefetched_batches(
                            teacher_samples, per_device_batch_size
                        ),
                        total=num_batches,
                        desc=f"Epoch {self.epoch} Teacher[{teacher_name}] targets",
                        disable=not self.show_progress_bar,
                    ):
                        # mu_T at each training step: (B, *latent) per step.
                        mu_teacher_steps = [
                            self._forward_step(
                                batch,
                                timestep_index,
                                guidance_scale=teacher_gs,
                                return_kwargs=["next_latents_mean"],
                            )[0].detach()
                            for timestep_index in train_timesteps
                        ]
                        # (B, num_train_steps, *latent)
                        mu_teacher_stacked = torch.stack(mu_teacher_steps, dim=1)
                        for j, sample in enumerate(batch.samples):
                            sample.extra_kwargs["mu_teacher"] = (
                                mu_teacher_stacked[j].to(self._mu_store_device).clone()
                            )
            # Belt-and-suspenders guard against a nested-autocast cache edge case.
            torch.clear_autocast_cache()

    def _distill(
        self,
        samples: List[BaseSample],
        train_timesteps: torch.Tensor,
    ) -> None:
        """PASS 2: student-only gradient loop matching mu_S to the cached mu_T."""
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = math.ceil(len(samples) / per_device_batch_size)

        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle unless disabled for pack-composition-dependent adapters.
            shuffled_samples = self._order_samples_for_optimize(samples, inner_epoch)

            self.adapter.train()
            # Per-teacher KL accumulators over the current gradient-accumulation window.
            # Fixed (num_teachers,) shape so the cross-rank reduce in `_log_distill_metrics`
            # is collective-safe regardless of which teachers each rank's micro-batches held.
            num_teachers = len(self._teacher_names)
            teacher_kl_sum = torch.zeros(num_teachers, device=device)
            teacher_kl_count = torch.zeros(num_teachers, device=device)
            grad_norm = None

            for batch in tqdm(
                self._iter_prefetched_batches(
                    shuffled_samples, per_device_batch_size
                ),
                total=num_batches,
                desc=f"Epoch {self.epoch} Distill",
                position=0,
                disable=not self.show_progress_bar,
            ):
                # Teacher index per sample in this (possibly source-mixed) micro-batch.
                teacher_idx = torch.tensor(
                    [self._teacher_index_for_sample(s) for s in batch.samples],
                    device=device,
                    dtype=torch.long,
                )  # (B,)
                # mu_teacher rides BaseSample.to() with the sample; ensure on device.
                mu_teacher_all = batch["mu_teacher"]
                if not isinstance(mu_teacher_all, torch.Tensor):
                    raise RuntimeError(
                        "Expected cached teacher means `mu_teacher` (a tensor) on every "
                        f"sample, got {type(mu_teacher_all).__name__}. PASS 1 "
                        "(_precompute_teacher_targets) must run before PASS 2."
                    )
                mu_teacher_all = mu_teacher_all.to(device)

                for idx, timestep_index in enumerate(
                    tqdm(
                        train_timesteps,
                        desc=f"Epoch {self.epoch} Timestep",
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    )
                ):
                    with self.accumulate_gradients():
                        with self.autocast():
                            # mu_S: (B, *latent) student transition mean at this step.
                            mu_S, std_dev_t, dt = self._forward_step(
                                batch,
                                timestep_index,
                                guidance_scale=self.training_args.guidance_scale,
                                return_kwargs=["next_latents_mean", "std_dev_t", "dt"],
                            )
                            # Each sample is matched to ITS OWN routed teacher: mu_teacher
                            # was cached per-sample in PASS 1, so a micro-batch may mix teachers.
                            mu_T = mu_teacher_all[:, idx]  # (B, *latent) this sample's teacher mean
                            # Per-sample MSE between student and teacher transition means.
                            per_sample_mse = (
                                (mu_S.float() - mu_T.float()).pow(2).flatten(1).mean(dim=1)
                            )  # (B,)
                            # Transition variance sigma_bar^2: 1.0 (ODE) or (B, 1, 1) (SDE).
                            denom = self.adapter.scheduler.get_kl_divergence_denominator(
                                std_dev_t, dt
                            )
                            if isinstance(denom, torch.Tensor):
                                # denom is per-sample-constant, so reduce (B,1,1) -> (B,).
                                denom = denom.reshape(per_sample_mse.shape[0], -1).mean(
                                    dim=1
                                )  # (B,)
                            per_sample_kl = 0.5 * (per_sample_mse / denom)  # (B,)
                            loss = per_sample_kl.mean()  # scalar (mean over batch)

                        self.accelerator.backward(loss)

                        # Accumulate per-teacher KL sums/counts for logging (detached).
                        with torch.no_grad():
                            teacher_kl_sum.index_add_(0, teacher_idx, per_sample_kl.detach())
                            teacher_kl_count.index_add_(
                                0, teacher_idx, torch.ones_like(per_sample_kl)
                            )

                        if self.accelerator.sync_gradients:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.adapter.get_trainable_parameters(),
                                self.training_args.max_grad_norm,
                            )
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            self._log_distill_metrics(
                                teacher_kl_sum, teacher_kl_count, grad_norm
                            )
                            self.step += 1
                            teacher_kl_sum.zero_()
                            teacher_kl_count.zero_()

    # =============================== Helpers ===============================
    def _log_distill_metrics(
        self,
        teacher_kl_sum: torch.Tensor,
        teacher_kl_count: torch.Tensor,
        grad_norm: Optional[torch.Tensor],
    ) -> None:
        """Globally reduce per-teacher KL sums/counts and log per-teacher + overall means.

        Logs one ``train/kl_div_{teacher_name}`` per teacher seen this window
        (the KL averaged over that teacher's samples x timesteps) plus the
        overall ``train/kl_div`` (averaged across all teachers). The reduce
        operates on fixed ``(num_teachers,)`` tensors, identical on every rank,
        so it is collective-safe even when teachers are unevenly distributed
        across ranks/micro-batches.
        """
        # Pack sum + count into one tensor so the cross-rank reduction is a single
        # collective (the pack-and-reduce idiom used across utils/dist.py).
        packed = torch.stack([teacher_kl_sum, teacher_kl_count])  # (2, num_teachers)
        packed = cast(torch.Tensor, self.accelerator.reduce(packed, reduction="sum"))
        g_sum, g_count = packed[0], packed[1]

        metrics: Dict[str, Any] = {}
        total_count = g_count.sum()
        if total_count > 0:
            metrics["kl_div"] = g_sum.sum() / total_count
        for teacher_idx, name in enumerate(self._teacher_names):
            if g_count[teacher_idx] > 0:
                metrics[f"kl_div_{name}"] = g_sum[teacher_idx] / g_count[teacher_idx]
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm

        self.log_data({f"train/{k}": v for k, v in metrics.items()}, step=self.step)

    @staticmethod
    def _select_train_step_indices(
        num_inference_steps: int,
        timestep_range: Union[float, Tuple[float, float]],
    ) -> torch.Tensor:
        """Trajectory step indices to distill on, from ``timestep_range``.

        ``timestep_range=(frac_lo, frac_hi)`` (a bare float ``f`` is treated as
        ``(0, f)``) selects the contiguous band of denoising transitions
        ``[int(T*frac_lo), int(T*frac_hi))`` where ``T = num_inference_steps``.
        Default ``0.99`` reproduces upstream DiffusionOPD's ``timestep_fraction``
        (distill the first 99% of steps, ``int(10*0.99)=9`` -> indices ``[0..8]``,
        skipping the near-clean tail). Deterministic and dynamics-agnostic, so it
        does NOT use the SDE-only ``scheduler.train_timesteps`` (empty under ODE),
        and gives identical indices in ``sample()`` and ``optimize()``. The band
        comes from :func:`resolve_distill_step_band`, the same resolver
        ``get_num_train_timesteps`` uses for the gradient-accumulation count.
        """
        lo, hi = resolve_distill_step_band(num_inference_steps, timestep_range)
        return torch.arange(lo, hi, dtype=torch.long)

    def _teacher_index_for_sample(self, sample: BaseSample) -> int:
        """Resolve a sample's teacher index from its dataset (``sample.source``).

        Single-dataset configs use a bare DataLoader that does not inject
        ``__source__`` (so ``sample.source`` is None); route those to the sole
        teacher of the only available dataset.
        """
        dataset = sample.source
        if dataset is None:
            if len(self._available_sources) == 1:
                dataset = next(iter(self._available_sources))
            else:
                raise RuntimeError(
                    f"DiffusionOPD sample is missing `source` but {len(self._available_sources)} "
                    "datasets are active; cannot route to a teacher. Multi-dataset rollouts must "
                    "carry `source` (set by MultiSourceTrainDataLoader)."
                )
        if dataset not in self._source_to_teacher:
            raise RuntimeError(
                f"Sample dataset {dataset!r} is not routed to any teacher. "
                f"Routing: {self._source_to_teacher}."
            )
        return self._source_to_teacher[dataset]

    def _forward_step(
        self,
        batch: Dict[str, Any],
        timestep_index: Union[int, torch.Tensor],
        guidance_scale: float,
        return_kwargs: List[str],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward the adapter at stored trajectory step ``timestep_index``.

        Replays the rollout transition ``x_j -> x_{j+1}`` (current/next latents
        and ``t``/``t_next`` come from the stored trajectory, fetched via the
        same index maps GRPO uses). Returns ``(mu, std_dev_t, dt)`` where ``mu``
        is the (validated non-None) transition mean ``next_latents_mean`` and
        ``std_dev_t``/``dt`` are the SDE statistics used by the loss denominator
        (present on the student pass, ``None``/zero under ODE).
        """
        latents_index_map = batch["latent_index_map"]
        num_timesteps = batch["timesteps"].shape[1]

        t = batch["timesteps"][:, timestep_index]
        t_next = (
            batch["timesteps"][:, timestep_index + 1]
            if timestep_index + 1 < num_timesteps
            else torch.zeros_like(t)
        )
        latents = batch["all_latents"][:, latents_index_map[timestep_index]]
        next_latents = batch["all_latents"][:, latents_index_map[timestep_index + 1]]

        forward_inputs = {
            **self.training_args,
            **batch,
            "t": t,
            "t_next": t_next,
            "latents": latents,
            "next_latents": next_latents,
            "compute_log_prob": False,
            "noise_level": self._student_noise_level,
            # guidance_scale set before filter_kwargs so it is dropped for adapters
            # whose forward() does not accept it (overrides the training_args value).
            "guidance_scale": guidance_scale,
        }
        forward_inputs = filter_kwargs(self.adapter.forward, **forward_inputs)
        forward_inputs["return_kwargs"] = list(return_kwargs)
        output = self.adapter.forward(**forward_inputs)

        if output.next_latents_mean is None:
            raise RuntimeError(
                "DiffusionOPD requires `next_latents_mean` from adapter.forward, got None. "
                f"Ensure the adapter/scheduler returns it (return_kwargs={list(return_kwargs)})."
            )
        return output.next_latents_mean, output.std_dev_t, output.dt
