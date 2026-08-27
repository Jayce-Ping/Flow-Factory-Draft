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

"""Train a deterministic few-step generator with trajectory distribution matching."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterator, List, Literal, Sequence, Tuple

import torch
from accelerate import Accelerator

from ...hparams import Arguments, TDMTrainingArguments
from ...hparams.training_args.dmd2 import DMD2_DEFAULT_OPTIMIZERS
from ...models.abc import BaseAdapter
from ...samples import (
    BaseSample,
    ComponentTimes,
    LatentState,
    NoisedState,
    StackedSampleBatch,
)
from ..abc import BaseTrainer
from ..execution import ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT
from .distillation_runtime import (
    as_role_microbatches,
    detach_state,
    generate_one_rollout_batch,
    query_score_velocity,
    record_distillation_metric,
    record_state_statistics,
    reference_forward_kwargs,
    reject_training_rewards,
    replay_forward_kwargs,
    require_velocity,
    role_repeat_progress,
    run_distillation_training_step,
    run_role_phase,
    validate_media_free_rollout,
    without_media_decoding,
)
from .distribution_matching import (
    tdm_conditional_renoise,
    tdm_fake_loss,
    tdm_generator_loss,
)
from .dmd2 import DMD2Trainer
from .tdm_trajectory import TDMBoundaryUnit, TDMTrajectoryRuntimeMixin


@dataclass(frozen=True)
class TDMGeneratorScoreTerms:
    """Share one live boundary, noised state, and frozen score queries."""

    loss: torch.Tensor
    boundary_state: LatentState
    times: ComponentTimes
    noised: NoisedState
    reference_velocity: LatentState
    fake_velocity: LatentState
    x0_real: LatentState
    x0_fake: LatentState


class TDMTrainer(TDMTrajectoryRuntimeMixin, BaseTrainer):
    """Optimize every boundary of a deterministic few-step generator trajectory."""

    paradigm: ClassVar[Literal["distillation"]] = "distillation"
    execution_contract = ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT

    def _optimizer_args_for_role(self, role_name: str):
        """Resolve this role's optimizer, falling back to TDM's published defaults.

        An ``optimizers`` list in the config file wins, which is also how a run puts
        one of these roles on Muon. Without one, the algorithm supplies the learning
        rates it was published with, because those numbers belong to the algorithm
        rather than to the framework. TDM reuses DMD2's generator and fake-score rates.
        """
        configured = self.config.optimizer_args.get_by_name(role_name)
        if configured is not None:
            return configured
        for default in DMD2_DEFAULT_OPTIMIZERS:
            if default.name == role_name:
                return default
        raise ValueError(
            f"expected an optimizer configuration for role {role_name!r}: no "
            f"`optimizers` entry and no TDM default"
        )

    def __init__(
        self,
        accelerator: Accelerator,
        config: Arguments,
        adapter: BaseAdapter,
    ) -> None:
        super().__init__(accelerator=accelerator, config=config, adapter=adapter)
        self.training_args: TDMTrainingArguments
        self._rollout_data_iter: Iterator[Any] | None = None
        self._rollout_dataloader_epoch = 0
        self._validate_trajectory_configuration()

    def _init_reward_model(self) -> Tuple[Dict[str, object], Dict[str, object]]:
        """Build the shared feedback runtime, which for this algorithm is eval-only.

        See :meth:`DMD2Trainer._init_reward_model`: the reward-free training contract is
        enforced in `Arguments`, and zeroing the runtime here removed eval monitoring
        along with it.

        Returns:
            Training and eval reward models; the training mapping is always empty.
        """
        return reject_training_rewards(self, algorithm_name="TDM")

    def _run_training_step(self) -> None:
        """Run GAS distinct trajectory rollouts and one fake/generator phase pair.

        Overriding only this keeps the shared rollout-iteration loop, so checkpointing and
        eval-time reward monitoring behave exactly as they do for every other
        trainer.
        """
        run_distillation_training_step(self)

    def sample(self) -> List[BaseSample]:
        """Collect the initial state and every generated ODE boundary."""
        self._validate_trajectory_configuration()
        self._validate_media_free_rollout()
        trajectory_indices = list(range(self.training_args.num_inference_steps + 1))
        with self._without_media_decoding():
            return generate_one_rollout_batch(
                self,
                reward_buffer=None,
                compute_log_prob=False,
                trajectory_indices=trajectory_indices,
                algorithm_name="TDM",
            )

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Perform no feedback work for the data-free objective."""
        del samples

    def optimize(self, samples: Sequence[Any]) -> None:
        """Run fake TTUR updates, then one generator step, over GAS replay batches."""
        if not samples:
            return
        microbatches = as_role_microbatches(
            samples,
            batch_size=self.training_args.per_device_batch_size,
            accumulation_steps=self.training_args.gradient_accumulation_steps,
            algorithm_name="TDM",
        )
        self.adapter.train()
        for _ in role_repeat_progress(
            self, role_name="fake", repeats=self.training_args.ttur_fake_updates
        ):
            self._fake_phase(microbatches)
        self._generator_phase(microbatches)

    def _mean_boundary_loss(
        self,
        units: Sequence[TDMBoundaryUnit],
        loss_fn,
    ) -> torch.Tensor:
        """Average one complete ordered boundary window into a single scalar."""
        losses = [loss_fn(unit) for unit in units]
        return torch.stack(losses).mean()

    def _fake_phase(self, microbatches: Sequence[Sequence[BaseSample]]) -> None:
        """Fit the fake score over one complete ordered boundary window per microbatch."""
        run_role_phase(
            self,
            "fake",
            microbatches,
            lambda batch: self._mean_boundary_loss(
                self._build_boundary_units(batch),
                self._fake_boundary_loss,
            ),
        )

    def _generator_phase(self, microbatches: Sequence[Sequence[BaseSample]]) -> None:
        """Update the generator over the identical ordered boundary window per microbatch."""
        run_role_phase(
            self,
            "generator",
            microbatches,
            lambda batch: self._mean_boundary_loss(
                self._build_boundary_units(batch),
                self._generator_boundary_loss,
            ),
        )

    def _fake_boundary_loss(self, unit: TDMBoundaryUnit) -> torch.Tensor:
        """Compute official conditionally-renoised fake DSM loss for one stage."""
        with torch.no_grad():
            batch, clean_state, model_noise = self._replay_generator_prediction(unit)
        detached_clean = detach_state(clean_state)
        mid_times = self.adapter.build_training_component_times(unit.interval_start, batch=batch)
        primary_times = self._sample_perturbation_times(unit)
        times = self.adapter.build_training_component_times(primary_times, batch=batch)
        self._validate_score_query_sigmas(
            times,
            primary_times,
            boundary_index=unit.boundary_index,
        )
        noised, importance = tdm_conditional_renoise(
            self.adapter,
            detached_clean,
            detach_state(model_noise),
            mid_times=mid_times,
            target_times=times,
            importance_clip=self.training_args.tdm_importance_clip,
        )
        with self.adapter.use_component_variant("fake"):
            with self.autocast():
                output = self.adapter.forward_state(
                    batch=batch,
                    state=noised.state,
                    times=times,
                    compute_log_prob=False,
                    return_fields=("velocity",),
                    **self._replay_forward_kwargs(batch),
                )
        velocity = require_velocity(output, algorithm_name="TDM", role_name="fake")
        predicted_clean = self.adapter.project_velocity_to_clean_state(
            noised.state,
            times,
            velocity,
        )
        primary_sigma = times.sigma[self.adapter.trajectory_component_order[0]]
        return tdm_fake_loss(
            self.adapter,
            predicted_clean,
            detached_clean,
            sigma=primary_sigma,
            importance=importance,
            snr_gamma=self.training_args.tdm_snr_gamma,
        )

    def _generator_boundary_loss(self, unit: TDMBoundaryUnit) -> torch.Tensor:
        """Replay one preceding transition with gradient before detached score queries."""
        return self._generator_score_terms(unit).loss

    def _generator_score_terms(self, unit: TDMBoundaryUnit) -> TDMGeneratorScoreTerms:
        """Replay a live clean prediction and query scores on conditional stage noise."""
        batch, clean_state, model_noise = self._replay_generator_prediction(unit)
        mid_times = self.adapter.build_training_component_times(unit.interval_start, batch=batch)
        primary_times = self._sample_perturbation_times(unit)
        times = self.adapter.build_training_component_times(primary_times, batch=batch)
        self._validate_score_query_sigmas(
            times,
            primary_times,
            boundary_index=unit.boundary_index,
        )
        noised, _ = tdm_conditional_renoise(
            self.adapter,
            detach_state(clean_state),
            detach_state(model_noise),
            mid_times=mid_times,
            target_times=times,
            importance_clip=self.training_args.tdm_importance_clip,
        )
        reference_velocity = query_score_velocity(
            self.adapter,
            batch,
            noised.state,
            times,
            role_name="reference",
            autocast=self.autocast,
            forward_kwargs=self._reference_forward_kwargs(batch),
            algorithm_name="TDM",
        )
        fake_velocity = query_score_velocity(
            self.adapter,
            batch,
            noised.state,
            times,
            role_name="fake",
            autocast=self.autocast,
            forward_kwargs=self._replay_forward_kwargs(batch),
            algorithm_name="TDM",
        )
        x0_real = self.adapter.project_velocity_to_clean_state(
            noised.state,
            times,
            reference_velocity,
        )
        x0_fake = self.adapter.project_velocity_to_clean_state(
            noised.state,
            times,
            fake_velocity,
        )
        record_state_statistics(self, "train/x0_gen", clean_state)
        record_state_statistics(self, "train/x0_real", x0_real)
        record_state_statistics(self, "train/x0_fake", x0_fake)
        record_distillation_metric(self, "train/boundary_index", unit.boundary_index)
        loss = tdm_generator_loss(
            self.adapter,
            clean_state,
            detach_state(x0_real),
            detach_state(x0_fake),
            use_huber=self.training_args.use_huber,
            huber_c=self.training_args.huber_c,
        )
        return TDMGeneratorScoreTerms(
            loss=loss,
            boundary_state=clean_state,
            times=times,
            noised=noised,
            reference_velocity=reference_velocity,
            fake_velocity=fake_velocity,
            # Detached: these are frozen score queries, and every consumer uses them as
            # a target or a scale rather than a path for the generator's gradient.
            x0_real=detach_state(x0_real),
            x0_fake=detach_state(x0_fake),
        )

    def _replay_generator_prediction(
        self,
        unit: TDMBoundaryUnit,
    ) -> Tuple[StackedSampleBatch, LatentState, LatentState]:
        """Replay a stage and return its live clean prediction and implied noise."""
        batch = self._stack_replay_unit(unit.samples)
        replay_step = self.adapter.get_replay_step(batch, unit.boundary_index - 1)
        with self.adapter.use_component_variant("generator"):
            with self.autocast():
                output = self.adapter.replay_generator_boundary(
                    batch,
                    unit.boundary_index,
                    return_fields=("velocity", "next_latents", "next_latents_mean"),
                    rtol=self.training_args.replay_rtol,
                    atol=self.training_args.replay_atol,
                    **self._replay_forward_kwargs(batch),
                )
        velocity = require_velocity(output, algorithm_name="TDM", role_name="generator")
        primary_name = self.adapter.trajectory_component_order[0]
        projection_times = self.adapter.build_training_component_times(
            replay_step.times.timestep[primary_name],
            batch=batch,
        )
        clean_state = self.adapter.project_velocity_to_clean_state(
            replay_step.state,
            projection_times,
            velocity,
        )
        direction = self.adapter.flow_velocity_direction
        if direction not in ("noise", "data"):
            raise ValueError(
                "TDM expected adapter.flow_velocity_direction in ('noise', 'data'), "
                f"received {direction!r}"
            )
        sign = 1.0 if direction == "noise" else -1.0
        model_noise = LatentState(
            {
                name: (
                    clean_state.components[name].to(torch.float32)
                    + sign * velocity.components[name].to(torch.float32)
                ).detach()
                for name in self.adapter.trajectory_component_order
            },
            active_masks=clean_state.active_masks,
        )
        return batch, clean_state, model_noise

    def _stack_replay_unit(
        self,
        replay_samples: Sequence[BaseSample],
    ) -> StackedSampleBatch:
        """Move and stack one boundary unit without generated media."""
        if not replay_samples:
            raise ValueError("expected a non-empty TDM boundary unit, received no samples")
        return BaseSample.stack([sample.to(self.accelerator.device) for sample in replay_samples])

    def _replay_forward_kwargs(self, batch: StackedSampleBatch) -> Dict[str, object]:
        """Return allow-listed adapter arguments not already owned by the batch."""
        return replay_forward_kwargs(self.training_args, batch)

    def _reference_forward_kwargs(self, batch: StackedSampleBatch) -> Dict[str, object]:
        """Return forward arguments for the real score, which alone may be guided."""
        return reference_forward_kwargs(self.training_args, batch)

    def _validate_media_free_rollout(self) -> None:
        """Require inference to expose a suppressible media reconstruction seam."""
        validate_media_free_rollout(self.adapter, algorithm_name="TDM")

    @contextmanager
    def _without_media_decoding(self) -> Iterator[None]:
        """Replace media reconstruction with shape-preserving empty outputs."""
        with without_media_decoding(self.adapter, algorithm_name="TDM"):
            yield
