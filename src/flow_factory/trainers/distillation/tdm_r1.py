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

"""Train TDM with a learned surrogate reward and frozen reference."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar, Dict, Iterator, List, Literal, Sequence, Tuple

import torch
from accelerate import Accelerator

from ...hparams import Arguments, TDMR1TrainingArguments
from ...hparams.training_args.tdm_r1 import TDM_R1_DEFAULT_OPTIMIZERS
from ...models.abc import BaseAdapter
from ...samples import BaseSample, LatentState
from ..abc import BaseTrainer
from .distillation_runtime import (
    as_role_microbatches,
    detach_state,
    generate_one_rollout_batch,
    query_score_velocity,
    reference_forward_kwargs,
    require_velocity,
    role_repeat_progress,
    run_distillation_training_step,
    run_role_phase,
)
from .group_preference import GroupPreferenceBatch, group_preference_loss
from .tdm import TDMBoundaryUnit, TDMTrainer


class TDMR1Trainer(BaseTrainer):
    """Reinforce deterministic TDM trajectories through a frozen-reference surrogate."""

    paradigm: ClassVar[Literal["decoupled"]] = "decoupled"

    def _optimizer_args_for_role(self, role_name: str):
        """Resolve this role's optimizer, falling back to TDM-R1's published defaults.

        An ``optimizers`` list in the config file wins, which is also how a run puts
        one of these roles on Muon. Without one, the algorithm supplies the learning
        rates it was published with, because those numbers belong to the algorithm
        rather than to the framework.
        """
        configured = self.config.optimizer_args.get_by_name(role_name)
        if configured is not None:
            return configured
        for default in TDM_R1_DEFAULT_OPTIMIZERS:
            if default.name == role_name:
                return default
        raise ValueError(
            f"expected an optimizer configuration for role {role_name!r}: no "
            f"`optimizers` entry and no TDM-R1 default"
        )

    def __init__(
        self,
        accelerator: Accelerator,
        config: Arguments,
        adapter: BaseAdapter,
    ) -> None:
        super().__init__(accelerator=accelerator, config=config, adapter=adapter)
        self.training_args: TDMR1TrainingArguments
        self._rollout_data_iter: Iterator[Any] | None = None
        self._rollout_dataloader_epoch = 0
        TDMTrainer._validate_trajectory_configuration(self)

    def _validate_trajectory_configuration(self) -> None:
        TDMTrainer._validate_trajectory_configuration(self)

    def _build_boundary_units(self, samples: Sequence[BaseSample]) -> List[TDMBoundaryUnit]:
        return TDMTrainer._build_boundary_units(self, samples)

    def _validate_sample_boundaries(self, samples: Tuple[BaseSample, ...], batch: Any) -> None:
        TDMTrainer._validate_sample_boundaries(self, samples, batch)

    def _validate_interval(self, *args: Any, **kwargs: Any) -> None:
        TDMTrainer._validate_interval(self, *args, **kwargs)

    def _require_matching_coordinate(self, *args: Any, **kwargs: Any) -> None:
        TDMTrainer._require_matching_coordinate(self, *args, **kwargs)

    def _validate_score_query_sigmas(self, *args: Any, **kwargs: Any) -> None:
        TDMTrainer._validate_score_query_sigmas(self, *args, **kwargs)

    @staticmethod
    def _validate_coordinate(*args: Any, **kwargs: Any) -> None:
        TDMTrainer._validate_coordinate(*args, **kwargs)

    @staticmethod
    def _normalize_coordinates(*args: Any, **kwargs: Any):
        return TDMTrainer._normalize_coordinates(*args, **kwargs)

    @classmethod
    def _coordinates_equal(cls, left: torch.Tensor, right: torch.Tensor) -> bool:
        return TDMTrainer._coordinates_equal(left, right)

    def _stack_replay_unit(self, replay_samples: Sequence[BaseSample]) -> Any:
        return TDMTrainer._stack_replay_unit(self, replay_samples)

    def _replay_forward_kwargs(self, batch: Any) -> Dict[str, object]:
        return TDMTrainer._replay_forward_kwargs(self, batch)

    def _reference_forward_kwargs(self, batch: Any) -> Dict[str, object]:
        """Return forward arguments for the real score, which alone may be guided."""
        return reference_forward_kwargs(self.training_args, batch)

    def _sample_perturbation_times(self, unit: TDMBoundaryUnit) -> torch.Tensor:
        return TDMTrainer._sample_perturbation_times(self, unit)

    def _fake_boundary_loss(self, unit: TDMBoundaryUnit) -> torch.Tensor:
        return TDMTrainer._fake_boundary_loss(self, unit)

    def _mean_boundary_loss(self, units: Sequence[TDMBoundaryUnit], loss_fn: Any) -> torch.Tensor:
        return TDMTrainer._mean_boundary_loss(self, units, loss_fn)

    def _run_training_step(self) -> None:
        """Run reward feedback followed by fake, surrogate, and generator updates.

        Overriding only this keeps the shared epoch loop, so checkpointing and
        eval-time reward monitoring behave exactly as they do for every other
        trainer.
        """
        run_distillation_training_step(self)

    def sample(self) -> List[BaseSample]:
        """Generate complete reward groups with every deterministic boundary stored."""
        self._validate_trajectory_configuration()
        trajectory_indices = list(range(self.training_args.trajectory_steps + 1))
        return generate_one_rollout_batch(
            self,
            reward_buffer=self.reward_buffer,
            compute_log_prob=False,
            trajectory_indices=trajectory_indices,
            algorithm_name="TDM-R1",
        )

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Finalize endpoint rewards and store group-normalized advantages."""
        rewards = self.reward_buffer.finalize(store_to_samples=True, split="all")
        self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=True,
            aggregation_func=self.training_args.advantage_aggregation,
        )
        advantage_metrics = self.advantage_processor.pop_advantage_metrics()
        if advantage_metrics:
            self.log_data(advantage_metrics, step=self.step)

    def optimize(self, samples: Sequence[Any]) -> None:
        """Run fake TTUR, one surrogate step, then one generator step."""
        if not samples:
            return
        microbatches = as_role_microbatches(
            samples,
            batch_size=self.training_args.per_device_batch_size,
            accumulation_steps=self.training_args.gradient_accumulation_steps,
            algorithm_name="TDM-R1",
        )
        self.adapter.train()
        for _ in role_repeat_progress(
            self, role_name="fake", repeats=self.training_args.ttur_fake_updates
        ):
            self._fake_phase(microbatches)
        self._surrogate_phase(microbatches)
        self._generator_phase(microbatches)

    def _fake_phase(self, microbatches: Sequence[Sequence[BaseSample]]) -> None:
        """Fit the fake score over the complete ordered boundary window."""
        TDMTrainer._fake_phase(self, microbatches)

    def _surrogate_phase(self, microbatches: Sequence[Sequence[BaseSample]]) -> None:
        """Update the surrogate with group preference on endpoint advantages."""
        run_role_phase(
            self,
            "surrogate",
            microbatches,
            lambda batch: self._mean_boundary_loss(
                self._build_boundary_units(batch),
                self._surrogate_boundary_loss,
            ),
        )

    def _generator_phase(self, microbatches: Sequence[Sequence[BaseSample]]) -> None:
        """Update the generator with TDM plus a scaled preference term."""
        run_role_phase(
            self,
            "generator",
            microbatches,
            lambda batch: self._mean_boundary_loss(
                self._build_boundary_units(batch),
                self._generator_boundary_loss,
            ),
        )

    def _surrogate_boundary_loss(self, unit: TDMBoundaryUnit) -> torch.Tensor:
        """Score one stored boundary with the surrogate and apply group preference."""
        trainable_values, reference_values = self._boundary_preference_values(
            unit,
            trainable_role="surrogate",
        )
        preference_batch = self._group_preference_batch(unit, trainable_values)
        return group_preference_loss(
            self.accelerator,
            preference_batch,
            trainable_values,
            reference_values,
            self.training_args.surrogate_preference_beta,
        )

    def _generator_boundary_loss(self, unit: TDMBoundaryUnit) -> torch.Tensor:
        """Combine the TDM generator term with a scaled live-boundary preference term."""
        terms = self._generator_score_terms(unit)
        trainable_values, reference_values = self._live_generator_preference_values(unit, terms)
        preference_batch = self._group_preference_batch(unit, trainable_values)
        preference_loss = group_preference_loss(
            self.accelerator,
            preference_batch,
            trainable_values,
            reference_values,
            self.training_args.surrogate_preference_beta,
        )
        return terms.loss + self.training_args.tdm_weight * preference_loss

    def _generator_score_terms(self, unit: TDMBoundaryUnit) -> Any:
        return TDMTrainer._generator_score_terms(self, unit)

    def _live_generator_preference_values(
        self,
        unit: TDMBoundaryUnit,
        terms: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score the live generated boundary, reusing the TDM reference query."""
        batch = self._stack_replay_unit(unit.samples)
        with self.adapter.use_component_variant("generator"):
            with self.autocast():
                trainable_output = self.adapter.forward_state(
                    batch=batch,
                    state=terms.noised.state,
                    times=terms.times,
                    compute_log_prob=False,
                    return_fields=("velocity",),
                    **self._replay_forward_kwargs(batch),
                )
        trainable_velocity = require_velocity(
            trainable_output,
            algorithm_name="TDM-R1",
            role_name="generator",
        )
        trainable_values = self._diffusion_density_values(
            terms.noised.target_velocity,
            trainable_velocity,
            state=terms.noised.state,
        )
        reference_values = self._diffusion_density_values(
            terms.noised.target_velocity,
            terms.reference_velocity,
            state=terms.noised.state,
        )
        return trainable_values, reference_values.detach()

    def _boundary_preference_values(
        self,
        unit: TDMBoundaryUnit,
        *,
        trainable_role: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-sample DSM values for a trainable role and frozen reference."""
        batch = self._stack_replay_unit(unit.samples)
        replay_step = self.adapter.get_replay_step(batch, unit.boundary_index - 1)
        boundary_state = detach_state(replay_step.next_state)
        primary_times = self._sample_perturbation_times(unit)
        times = self.adapter.build_training_component_times(primary_times, batch=batch)
        self._validate_score_query_sigmas(
            times,
            primary_times,
            boundary_index=unit.boundary_index,
        )
        noised = self.adapter.add_forward_process_noise(boundary_state, times)
        with self.adapter.use_component_variant(trainable_role):
            with self.autocast():
                trainable_output = self.adapter.forward_state(
                    batch=batch,
                    state=noised.state,
                    times=times,
                    compute_log_prob=False,
                    return_fields=("velocity",),
                    **self._replay_forward_kwargs(batch),
                )
        trainable_velocity = require_velocity(
            trainable_output,
            algorithm_name="TDM-R1",
            role_name=trainable_role,
        )
        reference_velocity = query_score_velocity(
            self.adapter,
            batch,
            noised.state,
            times,
            role_name="reference",
            autocast=self.autocast,
            forward_kwargs=self._reference_forward_kwargs(batch),
            algorithm_name="TDM-R1",
        )
        trainable_values = self._diffusion_density_values(
            noised.target_velocity,
            trainable_velocity,
            state=noised.state,
        )
        reference_values = self._diffusion_density_values(
            noised.target_velocity,
            reference_velocity,
            state=noised.state,
        )
        return trainable_values, reference_values.detach()

    def _diffusion_density_values(
        self,
        target_velocity: LatentState,
        predicted_velocity: LatentState,
        *,
        state: LatentState,
    ) -> torch.Tensor:
        """Return per-sample squared velocity error used as a preference logit."""
        expected_names = self.adapter.trajectory_component_order
        if target_velocity.component_names != expected_names:
            raise ValueError(
                "TDM-R1 expected target velocity component order "
                f"{expected_names}, received {target_velocity.component_names}"
            )
        if predicted_velocity.component_names != expected_names:
            raise ValueError(
                "TDM-R1 expected predicted velocity component order "
                f"{expected_names}, received {predicted_velocity.component_names}"
            )
        errors = {
            name: (target_velocity.components[name] - predicted_velocity.components[name]).square()
            for name in expected_names
        }
        return self.adapter.reduce_latent_values(errors, state=state)

    def _group_preference_batch(
        self,
        unit: TDMBoundaryUnit,
        values: torch.Tensor,
    ) -> GroupPreferenceBatch:
        """Build dense rank-local group indices for one complete replay unit."""
        unique_ids = torch.as_tensor(
            [int(sample.unique_id) for sample in unit.samples],
            device=values.device,
            dtype=torch.int64,
        )
        local_group_indices = torch.unique(unique_ids, return_inverse=True)[1]
        num_groups = int(torch.unique(local_group_indices).shape[0])
        group_size = self.training_args.group_size
        counts = torch.bincount(local_group_indices, minlength=num_groups)
        expected_counts = torch.full(
            (num_groups,),
            group_size,
            device=counts.device,
            dtype=counts.dtype,
        )
        if not torch.equal(counts, expected_counts):
            raise ValueError(
                "TDM-R1 expected every rank-local group to contain exactly "
                f"group_size={group_size} members in one microbatch, received "
                f"counts={counts.tolist()} for unique_ids={unique_ids.tolist()}"
            )
        advantages = self._advantages_for_samples(unit.samples, values)
        return GroupPreferenceBatch(
            local_group_indices=local_group_indices,
            num_groups=num_groups,
            group_size=group_size,
            advantages=advantages,
            reduce_across_ranks=False,
        )

    def _advantages_for_samples(
        self,
        samples: Sequence[BaseSample],
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Return validated endpoint advantages aligned with diffusion values."""
        advantage_rows = []
        for sample_index, sample in enumerate(samples):
            advantage = sample.extra_kwargs.get("advantage")
            if not isinstance(advantage, (Real, torch.Tensor)) or isinstance(advantage, bool):
                raise TypeError(
                    "TDM-R1 expected scalar numeric endpoint advantage in "
                    f"sample.extra_kwargs for sample_index={sample_index}, received "
                    f"{type(advantage).__name__}: {advantage!r}"
                )
            if isinstance(advantage, torch.Tensor) and not advantage.is_floating_point():
                raise TypeError(
                    "TDM-R1 expected floating endpoint advantage tensor for "
                    f"sample_index={sample_index}, received dtype={advantage.dtype}"
                )
            advantage_row = torch.as_tensor(
                advantage,
                device=values.device,
                dtype=values.dtype,
            )
            if advantage_row.ndim != 0:
                raise ValueError(
                    "TDM-R1 expected scalar endpoint advantage for "
                    f"sample_index={sample_index}, received shape "
                    f"{tuple(advantage_row.shape)}"
                )
            advantage_rows.append(advantage_row)
        advantages = torch.stack(advantage_rows)
        if advantages.shape != values.shape:
            raise ValueError(
                "TDM-R1 expected one endpoint advantage per diffusion value, received "
                f"advantages_shape={tuple(advantages.shape)} and "
                f"values_shape={tuple(values.shape)}"
            )
        clip_range = self.training_args.advantage_clip_range
        return advantages.clamp(-clip_range, clip_range)
