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
from typing import Any, ClassVar, Dict, Iterator, List, Literal, Mapping, Optional, Sequence, Tuple

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
    StructuredTrajectory,
)
from ..abc import BaseTrainer
from .dmd2 import DMD2Trainer
from .distillation_runtime import (
    as_role_microbatches,
    detach_state,
    generate_one_rollout_batch,
    query_score_velocity,
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
from .distribution_matching import tdm_fake_loss, tdm_generator_loss


@dataclass(frozen=True)
class TDMGeneratorScoreTerms:
    """Share one live boundary, noised state, and frozen score queries."""

    loss: torch.Tensor
    boundary_state: LatentState
    times: ComponentTimes
    noised: NoisedState
    reference_velocity: LatentState
    fake_velocity: LatentState


@dataclass(frozen=True)
class TDMBoundaryUnit:
    """Store one replay batch and its primary half-open perturbation interval."""

    samples: Tuple[BaseSample, ...]
    boundary_index: int
    interval_start: torch.Tensor
    interval_end: torch.Tensor


class TDMTrainer(BaseTrainer):
    """Optimize every boundary of a deterministic few-step generator trajectory."""

    paradigm: ClassVar[Literal["distillation"]] = "distillation"

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

    def _validate_trajectory_configuration(self) -> None:
        """Require one deterministic rollout with exactly K generated transitions."""
        trajectory_steps = self.training_args.trajectory_steps
        num_inference_steps = self.training_args.num_inference_steps
        if trajectory_steps != num_inference_steps:
            raise ValueError(
                "TDM requires train.trajectory_steps to equal train.num_inference_steps; "
                f"received trajectory_steps={trajectory_steps} and "
                f"num_inference_steps={num_inference_steps}"
            )
        if self.training_args.num_inner_epochs != 1:
            raise ValueError(
                "TDM requires train.num_inner_epochs=1 so every generator update owns a "
                "fresh rollout; "
                f"received train.num_inner_epochs={self.training_args.num_inner_epochs}"
            )
        non_ode = {
            name: scheduler.dynamics_type
            for name, scheduler in self.adapter.scheduler_group.items()
            if scheduler.dynamics_type != "ODE"
        }
        if non_ode:
            raise ValueError(
                "TDM requires deterministic ODE dynamics for every scheduler component; "
                f"received {non_ode!r}"
            )

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

        Overriding only this keeps the shared epoch loop, so checkpointing and
        eval-time reward monitoring behave exactly as they do for every other
        trainer.
        """
        run_distillation_training_step(self)

    def sample(self) -> List[BaseSample]:
        """Collect the initial state and every generated ODE boundary."""
        self._validate_trajectory_configuration()
        self._validate_media_free_rollout()
        trajectory_indices = list(range(self.training_args.trajectory_steps + 1))
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

    def _build_boundary_units(self, samples: Sequence[BaseSample]) -> List[TDMBoundaryUnit]:
        """Build all K ordered replay boundaries for one dataloader batch."""
        self._validate_trajectory_configuration()
        if not samples:
            return []
        batch_size = self.training_args.per_device_batch_size
        if len(samples) != batch_size:
            raise ValueError(
                "TDM boundary construction requires exactly one replay batch; "
                f"received samples={len(samples)} and per_device_batch_size={batch_size}"
            )

        units: List[TDMBoundaryUnit] = []
        replay_samples = tuple(samples)
        batch = self._stack_replay_unit(replay_samples)
        self._validate_sample_boundaries(replay_samples, batch)
        previous_interval_start: torch.Tensor | None = None
        for boundary_index in range(1, self.training_args.trajectory_steps + 1):
            replay_step = self.adapter.get_replay_step(
                batch,
                boundary_index - 1,
            )
            primary_name = self.adapter.trajectory_component_order[0]
            times = TDMTrainer._normalize_replay_times(self, replay_step.times, len(replay_samples))
            interval_end = times.timestep[primary_name]
            interval_start = times.next_timestep[primary_name]
            self._validate_interval(
                batch,
                times,
                boundary_index=boundary_index,
                interval_start=interval_start,
                interval_end=interval_end,
                previous_interval_start=previous_interval_start,
            )
            units.append(
                TDMBoundaryUnit(
                    samples=replay_samples,
                    boundary_index=boundary_index,
                    interval_start=interval_start.detach().clone(),
                    interval_end=interval_end.detach().clone(),
                )
            )
            previous_interval_start = interval_start
        return units

    def _normalize_replay_times(self, times: ComponentTimes, batch_size: int) -> ComponentTimes:
        """Return one real coordinate per sample for every component of a transition.

        The terminal transition's next coordinate is shared by the batch and synthesized
        as an integer zero. That is the number the schedule means, in a container nothing
        downstream accepts, and it reaches several independent checks -- normalize the
        whole transition once here instead of at each of them.

        Args:
            times: Coordinates as the replay accessor returned them.
            batch_size: Number of samples in the replay unit.

        Returns:
            The same coordinates, each shaped ``(B,)`` in the schedule's real dtype.
        """
        reference = next(iter(times.timestep.values()))

        def widen(mapping: Optional[Mapping[str, torch.Tensor]]) -> Optional[Dict[str, Any]]:
            if mapping is None:
                return None
            return {
                name: TDMTrainer._as_per_sample_coordinate(value, batch_size, like=reference)
                for name, value in mapping.items()
            }

        return ComponentTimes(
            timestep=widen(times.timestep),
            next_timestep=widen(times.next_timestep),
            sigma=widen(times.sigma),
            next_sigma=widen(times.next_sigma),
        )

    @staticmethod
    def _as_per_sample_coordinate(
        coordinate: torch.Tensor,
        batch_size: int,
        like: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a scheduler coordinate as one real value per sample.

        The terminal boundary is shared by the whole batch and is synthesized as an
        integer zero, so it arrives as a scalar of the wrong dtype. Both are the same
        number the schedule means; widen and convert once here rather than teach every
        consumer downstream to accept the other form.

        Args:
            coordinate: Scalar or per-sample scheduler coordinate.
            batch_size: Number of samples in the replay unit.
            like: Coordinate from the same boundary whose dtype and device to match.

        Returns:
            A ``(B,)`` tensor in the schedule's real dtype.

        Raises:
            ValueError: If the coordinate is neither scalar nor ``(B,)``.
        """
        if coordinate.ndim == 0:
            coordinate = coordinate.expand(batch_size)
        elif coordinate.ndim != 1 or coordinate.shape[0] != batch_size:
            raise ValueError(
                f"expected a scalar or ({batch_size},) scheduler coordinate, "
                f"received shape {tuple(coordinate.shape)}"
            )
        if like is not None and coordinate.dtype != like.dtype:
            coordinate = coordinate.to(dtype=like.dtype, device=like.device)
        return coordinate

    def _validate_sample_boundaries(
        self,
        samples: Tuple[BaseSample, ...],
        batch: StackedSampleBatch,
    ) -> None:
        """Require every one of the K + 1 rollout positions to be readable.

        Expressed through the same accessor the replay itself uses, because that is what
        the precondition is about. Reading ``sample.trajectory`` directly would restrict
        the algorithm to the adapters that publish a ``StructuredTrajectory`` -- today
        only the LTX2 pair -- while every other adapter stores the same boundaries in the
        legacy layout that ``get_replay_step`` understands.

        Where a structured trajectory is present its internals are checked too, since
        they say more than the accessor can.

        Args:
            samples: Rollout samples forming one replay unit.
            batch: The same samples, collated.

        Raises:
            ValueError: If any boundary is missing or inconsistently stored.
        """
        for boundary_index in range(self.training_args.trajectory_steps):
            try:
                self.adapter.get_replay_step(batch, boundary_index)
            except (KeyError, IndexError, ValueError, TypeError) as error:
                raise ValueError(
                    f"TDM requires all {self.training_args.trajectory_steps} rollout "
                    f"transitions to be stored; reading transition {boundary_index} failed. "
                    "The rollout must collect every boundary, so "
                    "`trajectory_indices` has to span the full schedule."
                ) from error

        # Unbound on purpose: TDM-R1 is a sibling that delegates method by method, so
        # `self` here need not carry TDM's private helpers.
        TDMTrainer._validate_structured_boundaries(self, samples)

    def _validate_structured_boundaries(self, samples: Tuple[BaseSample, ...]) -> None:
        """Check the stronger invariants an adapter that publishes structure can offer."""
        expected_length = self.training_args.trajectory_steps + 1
        component_order = self.adapter.trajectory_component_order
        expected_map = torch.arange(expected_length, dtype=torch.int64)
        for sample_index, sample in enumerate(samples):
            trajectory = sample.trajectory
            if not isinstance(trajectory, StructuredTrajectory):
                continue
            if trajectory.component_names != component_order:
                raise ValueError(
                    f"TDM expected trajectory component order {component_order}, received "
                    f"{trajectory.component_names} for sample_index={sample_index}"
                )
            primary_name = component_order[0]
            primary_map = trajectory.components[primary_name].state_index_map
            for name in component_order:
                component = trajectory.components[name]
                if component.timesteps.shape[-1] != expected_length:
                    raise ValueError(
                        f"TDM expected K + 1={expected_length} scheduler coordinates for "
                        f"component {name!r}, received {component.timesteps.shape[-1]} "
                        f"for sample_index={sample_index}"
                    )
                if component.states.shape[0] != expected_length:
                    raise ValueError(
                        f"TDM expected stored states length K + 1={expected_length} for "
                        f"component={name!r}, received {component.states.shape[0]} "
                        f"for sample_index={sample_index}"
                    )
                if component.state_index_map.shape[0] != expected_length:
                    raise ValueError(
                        f"TDM expected state_index_map length {expected_length} for component "
                        f"{name!r}, received {component.state_index_map.shape[0]} "
                        f"for sample_index={sample_index}"
                    )
                missing = torch.nonzero(component.state_index_map == -1).flatten().tolist()
                if missing:
                    raise ValueError(
                        f"TDM missing boundary states for component {name!r} at rollout "
                        f"positions={missing}, sample_index={sample_index}; all K + 1 "
                        "boundaries must be collected"
                    )
                if torch.unique(component.state_index_map).numel() != expected_length:
                    raise ValueError(
                        f"TDM requires a strictly one-to-one state_index_map for "
                        f"component={name!r}, received {component.state_index_map.tolist()} "
                        f"for sample_index={sample_index}"
                    )
                if name != primary_name and not torch.equal(
                    component.state_index_map.to(
                        device=primary_map.device,
                        dtype=primary_map.dtype,
                    ),
                    primary_map,
                ):
                    raise ValueError(
                        f"TDM requires component={name!r} state_index_map to match "
                        f"component={primary_name!r}; received "
                        f"{component.state_index_map.tolist()} versus {primary_map.tolist()} "
                        f"for sample_index={sample_index}"
                    )
                if not torch.equal(
                    component.state_index_map.to(
                        device=expected_map.device,
                        dtype=expected_map.dtype,
                    ),
                    expected_map,
                ):
                    raise ValueError(
                        "TDM requires dense aligned state_index_map=arange(K + 1); "
                        f"component={name!r}, expected={expected_map.tolist()}, "
                        f"received={component.state_index_map.tolist()}, "
                        f"sample_index={sample_index}"
                    )

    def _validate_interval(
        self,
        batch: StackedSampleBatch,
        stored_times: ComponentTimes,
        *,
        boundary_index: int,
        interval_start: torch.Tensor,
        interval_end: torch.Tensor,
        previous_interval_start: torch.Tensor | None,
    ) -> None:
        """Validate one descending transition as a half-open shared component interval."""
        if interval_start.shape != interval_end.shape or interval_start.ndim != 1:
            raise ValueError(
                f"TDM boundary_index={boundary_index} expected interval tensors shaped (B,), "
                f"received start={tuple(interval_start.shape)}, end={tuple(interval_end.shape)}"
            )
        self._validate_coordinate(
            interval_start,
            field="primary interval_start timestep",
            boundary_index=boundary_index,
        )
        self._validate_coordinate(
            interval_end,
            field="primary interval_end timestep",
            boundary_index=boundary_index,
        )
        if not bool((interval_start < interval_end).all().item()):
            raise ValueError(
                f"TDM boundary_index={boundary_index} received reversed or empty interval "
                f"[{interval_start.tolist()}, {interval_end.tolist()})"
            )
        if previous_interval_start is not None and not self._coordinates_equal(
            interval_end,
            previous_interval_start,
        ):
            comparison_end, comparison_previous = self._normalize_coordinates(
                interval_end,
                previous_interval_start,
            )
            topology = (
                "overlap" if bool((comparison_end > comparison_previous).any().item()) else "gap"
            )
            raise ValueError(
                f"TDM boundary_index={boundary_index} interval has an exact {topology}; "
                f"previous_start={previous_interval_start.tolist()}, "
                f"current_end={interval_end.tolist()}"
            )
        if boundary_index == self.training_args.trajectory_steps and not self._coordinates_equal(
            interval_start,
            torch.zeros_like(interval_start),
        ):
            raise ValueError(
                "TDM terminal interval must end at exact scheduler coordinate zero; "
                f"received terminal start={interval_start.tolist()}"
            )

        expected_order = self.adapter.trajectory_component_order
        for field_name in ("timestep", "next_timestep", "sigma", "next_sigma"):
            coordinates = getattr(stored_times, field_name)
            if coordinates is None:
                continue
            if tuple(coordinates) != expected_order:
                raise ValueError(
                    f"TDM stored {field_name} expected component order {expected_order}, "
                    f"received {tuple(coordinates)} at boundary_index={boundary_index}"
                )
            for name in expected_order:
                self._validate_coordinate(
                    coordinates[name],
                    field=f"stored {field_name}",
                    component=name,
                    boundary_index=boundary_index,
                )

        mapped_end = self.adapter.build_training_component_times(interval_end, batch=batch)
        mapped_start = self.adapter.build_training_component_times(interval_start, batch=batch)
        for mapped_name, mapped in (("interval_end", mapped_end), ("interval_start", mapped_start)):
            if tuple(mapped.timestep) != expected_order:
                raise ValueError(
                    f"TDM {mapped_name} expected mapped component order {expected_order}, "
                    f"received {tuple(mapped.timestep)} at boundary_index={boundary_index}"
                )
        for name in expected_order:
            self._validate_coordinate(
                mapped_end.timestep[name],
                field="mapped interval_end timestep",
                component=name,
                boundary_index=boundary_index,
            )
            self._validate_coordinate(
                mapped_start.timestep[name],
                field="mapped interval_start timestep",
                component=name,
                boundary_index=boundary_index,
            )
        if stored_times.sigma is not None:
            if mapped_end.sigma is None or mapped_start.sigma is None:
                raise ValueError(
                    "TDM shared interval mapping omitted component sigmas while the stored "
                    f"trajectory provides them at boundary_index={boundary_index}"
                )
            for name in expected_order:
                self._validate_coordinate(
                    mapped_end.sigma[name],
                    field="mapped interval_end sigma",
                    component=name,
                    boundary_index=boundary_index,
                )
                self._validate_coordinate(
                    mapped_start.sigma[name],
                    field="mapped interval_start sigma",
                    component=name,
                    boundary_index=boundary_index,
                )

        for name in expected_order:
            self._require_matching_coordinate(
                mapped_end.timestep[name],
                stored_times.timestep[name],
                component=name,
                endpoint="interval_end",
                boundary_index=boundary_index,
            )
            self._require_matching_coordinate(
                mapped_start.timestep[name],
                stored_times.next_timestep[name],
                component=name,
                endpoint="interval_start",
                boundary_index=boundary_index,
            )
            if stored_times.sigma is not None:
                self._require_matching_coordinate(
                    mapped_end.sigma[name],
                    stored_times.sigma[name],
                    component=name,
                    endpoint="interval_end sigma",
                    boundary_index=boundary_index,
                )
                self._require_matching_coordinate(
                    mapped_start.sigma[name],
                    stored_times.next_sigma[name],
                    component=name,
                    endpoint="interval_start sigma",
                    boundary_index=boundary_index,
                )

    def _require_matching_coordinate(
        self,
        mapped: torch.Tensor,
        stored: torch.Tensor,
        *,
        component: str,
        endpoint: str,
        boundary_index: int,
    ) -> None:
        """Require adapter mapping to reproduce one stored component endpoint."""
        if mapped.shape != stored.shape or not self._coordinates_equal(mapped, stored):
            raise ValueError(
                "TDM component schedules cannot define a shared interval through "
                "adapter.build_training_component_times with exact endpoints; "
                f"component={component!r}, exact endpoint={endpoint!r}, "
                f"boundary_index={boundary_index}, mapped={mapped.tolist()}, "
                f"stored={stored.tolist()}"
            )

    def _sample_perturbation_times(self, unit: TDMBoundaryUnit) -> torch.Tensor:
        """Sample strictly inside one primary scheduler interval."""
        self._validate_coordinate(
            unit.interval_start,
            field="primary interval_start timestep",
            boundary_index=unit.boundary_index,
        )
        self._validate_coordinate(
            unit.interval_end,
            field="primary interval_end timestep",
            boundary_index=unit.boundary_index,
        )
        open_start = torch.nextafter(unit.interval_start, unit.interval_end)
        open_end = torch.nextafter(unit.interval_end, unit.interval_start)
        if not bool((open_start <= open_end).all().item()):
            raise ValueError(
                f"TDM boundary_index={unit.boundary_index} interval has no representable "
                f"floating interior between start={unit.interval_start.tolist()} and "
                f"end={unit.interval_end.tolist()}"
            )
        random_fraction = torch.rand(
            unit.interval_start.shape,
            device=unit.interval_start.device,
            dtype=unit.interval_start.dtype,
        )
        precision = torch.finfo(random_fraction.dtype)
        random_fraction = random_fraction.clamp(
            min=precision.eps,
            max=1.0 - precision.eps,
        )
        sampled = unit.interval_start + (unit.interval_end - unit.interval_start) * random_fraction
        return torch.minimum(torch.maximum(sampled, open_start), open_end)

    def _validate_score_query_sigmas(
        self,
        times: ComponentTimes,
        primary_times: torch.Tensor,
        *,
        boundary_index: int,
    ) -> None:
        """Require finite positive component sigmas before score queries."""
        component_order = self.adapter.trajectory_component_order
        if times.sigma is None:
            raise ValueError(
                "TDM score-query sigma validation expected component sigmas; "
                f"boundary_index={boundary_index}, component={component_order[0]!r}, "
                f"tau={primary_times.tolist()}, received sigma=None"
            )
        if tuple(times.sigma) != component_order:
            raise ValueError(
                "TDM score-query sigma validation expected component order "
                f"{component_order}; boundary_index={boundary_index}, "
                f"tau={primary_times.tolist()}, received sigma components={tuple(times.sigma)}"
            )
        for name in component_order:
            sigma = times.sigma[name]
            if not isinstance(sigma, torch.Tensor) or not sigma.is_floating_point():
                received = (
                    f"dtype={sigma.dtype}, values={sigma.tolist()}"
                    if isinstance(sigma, torch.Tensor)
                    else f"{type(sigma).__name__}: {sigma!r}"
                )
                raise TypeError(
                    "TDM score-query sigma validation expected a floating tensor; "
                    f"boundary_index={boundary_index}, component={name!r}, "
                    f"tau={primary_times.tolist()}, received sigma={received}"
                )
            if not bool((torch.isfinite(sigma) & (sigma > 0)).all().item()):
                raise ValueError(
                    f"TDM score-query sigma validation boundary_index={boundary_index}, "
                    f"component={name!r}, tau={primary_times.tolist()} expected finite and "
                    f"strictly positive values, received sigma={sigma.tolist()}"
                )

    @staticmethod
    def _validate_coordinate(
        coordinate: torch.Tensor,
        *,
        field: str,
        boundary_index: int,
        component: str | None = None,
    ) -> None:
        """Require one finite real floating scheduler coordinate tensor."""
        component_context = "" if component is None else f", component={component!r}"
        if not isinstance(coordinate, torch.Tensor):
            raise TypeError(
                f"TDM {field}{component_context}, boundary_index={boundary_index} expected "
                f"torch.Tensor floating coordinates, received {type(coordinate).__name__}: "
                f"{coordinate!r}"
            )
        if not coordinate.is_floating_point():
            raise TypeError(
                f"TDM {field}{component_context}, boundary_index={boundary_index} expected "
                f"real floating coordinates, received dtype={coordinate.dtype}"
            )
        if not bool(torch.isfinite(coordinate).all().item()):
            raise ValueError(
                f"TDM {field}{component_context}, boundary_index={boundary_index} expected "
                f"finite floating coordinates, received {coordinate}"
            )

    @staticmethod
    def _normalize_coordinates(
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize two validated coordinates to one promoted dtype and device."""
        common_dtype = torch.promote_types(left.dtype, right.dtype)
        return (
            left.to(device=left.device, dtype=common_dtype),
            right.to(device=left.device, dtype=common_dtype),
        )

    @classmethod
    def _coordinates_equal(
        cls,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> bool:
        """Compare scheduler coordinates exactly after safe normalization."""
        if left.shape != right.shape:
            return False
        normalized_left, normalized_right = cls._normalize_coordinates(left, right)
        return torch.equal(normalized_left, normalized_right)

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
        """Compute fake DSM loss for one detached stored trajectory boundary."""
        batch = self._stack_replay_unit(unit.samples)
        replay_step = self.adapter.get_replay_step(
            batch,
            unit.boundary_index - 1,
        )
        boundary_state = detach_state(replay_step.next_state)
        primary_times = self._sample_perturbation_times(unit)
        times = self.adapter.build_training_component_times(primary_times, batch=batch)
        self._validate_score_query_sigmas(
            times,
            primary_times,
            boundary_index=unit.boundary_index,
        )
        noised = self.adapter.add_forward_process_noise(boundary_state, times)
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
        importance = torch.ones(
            next(iter(boundary_state.components.values())).shape[0],
            device=next(iter(boundary_state.components.values())).device,
            dtype=torch.float32,
        )
        primary_sigma = times.sigma[self.adapter.trajectory_component_order[0]]
        return tdm_fake_loss(
            self.adapter,
            predicted_clean,
            boundary_state,
            sigma=primary_sigma,
            importance=importance,
            snr_gamma=self.training_args.tdm_snr_gamma,
        )

    def _generator_boundary_loss(self, unit: TDMBoundaryUnit) -> torch.Tensor:
        """Replay one preceding transition with gradient before detached score queries."""
        return self._generator_score_terms(unit).loss

    def _generator_score_terms(self, unit: TDMBoundaryUnit) -> TDMGeneratorScoreTerms:
        """Replay a live boundary and query frozen scores once on the same noised state."""
        batch = self._stack_replay_unit(unit.samples)
        with self.adapter.use_component_variant("generator"):
            with self.autocast():
                generator_output = self.adapter.replay_generator_boundary(
                    batch,
                    unit.boundary_index,
                    return_fields=("velocity", "next_latents", "next_latents_mean"),
                    rtol=self.training_args.replay_rtol,
                    atol=self.training_args.replay_atol,
                    **self._replay_forward_kwargs(batch),
                )
        boundary_state = generator_output.next_state
        if boundary_state is None:
            raise ValueError(
                "TDM generator replay expected next_state, received None for "
                f"boundary_index={unit.boundary_index}"
            )

        primary_times = self._sample_perturbation_times(unit)
        times = self.adapter.build_training_component_times(primary_times, batch=batch)
        self._validate_score_query_sigmas(
            times,
            primary_times,
            boundary_index=unit.boundary_index,
        )
        noised = self.adapter.add_forward_process_noise(boundary_state, times)
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
        loss = tdm_generator_loss(
            self.adapter,
            boundary_state,
            detach_state(x0_real),
            detach_state(x0_fake),
            use_huber=self.training_args.use_huber,
            huber_c=self.training_args.huber_c,
        )
        return TDMGeneratorScoreTerms(
            loss=loss,
            boundary_state=boundary_state,
            times=times,
            noised=noised,
            reference_velocity=reference_velocity,
            fake_velocity=fake_velocity,
        )

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
