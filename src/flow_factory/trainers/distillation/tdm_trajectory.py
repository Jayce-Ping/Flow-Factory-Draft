"""Trajectory topology, coordinates, and boundary construction for TDM trainers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ...samples import (
    BaseSample,
    ComponentTimes,
    StackedSampleBatch,
    StructuredTrajectory,
)


@dataclass(frozen=True)
class TDMBoundaryUnit:
    """Store one replay batch and its primary half-open perturbation interval."""

    samples: tuple[BaseSample, ...]
    boundary_index: int
    interval_start: torch.Tensor
    interval_end: torch.Tensor


class TDMTrajectoryRuntimeMixin:
    """Validate and construct dense deterministic TDM trajectory boundaries."""

    def _validate_trajectory_configuration(self) -> None:
        """Require one deterministic rollout with exactly K generated transitions."""
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

    def _build_boundary_units(
        self,
        samples: Sequence[BaseSample],
    ) -> list[TDMBoundaryUnit]:
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

        units: list[TDMBoundaryUnit] = []
        replay_samples = tuple(samples)
        batch = self._stack_replay_unit(replay_samples)
        self._validate_sample_boundaries(replay_samples, batch)
        previous_interval_start: torch.Tensor | None = None
        for boundary_index in range(1, self.training_args.num_inference_steps + 1):
            replay_step = self.adapter.get_replay_step(
                batch,
                boundary_index - 1,
            )
            primary_name = self.adapter.trajectory_component_order[0]
            times = TDMTrajectoryRuntimeMixin._normalize_replay_times(
                self,
                replay_step.times,
                len(replay_samples),
            )
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

    def _normalize_replay_times(
        self,
        times: ComponentTimes,
        batch_size: int,
    ) -> ComponentTimes:
        """Return one real coordinate per sample for every component of a transition."""
        reference = next(iter(times.timestep.values()))

        def widen(
            mapping: Mapping[str, torch.Tensor] | None,
        ) -> dict[str, Any] | None:
            if mapping is None:
                return None
            return {
                name: TDMTrajectoryRuntimeMixin._as_per_sample_coordinate(
                    value,
                    batch_size,
                    like=reference,
                )
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
        like: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a scheduler coordinate as one real value per sample."""
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
        samples: tuple[BaseSample, ...],
        batch: StackedSampleBatch,
    ) -> None:
        """Require every one of the K + 1 rollout positions to be readable."""
        for boundary_index in range(self.training_args.num_inference_steps):
            try:
                self.adapter.get_replay_step(batch, boundary_index)
            except (KeyError, IndexError, ValueError, TypeError) as error:
                raise ValueError(
                    f"TDM requires all {self.training_args.num_inference_steps} rollout "
                    f"transitions to be stored; reading transition {boundary_index} failed. "
                    "The rollout must collect every boundary, so "
                    "`trajectory_indices` has to span the full schedule."
                ) from error

        TDMTrajectoryRuntimeMixin._validate_structured_boundaries(self, samples)

    def _validate_structured_boundaries(self, samples: tuple[BaseSample, ...]) -> None:
        """Check the stronger invariants an adapter that publishes structure can offer."""
        expected_length = self.training_args.num_inference_steps + 1
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
                        "TDM requires a strictly one-to-one state_index_map for "
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
        if (
            boundary_index == self.training_args.num_inference_steps
            and not self._coordinates_equal(
                interval_start,
                torch.zeros_like(interval_start),
            )
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
        for mapped_name, mapped in (
            ("interval_end", mapped_end),
            ("interval_start", mapped_start),
        ):
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
