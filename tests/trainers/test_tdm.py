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

from contextlib import contextmanager, nullcontext
from types import MethodType, SimpleNamespace
from typing import Any, Iterator, Mapping, Optional, Tuple

import pytest
import torch
from accelerate import Accelerator

from flow_factory.hparams import Arguments, TDMTrainingArguments
from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import (
    BaseSample,
    ComponentTimes,
    ComponentTrajectory,
    LatentState,
    MultiModalStepOutput,
    StackedSampleBatch,
    StructuredTrajectory,
)
from flow_factory.scheduler import SDESchedulerOutput
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.distillation.tdm import TDMBoundaryUnit, TDMTrainer
from flow_factory.trainers.role_optimization import (
    OptimizationRole,
    RoleOptimizationCoordinator,
    RoleOptimizerConfig,
)


class TinyTDMAdapter(BaseAdapter):
    """Expose a two-component aligned schedule for TDM unit tests."""

    trajectory_component_order = ("video", "audio")

    def __init__(self) -> None:
        self.scheduler_group = {
            "video": SimpleNamespace(dynamics_type="ODE"),
            "audio": SimpleNamespace(dynamics_type="ODE"),
        }

    def load_pipeline(self) -> Any:
        raise NotImplementedError

    def decode_latents(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("TDM must not decode generated media")

    def inference(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward(self, *args: Any, **kwargs: Any) -> SDESchedulerOutput:
        raise NotImplementedError

    def train(self, mode: bool = True) -> None:
        del mode

    def build_training_component_times(
        self,
        primary_timesteps: torch.Tensor,
        *,
        batch: Optional[StackedSampleBatch] = None,
    ) -> ComponentTimes:
        del batch
        audio_timesteps = primary_timesteps * 0.5
        return ComponentTimes(
            timestep={"video": primary_timesteps, "audio": audio_timesteps},
            next_timestep={
                "video": torch.zeros_like(primary_timesteps),
                "audio": torch.zeros_like(audio_timesteps),
            },
            sigma={
                "video": primary_timesteps / 1000,
                "audio": audio_timesteps / 500,
            },
            next_sigma={
                "video": torch.zeros_like(primary_timesteps),
                "audio": torch.zeros_like(audio_timesteps),
            },
        )

    def _forward_state(
        self,
        *,
        batch: StackedSampleBatch,
        state: LatentState,
        times: ComponentTimes,
        next_state: Optional[LatentState],
        compute_log_prob: bool,
        return_fields: Tuple[str, ...],
        noise_level: Optional[float],
        forward_kwargs: Mapping[str, Any],
    ) -> MultiModalStepOutput:
        del batch, times, next_state, compute_log_prob, return_fields, noise_level, forward_kwargs
        return MultiModalStepOutput(next_state=state, velocity=state)


class ObjectiveBundle(torch.nn.Module):
    """Own generator, fake, and forbidden auxiliary parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.generator = torch.nn.Parameter(torch.tensor(1.0))
        self.fake = torch.nn.Parameter(torch.tensor(-0.2))
        self.auxiliary = torch.nn.Parameter(torch.tensor(0.7))


class ObjectiveTDMAdapter(BaseAdapter):
    """Exercise real TDM replay, score-query, and component RNG paths."""

    trajectory_component_order = ("video", "audio")

    def __init__(self, bundle: ObjectiveBundle) -> None:
        self.bundle = bundle
        self.active_role = "generator"
        self.events: list[tuple[str, bool, int, int]] = []
        self.noise_draws: list[LatentState] = []
        self.mapped_primary_times: list[torch.Tensor] = []
        self.mapped_component_times: list[ComponentTimes] = []
        self.scheduler_group = {
            "video": SimpleNamespace(dynamics_type="ODE"),
            "audio": SimpleNamespace(dynamics_type="ODE"),
        }

    def load_pipeline(self) -> Any:
        raise NotImplementedError

    def decode_latents(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("TDM must not decode generated media")

    def inference(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward(self, *args: Any, **kwargs: Any) -> SDESchedulerOutput:
        raise NotImplementedError

    def train(self, mode: bool = True) -> None:
        del mode

    def build_training_component_times(
        self,
        primary_timesteps: torch.Tensor,
        *,
        batch: Optional[StackedSampleBatch] = None,
    ) -> ComponentTimes:
        del batch
        audio_timesteps = primary_timesteps * 0.5
        times = ComponentTimes(
            timestep={"video": primary_timesteps, "audio": audio_timesteps},
            next_timestep={
                "video": torch.zeros_like(primary_timesteps),
                "audio": torch.zeros_like(audio_timesteps),
            },
            sigma={
                "video": primary_timesteps / 1000,
                "audio": audio_timesteps / 500,
            },
            next_sigma={
                "video": torch.zeros_like(primary_timesteps),
                "audio": torch.zeros_like(audio_timesteps),
            },
        )
        self.mapped_primary_times.append(primary_timesteps.detach().clone())
        self.mapped_component_times.append(times)
        return times

    @contextmanager
    def use_component_variant(self, role_name: str) -> Iterator[None]:
        # Mirror the real registry: only declared trainable variants resolve. The
        # frozen reference is a parameter snapshot and never a declared variant.
        if role_name not in ("generator", "fake"):
            raise KeyError(f"component variant {role_name!r} is not declared")
        previous = self.active_role
        self.active_role = role_name
        try:
            yield
        finally:
            self.active_role = previous

    @contextmanager
    def use_ref_parameters(self) -> Iterator[None]:
        previous = self.active_role
        self.active_role = "reference"
        try:
            yield
        finally:
            self.active_role = previous

    def add_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        *,
        generator: torch.Generator | None = None,
    ) -> Any:
        del generator
        noise = LatentState(
            {
                name: torch.randn_like(clean_state.components[name])
                for name in self.trajectory_component_order
            }
        )
        self.noise_draws.append(
            LatentState(
                {name: values.detach().clone() for name, values in noise.components.items()}
            )
        )
        return self.apply_forward_process_noise(clean_state, times, noise)

    def _project_clean_to_score_state(
        self,
        state: LatentState,
        times: ComponentTimes,
        clean_state: LatentState,
    ) -> LatentState:
        return self._project_flow_match_clean_to_score_state(state, times, clean_state)

    def _forward_state(
        self,
        *,
        batch: StackedSampleBatch,
        state: LatentState,
        times: ComponentTimes,
        next_state: Optional[LatentState],
        compute_log_prob: bool,
        return_fields: Tuple[str, ...],
        noise_level: Optional[float],
        forward_kwargs: Mapping[str, Any],
    ) -> MultiModalStepOutput:
        del batch, next_state, compute_log_prob, return_fields, noise_level, forward_kwargs
        self.events.append((self.active_role, torch.is_grad_enabled(), id(state), id(times)))
        if self.active_role == "generator":
            generated = LatentState(
                {name: values + self.bundle.generator for name, values in state.components.items()},
                active_masks=state.active_masks,
            )
            velocity = LatentState(
                {
                    name: self.bundle.generator.expand_as(values)
                    for name, values in state.components.items()
                },
                active_masks=state.active_masks,
            )
            return MultiModalStepOutput(
                next_state=generated,
                next_state_mean=generated,
                velocity=velocity,
            )
        role_value = (
            self.bundle.fake if self.active_role == "fake" else torch.zeros_like(self.bundle.fake)
        )
        return MultiModalStepOutput(
            velocity=LatentState(
                {name: role_value.expand_as(values) for name, values in state.components.items()},
                active_masks=state.active_masks,
            )
        )


def _role_config(role_name: str, learning_rate: float) -> RoleOptimizerConfig:
    return RoleOptimizerConfig(
        role_name=role_name,  # type: ignore[arg-type]
        learning_rate=learning_rate,
        adam_betas=(0.8, 0.9),
        adam_weight_decay=0.0,
        adam_epsilon=1e-8,
        max_grad_norm=100.0,
    )


def _sample(
    *,
    state_index_map: torch.Tensor | None = None,
    audio_state_index_map: torch.Tensor | None = None,
    video_states: torch.Tensor | None = None,
    audio_states: torch.Tensor | None = None,
    video_times: torch.Tensor | None = None,
    audio_times: torch.Tensor | None = None,
) -> BaseSample:
    index_map = (
        torch.tensor([0, 1, 2], dtype=torch.int64) if state_index_map is None else state_index_map
    )
    video_schedule = torch.tensor([1000.0, 500.0, 0.0]) if video_times is None else video_times
    audio_schedule = torch.tensor([500.0, 250.0, 0.0]) if audio_times is None else audio_times
    return BaseSample(
        prompt_embeds=torch.tensor([1.0]),
        trajectory=StructuredTrajectory(
            components={
                "video": ComponentTrajectory(
                    states=(
                        torch.tensor([[0.0], [1.0], [2.0]])
                        if video_states is None
                        else video_states
                    ),
                    timesteps=video_schedule,
                    sigmas=video_schedule / 1000,
                    state_index_map=index_map,
                ),
                "audio": ComponentTrajectory(
                    states=(
                        torch.tensor([[10.0], [11.0], [12.0]])
                        if audio_states is None
                        else audio_states
                    ),
                    timesteps=audio_schedule,
                    sigmas=audio_schedule / 500,
                    state_index_map=(
                        index_map.clone()
                        if audio_state_index_map is None
                        else audio_state_index_map
                    ),
                ),
            }
        ),
    )


def _trainer(**overrides: Any) -> TDMTrainer:
    trainer = object.__new__(TDMTrainer)
    defaults = {
        "trajectory_steps": 2,
        "num_inference_steps": 2,
        "num_inner_epochs": 1,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "ttur_fake_updates": 1,
        "use_huber": False,
        "huber_c": 1e-3,
        "tdm_snr_gamma": 5.0,
        "replay_rtol": 1e-6,
        "replay_atol": 1e-6,
    }
    defaults.update(overrides)
    trainer.training_args = SimpleNamespace(**defaults)
    trainer.adapter = TinyTDMAdapter()
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), is_local_main_process=True)
    trainer.log_args = SimpleNamespace(verbose=False)
    trainer.autocast = nullcontext
    trainer.step = 0
    trainer.epoch = 0
    return trainer


def _objective_trainer() -> tuple[TDMTrainer, ObjectiveTDMAdapter, ObjectiveBundle]:
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1)
    bundle = ObjectiveBundle()
    adapter = ObjectiveTDMAdapter(bundle)
    configs = {
        "generator": _role_config("generator", 0.03),
        "fake": _role_config("fake", 0.07),
    }
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [bundle.generator],
                "role_name": "generator",
                "lr": configs["generator"].learning_rate,
                "betas": configs["generator"].adam_betas,
                "weight_decay": configs["generator"].adam_weight_decay,
                "eps": configs["generator"].adam_epsilon,
            },
            {
                "params": [bundle.fake],
                "role_name": "fake",
                "lr": configs["fake"].learning_rate,
                "betas": configs["fake"].adam_betas,
                "weight_decay": configs["fake"].adam_weight_decay,
                "eps": configs["fake"].adam_epsilon,
            },
        ]
    )
    roles = {
        "generator": OptimizationRole(configs["generator"], (bundle.generator,), (0,)),
        "fake": OptimizationRole(configs["fake"], (bundle.fake,), (1,)),
    }
    trainer = object.__new__(TDMTrainer)
    trainer.accelerator = accelerator
    trainer.adapter = adapter
    trainer.model_bundle = bundle
    trainer.optimizer = optimizer
    trainer.optimization_roles = roles
    trainer.role_optimization = RoleOptimizationCoordinator(
        accelerator,
        bundle,
        optimizer,
        roles,
    )
    trainer.training_args = TDMTrainingArguments(
        num_inference_steps=2,
        trajectory_steps=2,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        ttur_fake_updates=1,
        use_huber=False,
        replay_rtol=0,
        replay_atol=0,
    )
    trainer.autocast = nullcontext
    trainer.step = 0
    trainer.epoch = 0
    trainer.log_args = SimpleNamespace(verbose=False)
    return trainer, adapter, bundle


def test_tdm_is_direct_deterministic_distillation_trainer() -> None:
    assert TDMTrainer.__bases__ == (BaseTrainer,)
    assert TDMTrainer.paradigm == "distillation"


def test_tdm_config_counts_every_boundary_in_one_generator_window() -> None:
    training_args = TDMTrainingArguments()

    assert training_args.gradient_step_per_epoch == 1
    assert training_args.get_num_train_timesteps(SimpleNamespace()) == 1

    resolved = Arguments.from_dict(
        {
            "train": {
                "trainer_type": "tdm",
                "trajectory_steps": 4,
                "num_inference_steps": 4,
                "unique_sample_num_per_epoch": 8,
                "per_device_batch_size": 1,
            },
            "scheduler": {"dynamics_type": "ODE"},
        }
    ).training_args
    assert resolved.num_batches_per_epoch == 8
    assert resolved.gradient_accumulation_steps == 1


@pytest.mark.parametrize("manual_gas", [8, 16, 64])
def test_tdm_config_accepts_manual_gas(manual_gas: int) -> None:
    training_args = Arguments.from_dict(
        {
            "train": {
                "trainer_type": "tdm",
                "trajectory_steps": 4,
                "num_inference_steps": 4,
                "unique_sample_num_per_epoch": 8,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": manual_gas,
            },
            "scheduler": {"dynamics_type": "ODE"},
        }
    ).training_args

    assert training_args.gradient_accumulation_steps == manual_gas


def test_tdm_builds_every_boundary_with_half_open_non_overlapping_intervals() -> None:
    trainer = _trainer()
    sample = _sample()

    units = trainer._build_boundary_units([sample])

    assert [unit.boundary_index for unit in units] == [1, 2]
    torch.testing.assert_close(units[0].interval_start, torch.tensor([500.0]))
    torch.testing.assert_close(units[0].interval_end, torch.tensor([1000.0]))
    torch.testing.assert_close(units[1].interval_start, torch.tensor([0.0]))
    torch.testing.assert_close(units[1].interval_end, torch.tensor([500.0]))
    assert all(isinstance(unit, TDMBoundaryUnit) for unit in units)
    assert all(len(unit.samples) == 1 and unit.samples[0] is sample for unit in units)


def test_tdm_samples_inside_each_boundary_half_open_interval() -> None:
    trainer = _trainer()
    units = trainer._build_boundary_units([_sample()])
    torch.manual_seed(7)

    for unit in units:
        sampled = torch.stack([trainer._sample_perturbation_times(unit) for _ in range(32)])
        assert bool((sampled >= unit.interval_start).all())
        assert bool((sampled < unit.interval_end).all())


@pytest.mark.parametrize(
    "random_value", [0.0, torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)).item()]
)
def test_tdm_sampling_uses_representable_open_interval_bounds(
    monkeypatch: pytest.MonkeyPatch,
    random_value: float,
) -> None:
    trainer = _trainer()
    terminal_unit = trainer._build_boundary_units([_sample()])[-1]

    def deterministic_rand(*shape: Any, **kwargs: Any) -> torch.Tensor:
        del shape
        return torch.full(
            terminal_unit.interval_start.shape,
            random_value,
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        )

    monkeypatch.setattr(torch, "rand", deterministic_rand)
    sampled = trainer._sample_perturbation_times(terminal_unit)

    assert bool((sampled > terminal_unit.interval_start).all())
    assert bool((sampled < terminal_unit.interval_end).all())
    assert bool((sampled > 0).all())


def test_tdm_terminal_generator_score_path_keeps_mapped_sigmas_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, adapter, _ = _objective_trainer()
    terminal_unit = trainer._build_boundary_units([_sample()])[-1]

    def zero_rand(*shape: Any, **kwargs: Any) -> torch.Tensor:
        del shape
        return torch.zeros(
            terminal_unit.interval_start.shape,
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        )

    monkeypatch.setattr(torch, "rand", zero_rand)
    loss = trainer._generator_boundary_loss(terminal_unit)

    sampled_tau = adapter.mapped_primary_times[-1]
    mapped_times = adapter.mapped_component_times[-1]
    assert bool((sampled_tau > 0).all())
    assert mapped_times.sigma is not None
    assert all(
        bool((mapped_times.sigma[name] > 0).all()) for name in adapter.trajectory_component_order
    )
    assert bool(torch.isfinite(loss).item())
    assert [event[0] for event in adapter.events] == ["generator", "reference", "fake"]


def test_tdm_rejects_zero_custom_mapped_sigma_before_score_queries() -> None:
    trainer, adapter, _ = _objective_trainer()
    terminal_unit = trainer._build_boundary_units([_sample()])[-1]
    original_mapping = adapter.build_training_component_times

    def zero_audio_sigma(
        primary_timesteps: torch.Tensor,
        *,
        batch: StackedSampleBatch | None = None,
    ) -> ComponentTimes:
        mapped = original_mapping(primary_timesteps, batch=batch)
        assert mapped.sigma is not None
        return ComponentTimes(
            timestep=mapped.timestep,
            next_timestep=mapped.next_timestep,
            sigma={
                "video": mapped.sigma["video"],
                "audio": torch.zeros_like(mapped.sigma["audio"]),
            },
            next_sigma=mapped.next_sigma,
        )

    adapter.build_training_component_times = zero_audio_sigma
    with pytest.raises(
        ValueError,
        match=(
            r"boundary_index=2.*component='audio'.*tau=.*finite and strictly positive.*"
            r"received sigma="
        ),
    ):
        trainer._generator_boundary_loss(terminal_unit)

    assert [event[0] for event in adapter.events] == ["generator"]


def test_tdm_rejects_interval_without_representable_interior() -> None:
    trainer = _trainer()
    start = torch.tensor([1.0])
    end = torch.nextafter(start, torch.tensor([float("inf")]))
    unit = TDMBoundaryUnit(
        samples=(_sample(),),
        boundary_index=1,
        interval_start=start,
        interval_end=end,
    )

    with pytest.raises(ValueError, match=r"no representable floating interior"):
        trainer._sample_perturbation_times(unit)


@pytest.mark.parametrize(
    ("video_times", "expected_type"),
    [
        (torch.tensor([1000, 500, 0], dtype=torch.int64), "int64"),
        (torch.tensor([True, True, False]), "bool"),
        (torch.tensor([1000 + 0j, 500 + 0j, 0 + 0j]), "complex"),
    ],
)
def test_tdm_rejects_nonfloating_primary_coordinates(
    video_times: torch.Tensor,
    expected_type: str,
) -> None:
    trainer = _trainer()

    with pytest.raises(
        TypeError,
        match=rf"primary.*timestep.*floating.*{expected_type}",
    ):
        trainer._build_boundary_units([_sample(video_times=video_times)])


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_tdm_rejects_nonfinite_primary_coordinates(bad_value: float) -> None:
    trainer = _trainer()
    video_times = torch.tensor([1000.0, 500.0, bad_value])

    with pytest.raises(
        ValueError,
        match=r"primary.*timestep.*boundary_index=2.*finite",
    ):
        trainer._build_boundary_units([_sample(video_times=video_times)])


@pytest.mark.parametrize(
    ("bad_value", "error_type", "match"),
    [
        (torch.tensor([500], dtype=torch.int64), TypeError, r"floating.*int64"),
        (torch.tensor([True]), TypeError, r"floating.*bool"),
        (torch.tensor([500 + 0j]), TypeError, r"floating.*complex"),
        (torch.tensor([float("nan")]), ValueError, r"finite"),
        (torch.tensor([float("inf")]), ValueError, r"finite"),
    ],
)
def test_tdm_rejects_invalid_component_coordinates(
    bad_value: torch.Tensor,
    error_type: type[Exception],
    match: str,
) -> None:
    trainer = _trainer()

    def invalid_component_times(
        primary_timesteps: torch.Tensor,
        *,
        batch: StackedSampleBatch | None = None,
    ) -> ComponentTimes:
        del batch
        invalid = bad_value.to(primary_timesteps.device).expand_as(primary_timesteps)
        return ComponentTimes(
            timestep={"video": primary_timesteps, "audio": invalid},
            next_timestep={"video": primary_timesteps, "audio": invalid},
        )

    trainer.adapter.build_training_component_times = invalid_component_times
    with pytest.raises(
        error_type,
        match=rf"interval_end.*component='audio'.*{match}",
    ):
        trainer._build_boundary_units([_sample()])


@pytest.mark.parametrize(
    ("current_end", "previous_start", "kind"),
    [
        (500.0 + 1e-4, 500.0, "overlap"),
        (500.0 - 1e-4, 500.0, "gap"),
    ],
)
def test_tdm_rejects_tiny_nonzero_interval_gap_or_overlap(
    current_end: float,
    previous_start: float,
    kind: str,
) -> None:
    trainer = _trainer()
    batch = trainer._stack_replay_unit((_sample(),))
    times = trainer.adapter.build_training_component_times(torch.tensor([current_end]), batch=batch)

    with pytest.raises(ValueError, match=rf"exact.*{kind}"):
        trainer._validate_interval(
            batch,
            times,
            boundary_index=2,
            interval_start=torch.tensor([0.0]),
            interval_end=torch.tensor([current_end]),
            previous_interval_start=torch.tensor([previous_start]),
        )


def test_tdm_rejects_tiny_nonzero_terminal_coordinate() -> None:
    trainer = _trainer()
    tiny_terminal = torch.tensor([1e-7])
    batch = trainer._stack_replay_unit((_sample(),))
    end_times = trainer.adapter.build_training_component_times(torch.tensor([500.0]), batch=batch)
    start_times = trainer.adapter.build_training_component_times(tiny_terminal, batch=batch)
    times = ComponentTimes(
        timestep=end_times.timestep,
        next_timestep=start_times.timestep,
        sigma=end_times.sigma,
        next_sigma=start_times.sigma,
    )

    with pytest.raises(ValueError, match=r"terminal.*exact.*zero"):
        trainer._validate_interval(
            batch,
            times,
            boundary_index=2,
            interval_start=tiny_terminal,
            interval_end=torch.tensor([500.0]),
            previous_interval_start=torch.tensor([500.0]),
        )


def test_tdm_rejects_tiny_component_endpoint_mismatch() -> None:
    trainer = _trainer()
    trainer.training_args.replay_atol = 1.0
    audio_times = torch.tensor([500.0, 250.0 + 1e-4, 0.0])

    with pytest.raises(
        ValueError,
        match=r"component='audio'.*exact.*endpoint='interval_start'",
    ):
        trainer._build_boundary_units([_sample(audio_times=audio_times)])


@pytest.mark.parametrize(
    ("sample_kwargs", "match"),
    [
        ({"state_index_map": torch.tensor([0, 0, 1])}, "one-to-one"),
        ({"state_index_map": torch.tensor([0, 2, 1])}, "aligned.*arange"),
        (
            {"audio_state_index_map": torch.tensor([0, 2, 1])},
            "component='audio'.*match.*component='video'",
        ),
        (
            {"video_states": torch.tensor([[0.0], [1.0], [2.0], [3.0]])},
            "stored states.*K \\+ 1=3",
        ),
    ],
)
def test_tdm_rejects_non_dense_or_misaligned_state_maps(
    sample_kwargs: dict[str, torch.Tensor],
    match: str,
) -> None:
    trainer = _trainer()

    with pytest.raises(ValueError, match=match):
        trainer._build_boundary_units([_sample(**sample_kwargs)])


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        # A hole in the map is now caught by the replay accessor, which reports which
        # transition could not be read rather than which map entry was -1.
        ({"state_index_map": torch.tensor([0, -1, 2])}, "reading transition .* failed"),
        ({"video_times": torch.tensor([1000.0, 500.0, 100.0])}, "gap|terminal"),
        ({"video_times": torch.tensor([1000.0, 1100.0, 0.0])}, "reversed"),
        ({"audio_times": torch.tensor([500.0, 300.0, 0.0])}, "shared interval"),
    ],
)
def test_tdm_rejects_invalid_boundary_geometry(
    overrides: dict[str, torch.Tensor],
    match: str,
) -> None:
    trainer = _trainer()

    with pytest.raises(ValueError, match=match):
        trainer._build_boundary_units([_sample(**overrides)])


def test_tdm_rejects_non_ode_or_k_mismatch_before_building_units() -> None:
    non_ode = _trainer()
    non_ode.adapter.scheduler_group["audio"].dynamics_type = "Flow-SDE"
    with pytest.raises(ValueError, match=r"deterministic ODE.*audio.*Flow-SDE"):
        non_ode._build_boundary_units([_sample()])

    mismatch = _trainer(trajectory_steps=3, num_inference_steps=2)
    with pytest.raises(ValueError, match=r"trajectory_steps=3.*num_inference_steps=2"):
        mismatch._build_boundary_units([_sample()])


def test_tdm_optimize_runs_fake_ttur_then_generator_over_identical_ordered_units() -> None:
    trainer = _trainer(ttur_fake_updates=3)
    events: list[tuple[str, tuple[int, ...]]] = []

    def record(self: TDMTrainer, microbatches: list, name: str) -> None:
        units = self._build_boundary_units(microbatches[0])
        events.append((name, tuple(unit.boundary_index for unit in units)))

    trainer._fake_phase = MethodType(lambda self, units: record(self, units, "fake"), trainer)
    trainer._generator_phase = MethodType(
        lambda self, units: record(self, units, "generator"), trainer
    )

    trainer.optimize([_sample()])

    assert events == [("fake", (1, 2)), ("fake", (1, 2)), ("fake", (1, 2)), ("generator", (1, 2))]


def test_tdm_optimize_rejects_gas_microbatch_count_mismatch() -> None:
    trainer = _trainer(gradient_accumulation_steps=2)

    with pytest.raises(ValueError, match=r"TDM optimize expected 2 role microbatches.*received 1"):
        trainer.optimize([_sample()])


def test_tdm_generator_loss_is_finite_and_queries_reference_and_fake() -> None:
    trainer, adapter, bundle = _objective_trainer()
    unit = trainer._build_boundary_units([_sample()])[0]
    trainer._sample_perturbation_times = MethodType(
        lambda self, selected: (selected.interval_start + selected.interval_end) / 2,
        trainer,
    )

    loss = trainer._generator_boundary_loss(unit)
    (gradient,) = torch.autograd.grad(loss, (bundle.generator,))

    assert torch.isfinite(loss)
    assert torch.isfinite(gradient)
    assert adapter.events[0][:2] == ("generator", True)


def test_tdm_replay_mismatch_fails_before_score_queries() -> None:
    trainer, adapter, _ = _objective_trainer()
    mismatched = _sample(
        video_states=torch.tensor([[0.0], [9.0], [10.0]]),
    )
    unit = trainer._build_boundary_units([mismatched])[0]

    with pytest.raises(ValueError, match=r"replay_generator_boundary.*mismatch|replay.*mismatch"):
        trainer._generator_boundary_loss(unit)

    assert [event[0] for event in adapter.events] == ["generator"]


def test_tdm_reference_and_fake_receive_identical_detached_state_and_times_objects() -> None:
    trainer, adapter, _ = _objective_trainer()
    unit = trainer._build_boundary_units([_sample()])[0]

    trainer._generator_boundary_loss(unit)

    generator_event, reference_event, fake_event = adapter.events
    assert generator_event[:2] == ("generator", True)
    assert reference_event[:2] == ("reference", False)
    assert fake_event[:2] == ("fake", False)
    assert reference_event[2:] == fake_event[2:]


def test_tdm_generator_graph_owns_one_transition_and_no_auxiliary_parameters() -> None:
    trainer, adapter, bundle = _objective_trainer()
    unit = trainer._build_boundary_units([_sample()])[0]

    trainer._generator_boundary_loss(unit).backward()

    assert bundle.generator.grad is not None
    assert bundle.fake.grad is None
    assert bundle.auxiliary.grad is None
    assert sum(role == "generator" and grad for role, grad, _, _ in adapter.events) == 1


def test_tdm_fake_units_draw_fresh_noise_for_every_component_and_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, adapter, _ = _objective_trainer()
    first_unit, second_unit = trainer._build_boundary_units([_sample()])
    torch.manual_seed(31)
    original_randn_like = torch.randn_like
    draws: list[torch.Tensor] = []

    def recording_randn_like(value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        draw = original_randn_like(value, *args, **kwargs)
        draws.append(draw.detach().clone())
        return draw

    monkeypatch.setattr(torch, "randn_like", recording_randn_like)

    trainer._fake_boundary_loss(first_unit)
    trainer._fake_boundary_loss(second_unit)
    trainer._fake_boundary_loss(first_unit)

    component_count = len(adapter.trajectory_component_order)
    assert len(draws) == 3 * component_count
    grouped = [
        draws[offset : offset + component_count]
        for offset in range(0, len(draws), component_count)
    ]
    for draw in grouped:
        assert not torch.equal(draw[0], draw[1])
    for component_index in range(component_count):
        component_draws = [draw[component_index] for draw in grouped]
        assert all(
            not torch.equal(component_draws[left], component_draws[right])
            for left in range(len(component_draws))
            for right in range(left + 1, len(component_draws))
        )


def test_tdm_media_suppression_restores_decoder_after_rollout_scope() -> None:
    trainer = _trainer()

    def decoder(video_latents: torch.Tensor, audio_latents: torch.Tensor) -> Any:
        del video_latents, audio_latents
        raise AssertionError("TDM must not decode generated media")

    trainer.adapter.decode_latents = decoder

    with trainer._without_media_decoding():
        assert trainer.adapter.decode_latents(
            torch.zeros(2, 3),
            torch.zeros(2, 4),
        ) == ([None, None], [None, None])

    assert trainer.adapter.decode_latents is decoder
    with pytest.raises(AssertionError, match="must not decode"):
        trainer.adapter.decode_latents(torch.zeros(2, 3), torch.zeros(2, 4))


def test_tdm_real_role_phases_advance_exact_gas_counters() -> None:
    trainer, _, _ = _objective_trainer()
    microbatches = [[_sample()]]

    trainer._fake_phase(microbatches)
    trainer._generator_phase(microbatches)

    assert trainer.optimization_roles["fake"].step == 1
    assert trainer.optimization_roles["generator"].step == 1
    assert trainer.step == 1
