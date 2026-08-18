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

"""Train a one-step generator with data-free DMD2 distribution matching."""

from __future__ import annotations

from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import torch
from accelerate import Accelerator

from ...hparams import Arguments, DMD2TrainingArguments
from ...hparams.training_args.dmd2 import DMD2_DEFAULT_OPTIMIZERS
from ...models.abc import BaseAdapter
from ...samples import BaseSample, ComponentTimes, LatentState, StackedSampleBatch
from ..abc import BaseTrainer
from .distillation_runtime import (
    as_role_microbatches,
    detach_state,
    generate_one_rollout_batch,
    query_score_velocity,
    replay_forward_kwargs,
    require_velocity,
    run_distillation_training_step,
    run_role_phase,
    validate_media_free_rollout,
    without_media_decoding,
)
from .distribution_matching import dmd_generator_loss, flow_matching_loss

if TYPE_CHECKING:
    from ..rewards import RewardBuffer


class DMD2Trainer(BaseTrainer):
    """Optimize a deterministic one-step generator without real training data."""

    paradigm: ClassVar[Literal["distillation"]] = "distillation"

    def _optimizer_args_for_role(self, role_name: str):
        """Resolve this role's optimizer, falling back to DMD2's published defaults.

        An ``optimizers`` list in the config file wins, which is also how a run puts
        one of these roles on Muon. Without one, the algorithm supplies the learning
        rates it was published with, because those numbers belong to the algorithm
        rather than to the framework.
        """
        configured = self.config.optimizer_args.get_by_name(role_name)
        if configured is not None:
            return configured
        for default in DMD2_DEFAULT_OPTIMIZERS:
            if default.name == role_name:
                return default
        raise ValueError(
            f"expected an optimizer configuration for role {role_name!r}: no "
            f"`optimizers` entry and no DMD2 default"
        )

    _BOUNDARY_INDEX: ClassVar[int] = 1

    def __init__(
        self,
        accelerator: Accelerator,
        config: Arguments,
        adapter: BaseAdapter,
    ) -> None:
        training_args = config.training_args
        if training_args.num_inference_steps != 1:
            raise ValueError(
                "DMD2 requires train.num_inference_steps=1 for its one-step generator; "
                f"received train.num_inference_steps={training_args.num_inference_steps}. "
                "Use TDM for few-step trajectory distribution matching."
            )
        super().__init__(accelerator=accelerator, config=config, adapter=adapter)
        self.training_args: DMD2TrainingArguments
        self._validate_one_step_configuration()
        self._rollout_data_iter: Optional[Iterator[Any]] = None
        self._rollout_dataloader_epoch = 0
        non_ode = {
            name: scheduler.dynamics_type
            for name, scheduler in self.adapter.scheduler_group.items()
            if scheduler.dynamics_type != "ODE"
        }
        if non_ode:
            raise ValueError(
                "DMD2 requires deterministic ODE generation for boundary replay; "
                f"received scheduler dynamics {non_ode!r}"
            )

    def _validate_one_step_configuration(self) -> None:
        """Reject generation schedules that need an unimplemented multi-step objective."""
        num_inference_steps = self.training_args.num_inference_steps
        if num_inference_steps != 1:
            raise ValueError(
                "DMD2 requires train.num_inference_steps=1 for its one-step generator; "
                f"received train.num_inference_steps={num_inference_steps}. "
                "Use TDM for few-step trajectory distribution matching."
            )
        if self.training_args.num_inner_epochs != 1:
            raise ValueError(
                "DMD2 requires train.num_inner_epochs=1 so every generator update owns a "
                "fresh rollout boundary; "
                f"received train.num_inner_epochs={self.training_args.num_inner_epochs}. "
                "Start a fresh rollout before another generator update."
            )

    def _init_reward_model(self) -> Tuple[Dict[str, object], Dict[str, object]]:
        """Initialize an explicitly empty feedback runtime."""
        self.reward_loader = None
        self.reward_models = {}
        self.eval_reward_models = {}
        self.reward_buffer = None
        self.eval_dataset_reward_buffers = {}
        self._eval_dataset_configs = {}
        self.advantage_processor = None
        return self.reward_models, self.eval_reward_models

    def _run_training_step(self) -> None:
        """Run GAS distinct rollouts, then fake TTUR updates, then one generator step.

        Overriding only this keeps the shared epoch loop, so checkpointing and
        eval-time reward monitoring behave exactly as they do for every other
        trainer.
        """
        run_distillation_training_step(self)

    def sample(self) -> List[BaseSample]:
        """Collect the initial state and generated boundary of one fresh rollout."""
        self._validate_one_step_configuration()
        self._validate_media_free_rollout()
        with self._without_media_decoding():
            return self.generate_samples(
                reward_buffer=None,
                compute_log_prob=False,
                trajectory_indices=[0, self._BOUNDARY_INDEX],
            )

    def generate_samples(
        self,
        reward_buffer: Optional[RewardBuffer] = None,
        compute_log_prob: bool = False,
        trajectory_indices: Optional[List[int]] = None,
        **extra_inference_kwargs: Any,
    ) -> List[BaseSample]:
        """Generate exactly one fresh rollout batch for one outer iteration.

        Unlike the epoch-sized base implementation, DMD2 consumes one
        dataloader batch per outer iteration; the iterator rolls over with a
        reseeded epoch only after a full pass through the prompt dataloader.
        """
        del extra_inference_kwargs
        if reward_buffer is not None:
            raise ValueError(
                "DMD2 is data-free and computes no rewards; expected "
                f"reward_buffer=None, received {type(reward_buffer).__name__}"
            )
        return generate_one_rollout_batch(
            self,
            reward_buffer=None,
            compute_log_prob=compute_log_prob,
            trajectory_indices=trajectory_indices,
            algorithm_name="DMD2",
        )

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Perform no feedback work for the data-free objective."""
        del samples

    def optimize(self, samples: Sequence[Any]) -> None:
        """Run fake TTUR updates, then one generator step, over GAS replay units."""
        if not samples:
            return
        self._validate_one_step_configuration()
        replay_units = as_role_microbatches(
            samples,
            batch_size=self.training_args.per_device_batch_size,
            accumulation_steps=self.training_args.gradient_accumulation_steps,
            algorithm_name="DMD2",
        )
        self.adapter.train()
        for _ in range(self.training_args.ttur_fake_updates):
            self._fake_phase(replay_units)
        self._generator_phase(replay_units)

    def _fake_phase(self, replay_units: Sequence[Sequence[BaseSample]]) -> None:
        """Fit the fake score on fresh perturbations of generated boundaries."""
        run_role_phase(
            self,
            "fake",
            replay_units,
            lambda unit: self._fake_replay_loss(self._stack_replay_unit(list(unit))),
        )

    def _generator_phase(self, replay_units: Sequence[Sequence[BaseSample]]) -> None:
        """Update the generator from detached reference/fake score differences."""
        run_role_phase(
            self,
            "generator",
            replay_units,
            lambda unit: self._generator_replay_loss(self._stack_replay_unit(list(unit))),
        )

    def _stack_replay_unit(self, replay_unit: Sequence[BaseSample]) -> StackedSampleBatch:
        """Move and stack one replay unit without touching generated media."""
        if not replay_unit:
            raise ValueError("expected a non-empty DMD2 replay unit, received no samples")
        return BaseSample.stack([sample.to(self.accelerator.device) for sample in replay_unit])

    def _validate_media_free_rollout(self) -> None:
        """Require inference to expose the adapter media reconstruction seam."""
        validate_media_free_rollout(self.adapter, algorithm_name="DMD2")

    @contextmanager
    def _without_media_decoding(self) -> Iterator[None]:
        """Replace rollout media reconstruction with shape-preserving empty outputs."""
        with without_media_decoding(self.adapter, algorithm_name="DMD2"):
            yield

    def _fake_replay_loss(self, batch: StackedSampleBatch) -> torch.Tensor:
        """Compute fake clean-state denoising loss for one replay unit."""
        boundary_state = detach_state(self.adapter.get_terminal_state(batch))
        times = self._sample_perturbation_times(boundary_state, batch)
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
        velocity = require_velocity(output, algorithm_name="DMD2", role_name="fake")
        return flow_matching_loss(
            self.adapter,
            velocity,
            noised.target_velocity,
            state=noised.state,
        )

    def _generator_replay_loss(self, batch: StackedSampleBatch) -> torch.Tensor:
        """Compute the one-step generator pseudo-loss for one replay unit."""
        with self.adapter.use_component_variant("generator"):
            with self.autocast():
                generator_output = self.adapter.replay_generator_boundary(
                    batch,
                    self._BOUNDARY_INDEX,
                    return_fields=("velocity", "next_latents", "next_latents_mean"),
                    rtol=self.training_args.replay_rtol,
                    atol=self.training_args.replay_atol,
                    **self._replay_forward_kwargs(batch),
                )
        boundary_state = generator_output.next_state
        if boundary_state is None:
            raise ValueError(
                "DMD2 generator boundary replay expected next_state, received None "
                f"for boundary_index={self._BOUNDARY_INDEX}"
            )

        detached_boundary = detach_state(boundary_state)
        times = self._sample_perturbation_times(detached_boundary, batch)
        noised = self.adapter.add_forward_process_noise(detached_boundary, times)
        reference_velocity = query_score_velocity(
            self.adapter,
            batch,
            noised.state,
            times,
            role_name="reference",
            autocast=self.autocast,
            forward_kwargs=self._replay_forward_kwargs(batch),
            algorithm_name="DMD2",
        )
        fake_velocity = query_score_velocity(
            self.adapter,
            batch,
            noised.state,
            times,
            role_name="fake",
            autocast=self.autocast,
            forward_kwargs=self._replay_forward_kwargs(batch),
            algorithm_name="DMD2",
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
        return dmd_generator_loss(
            self.adapter,
            boundary_state,
            detach_state(x0_real),
            detach_state(x0_fake),
        )

    def _sample_perturbation_times(
        self,
        boundary_state: LatentState,
        batch: StackedSampleBatch,
    ) -> ComponentTimes:
        """Draw one fresh forward-process coordinate per sample."""
        primary_name = boundary_state.component_names[0]
        component = boundary_state.components[primary_name]
        lower, upper = self.training_args.perturbation_timestep_range
        sigma = torch.rand(
            component.shape[0],
            device=component.device,
            dtype=torch.float32,
        )
        sigma = lower + (upper - lower) * sigma
        primary_timesteps = sigma * 1000.0
        return self.adapter.build_training_component_times(primary_timesteps, batch=batch)

    def _replay_forward_kwargs(self, batch: StackedSampleBatch) -> Dict[str, object]:
        """Return allow-listed adapter arguments not already owned by the batch."""
        return replay_forward_kwargs(self.training_args, batch)
