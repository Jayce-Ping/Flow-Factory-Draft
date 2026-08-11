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

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import pytest
import torch

from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import (
    BaseSample,
    ComponentTimes,
    LatentState,
    MultiModalStepOutput,
    ReplayStep,
)
from flow_factory.scheduler import SchedulerGroup, SDESchedulerOutput
from flow_factory.trainers.dppo import DPPOTrainer, gaussian_kl_div
from flow_factory.trainers.grpo import GRPOGuardTrainer, GRPOTrainer


class TrainingArgsFake(dict):
    """Mapping/attribute hybrid mirroring ``ArgABC`` unpacking behaviour."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


class SchedulerFake:
    """Small scheduler-like object exposing the dynamics used by DPPO."""

    def __init__(self, dynamics_type: str = "Flow-SDE") -> None:
        self.dynamics_type = dynamics_type
        self.noise_level = 0.7
        self.train_timesteps = torch.tensor([0, 1])

    def step(self) -> None:
        """Provide scheduler compatibility."""


class AdapterFake(BaseAdapter):
    """Minimal concrete adapter recording replay forward arguments."""

    def load_pipeline(self) -> Any:
        """Return an unused pipeline fake."""
        raise NotImplementedError

    def decode_latents(self, latents: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return latents unchanged."""
        return latents

    def inference(self, **kwargs: Any) -> List[BaseSample]:
        """Return no samples."""
        return []

    def forward(self, **kwargs: Any) -> SDESchedulerOutput:
        """Record kwargs and return a deterministic scheduler output."""
        self.forward_kwargs = kwargs
        latents = kwargs["latents"]
        # Schedulers broadcast per-sample statistics to (B, 1, ..., 1).
        broadcast_shape = (latents.shape[0],) + (1,) * (latents.ndim - 1)
        return SDESchedulerOutput(
            next_latents=latents + 1,
            next_latents_mean=latents + 2,
            std_dev_t=torch.full(broadcast_shape, 0.25),
            dt=torch.full(broadcast_shape, -0.5),
            log_prob=torch.tensor([0.75, 0.5]),
            velocity=latents + 3,
        )

    @contextmanager
    def use_ref_parameters(self) -> Iterator[None]:
        """Record reference-parameter scope without touching real weights."""
        self.ref_scope_entered = True
        yield


class StructuredAdapterFake(AdapterFake):
    """Adapter fake declaring the structured video/audio component contract."""

    trajectory_component_order = ("video", "audio")


class MaskedAdapterFake(AdapterFake):
    """Adapter whose latent keeps only its first two positions active."""

    active_numel = 2

    def get_state_active_numel(self, state: LatentState) -> Dict[str, int]:
        """Count only the active latent prefix."""
        return {"latent": self.active_numel}

    def reduce_component_latent_values(
        self, values: Dict[str, torch.Tensor], *, state: Optional[LatentState] = None
    ) -> Dict[str, torch.Tensor]:
        """Average each component over its active prefix only."""
        return {
            "latent": values["latent"].flatten(1)[:, : self.active_numel].mean(dim=1),
        }


def _adapter(dynamics_type: str = "Flow-SDE") -> AdapterFake:
    adapter = object.__new__(AdapterFake)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake(dynamics_type))
    adapter.scheduler_group = adapter.build_scheduler_group()
    return adapter


class DynamicMaskAdapterFake(AdapterFake):
    """Adapter reducing only the positions the replay state marks active."""

    def _reduce_latent_values(
        self,
        values: Dict[str, torch.Tensor],
        *,
        active_numel: Optional[Dict[str, int]] = None,
        state: Optional[LatentState] = None,
    ) -> torch.Tensor:
        """Average the elements the per-sample state mask selects."""
        if state is None:
            raise ValueError("expected state context for DynamicMaskAdapterFake reduction")
        mask = state.components["latent"]
        values_flat = values["latent"].reshape(mask.shape[0], -1)
        return (values_flat * mask).sum(dim=1) / mask.sum(dim=1)


def _dynamic_mask_adapter() -> DynamicMaskAdapterFake:
    adapter = object.__new__(DynamicMaskAdapterFake)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake())
    adapter.scheduler_group = adapter.build_scheduler_group()
    return adapter


def _masked_adapter() -> MaskedAdapterFake:
    adapter = object.__new__(MaskedAdapterFake)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake())
    adapter.scheduler_group = adapter.build_scheduler_group()
    return adapter


def _structured_adapter(
    video_dynamics: str = "Flow-SDE", audio_dynamics: str = "CPS"
) -> StructuredAdapterFake:
    adapter = object.__new__(StructuredAdapterFake)
    video = SchedulerFake(video_dynamics)
    adapter.pipeline = SimpleNamespace(scheduler=video)
    adapter.scheduler_group = SchedulerGroup(
        {"video": video, "audio": SchedulerFake(audio_dynamics)},
        primary_name="video",
    )
    return adapter


def _trainer(cls: type, adapter: BaseAdapter, **training_args: Any) -> Any:
    trainer = object.__new__(cls)
    trainer.adapter = adapter
    trainer.training_args = TrainingArgsFake(training_args)
    return trainer


def _replay(state: Dict[str, torch.Tensor], log_prob: Optional[torch.Tensor] = None) -> ReplayStep:
    names = tuple(state)
    batch_size = state[names[0]].shape[0]
    return ReplayStep(
        state=LatentState(dict(state)),
        next_state=LatentState({name: value + 1 for name, value in state.items()}),
        times=ComponentTimes(
            timestep={name: torch.full((batch_size,), 500.0) for name in names},
            next_timestep={name: torch.zeros(batch_size) for name in names},
        ),
        log_prob=log_prob,
        component_log_probs=None if log_prob is None else {name: log_prob for name in names},
    )


def _legacy_batch() -> Any:
    return BaseSample.stack(
        [
            BaseSample(
                timesteps=torch.tensor([1000.0, 500.0, 0.0]),
                all_latents=torch.tensor([[1.0], [2.0], [3.0]]),
                latent_index_map=torch.tensor([0, 1, 2]),
                log_probs=torch.tensor([0.1, 0.2]),
                log_prob_index_map=torch.tensor([0, 1]),
                height=64,
                prompt_embeds=torch.tensor([4.0]),
            ),
            BaseSample(
                timesteps=torch.tensor([1000.0, 500.0, 0.0]),
                all_latents=torch.tensor([[10.0], [20.0], [30.0]]),
                latent_index_map=torch.tensor([0, 1, 2]),
                log_probs=torch.tensor([0.3, 0.4]),
                log_prob_index_map=torch.tensor([0, 1]),
                height=64,
                prompt_embeds=torch.tensor([5.0]),
            ),
        ]
    )


def _sparse_legacy_batch() -> Any:
    """Legacy rollout storing only the last two of three denoising transitions."""
    samples = []
    for offset in (0.0, 100.0):
        samples.append(
            BaseSample(
                timesteps=torch.tensor([1000.0, 700.0, 300.0]),
                all_latents=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) + offset,
                latent_index_map=torch.tensor([-1, 0, 1, 2]),
                log_probs=torch.tensor([0.1, 0.2]) + offset,
                log_prob_index_map=torch.tensor([-1, 0, 1]),
                extra_kwargs={
                    "velocity": torch.tensor([[7.0, 8.0], [9.0, 10.0]]) + offset,
                    "callback_index_map": torch.tensor([-1, 0, 1]),
                },
            )
        )
    return BaseSample.stack(samples)


@pytest.mark.parametrize("timestep_index", [1, 2])
def test_replay_hooks_reproduce_legacy_sparse_index_expressions(timestep_index: int) -> None:
    adapter = _adapter()
    batch = _sparse_legacy_batch()
    latent_index_map = batch["latent_index_map"]
    log_prob_index_map = batch["log_prob_index_map"]
    callback_index_map = batch["callback_index_map"][0]
    num_timesteps = batch["timesteps"].shape[1]

    replay = adapter.get_replay_step(batch, timestep_index)
    callback = adapter.get_replay_callback(batch, timestep_index, "velocity")

    assert torch.equal(
        replay.state.components["latent"],
        batch["all_latents"][:, latent_index_map[timestep_index]],
    )
    assert torch.equal(
        replay.next_state.components["latent"],
        batch["all_latents"][:, latent_index_map[timestep_index + 1]],
    )
    assert torch.equal(replay.log_prob, batch["log_probs"][:, log_prob_index_map[timestep_index]])
    assert torch.equal(replay.times.timestep["latent"], batch["timesteps"][:, timestep_index])
    expected_next_timestep = (
        batch["timesteps"][:, timestep_index + 1]
        if timestep_index + 1 < num_timesteps
        else torch.tensor(0)
    )
    assert torch.equal(replay.times.next_timestep["latent"], expected_next_timestep)
    assert torch.equal(
        callback.components["latent"],
        batch["velocity"][:, callback_index_map[timestep_index]],
    )


def _legacy_squared_error_kl(new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    """Reproduce the pre-migration GRPO reference-KL reduction."""
    return torch.mean(torch.mean((new - old) ** 2, dim=tuple(range(1, new.ndim)), keepdim=True))


def test_grpo_reference_kl_matches_legacy_velocity_formula() -> None:
    torch.manual_seed(0)
    new_velocity = torch.randn(2, 3, 4)
    ref_velocity = torch.randn(2, 3, 4)
    trainer = _trainer(GRPOTrainer, _adapter(), kl_type="v-based")
    replay = _replay({"latent": torch.zeros(2, 3, 4)})

    kl_div = trainer._reference_kl_divergence(
        MultiModalStepOutput(velocity=LatentState({"latent": new_velocity})),
        MultiModalStepOutput(velocity=LatentState({"latent": ref_velocity})),
        replay,
    )

    assert torch.equal(kl_div, _legacy_squared_error_kl(new_velocity, ref_velocity))


def test_grpo_reference_kl_matches_legacy_next_state_mean_formula() -> None:
    torch.manual_seed(1)
    new_mean = torch.randn(2, 6)
    ref_mean = torch.randn(2, 6)
    trainer = _trainer(GRPOTrainer, _adapter(), kl_type="x-based")
    replay = _replay({"latent": torch.zeros(2, 6)})

    kl_div = trainer._reference_kl_divergence(
        MultiModalStepOutput(next_state_mean=LatentState({"latent": new_mean})),
        MultiModalStepOutput(next_state_mean=LatentState({"latent": ref_mean})),
        replay,
    )

    assert torch.equal(kl_div, _legacy_squared_error_kl(new_mean, ref_mean))


def test_grpo_reference_kl_reduces_raw_elements_across_components() -> None:
    """A global element mean, not an unweighted average of component means."""
    torch.manual_seed(2)
    new_video, ref_video = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    new_audio, ref_audio = torch.randn(2, 5), torch.randn(2, 5)
    trainer = _trainer(GRPOTrainer, _structured_adapter(), kl_type="v-based")
    replay = _replay({"video": torch.zeros(2, 3, 4), "audio": torch.zeros(2, 5)})

    kl_div = trainer._reference_kl_divergence(
        MultiModalStepOutput(velocity=LatentState({"video": new_video, "audio": new_audio})),
        MultiModalStepOutput(velocity=LatentState({"video": ref_video, "audio": ref_audio})),
        replay,
    )

    video_sum = (new_video - ref_video).flatten(1).pow(2).sum(dim=1)
    audio_sum = (new_audio - ref_audio).flatten(1).pow(2).sum(dim=1)
    expected = torch.mean((video_sum + audio_sum) / 17)
    assert torch.equal(kl_div, expected)


def test_grpo_replay_requires_stored_joint_log_probability() -> None:
    trainer = _trainer(GRPOTrainer, _adapter())
    replay = _replay({"latent": torch.zeros(2, 3)})

    with pytest.raises(
        ValueError,
        match=r"stored rollout log_prob.*GRPOTrainer.*step_index=3.*batch size 2"
        r".*NoneType.*compute_log_prob",
    ):
        trainer._require_replay_log_prob(replay, 3)


@pytest.mark.parametrize(
    "trainer_cls", [GRPOTrainer, GRPOGuardTrainer, DPPOTrainer], ids=["grpo", "guard", "dppo"]
)
def test_stored_joint_log_probability_must_be_one_scalar_per_sample(trainer_cls: type) -> None:
    """The replay state, not the stored tensor, defines the expected batch size."""
    trainer = _trainer(trainer_cls, _adapter())
    replay = ReplayStep(
        state=LatentState({"latent": torch.zeros(2, 3)}),
        next_state=LatentState({"latent": torch.zeros(2, 3)}),
        times=ComponentTimes(
            timestep={"latent": torch.zeros(2)}, next_timestep={"latent": torch.zeros(2)}
        ),
    )
    object.__setattr__(replay, "log_prob", torch.zeros(2, 3))

    with pytest.raises(
        ValueError,
        match=rf"stored rollout log_prob.*{trainer_cls.__name__}.*step_index=3.*batch size 2"
        r".*\(2, 3\)",
    ):
        trainer._require_replay_log_prob(replay, 3)


def test_stored_joint_log_probability_returns_the_validated_tensor() -> None:
    trainer = _trainer(GRPOTrainer, _adapter())
    log_prob = torch.tensor([0.25, -0.5])
    replay = _replay({"latent": torch.zeros(2, 3)}, log_prob=log_prob)

    assert trainer._require_replay_log_prob(replay, 0) is log_prob


def test_grpo_reference_kl_passes_replay_state_to_the_global_reducer() -> None:
    """A dynamic mask adapter needs state context to weight only active elements."""
    torch.manual_seed(21)
    new_velocity, ref_velocity = torch.randn(2, 4), torch.randn(2, 4)
    adapter = _dynamic_mask_adapter()
    trainer = _trainer(GRPOTrainer, adapter, kl_type="v-based")
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    replay = _replay({"latent": mask})

    kl_div = trainer._reference_kl_divergence(
        MultiModalStepOutput(velocity=LatentState({"latent": new_velocity})),
        MultiModalStepOutput(velocity=LatentState({"latent": ref_velocity})),
        replay,
    )

    squared = (new_velocity - ref_velocity) ** 2
    expected = torch.mean((squared * mask).sum(dim=1) / mask.sum(dim=1))
    assert torch.equal(kl_div, expected)


def test_dppo_trust_region_kl_passes_replay_state_to_the_global_reducer() -> None:
    torch.manual_seed(22)
    new_velocity, old_velocity = torch.randn(2, 4), torch.randn(2, 4)
    adapter = _dynamic_mask_adapter()
    trainer = _trainer(DPPOTrainer, adapter, kl_mask_type="v-based")
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    replay = _replay({"latent": mask})

    kl_new_old = trainer._trust_region_kl(
        MultiModalStepOutput(velocity=LatentState({"latent": new_velocity})),
        replay,
        LatentState({"latent": old_velocity}),
    )

    squared = (new_velocity - old_velocity) ** 2
    expected = (squared * mask).sum(dim=1) / mask.sum(dim=1)
    assert torch.equal(kl_new_old, expected)


def test_grpo_replay_forward_kwargs_keep_batch_precedence() -> None:
    trainer = _trainer(GRPOTrainer, _adapter(), height=512, guidance_scale=3.5)

    kwargs = trainer._replay_forward_kwargs(_legacy_batch())

    assert "height" not in kwargs
    assert kwargs == {"guidance_scale": 3.5}


def test_grpo_replay_forward_passes_state_and_training_arguments() -> None:
    adapter = _adapter()
    trainer = _trainer(GRPOTrainer, adapter, guidance_scale=3.5)
    batch = _legacy_batch()
    replay = adapter.get_replay_step(batch, 0)

    trainer._replay_forward(batch, replay, ("log_prob", "dt"))

    assert adapter.forward_kwargs["latents"] is replay.state.components["latent"]
    assert adapter.forward_kwargs["next_latents"] is replay.next_state.components["latent"]
    assert adapter.forward_kwargs["compute_log_prob"] is True
    assert adapter.forward_kwargs["noise_level"] == 0.7
    assert adapter.forward_kwargs["guidance_scale"] == 3.5
    assert adapter.forward_kwargs["return_kwargs"] == ("log_prob", "dt")


def _guard_output(
    new_mean: Dict[str, torch.Tensor],
    new_log_probs: Dict[str, torch.Tensor],
    std_dev_t: Optional[Dict[str, torch.Tensor]],
    dt: Optional[Dict[str, torch.Tensor]],
) -> MultiModalStepOutput:
    joint = None
    for value in new_log_probs.values():
        joint = value if joint is None else joint + value
    return MultiModalStepOutput(
        next_state_mean=LatentState(dict(new_mean)),
        std_dev_t=std_dev_t,
        dt=dt,
        log_prob=joint,
        component_log_probs=dict(new_log_probs),
    )


@pytest.mark.parametrize("latent_shape", [(2, 3, 4), (2, 3, 4, 5)])
def test_guard_ratio_normalizes_broadcast_scheduler_statistics(latent_shape: Any) -> None:
    """Schedulers emit ``(B, 1, ...)`` statistics; the scalar formula needs ``(B,)``."""
    torch.manual_seed(3)
    new_mean, old_mean = torch.randn(*latent_shape), torch.randn(*latent_shape)
    new_log_prob, old_log_prob = torch.randn(2), torch.randn(2)
    broadcast_shape = (2,) + (1,) * (len(latent_shape) - 1)
    std_dev_t = torch.tensor([0.3, 0.35]).reshape(broadcast_shape)
    dt = torch.tensor([-0.4, -0.45]).reshape(broadcast_shape)
    trainer = _trainer(GRPOGuardTrainer, _adapter())
    replay = _replay({"latent": torch.zeros(*latent_shape)}, log_prob=old_log_prob)
    output = _guard_output(
        {"latent": new_mean}, {"latent": new_log_prob}, {"latent": std_dev_t}, {"latent": dt}
    )

    ratio = trainer._guard_ratio(output, replay, LatentState({"latent": old_mean}))

    scale_factor = torch.sqrt(-dt.reshape(2)) * std_dev_t.reshape(2)
    mse = (new_mean - old_mean).flatten(1).pow(2).mean(dim=1)
    expected = torch.exp((new_log_prob - old_log_prob) * scale_factor + mse / (2 * scale_factor))
    assert ratio.shape == (2,)
    assert torch.equal(ratio, expected)


def test_guard_ratio_matches_the_legacy_formula_for_a_single_sample() -> None:
    """With one sample the legacy broadcast was harmless, so values must match."""
    torch.manual_seed(12)
    new_mean, old_mean = torch.randn(1, 3, 4), torch.randn(1, 3, 4)
    new_log_prob, old_log_prob = torch.randn(1), torch.randn(1)
    std_dev_t, dt = torch.full((1, 1, 1), 0.3), torch.full((1, 1, 1), -0.4)
    trainer = _trainer(GRPOGuardTrainer, _adapter())
    replay = _replay({"latent": torch.zeros(1, 3, 4)}, log_prob=old_log_prob)
    output = _guard_output(
        {"latent": new_mean}, {"latent": new_log_prob}, {"latent": std_dev_t}, {"latent": dt}
    )

    ratio = trainer._guard_ratio(output, replay, LatentState({"latent": old_mean}))

    legacy_scale = torch.sqrt(-dt) * std_dev_t
    legacy_mse = (new_mean - old_mean).flatten(1).pow(2).mean(dim=1)
    legacy = torch.exp(
        (new_log_prob - old_log_prob) * legacy_scale + legacy_mse / (2 * legacy_scale)
    )
    assert legacy.shape == (1, 1, 1)
    assert torch.equal(ratio, legacy.reshape(1))


def test_guard_ratio_weights_two_components_by_active_degrees_of_freedom() -> None:
    torch.manual_seed(4)
    new_video, old_video = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    new_audio, old_audio = torch.randn(2, 5), torch.randn(2, 5)
    new_log_probs = {"video": torch.randn(2), "audio": torch.randn(2)}
    old_log_probs = {"video": torch.randn(2), "audio": torch.randn(2)}
    std = {
        "video": torch.tensor([0.3, 0.32]).reshape(2, 1, 1),
        "audio": torch.tensor([0.6, 0.62]).reshape(2, 1),
    }
    dt = {
        "video": torch.tensor([-0.4, -0.42]).reshape(2, 1, 1),
        "audio": torch.tensor([-0.2, -0.22]).reshape(2, 1),
    }
    trainer = _trainer(GRPOGuardTrainer, _structured_adapter())
    replay = ReplayStep(
        state=LatentState({"video": torch.zeros(2, 3, 4), "audio": torch.zeros(2, 5)}),
        next_state=LatentState({"video": torch.zeros(2, 3, 4), "audio": torch.zeros(2, 5)}),
        times=ComponentTimes(
            timestep={"video": torch.zeros(2), "audio": torch.zeros(2)},
            next_timestep={"video": torch.zeros(2), "audio": torch.zeros(2)},
        ),
        log_prob=old_log_probs["video"] + old_log_probs["audio"],
        component_log_probs=old_log_probs,
    )
    output = _guard_output({"video": new_video, "audio": new_audio}, new_log_probs, std, dt)

    ratio = trainer._guard_ratio(
        output, replay, LatentState({"video": old_video, "audio": old_audio})
    )

    terms = {}
    for name, new_value, old_value in (
        ("video", new_video, old_video),
        ("audio", new_audio, old_audio),
    ):
        scale = std[name].reshape(2)
        if name == "video":
            scale = torch.sqrt(-dt[name].reshape(2)) * scale
        mse = (new_value - old_value).flatten(1).pow(2).mean(dim=1)
        terms[name] = (new_log_probs[name] - old_log_probs[name]) * scale + mse / (2 * scale)
    expected = torch.exp((terms["video"] * 12 + terms["audio"] * 5) / 17)
    assert torch.equal(ratio, expected)


def test_guard_ratio_rejects_a_zero_stochastic_transition_scale() -> None:
    trainer = _trainer(GRPOGuardTrainer, _adapter(dynamics_type="CPS"))
    replay = _replay({"latent": torch.zeros(1, 4)}, log_prob=torch.zeros(1))
    output = _guard_output(
        {"latent": torch.zeros(1, 4)},
        {"latent": torch.zeros(1)},
        {"latent": torch.zeros(1, 1)},
        {"latent": torch.full((1, 1), -0.5)},
    )

    with pytest.raises(
        ValueError,
        match=r"GRPO-Guard ratio.*component 'latent'.*strictly positive.*CPS",
    ):
        trainer._guard_ratio(output, replay, LatentState({"latent": torch.zeros(1, 4)}))


def test_guard_statistic_normalization_rejects_multiple_non_batch_values() -> None:
    trainer = _trainer(GRPOGuardTrainer, _adapter())

    with pytest.raises(
        ValueError,
        match=r"std_dev_t.*component 'latent'.*one value per sample.*\(2, 3, 1\)",
    ):
        trainer._scalar_component_statistic(
            {"latent": torch.full((2, 3, 1), 0.3)}, "std_dev_t", "latent", 2
        )


def test_guard_statistic_normalization_rejects_a_mismatched_batch_size() -> None:
    """The replay batch size, not the policy output's own state, drives the formula."""
    trainer = _trainer(GRPOGuardTrainer, _adapter())

    with pytest.raises(ValueError, match=r"std_dev_t.*component 'latent'.*batch size 2.*\(1, 1\)"):
        trainer._scalar_component_statistic(
            {"latent": torch.full((1, 1), 0.3)}, "std_dev_t", "latent", 2
        )


def test_guard_mean_squared_error_uses_overridable_component_reducer() -> None:
    """A masked adapter reduces each component over active elements only."""
    torch.manual_seed(9)
    new_mean, old_mean = torch.randn(2, 4), torch.randn(2, 4)
    new_log_prob, old_log_prob = torch.randn(2), torch.randn(2)
    std_dev_t = torch.tensor([0.3, 0.35]).reshape(2, 1)
    dt = torch.tensor([-0.4, -0.45]).reshape(2, 1)
    trainer = _trainer(GRPOGuardTrainer, _masked_adapter())
    replay = _replay({"latent": torch.zeros(2, 4)}, log_prob=old_log_prob)
    output = _guard_output(
        {"latent": new_mean}, {"latent": new_log_prob}, {"latent": std_dev_t}, {"latent": dt}
    )

    ratio = trainer._guard_ratio(output, replay, LatentState({"latent": old_mean}))

    scale_factor = torch.sqrt(-dt.reshape(2)) * std_dev_t.reshape(2)
    active_mse = (new_mean - old_mean)[:, :2].pow(2).mean(dim=1)
    expected = torch.exp(
        (new_log_prob - old_log_prob) * scale_factor + active_mse / (2 * scale_factor)
    )
    assert torch.equal(ratio, expected)


def test_guard_ratio_requires_a_well_formed_policy_log_probability() -> None:
    log_prob = torch.zeros(2)
    trainer = _trainer(GRPOGuardTrainer, _adapter())
    replay = _replay({"latent": torch.zeros(2, 4)}, log_prob=log_prob)
    output = MultiModalStepOutput(
        next_state_mean=LatentState({"latent": torch.zeros(2, 4)}),
        std_dev_t={"latent": torch.full((2, 1), 0.3)},
        dt={"latent": torch.full((2, 1), -0.4)},
        component_log_probs={"latent": log_prob},
    )

    with pytest.raises(
        ValueError, match=r"policy log_prob.*GRPOGuardTrainer.*step_index=4.*NoneType"
    ):
        trainer._guard_ratio(output, replay, LatentState({"latent": torch.zeros(2, 4)}), 4)


def test_guard_ratio_requires_component_standard_deviation() -> None:
    trainer = _trainer(GRPOGuardTrainer, _adapter())
    log_prob = torch.zeros(2)
    replay = _replay({"latent": torch.zeros(2, 4)}, log_prob=log_prob)
    output = _guard_output(
        {"latent": torch.zeros(2, 4)}, {"latent": log_prob}, None, {"latent": torch.zeros(2, 1)}
    )

    with pytest.raises(
        ValueError, match=r"policy output std_dev_t.*GRPOGuardTrainer.*\('latent',\)"
    ):
        trainer._guard_ratio(output, replay, LatentState({"latent": torch.zeros(2, 4)}))


def test_guard_ratio_requires_old_component_log_probabilities() -> None:
    trainer = _trainer(GRPOGuardTrainer, _adapter())
    log_prob = torch.zeros(2)
    replay = _replay({"latent": torch.zeros(2, 4)})
    output = _guard_output(
        {"latent": torch.zeros(2, 4)},
        {"latent": log_prob},
        {"latent": torch.full((2, 1), 0.3)},
        {"latent": torch.full((2, 1), -0.4)},
    )

    with pytest.raises(
        ValueError, match=r"stored rollout component_log_probs.*GRPOGuardTrainer.*\('latent',\)"
    ):
        trainer._guard_ratio(output, replay, LatentState({"latent": torch.zeros(2, 4)}))


def test_dppo_velocity_mask_kl_matches_legacy_formula() -> None:
    torch.manual_seed(5)
    new_velocity, old_velocity = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    trainer = _trainer(DPPOTrainer, _adapter(), kl_mask_type="v-based")
    replay = _replay({"latent": torch.zeros(2, 3, 4)})

    kl_new_old = trainer._trust_region_kl(
        MultiModalStepOutput(velocity=LatentState({"latent": new_velocity})),
        replay,
        LatentState({"latent": old_velocity}),
    )

    squared = (new_velocity - old_velocity) ** 2
    assert torch.equal(kl_new_old, squared.mean(dim=tuple(range(1, squared.ndim))))


def test_dppo_state_mask_kl_matches_legacy_gaussian_formula() -> None:
    torch.manual_seed(6)
    new_mean, old_mean = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    std_dev_t = torch.tensor([0.3, 0.32]).reshape(2, 1, 1)
    dt = torch.tensor([-0.4, -0.42]).reshape(2, 1, 1)
    trainer = _trainer(DPPOTrainer, _adapter(), kl_mask_type="x-based")
    replay = _replay({"latent": torch.zeros(2, 3, 4)})

    kl_new_old = trainer._trust_region_kl(
        MultiModalStepOutput(
            next_state_mean=LatentState({"latent": new_mean}),
            std_dev_t={"latent": std_dev_t},
            dt={"latent": dt},
        ),
        replay,
        LatentState({"latent": old_mean}),
    )

    sigma = std_dev_t * torch.sqrt(-dt)
    kl_elem = gaussian_kl_div(new_mean, old_mean, sigma)
    assert torch.equal(kl_new_old, kl_elem.mean(dim=tuple(range(1, kl_elem.ndim))))


def test_dppo_state_mask_uses_each_component_scheduler_denominator() -> None:
    """Per-component sigma, then one raw global element mean across components."""
    torch.manual_seed(7)
    new_video, old_video = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    new_audio, old_audio = torch.randn(2, 5), torch.randn(2, 5)
    std = {
        "video": torch.tensor([0.3, 0.32]).reshape(2, 1, 1),
        "audio": torch.tensor([0.6, 0.62]).reshape(2, 1),
    }
    dt = {
        "video": torch.tensor([-0.4, -0.42]).reshape(2, 1, 1),
        "audio": torch.tensor([-0.2, -0.22]).reshape(2, 1),
    }
    trainer = _trainer(
        DPPOTrainer,
        _structured_adapter(video_dynamics="Flow-SDE", audio_dynamics="CPS"),
        kl_mask_type="x-based",
    )
    replay = _replay({"video": torch.zeros(2, 3, 4), "audio": torch.zeros(2, 5)})

    kl_new_old = trainer._trust_region_kl(
        MultiModalStepOutput(
            next_state_mean=LatentState({"video": new_video, "audio": new_audio}),
            std_dev_t=std,
            dt=dt,
        ),
        replay,
        LatentState({"video": old_video, "audio": old_audio}),
    )

    video_sigma = std["video"] * torch.sqrt(-dt["video"])
    audio_sigma = std["audio"]
    video_sum = gaussian_kl_div(new_video, old_video, video_sigma).flatten(1).sum(dim=1)
    audio_sum = gaussian_kl_div(new_audio, old_audio, audio_sigma).flatten(1).sum(dim=1)
    assert torch.equal(kl_new_old, (video_sum + audio_sum) / 17)


@pytest.mark.parametrize(
    "trainer_cls", [GRPOTrainer, GRPOGuardTrainer, DPPOTrainer], ids=["grpo", "guard", "dppo"]
)
def test_policy_log_probability_must_be_present_before_ppo_arithmetic(trainer_cls: type) -> None:
    trainer = _trainer(trainer_cls, _adapter())
    output = MultiModalStepOutput(next_state_mean=LatentState({"latent": torch.zeros(2, 4)}))

    with pytest.raises(
        ValueError,
        match=rf"policy log_prob.*{trainer_cls.__name__}.*step_index=2.*batch size 2.*NoneType",
    ):
        trainer._require_policy_log_prob(output, 2, 2)


@pytest.mark.parametrize(
    "trainer_cls", [GRPOTrainer, GRPOGuardTrainer, DPPOTrainer], ids=["grpo", "guard", "dppo"]
)
def test_policy_log_probability_must_be_one_scalar_per_sample(trainer_cls: type) -> None:
    trainer = _trainer(trainer_cls, _adapter())
    output = MultiModalStepOutput(log_prob=torch.zeros(2, 3, 4))

    with pytest.raises(
        ValueError,
        match=rf"policy log_prob.*{trainer_cls.__name__}.*step_index=2.*batch size 2"
        r".*\(2, 3, 4\)",
    ):
        trainer._require_policy_log_prob(output, 2, 2)


def test_policy_log_probability_returns_validated_tensor() -> None:
    trainer = _trainer(GRPOTrainer, _adapter())
    log_prob = torch.tensor([0.25, -0.5])
    output = MultiModalStepOutput(log_prob=log_prob)

    assert trainer._require_policy_log_prob(output, 0, 2) is log_prob


def test_return_fields_use_a_deterministic_canonical_order() -> None:
    trainer = _trainer(GRPOTrainer, _adapter())

    assert trainer._canonical_return_fields({"dt", "log_prob", "velocity"}) == (
        "log_prob",
        "dt",
        "velocity",
    )
    assert trainer._canonical_return_fields({"velocity", "log_prob"}) == ("log_prob", "velocity")
    assert trainer._canonical_return_fields(["std_dev_t", "next_latents_mean"]) == (
        "next_latents_mean",
        "std_dev_t",
    )


def test_return_fields_reject_unknown_scheduler_output_names() -> None:
    trainer = _trainer(GRPOTrainer, _adapter())

    with pytest.raises(ValueError, match=r"unknown return field 'entropy'"):
        trainer._canonical_return_fields({"log_prob", "entropy"})


def test_dppo_effective_sigma_rejects_ode_component() -> None:
    trainer = _trainer(
        DPPOTrainer,
        _structured_adapter(video_dynamics="Flow-SDE", audio_dynamics="ODE"),
        kl_mask_type="x-based",
    )

    with pytest.raises(ValueError, match=r"component 'audio'.*'ODE'.*Flow-SDE"):
        trainer._effective_sigma("audio", torch.tensor([0.3]), torch.tensor([-0.4]))


def test_dppo_reference_forward_applies_kl_guidance_scale_override() -> None:
    adapter = _adapter()
    trainer = _trainer(DPPOTrainer, adapter, guidance_scale=4.5)
    batch = _legacy_batch()
    replay = adapter.get_replay_step(batch, 0)

    trainer._reference_forward(batch, replay, ("velocity",), guidance_scale=1.0)

    assert adapter.ref_scope_entered is True
    assert adapter.forward_kwargs["guidance_scale"] == 1.0
    assert adapter.forward_kwargs["compute_log_prob"] is False
    assert adapter.forward_kwargs["return_kwargs"] == ("velocity",)
