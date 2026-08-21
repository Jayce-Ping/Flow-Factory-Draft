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

"""Drive every algorithm's production formulas on real LTX2 rollout trajectories.

Each test replays a transition from an actual LTX2 rollout — real component
states, per-component schedules, per-scheduler statistics and the I2AV
conditioning mask — and compares a trainer's production helper against an
independently written expectation.
"""

from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Tuple

import pytest
import torch
from ltx2_fakes import (
    AUDIO_SCHEDULER_OFFSET,
    AUDIO_SEQ_LEN,
    BATCH_SIZE,
    CHANNELS,
    FRAME_SEQ_LEN,
    GENERATED_VIDEO_NUMEL,
    VIDEO_SCHEDULER_OFFSET,
    VIDEO_SEQ_LEN,
)
from ltx2_inference_fakes import SEED, condition_images, inference_adapter, inference_kwargs

from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter, LTX2I2AVSample
from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter, LTX2Sample
from flow_factory.samples import LatentState, MultiModalStepOutput, NoisedState
from flow_factory.trainers.distillation.opd.common import (
    compute_structured_distillation_loss,
    project_distillation_target_state,
)
from flow_factory.trainers.distillation.opd.trainer import DiffusionOPDTrainer
from flow_factory.trainers.rl.awm import AWMTrainer
from flow_factory.trainers.rl.crd import CRDTrainer
from flow_factory.trainers.rl.dgpo import DGPOTrainer
from flow_factory.trainers.rl.dpo import DPOTrainer
from flow_factory.trainers.rl.dppo import DPPOTrainer
from flow_factory.trainers.rl.grpo import GRPOGuardTrainer, GRPOTrainer
from flow_factory.trainers.rl.nft import DiffusionNFTTrainer

COMPONENTS = ("video", "audio")
AUDIO_NUMEL = AUDIO_SEQ_LEN * CHANNELS
VIDEO_NUMEL = VIDEO_SEQ_LEN * CHANNELS
REPLAY_STEP = 0
RETURN_FIELDS = ("next_latents", "next_latents_mean", "std_dev_t", "dt", "log_prob", "velocity")


class TrainingArgsFake(dict):
    """Mapping/attribute hybrid mirroring ``ArgABC`` unpacking behaviour."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


def _rollout(cls: type, sample_type: type, *, one_group: bool = False) -> SimpleNamespace:
    """Run one rollout and expose its collated batch, replay step and step output.

    ``one_group`` gives every sample the same prompt identity, so the derived
    ``unique_id`` places the whole micro-batch in one advantage group.
    """
    kwargs = inference_kwargs()
    if one_group:
        kwargs["prompt"] = ["a"] * BATCH_SIZE
        kwargs["prompt_ids"] = torch.zeros(BATCH_SIZE, 2, dtype=torch.int64)
    if cls is LTX2_I2AV_Adapter:
        kwargs["condition_images"] = condition_images()
    torch.manual_seed(SEED)
    adapter, _ = inference_adapter(cls)
    samples = adapter.inference(**kwargs)
    batch = sample_type.stack(samples)
    replay = adapter.get_replay_step(batch, REPLAY_STEP)
    output = adapter.forward_state(
        batch=batch,
        state=replay.state,
        times=replay.times,
        next_state=replay.next_state,
        compute_log_prob=True,
        return_fields=RETURN_FIELDS,
        guidance_scale=1.0,
    )
    return SimpleNamespace(
        adapter=adapter,
        samples=samples,
        batch=batch,
        replay=replay,
        output=output,
        conditioned=cls is LTX2_I2AV_Adapter,
    )


@pytest.fixture(scope="module")
def t2av() -> SimpleNamespace:
    """Return the unconditioned T2AV rollout context."""
    return _rollout(LTX2_T2AV_Adapter, LTX2Sample)


@pytest.fixture(scope="module")
def i2av() -> SimpleNamespace:
    """Return the conditioned I2AV rollout context."""
    return _rollout(LTX2_I2AV_Adapter, LTX2I2AVSample)


@pytest.fixture(params=["t2av", "i2av"])
def rollout(request: Any) -> SimpleNamespace:
    """Return each LTX2 rollout context in turn."""
    return request.getfixturevalue(request.param)


def _trainer(cls: type, rollout: SimpleNamespace, **attributes: Any) -> Any:
    """Build a trainer shell holding only what a production formula reads."""
    trainer = object.__new__(cls)
    trainer.adapter = rollout.adapter
    trainer.training_args = TrainingArgsFake(attributes.pop("training_args", {}))
    for name, value in attributes.items():
        setattr(trainer, name, value)
    return trainer


def _active(rollout: SimpleNamespace, name: str, values: torch.Tensor) -> torch.Tensor:
    """Drop the I2AV conditioning tokens an independent expectation must ignore."""
    if rollout.conditioned and name == "video":
        return values[:, FRAME_SEQ_LEN:]
    return values


def _active_numel(rollout: SimpleNamespace) -> Dict[str, int]:
    video = GENERATED_VIDEO_NUMEL if rollout.conditioned else VIDEO_NUMEL
    return {"video": video, "audio": AUDIO_NUMEL}


def _component_means(
    rollout: SimpleNamespace, values: Mapping[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Average each component over its active elements only."""
    return {
        name: _active(rollout, name, values[name]).reshape(BATCH_SIZE, -1).mean(dim=1)
        for name in COMPONENTS
    }


def _joint_mean(rollout: SimpleNamespace, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Average every active element of every component exactly once."""
    total: Optional[torch.Tensor] = None
    count = 0
    for name in COMPONENTS:
        active = _active(rollout, name, values[name]).reshape(BATCH_SIZE, -1)
        summed = active.sum(dim=1)
        total = summed if total is None else total + summed
        count += active.shape[1]
    return total / count


def _dof_weighted(rollout: SimpleNamespace, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Combine already-reduced component scalars by their active degrees of freedom."""
    numel = _active_numel(rollout)
    total = sum(numel.values())
    return sum(values[name] * numel[name] for name in COMPONENTS) / total


def _shift(state: LatentState, factor: float) -> LatentState:
    """Perturb every component of a state, keeping the component order."""
    return LatentState({name: state.components[name] * factor for name in COMPONENTS})


def _output(rollout: SimpleNamespace, **overrides: Any) -> MultiModalStepOutput:
    """Rebuild the real step output with selected fields replaced."""
    source = rollout.output
    fields = {
        "next_state": source.next_state,
        "next_state_mean": source.next_state_mean,
        "std_dev_t": source.std_dev_t,
        "dt": source.dt,
        "log_prob": source.log_prob,
        "component_log_probs": source.component_log_probs,
        "velocity": source.velocity,
    }
    fields.update(overrides)
    return MultiModalStepOutput(**fields)


def _noised(rollout: SimpleNamespace, seed: int = 11) -> Tuple[NoisedState, Any]:
    """Draw one forward-process noising of the replayed state."""
    replay = rollout.replay
    noised = rollout.adapter.add_forward_process_noise(
        replay.state, replay.times, generator=torch.Generator().manual_seed(seed)
    )
    return noised, replay.times


def _velocity(rollout: SimpleNamespace, factor: float) -> LatentState:
    return _shift(rollout.output.velocity, factor)


def _statistic(values: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    return values[name].reshape(BATCH_SIZE)


def test_the_grpo_joint_ratio_replays_the_stored_transition_at_unit_ratio(
    rollout: SimpleNamespace,
) -> None:
    trainer = _trainer(GRPOTrainer, rollout)

    old_log_prob = trainer._require_replay_log_prob(rollout.replay, REPLAY_STEP)
    new_log_prob = trainer._require_policy_log_prob(rollout.output, REPLAY_STEP, BATCH_SIZE)

    assert torch.equal(new_log_prob, old_log_prob)
    assert torch.equal(torch.exp(new_log_prob - old_log_prob), torch.ones(BATCH_SIZE))


def test_the_grpo_joint_log_probability_weights_both_component_schedulers(
    rollout: SimpleNamespace,
) -> None:
    numel = _active_numel(rollout)
    components = rollout.output.component_log_probs

    joint = rollout.output.log_prob

    expected = (components["video"] * numel["video"] + components["audio"] * numel["audio"]) / (
        numel["video"] + numel["audio"]
    )
    assert torch.allclose(joint, expected)


@pytest.mark.parametrize(
    ("kl_type", "field"), [("v-based", "velocity"), ("x-based", "next_state_mean")]
)
def test_the_grpo_reference_kl_reduces_only_the_active_component_elements(
    rollout: SimpleNamespace, kl_type: str, field: str
) -> None:
    trainer = _trainer(GRPOTrainer, rollout, training_args={"kl_type": kl_type})
    reference = _output(rollout, **{field: _shift(getattr(rollout.output, field), 1.25)})

    kl = trainer._reference_kl_divergence(rollout.output, reference, rollout.replay)

    errors = {
        name: (
            getattr(rollout.output, field).components[name]
            - getattr(reference, field).components[name]
        )
        ** 2
        for name in COMPONENTS
    }
    assert torch.allclose(kl, _joint_mean(rollout, errors).mean())


def test_the_guard_ratio_reweights_each_component_by_its_own_scheduler(
    rollout: SimpleNamespace,
) -> None:
    trainer = _trainer(GRPOGuardTrainer, rollout)
    old_state_mean = _shift(rollout.output.next_state_mean, 0.75)

    ratio = trainer._guard_ratio(rollout.output, rollout.replay, old_state_mean, REPLAY_STEP)

    squared = {
        name: (rollout.output.next_state_mean.components[name] - old_state_mean.components[name])
        ** 2
        for name in COMPONENTS
    }
    component_mse = _component_means(rollout, squared)
    terms = {}
    for name in COMPONENTS:
        scale = torch.sqrt(-_statistic(rollout.output.dt, name)) * _statistic(
            rollout.output.std_dev_t, name
        )
        delta = rollout.output.component_log_probs[name] - rollout.replay.component_log_probs[name]
        terms[name] = delta * scale + component_mse[name] / (2 * scale)
    assert torch.allclose(ratio, torch.exp(_dof_weighted(rollout, terms)))


def test_the_guard_ratio_is_one_when_the_stored_transition_is_replayed(
    rollout: SimpleNamespace,
) -> None:
    trainer = _trainer(GRPOGuardTrainer, rollout)

    ratio = trainer._guard_ratio(
        rollout.output, rollout.replay, rollout.output.next_state_mean, REPLAY_STEP
    )

    assert torch.allclose(ratio, torch.ones(BATCH_SIZE))


def test_the_dppo_velocity_trust_region_reduces_the_active_elements(
    rollout: SimpleNamespace,
) -> None:
    trainer = _trainer(DPPOTrainer, rollout, training_args={"kl_mask_type": "v-based"})
    old_velocity = _shift(rollout.output.velocity, 1.5)

    kl = trainer._trust_region_kl(rollout.output, rollout.replay, old_velocity)

    errors = {
        name: (rollout.output.velocity.components[name] - old_velocity.components[name]) ** 2
        for name in COMPONENTS
    }
    assert torch.allclose(kl, _joint_mean(rollout, errors))


def test_the_dppo_state_trust_region_uses_each_components_own_scheduler_variance(
    rollout: SimpleNamespace,
) -> None:
    trainer = _trainer(DPPOTrainer, rollout, training_args={"kl_mask_type": "x-based"})
    old_state = _shift(rollout.output.next_state_mean, 0.5)

    kl = trainer._trust_region_kl(rollout.output, rollout.replay, old_state)

    errors = {}
    for name in COMPONENTS:
        sigma = rollout.output.std_dev_t[name] * torch.sqrt(-rollout.output.dt[name])
        errors[name] = (
            rollout.output.next_state_mean.components[name] - old_state.components[name]
        ) ** 2 / (2 * sigma**2)
    assert torch.allclose(kl, _joint_mean(rollout, errors))
    video_sigma = VIDEO_SCHEDULER_OFFSET * VIDEO_SCHEDULER_OFFSET**0.5
    audio_sigma = AUDIO_SCHEDULER_OFFSET * AUDIO_SCHEDULER_OFFSET**0.5
    assert video_sigma != audio_sigma


def test_the_nft_matching_losses_target_each_components_own_x0(
    rollout: SimpleNamespace,
) -> None:
    trainer = _trainer(DiffusionNFTTrainer, rollout, nft_beta=0.4)
    noised, times = _noised(rollout)
    new_velocity = _velocity(rollout, 1.0)
    old_velocity = _velocity(rollout, 0.8)

    positive, negative = trainer._matching_losses(
        rollout.replay.state, noised, times, new_velocity, old_velocity
    )

    beta = 0.4
    predictions = {
        "positive": {
            name: beta * new_velocity.components[name] + (1 - beta) * old_velocity.components[name]
            for name in COMPONENTS
        },
        "negative": {
            name: (1.0 + beta) * old_velocity.components[name]
            - beta * new_velocity.components[name]
            for name in COMPONENTS
        },
    }
    expected = {}
    for arm, prediction in predictions.items():
        x0 = {}
        for name in COMPONENTS:
            sigma = times.sigma[name].reshape(BATCH_SIZE, 1, 1)
            x0[name] = noised.state.components[name] - sigma * prediction[name]
        clean = rollout.replay.state
        # The production formula divides each component by its own detached mean
        # absolute x0 deviation over that component's active elements.
        scales = _component_means(
            rollout,
            {
                name: torch.abs(x0[name].double() - clean.components[name].double())
                for name in COMPONENTS
            },
        )
        expected[arm] = _joint_mean(
            rollout,
            {
                name: (x0[name] - clean.components[name]) ** 2
                / scales[name].clip(min=1e-5).reshape(BATCH_SIZE, 1, 1)
                for name in COMPONENTS
            },
        )
    assert torch.allclose(positive, expected["positive"])
    assert torch.allclose(negative, expected["negative"])


def test_the_nft_reference_kl_reduces_the_active_elements(rollout: SimpleNamespace) -> None:
    trainer = _trainer(DiffusionNFTTrainer, rollout, nft_beta=0.4)
    noised, _ = _noised(rollout)
    velocity = _velocity(rollout, 1.0)
    reference = _velocity(rollout, 0.5)

    kl = trainer._velocity_kl(velocity, reference, noised)

    errors = {
        name: (velocity.components[name] - reference.components[name]) ** 2 for name in COMPONENTS
    }
    assert torch.allclose(kl, _joint_mean(rollout, errors))


def test_the_awm_matching_log_probability_weights_each_component_by_its_own_sigma(
    rollout: SimpleNamespace,
) -> None:
    trainer = _trainer(AWMTrainer, rollout, weighting="t", ghuber_power=0.25)
    noised, times = _noised(rollout)
    velocity = _velocity(rollout, 1.1)

    log_prob = trainer._matching_log_prob(velocity, noised, times)

    component_values = _component_means(
        rollout,
        {
            name: -(
                (
                    velocity.components[name].double()
                    - noised.target_velocity.components[name].double()
                )
                ** 2
            )
            for name in COMPONENTS
        },
    )
    weighted = {name: component_values[name] * times.sigma[name].double() for name in COMPONENTS}
    assert log_prob.dtype == torch.float32
    assert torch.allclose(log_prob, _dof_weighted(rollout, weighted).float())


def test_the_awm_velocity_kl_reduces_the_active_elements(rollout: SimpleNamespace) -> None:
    trainer = _trainer(AWMTrainer, rollout)
    noised, _ = _noised(rollout)
    velocity = _velocity(rollout, 1.0)
    other = _velocity(rollout, 0.6)

    kl = trainer._velocity_kl(velocity, other, noised)

    errors = {
        name: (velocity.components[name] - other.components[name]) ** 2 for name in COMPONENTS
    }
    assert torch.allclose(kl, _joint_mean(rollout, errors))


def test_the_dpo_arms_share_one_component_noise_draw(rollout: SimpleNamespace) -> None:
    replay = rollout.replay
    noise = rollout.adapter.add_forward_process_noise(
        replay.state, replay.times, generator=torch.Generator().manual_seed(7)
    ).noise

    winner = rollout.adapter.apply_forward_process_noise(replay.state, replay.times, noise)
    loser = rollout.adapter.apply_forward_process_noise(replay.state, replay.times, noise)

    for name in COMPONENTS:
        assert torch.equal(winner.state.components[name], loser.state.components[name])
        assert torch.equal(
            winner.target_velocity.components[name], loser.target_velocity.components[name]
        )


def test_the_dpo_preference_errors_reduce_only_the_active_elements(
    rollout: SimpleNamespace,
) -> None:
    trainer = _trainer(DPOTrainer, rollout, training_args={"beta": 2.0})
    noised, _ = _noised(rollout)
    policy = _velocity(rollout, 1.2)
    reference = _velocity(rollout, 0.9)

    policy_error = trainer._arm_velocity_error(policy, noised)
    reference_error = trainer._arm_velocity_error(reference, noised)
    loss, metrics = trainer._preference_loss(
        policy_error, policy_error * 1.5, reference_error, reference_error * 1.5
    )

    expected = _joint_mean(
        rollout,
        {
            name: (
                policy.components[name].float() - noised.target_velocity.components[name].float()
            )
            ** 2
            for name in COMPONENTS
        },
    )
    assert torch.allclose(policy_error, expected)
    assert loss.shape == ()
    assert metrics["implicit_reward_chosen"].shape == (BATCH_SIZE,)


@pytest.mark.parametrize(
    ("cls", "sample_type"),
    [(LTX2_T2AV_Adapter, LTX2Sample), (LTX2_I2AV_Adapter, LTX2I2AVSample)],
    ids=["t2av", "i2av"],
)
def test_the_dgpo_group_noise_is_shared_within_a_group_per_component(
    cls: type, sample_type: type
) -> None:
    rollout = _rollout(cls, sample_type, one_group=True)
    trainer = _trainer(
        DGPOTrainer,
        rollout,
        training_args={"seed": 1234},
        epoch=0,
        use_shared_noise=True,
    )
    assert rollout.samples[0].unique_id == rollout.samples[1].unique_id

    noise = trainer._shared_group_noise(rollout.replay.state, rollout.samples, 0, timestep_index=0)

    for name in COMPONENTS:
        rows = noise.components[name]
        assert rows.shape == rollout.replay.state.components[name].shape
        assert torch.equal(rows[0], rows[1])
    assert not torch.equal(
        noise.components["video"].reshape(-1)[:AUDIO_NUMEL],
        noise.components["audio"].reshape(-1)[:AUDIO_NUMEL],
    )


def test_the_dgpo_denoising_score_matching_reduces_the_active_elements(
    rollout: SimpleNamespace,
) -> None:
    trainer = _trainer(DGPOTrainer, rollout)
    noised, _ = _noised(rollout)
    prediction = _velocity(rollout, 1.3)

    dsm = trainer._compute_dsm_loss(noised.target_velocity, prediction, noised)

    errors = {
        name: (prediction.components[name] - noised.target_velocity.components[name]).float() ** 2
        for name in COMPONENTS
    }
    assert torch.allclose(dsm, _joint_mean(rollout, errors))


@pytest.mark.parametrize("adaptive_logp", [False, True])
def test_the_crd_implicit_reward_normalizes_each_component_separately(
    rollout: SimpleNamespace, adaptive_logp: bool
) -> None:
    trainer = _trainer(CRDTrainer, rollout, adaptive_logp=adaptive_logp)
    noised, _ = _noised(rollout)
    velocity = _velocity(rollout, 1.4)
    old_velocity = _velocity(rollout, 0.7)

    reward = trainer._implicit_reward(velocity, old_velocity, noised)

    target = noised.target_velocity
    current = {
        name: (velocity.components[name] - target.components[name]) ** 2 for name in COMPONENTS
    }
    old = {
        name: (old_velocity.components[name] - target.components[name]) ** 2 for name in COMPONENTS
    }
    if adaptive_logp:
        current_weights = _component_means(
            rollout,
            {
                name: torch.abs(
                    velocity.components[name].double() - target.components[name].double()
                )
                for name in COMPONENTS
            },
        )
        old_weights = _component_means(
            rollout,
            {
                name: torch.abs(
                    old_velocity.components[name].double() - target.components[name].double()
                )
                for name in COMPONENTS
            },
        )
        values = {
            name: -(
                current[name] / current_weights[name].clip(min=1e-5).reshape(BATCH_SIZE, 1, 1)
                - old[name] / old_weights[name].clip(min=1e-5).reshape(BATCH_SIZE, 1, 1)
            )
            for name in COMPONENTS
        }
    else:
        values = {name: -(current[name] - old[name]) for name in COMPONENTS}
    assert torch.allclose(reward, _joint_mean(rollout, values), atol=1e-6)


def test_the_opd_denominators_follow_each_components_own_scheduler(
    rollout: SimpleNamespace,
) -> None:
    trainer = _trainer(DiffusionOPDTrainer, rollout)

    denominators = trainer._component_kl_denominators(rollout.output, rollout.replay, REPLAY_STEP)

    assert tuple(denominators) == COMPONENTS
    for name in COMPONENTS:
        expected = _statistic(rollout.output.std_dev_t, name) ** 2 * -_statistic(
            rollout.output.dt, name
        )
        assert torch.allclose(denominators[name], expected)
    assert not torch.allclose(denominators["video"], denominators["audio"])


@pytest.mark.parametrize("loss_target", ["xt", "v", "x0"])
def test_the_opd_projection_maps_every_component_into_the_target_space(
    rollout: SimpleNamespace, loss_target: str
) -> None:
    projected = project_distillation_target_state(
        rollout.adapter,
        loss_target=loss_target,
        state=rollout.replay.state,
        output=rollout.output,
        times=rollout.replay.times,
    )

    assert projected.component_names == COMPONENTS
    for name in COMPONENTS:
        state = rollout.replay.state.components[name]
        if loss_target == "xt":
            expected = rollout.output.next_state_mean.components[name]
        elif loss_target == "v":
            expected = rollout.output.velocity.components[name]
        else:
            sigma = rollout.replay.times.sigma[name].reshape(BATCH_SIZE, 1, 1)
            expected = (
                state.float() - sigma.float() * rollout.output.velocity.components[name].float()
            )
        assert torch.allclose(projected.components[name], expected)


def test_the_opd_distillation_loss_divides_each_component_by_its_own_denominator(
    rollout: SimpleNamespace,
) -> None:
    trainer = _trainer(DiffusionOPDTrainer, rollout)
    denominators = trainer._component_kl_denominators(rollout.output, rollout.replay, REPLAY_STEP)
    student = rollout.output.next_state_mean
    teacher = _shift(student, 0.9)

    loss = compute_structured_distillation_loss(
        rollout.adapter,
        student_target=student,
        teacher_target=teacher,
        state=rollout.replay.state,
        self_normalize=False,
        denominators=denominators,
    )

    errors = {
        name: (student.components[name].float() - teacher.components[name].float()) ** 2
        / denominators[name].reshape(BATCH_SIZE, 1, 1)
        for name in COMPONENTS
    }
    assert torch.allclose(loss, _joint_mean(rollout, errors))


def test_every_algorithm_reduction_ignores_the_i2av_conditioning_frame(
    i2av: SimpleNamespace,
) -> None:
    noised, times = _noised(i2av)
    poisoned = {name: torch.zeros_like(noised.state.components[name]) for name in COMPONENTS}
    poisoned["video"][:, :FRAME_SEQ_LEN] = 1e6
    velocity = LatentState(poisoned)

    reduced = i2av.adapter.reduce_component_latent_values(poisoned, state=noised.state)
    joint = i2av.adapter.reduce_latent_values(poisoned, state=noised.state)
    matching = _trainer(AWMTrainer, i2av, weighting="t", ghuber_power=0.25)._matching_log_prob(
        velocity, noised, times
    )

    assert torch.equal(reduced["video"], torch.zeros(BATCH_SIZE))
    assert torch.equal(joint, torch.zeros(BATCH_SIZE))
    assert bool(torch.isfinite(matching).all())
