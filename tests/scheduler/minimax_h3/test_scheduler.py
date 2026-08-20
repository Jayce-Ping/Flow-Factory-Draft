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

import pytest
import torch

from flow_factory.scheduler import MiniMaxH3SDEScheduler
from flow_factory.scheduler.registry import get_sde_scheduler_class

ORACLE_COMMIT = "huggingface/diffusers@f53d552036a0d1bd5570782a39cd40cfabf112bc"


@pytest.mark.parametrize("shift", [12.0, 3.0])
def test_schedule_matches_upstream_transcription_with_n_plus_one_points(shift: float) -> None:
    """Freeze the sigma-grid formula from the upstream oracle commit."""
    scheduler = MiniMaxH3SDEScheduler(shift=shift)
    scheduler.set_timesteps(4)
    base = torch.linspace(1.0, 0.0, 5, dtype=torch.float32)
    expected = shift * base / (1 + (shift - 1) * base)

    assert torch.equal(scheduler.sigmas.cpu(), expected)
    assert torch.equal(scheduler.timesteps.cpu(), expected[:-1] * 1000)
    assert torch.equal(scheduler.model_timesteps.cpu(), 1 - expected[:-1])
    assert scheduler.sigmas[-1].item() == 0.0


def test_explicit_schedule_is_validated_and_used_without_shifting() -> None:
    scheduler = MiniMaxH3SDEScheduler(shift=12.0)
    scheduler.set_timesteps(sigmas=[1.0, 0.7, 0.2, 0.0])
    assert torch.equal(scheduler.sigmas, torch.tensor([1.0, 0.7, 0.2, 0.0]))

    with pytest.raises(ValueError, match=r"strictly decreasing.*ending.*0"):
        scheduler.set_timesteps(sigmas=[1.0, 0.7, 0.7, 0.0])
    with pytest.raises(ValueError, match=r"num_inference_steps.*positive"):
        scheduler.set_timesteps(0)


def test_dedup_transition_mismatch_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    original = torch.unique_consecutive

    def collapse(values: torch.Tensor) -> torch.Tensor:
        return original(values)[::2]

    monkeypatch.setattr(torch, "unique_consecutive", collapse)
    scheduler = MiniMaxH3SDEScheduler(shift=12.0)
    with pytest.raises(ValueError, match=r"expected 4 transitions.*deduplication"):
        scheduler.set_timesteps(4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ode_matches_upstream_data_ward_formula(dtype: torch.dtype) -> None:
    """Freeze the upstream x0 blend, including the 1-(1-sigma) round trip."""
    scheduler = MiniMaxH3SDEScheduler(shift=3.0, dynamics_type="ODE")
    scheduler.set_timesteps(sigmas=[1.0, 0.3, 0.0])
    sample = torch.tensor([[[0.25, -0.75]]], dtype=dtype)
    velocity = torch.tensor([[[1.5, -0.5]]], dtype=dtype)
    sigma = torch.tensor(0.3, dtype=torch.float32)
    sigma_next = torch.tensor(0.0, dtype=torch.float32)

    output = scheduler.step(
        velocity,
        scheduler.timesteps[1],
        sample,
        sigma=sigma,
        sigma_next=sigma_next,
    )
    model_t = 1 - sigma
    sigma_from_model_t = 1 - model_t
    x0 = sample + sigma_from_model_t.to(dtype) * velocity
    expected = (
        (sigma_next / sigma) * sample.float()
        + (1 - sigma_next / sigma) * x0.float()
    ).to(dtype)

    assert torch.equal(output.next_latents, expected)
    assert output.velocity is velocity
    assert torch.equal(output.log_prob, torch.zeros(1))


def test_flow_sde_rollout_replay_preserves_rng_and_storage_round_trip() -> None:
    scheduler = MiniMaxH3SDEScheduler(
        shift=3.0, dynamics_type="Flow-SDE", noise_level=0.6
    )
    scheduler.set_timesteps(3)
    sample = torch.tensor([[[0.25, -0.75]]], dtype=torch.bfloat16)
    velocity = torch.tensor([[[1.5, -0.5]]], dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(7)
    output = scheduler.step(
        velocity,
        scheduler.timesteps[0],
        sample,
        generator=generator,
    )
    post_draw = torch.randn(3, generator=generator)

    replay_generator = torch.Generator().manual_seed(7)
    replay = scheduler.step(
        velocity,
        scheduler.timesteps[0],
        sample,
        next_latents=output.next_latents,
        sigma=scheduler.sigmas[0],
        sigma_next=scheduler.sigmas[1],
        generator=replay_generator,
    )
    torch.randn(velocity.shape, generator=replay_generator)
    expected_post_draw = torch.randn(3, generator=replay_generator)

    assert torch.equal(output.next_latents, output.next_latents.to(torch.bfloat16).float())
    assert torch.equal(output.log_prob, replay.log_prob)
    assert torch.equal(post_draw, expected_post_draw)
    assert replay.velocity is velocity


@pytest.mark.parametrize("dynamics_type", ["Dance-SDE", "CPS"])
def test_other_dynamics_smoke_and_lifecycle(dynamics_type: str) -> None:
    scheduler = MiniMaxH3SDEScheduler(
        shift=3.0, dynamics_type=dynamics_type, noise_level=0.4, seed=5
    )
    scheduler.set_timesteps(3)
    scheduler.set_seed(11)
    assert scheduler.seed == 11
    scheduler.eval()
    assert scheduler.is_eval
    scheduler.train()
    assert not scheduler.is_eval
    assert scheduler.get_train_timesteps().ndim == 1

    output = scheduler.step(
        torch.ones(2, 3),
        scheduler.timesteps[0],
        torch.zeros(2, 3),
        generator=torch.Generator().manual_seed(1),
    )
    assert output.next_latents.shape == (2, 3)
    assert output.log_prob.shape == (2,)


def test_registry_resolves_unreleased_upstream_class_name() -> None:
    fake = type("MiniMaxH3Scheduler", (), {})
    assert get_sde_scheduler_class(fake) is MiniMaxH3SDEScheduler


def test_scale_noise_uses_h3_clean_time_semantics() -> None:
    scheduler = MiniMaxH3SDEScheduler()
    clean = torch.tensor([[2.0]])
    noise = torch.tensor([[6.0]])
    assert torch.equal(
        scheduler.scale_noise(clean, torch.tensor([0.25]), noise),
        torch.tensor([[5.0]]),
    )


def test_implicit_steps_track_the_resolved_schedule_index() -> None:
    scheduler = MiniMaxH3SDEScheduler(dynamics_type="ODE")
    scheduler.set_timesteps(3)
    assert scheduler.step_index is None

    scheduler.step(torch.ones(1, 2), scheduler.timesteps[1], torch.zeros(1, 2))

    assert scheduler.step_index == 2
