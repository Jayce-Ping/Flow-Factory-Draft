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

import math

import pytest
import torch

from flow_factory.hparams import SchedulerArguments
from flow_factory.scheduler import MiniMaxH3SDEScheduler, load_scheduler


def test_scalar_implicit_begin_index_and_monotonic_step_lifecycle() -> None:
    scheduler = MiniMaxH3SDEScheduler(dynamics_type="ODE")
    scheduler.set_timesteps(4)
    scheduler.set_begin_index(1)

    scheduler.step(torch.ones(1, 2), scheduler.timesteps[1], torch.zeros(1, 2))
    assert scheduler.step_index == 2
    scheduler.step(torch.ones(1, 2), scheduler.timesteps[2], torch.zeros(1, 2))
    assert scheduler.step_index == 3


def test_batched_implicit_timesteps_gather_independent_indices_without_global_step() -> None:
    scheduler = MiniMaxH3SDEScheduler(dynamics_type="ODE")
    scheduler.set_timesteps(sigmas=[1.0, 0.6, 0.2, 0.0])
    latents = torch.tensor([[1.0], [2.0]])
    velocity = torch.tensor([[0.5], [-0.25]])

    output = scheduler.step(
        velocity,
        scheduler.timesteps[torch.tensor([0, 2])],
        latents,
    )

    sigma = torch.tensor([1.0, 0.2]).unsqueeze(1)
    sigma_next = torch.tensor([0.6, 0.0]).unsqueeze(1)
    expected = (sigma_next / sigma) * latents + (1 - sigma_next / sigma) * (
        latents + sigma * velocity
    )
    assert torch.equal(output.next_latents, expected)
    assert scheduler.step_index is None


def test_step_rejects_conflicting_dynamics_override() -> None:
    scheduler = MiniMaxH3SDEScheduler(dynamics_type="Flow-SDE")
    scheduler.set_timesteps(3)
    with pytest.raises(ValueError, match=r"configured dynamics_type.*Flow-SDE.*ODE"):
        scheduler.step(
            torch.ones(1, 2),
            scheduler.timesteps[0],
            torch.zeros(1, 2),
            dynamics_type="ODE",
        )


@pytest.mark.parametrize("dynamics_type", ["Flow-SDE", "Dance-SDE", "CPS"])
def test_zero_variance_log_prob_fails_with_context(dynamics_type: str) -> None:
    scheduler = MiniMaxH3SDEScheduler(
        dynamics_type=dynamics_type,
        noise_level=0.0,
    )
    scheduler.set_timesteps(3)
    with pytest.raises(
        ValueError,
        match=rf"zero transition variance.*{dynamics_type}.*noise_level.*0",
    ):
        scheduler.step(
            torch.ones(1, 2),
            scheduler.timesteps[0],
            torch.zeros(1, 2),
            compute_log_prob=True,
        )


def test_constructor_and_return_fields_fail_fast_on_unknown_values() -> None:
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'unknown'"):
        MiniMaxH3SDEScheduler(unknown=1)
    scheduler = MiniMaxH3SDEScheduler(dynamics_type="ODE")
    scheduler.set_timesteps(2)
    with pytest.raises(ValueError, match=r"unknown return fields.*mystery"):
        scheduler.step(
            torch.ones(1, 2),
            scheduler.timesteps[0],
            torch.zeros(1, 2),
            return_kwargs=("next_latents", "mystery"),
        )


def test_loader_accepts_shift_and_all_scheduler_arguments() -> None:
    upstream = type("MiniMaxH3Scheduler", (), {})()
    upstream.config = {"shift": 12.0}
    scheduler = load_scheduler(
        upstream,
        SchedulerArguments(
            dynamics_type="Flow-SDE",
            noise_level=0.5,
            num_sde_steps=1,
            sde_steps=[0],
            seed=17,
        ),
    )
    assert scheduler.shift == 12.0
    assert scheduler.seed == 17


@pytest.mark.parametrize("shift", [float("nan"), float("inf"), 0.0])
def test_shift_requires_a_finite_positive_value(shift: float) -> None:
    with pytest.raises(ValueError, match=r"positive finite shift"):
        MiniMaxH3SDEScheduler(shift=shift)


def test_step_rejects_empty_batches() -> None:
    scheduler = MiniMaxH3SDEScheduler(dynamics_type="ODE")
    scheduler.set_timesteps(2)
    with pytest.raises(ValueError, match=r"non-empty batch"):
        scheduler.step(
            torch.empty(0, 2),
            scheduler.timesteps[0],
            torch.empty(0, 2),
        )


@pytest.mark.parametrize("dynamics_type", ["Flow-SDE", "Dance-SDE", "CPS"])
def test_sde_formula_oracle_is_independent_and_complete(dynamics_type: str) -> None:
    noise_level = 0.35
    scheduler = MiniMaxH3SDEScheduler(
        dynamics_type=dynamics_type,
        noise_level=noise_level,
    )
    scheduler.set_timesteps(sigmas=[1.0, 0.6, 0.2, 0.0])
    latents = torch.tensor([[0.25, -0.75]])
    velocity_h3 = torch.tensor([[1.5, -0.5]])
    sigma = torch.tensor(0.6)
    sigma_next = torch.tensor(0.2)
    dt = sigma_next - sigma
    generator = torch.Generator().manual_seed(23)

    output = scheduler.step(
        velocity_h3,
        sigma * 1000,
        latents,
        sigma=sigma,
        sigma_next=sigma_next,
        generator=generator,
    )

    standard_velocity = -velocity_h3
    x0 = latents + (1 - (1 - sigma)) * velocity_h3
    if dynamics_type == "Flow-SDE":
        std = torch.sqrt(sigma / (1 - sigma)) * noise_level
        mean = latents * (1 + std**2 / (2 * sigma) * dt) + standard_velocity * (
            1 + std**2 * (1 - sigma) / (2 * sigma)
        ) * dt
        scale = std * torch.sqrt(-dt)
        denominator = std**2 * (-dt)
    elif dynamics_type == "Dance-SDE":
        std = torch.tensor(noise_level)
        log_term = 0.5 * std**2 * (latents - x0 * (1 - sigma)) / sigma**2
        mean = latents + (standard_velocity + log_term) * dt
        scale = std * torch.sqrt(-dt)
        denominator = std**2 * (-dt)
    else:
        std = sigma_next * torch.sin(torch.tensor(noise_level) * torch.pi / 2)
        noise_endpoint = latents + standard_velocity * (1 - sigma)
        mean = x0 * (1 - sigma_next) + noise_endpoint * torch.sqrt(sigma_next**2 - std**2)
        scale = std
        denominator = std**2
    variance_noise = torch.randn(latents.shape, generator=torch.Generator().manual_seed(23))
    sampled = (mean + scale * variance_noise).to(latents.dtype).float()
    if dynamics_type == "CPS":
        expected_log_prob = -((sampled - mean) ** 2).mean(dim=1)
    else:
        expected_log_prob = (
            -((sampled - mean) ** 2) / (2 * scale**2)
            - torch.log(scale)
            - math.log(math.sqrt(2 * math.pi))
        ).mean(dim=1)

    assert torch.allclose(output.next_latents_mean, mean)
    assert torch.allclose(output.std_dev_t, std.reshape(1, 1))
    assert torch.allclose(output.dt, dt.reshape(1, 1))
    assert torch.equal(output.next_latents, sampled)
    assert torch.allclose(output.log_prob, expected_log_prob)
    assert output.velocity is velocity_h3
    assert torch.allclose(
        scheduler.get_kl_divergence_denominator(output.std_dev_t, output.dt),
        denominator.reshape(1, 1),
    )
    replay = scheduler.step(
        velocity_h3,
        sigma * 1000,
        latents,
        next_latents=sampled,
        sigma=sigma,
        sigma_next=sigma_next,
    )
    assert torch.equal(replay.next_latents, sampled)
    assert torch.equal(replay.log_prob, output.log_prob)
