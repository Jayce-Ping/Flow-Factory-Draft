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

"""Verify ODE rollout and differentiable replay share one precision boundary."""

from __future__ import annotations

from typing import Type

import pytest
import torch

from flow_factory.scheduler.flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSDEScheduler,
)
from flow_factory.scheduler.unipc_multistep import UniPCMultistepSDEScheduler


@pytest.mark.parametrize(
    "scheduler_type",
    [FlowMatchEulerDiscreteSDEScheduler, UniPCMultistepSDEScheduler],
)
@pytest.mark.parametrize("storage_dtype", [torch.float16, torch.bfloat16])
def test_ode_mean_round_trips_through_input_dtype_with_gradient(
    scheduler_type: Type,
    storage_dtype: torch.dtype,
) -> None:
    """The replayed mean must be the exact point rollout stores and reuses.

    Scheduler arithmetic deliberately runs in fp32, but inference persists the
    resulting boundary in ``latent_storage_dtype``. Returning the unquantized mean
    during training makes replay supervise a neighboring point once LoRA weights are
    non-zero. The cast must remain in the graph so the generator still receives a
    gradient through the one replayed transition.
    """
    scheduler = scheduler_type(dynamics_type="ODE")
    latents = torch.tensor([[1.25, -0.75]], dtype=storage_dtype)
    velocity = torch.tensor([[0.1234, -0.4321]], dtype=torch.float32, requires_grad=True)

    output = scheduler.step(
        velocity=velocity,
        timestep=torch.tensor(750.0),
        timestep_next=torch.tensor(250.0),
        latents=latents,
        compute_log_prob=False,
    )

    raw_fp32_mean = latents.float() - 0.5 * velocity
    expected = raw_fp32_mean.to(storage_dtype).float()
    assert output.next_latents_mean.dtype == torch.float32
    assert output.next_latents.dtype == torch.float32
    torch.testing.assert_close(output.next_latents_mean, expected, rtol=0, atol=0)
    torch.testing.assert_close(output.next_latents, expected, rtol=0, atol=0)
    assert output.next_latents_mean.requires_grad

    output.next_latents_mean.sum().backward()
    assert velocity.grad is not None
    assert torch.count_nonzero(velocity.grad) == velocity.numel()


@pytest.mark.parametrize(
    "scheduler_type",
    [FlowMatchEulerDiscreteSDEScheduler, UniPCMultistepSDEScheduler],
)
def test_ode_replay_uses_the_stored_boundary_but_recomputes_the_same_mean(
    scheduler_type: Type,
) -> None:
    """Supplying rollout's stored next state must not change the recomputed mean."""
    scheduler = scheduler_type(dynamics_type="ODE")
    latents = torch.tensor([[1.25, -0.75]], dtype=torch.float16)
    velocity = torch.tensor([[0.1234, -0.4321]], dtype=torch.float32, requires_grad=True)
    stored = (latents.float() - 0.5 * velocity.detach()).to(torch.float16)

    output = scheduler.step(
        velocity=velocity,
        timestep=torch.tensor(750.0),
        timestep_next=torch.tensor(250.0),
        latents=latents,
        next_latents=stored,
        compute_log_prob=False,
    )

    expected = stored.float()
    torch.testing.assert_close(output.next_latents_mean, expected, rtol=0, atol=0)
    torch.testing.assert_close(output.next_latents, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "scheduler_type",
    [FlowMatchEulerDiscreteSDEScheduler, UniPCMultistepSDEScheduler],
)
def test_sde_and_ode_step_selection_share_epoch_seeded_randomness(
    scheduler_type: Type,
) -> None:
    """The scheduler seed controls both random SDE subsets and ODE replay draws."""
    scheduler = scheduler_type(
        dynamics_type="ODE",
        seed=17,
        sde_steps=[0, 1, 2, 3],
        num_sde_steps=2,
    )
    scheduler.timesteps = torch.arange(4)

    sde_first = scheduler.current_sde_steps.clone()
    ode_first = [scheduler.sample_ode_step_index(draw) for draw in range(200)]
    scheduler.set_seed(18)
    sde_second = scheduler.current_sde_steps.clone()
    ode_second = [scheduler.sample_ode_step_index(draw) for draw in range(200)]
    scheduler.set_seed(17)

    torch.testing.assert_close(scheduler.current_sde_steps, sde_first, rtol=0, atol=0)
    assert [scheduler.sample_ode_step_index(draw) for draw in range(200)] == ode_first
    assert not torch.equal(sde_first, sde_second)
    assert ode_first != ode_second
    assert set(ode_first) == {0, 1, 2, 3}
