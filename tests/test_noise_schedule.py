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

from flow_factory.utils.noise_schedule import (
    TIMESTEP_MAX,
    flow_match_coordinates_close,
    flow_match_sigma,
)


def test_flow_match_sigma_preserves_dtype_and_exact_endpoints() -> None:
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        timesteps = torch.tensor([0.0, TIMESTEP_MAX], dtype=dtype)
        sigmas = flow_match_sigma(timesteps)

        assert sigmas.dtype == dtype
        torch.testing.assert_close(
            sigmas,
            torch.tensor([0.0, 1.0], dtype=dtype),
            rtol=0,
            atol=0,
        )


def test_flow_match_sigma_keeps_representable_upper_interior_strict() -> None:
    endpoint = torch.tensor([TIMESTEP_MAX], dtype=torch.float32)
    interior = torch.nextafter(endpoint, torch.zeros_like(endpoint))

    sigma = flow_match_sigma(interior)

    assert interior.item() < TIMESTEP_MAX
    assert sigma.item() < 1.0


def test_flow_match_coordinates_accept_one_ulp_but_reject_two() -> None:
    timestep = torch.tensor([990.4219970703125], dtype=torch.float32)
    expected = flow_match_sigma(timestep)
    direction = torch.full_like(expected, float("inf"))
    one_ulp = torch.nextafter(expected, direction)
    two_ulps = torch.nextafter(one_ulp, direction)

    assert flow_match_coordinates_close(timestep, one_ulp)
    assert not flow_match_coordinates_close(timestep, two_ulps)

    binade_timestep = torch.tensor([500.0], dtype=torch.float32)
    binade_sigma = torch.tensor([0.5], dtype=torch.float32)
    lower = torch.full_like(binade_sigma, -float("inf"))
    one_ulp_down = torch.nextafter(binade_sigma, lower)
    two_ulps_down = torch.nextafter(one_ulp_down, lower)
    assert flow_match_coordinates_close(binade_timestep, one_ulp_down)
    assert not flow_match_coordinates_close(binade_timestep, two_ulps_down)


def test_flow_match_coordinates_reject_clamped_out_of_range_values() -> None:
    endpoint = torch.tensor([TIMESTEP_MAX], dtype=torch.float32)
    outside = torch.nextafter(endpoint, torch.full_like(endpoint, float("inf")))

    assert not flow_match_coordinates_close(outside, torch.ones_like(outside))


@pytest.mark.parametrize(
    "timesteps",
    [
        torch.tensor([-1.0]),
        torch.tensor([TIMESTEP_MAX + 1]),
        torch.tensor([float("nan")]),
    ],
)
def test_flow_match_sigma_rejects_invalid_scheduler_coordinates(
    timesteps: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match=r"t_scheduler.*\[0, 1000\]"):
        flow_match_sigma(timesteps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flow_match_sigma_matches_cpu_at_upper_interior_on_cuda() -> None:
    endpoint = torch.tensor([TIMESTEP_MAX], dtype=torch.float32)
    interior = torch.nextafter(endpoint, torch.zeros_like(endpoint))

    cpu_sigma = flow_match_sigma(interior)
    cuda_sigma = flow_match_sigma(interior.cuda()).cpu()

    assert cuda_sigma.item() < 1.0
    torch.testing.assert_close(cuda_sigma, cpu_sigma, rtol=0, atol=0)
