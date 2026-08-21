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

from flow_factory.utils.noise_schedule import TIMESTEP_MAX, flow_match_sigma


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flow_match_sigma_matches_cpu_at_upper_interior_on_cuda() -> None:
    endpoint = torch.tensor([TIMESTEP_MAX], dtype=torch.float32)
    interior = torch.nextafter(endpoint, torch.zeros_like(endpoint))

    cpu_sigma = flow_match_sigma(interior)
    cuda_sigma = flow_match_sigma(interior.cuda()).cpu()

    assert cuda_sigma.item() < 1.0
    torch.testing.assert_close(cuda_sigma, cpu_sigma, rtol=0, atol=0)
