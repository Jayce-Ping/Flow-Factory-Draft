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

from flow_factory.utils.noise_schedule import TIMESTEP_MAX, TimeSampler, flow_match_sigma


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


@pytest.mark.parametrize(
    "sampler_name",
    ["independent_logit_normal_shifted", "independent_uniform"],
)
def test_independent_time_samplers_materialize_distinct_batch_draws(
    sampler_name: str,
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(123)
    sampler = getattr(TimeSampler, sampler_name)

    timesteps = sampler(
        batch_size=4,
        num_timesteps=3,
        timestep_range=(0.2, 0.8),
        generator=generator,
    )

    assert timesteps.shape == (3, 4)
    assert timesteps.stride() != (1, 0)
    assert torch.all(timesteps >= 200.0)
    assert torch.all(timesteps <= 800.0)
    assert all(torch.unique(row).numel() > 1 for row in timesteps)


@pytest.mark.parametrize(
    "sampler_name",
    ["independent_logit_normal_shifted", "independent_uniform"],
)
def test_independent_time_samplers_are_generator_reproducible(sampler_name: str) -> None:
    sampler = getattr(TimeSampler, sampler_name)
    first_generator = torch.Generator(device="cpu").manual_seed(91)
    second_generator = torch.Generator(device="cpu").manual_seed(91)

    first = sampler(2, 4, 0.99, generator=first_generator)
    second = sampler(2, 4, 0.99, generator=second_generator)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("sampler_name", ["logit_normal_shifted", "uniform"])
def test_legacy_time_samplers_keep_shared_batch_coordinates(sampler_name: str) -> None:
    sampler = getattr(TimeSampler, sampler_name)
    generator = torch.Generator(device="cpu").manual_seed(321)

    timesteps = sampler(4, 3, (0.2, 0.8), generator=generator)

    assert timesteps.shape == (3, 4)
    for row in timesteps:
        torch.testing.assert_close(row, row[0].expand_as(row), rtol=0, atol=0)


@pytest.mark.parametrize("field_name", ["batch_size", "num_timesteps"])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.0])
def test_independent_time_samplers_validate_materialized_shape(
    field_name: str,
    invalid: object,
) -> None:
    kwargs = {"batch_size": 2, "num_timesteps": 3, field_name: invalid}
    expected_error = TypeError if type(invalid) is not int else ValueError

    with pytest.raises(expected_error):
        TimeSampler.independent_uniform(timestep_range=0.99, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flow_match_sigma_matches_cpu_at_upper_interior_on_cuda() -> None:
    endpoint = torch.tensor([TIMESTEP_MAX], dtype=torch.float32)
    interior = torch.nextafter(endpoint, torch.zeros_like(endpoint))

    cpu_sigma = flow_match_sigma(interior)
    cuda_sigma = flow_match_sigma(interior.cuda()).cpu()

    assert cuda_sigma.item() < 1.0
    torch.testing.assert_close(cuda_sigma, cpu_sigma, rtol=0, atol=0)
