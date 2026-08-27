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

from flow_factory.utils.precision import native_ulp_spacing, within_one_native_ulp


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
def test_native_ulp_comparison_accepts_one_step_and_rejects_two(dtype: torch.dtype) -> None:
    center = torch.tensor([0.0, 0.5, 1.0], dtype=dtype)
    higher = torch.full_like(center, float("inf"))
    one_ulp = torch.nextafter(center, higher)
    two_ulps = torch.nextafter(one_ulp, higher)

    assert within_one_native_ulp(center, one_ulp)
    assert not within_one_native_ulp(center, two_ulps)


def test_native_ulp_comparison_uses_directional_spacing_at_a_binade_boundary() -> None:
    boundary = torch.tensor([0.5], dtype=torch.float32)
    lower = torch.full_like(boundary, -float("inf"))
    one_ulp_down = torch.nextafter(boundary, lower)
    two_ulps_down = torch.nextafter(one_ulp_down, lower)

    assert within_one_native_ulp(boundary, one_ulp_down)
    assert not within_one_native_ulp(boundary, two_ulps_down)


def test_native_ulp_comparison_uses_the_larger_mixed_dtype_spacing() -> None:
    low_precision = torch.tensor([1.0], dtype=torch.float16)
    high_precision = torch.nextafter(
        low_precision,
        torch.full_like(low_precision, float("inf")),
    ).to(torch.float64)

    assert within_one_native_ulp(low_precision, high_precision)


def test_native_ulp_spacing_does_not_collapse_when_reference_casts_to_source() -> None:
    source = torch.tensor([1.0], dtype=torch.float16)
    reference = torch.tensor([1.0001], dtype=torch.float64)
    assert reference.to(source).item() == source.item()

    spacing = native_ulp_spacing(source, reference)

    expected = torch.nextafter(source, torch.full_like(source, float("inf"))).to(torch.float64)
    torch.testing.assert_close(spacing, expected - source.to(torch.float64), rtol=0, atol=0)
    assert spacing.item() > 0


def test_native_ulp_comparison_rejects_nonfinite_values() -> None:
    assert not within_one_native_ulp(torch.tensor([float("nan")]), torch.tensor([float("nan")]))
    assert not within_one_native_ulp(torch.tensor([float("inf")]), torch.tensor([float("inf")]))


def test_native_ulp_comparison_rejects_large_gap_above_a_dtype_maximum() -> None:
    dtype_maximum = torch.tensor([torch.finfo(torch.float16).max], dtype=torch.float16)
    much_larger = torch.tensor([1e10], dtype=torch.float32)

    assert not within_one_native_ulp(dtype_maximum, much_larger)
