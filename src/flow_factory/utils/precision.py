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

"""Floating-point comparison primitives that preserve each tensor's native dtype."""

import torch


def native_ulp_spacing(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Return each value's native ULP spacing in the direction of ``reference``.

    The direction is chosen in float64 before calling ``nextafter`` toward an
    infinity in ``values.dtype``. This avoids a zero spacing when casting the
    reference to a lower-precision dtype would round it back to ``values``.

    Args:
        values: Finite floating values whose native spacing is requested.
        reference: Finite floating values that choose the direction.

    Returns:
        Absolute adjacent-value spacing represented in float64.

    Raises:
        TypeError: If either input is not a floating tensor.
        ValueError: If the inputs have different shapes or devices.
    """
    _validate_floating_pair(values, reference)
    return _native_ulp_spacing_unchecked(values, reference)


def within_one_native_ulp(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether finite tensor pairs differ by at most one native ULP.

    Each side contributes spacing from its own dtype, and the larger directional
    spacing is used. This makes mixed-precision comparisons symmetric while still
    rejecting a semantic difference that exceeds both stored representations.

    Args:
        left: First floating tensor.
        right: Second floating tensor with the same shape and device.

    Returns:
        ``True`` when every pair is finite and within one native ULP.

    Raises:
        TypeError: If either input is not a floating tensor.
        ValueError: If the inputs have different shapes or devices.
    """
    _validate_floating_pair(left, right)
    if not bool((torch.isfinite(left) & torch.isfinite(right)).all()):
        return False
    left_float64 = left.to(torch.float64)
    right_float64 = right.to(torch.float64)
    difference = (left_float64 - right_float64).abs()
    tolerance = torch.maximum(
        _native_ulp_spacing_unchecked(left, right),
        _native_ulp_spacing_unchecked(right, left),
    )
    return bool((difference <= tolerance).all())


def _native_ulp_spacing_unchecked(
    values: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    values_float64 = values.to(torch.float64)
    reference_float64 = reference.to(torch.float64)
    toward_higher = reference_float64 >= values_float64
    direction = torch.where(
        toward_higher,
        torch.full_like(values, float("inf")),
        torch.full_like(values, -float("inf")),
    )
    adjacent_native = torch.nextafter(values, direction)
    opposite_direction = torch.where(
        toward_higher,
        torch.full_like(values, -float("inf")),
        torch.full_like(values, float("inf")),
    )
    adjacent_native = torch.where(
        torch.isfinite(adjacent_native),
        adjacent_native,
        torch.nextafter(values, opposite_direction),
    )
    adjacent = adjacent_native.to(torch.float64)
    return (adjacent - values_float64).abs()


def _validate_floating_pair(left: torch.Tensor, right: torch.Tensor) -> None:
    for field, values in (("left", left), ("right", right)):
        if not isinstance(values, torch.Tensor) or not values.is_floating_point():
            raise TypeError(
                f"expected floating torch.Tensor {field}, received "
                f"{type(values).__name__}/{getattr(values, 'dtype', None)}"
            )
    if left.shape != right.shape:
        raise ValueError(
            "expected floating tensor shapes to match, received "
            f"left={tuple(left.shape)} and right={tuple(right.shape)}"
        )
    if left.device != right.device:
        raise ValueError(
            "expected floating tensor devices to match, received "
            f"left={left.device} and right={right.device}"
        )
