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

"""Reduce per-component transition statistics to one joint policy quantity."""

from typing import Mapping, Optional

import torch


def reduce_component_log_probs(
    component_log_probs: Mapping[str, torch.Tensor],
    component_dofs: Mapping[str, int],
) -> torch.Tensor:
    """Combine per-component log probabilities by stochastic degrees of freedom.

    A multi-component model steps each component with its own scheduler, and each
    returns a per-sample log probability already meaned over that component's own
    dimensions. Weighting by per-sample element counts reproduces the mean a single
    scheduler over the concatenated components would produce, which keeps the joint
    log probability on the same scale as a single-component policy.

    Callers own the rank contract: this reduces elementwise over whatever shape the
    components agree on, so both a per-step ``(B,)`` and a stored ``(B, T)`` work.

    Args:
        component_log_probs: Log probabilities of one identical shape by component.
        component_dofs: Per-sample scalar element count by component.

    Returns:
        Joint log probability shaped like each component input.

    Raises:
        TypeError: If any log probability is not a tensor.
        ValueError: If the mappings disagree, a count is not a positive int, or the
            log probabilities do not share shape, dtype, and device.
    """
    expected = tuple(component_log_probs)
    if tuple(component_dofs) != expected:
        raise ValueError(
            f"expected component_dofs component order {expected}, received {tuple(component_dofs)}"
        )
    if not expected:
        raise ValueError("expected at least one component log probability, received none")
    reference: Optional[torch.Tensor] = None
    for name in expected:
        values = component_log_probs[name]
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                f"expected torch.Tensor component log probability for {name!r}, "
                f"received {type(values).__name__}: {values!r}"
            )
        if reference is None:
            reference = values
        elif (
            values.shape != reference.shape
            or values.dtype != reference.dtype
            or values.device != reference.device
        ):
            raise ValueError(
                f"expected matching shape/dtype/device across component log probabilities; "
                f"{name!r} has {tuple(values.shape)}/{values.dtype}/{values.device} against "
                f"{tuple(reference.shape)}/{reference.dtype}/{reference.device}"
            )
        dof = component_dofs[name]
        if not isinstance(dof, int) or isinstance(dof, bool) or dof <= 0:
            raise ValueError(
                f"expected positive int degrees of freedom for {name!r}, received {dof!r}"
            )

    total = sum(component_dofs[name] for name in expected)
    weighted = sum(component_log_probs[name] * component_dofs[name] for name in expected)
    return weighted / total
