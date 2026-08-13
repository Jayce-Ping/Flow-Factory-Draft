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

import torch



def _expand_active_mask(
    mask: torch.Tensor,
    values: torch.Tensor,
    *,
    component: str,
    operation: str,
) -> torch.Tensor:
    """Broadcast one static mask onto the values it selects."""
    if mask.ndim != values.ndim or any(
        mask_dim not in (1, value_dim) for mask_dim, value_dim in zip(mask.shape, values.shape)
    ):
        raise ValueError(
            f"expected the {operation} active mask for component {component!r} to broadcast onto "
            f"the value shape {tuple(values.shape)}, received mask shape {tuple(mask.shape)}"
        )
    return mask.expand(values.shape)


def _active_element_counts(
    expanded_mask: torch.Tensor,
    *,
    component: str,
) -> torch.Tensor:
    """Return one positive active element count per sample."""
    counts = expanded_mask.reshape(expanded_mask.shape[0], -1).sum(dim=1)
    minimum = int(counts.min().item())
    if minimum <= 0:
        raise ValueError(
            f"expected component {component!r} active_mask to leave a positive active element "
            f"count per sample, received {minimum} for at least one sample with counts "
            f"{counts.tolist()}"
        )
    return counts
