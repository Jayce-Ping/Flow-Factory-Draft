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

from flow_factory.models.component_reduction import reduce_component_log_probs


def test_shared_reduction_reproduces_the_two_modality_arithmetic_bit_for_bit() -> None:
    """LTX2 keeps its exact values after delegating to the shared reducer."""
    torch.manual_seed(0)
    video = torch.randn(4, dtype=torch.float32)
    audio = torch.randn(4, dtype=torch.float32)
    n_video, n_audio = 3072, 512

    expected = (video * n_video + audio * n_audio) / (n_video + n_audio)
    reduced = reduce_component_log_probs(
        {"video": video, "audio": audio},
        {"video": n_video, "audio": n_audio},
    )

    assert torch.equal(reduced, expected)


def test_reduction_weights_by_degrees_of_freedom_not_by_component_count() -> None:
    """A component with more stochastic elements pulls the joint value further."""
    dominant = reduce_component_log_probs(
        {"video": torch.tensor([1.0]), "audio": torch.tensor([0.0])},
        {"video": 9, "audio": 1},
    )

    assert torch.allclose(dominant, torch.tensor([0.9]))


def test_reduction_supports_stored_per_transition_log_probs() -> None:
    """Reduction is elementwise, so a stored ``(B, T)`` trajectory works unchanged."""
    video = torch.zeros(2, 3)
    audio = torch.ones(2, 3)

    reduced = reduce_component_log_probs(
        {"video": video, "audio": audio}, {"video": 1, "audio": 1}
    )

    assert reduced.shape == (2, 3)
    assert torch.allclose(reduced, torch.full((2, 3), 0.5))


def test_reduction_rejects_component_order_disagreement() -> None:
    """The two mappings are one contract; a silent reorder would misweight."""
    with pytest.raises(ValueError, match="component_dofs component order"):
        reduce_component_log_probs(
            {"video": torch.zeros(1), "audio": torch.zeros(1)},
            {"audio": 1, "video": 1},
        )


def test_reduction_rejects_mismatched_component_shapes() -> None:
    """Broadcasting two disagreeing shapes would silently invent a joint value."""
    with pytest.raises(ValueError, match="matching shape/dtype/device"):
        reduce_component_log_probs(
            {"video": torch.zeros(2), "audio": torch.zeros(3)},
            {"video": 1, "audio": 1},
        )


def test_reduction_rejects_non_positive_degrees_of_freedom() -> None:
    """A zero or negative element count cannot weight a mean."""
    with pytest.raises(ValueError, match="positive int degrees of freedom"):
        reduce_component_log_probs({"video": torch.zeros(1)}, {"video": 0})
