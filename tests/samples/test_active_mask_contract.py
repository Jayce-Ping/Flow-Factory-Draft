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

from typing import List, Optional

import pytest
import torch

from flow_factory.samples import ComponentTrajectory, LatentState, StructuredTrajectory


def _component(active_mask: Optional[torch.Tensor] = None) -> ComponentTrajectory:
    """Per-sample video trajectory with two tokens of one channel each."""
    return ComponentTrajectory(
        states=torch.arange(12, dtype=torch.float32).reshape(2, 3, 2),
        timesteps=torch.tensor([1000.0, 0.0]),
        state_index_map=torch.tensor([0, 1]),
        active_mask=active_mask,
    )


def _audio_component(active_mask: Optional[torch.Tensor] = None) -> ComponentTrajectory:
    return ComponentTrajectory(
        states=torch.arange(4, dtype=torch.float32).reshape(2, 1, 2),
        timesteps=torch.tensor([1000.0, 0.0]),
        state_index_map=torch.tensor([0, 1]),
        active_mask=active_mask,
    )


def _trajectory(masked: bool, *, audio_masked: Optional[bool] = None) -> StructuredTrajectory:
    audio_masked = masked if audio_masked is None else audio_masked
    return StructuredTrajectory(
        components={
            "video": _component(
                torch.tensor([[False], [True], [True]]) if masked else None,
            ),
            "audio": _audio_component(torch.ones(1, 1, dtype=torch.bool) if audio_masked else None),
        }
    )


def test_unbatched_component_mask_broadcasts_to_one_stored_state() -> None:
    component = _component(torch.tensor([[False], [True], [True]]))

    assert component.active_mask.shape == (3, 1)
    assert component.states[0].shape == (3, 2)


def test_batched_component_mask_broadcasts_to_one_batched_state() -> None:
    component = ComponentTrajectory(
        states=torch.zeros(2, 3, 4, 2),
        timesteps=torch.zeros(2, 3),
        state_index_map=torch.tensor([0, 1, 2]),
        active_mask=torch.ones(2, 4, 1, dtype=torch.bool),
    )

    assert component.active_mask.shape == (2, 4, 1)


def test_component_mask_rejects_a_non_boolean_dtype() -> None:
    with pytest.raises(TypeError, match=r"ComponentTrajectory.active_mask.*bool.*torch.float32"):
        _component(torch.ones(3, 1))


def test_component_mask_rejects_a_shape_that_does_not_broadcast_to_one_state() -> None:
    with pytest.raises(
        ValueError,
        match=r"ComponentTrajectory.active_mask.*\(3, 2\).*received.*\(4, 1\)",
    ):
        _component(torch.ones(4, 1, dtype=torch.bool))


def test_batched_component_mask_rejects_a_missing_batch_axis() -> None:
    with pytest.raises(
        ValueError,
        match=r"ComponentTrajectory.active_mask.*\(2, 4, 2\).*received.*\(4, 1\)",
    ):
        ComponentTrajectory(
            states=torch.zeros(2, 3, 4, 2),
            timesteps=torch.zeros(2, 3),
            state_index_map=torch.tensor([0, 1, 2]),
            active_mask=torch.ones(4, 1, dtype=torch.bool),
        )


def test_component_device_traversal_visits_the_mask() -> None:
    component = _component(torch.tensor([[False], [True], [True]]))
    visited: List[torch.Tensor] = []

    component.map_tensors(lambda tensor: visited.append(tensor) or tensor)

    assert any(tensor.dtype is torch.bool for tensor in visited)


def test_structured_stack_stacks_component_masks_along_the_batch() -> None:
    stacked = StructuredTrajectory.stack([_trajectory(True), _trajectory(True)])

    assert stacked.components["video"].active_mask.shape == (2, 3, 1)
    assert stacked.components["audio"].active_mask.shape == (2, 1, 1)


def test_structured_stack_keeps_maskless_components_maskless() -> None:
    stacked = StructuredTrajectory.stack([_trajectory(False), _trajectory(False)])

    assert stacked.components["video"].active_mask is None
    assert stacked.components["audio"].active_mask is None


def test_structured_stack_rejects_mixed_mask_presence_across_samples() -> None:
    with pytest.raises(
        ValueError,
        match=r"identical active_mask presence for component 'video'.*\[True, False\]",
    ):
        StructuredTrajectory.stack([_trajectory(True), _trajectory(False)])


def test_latent_state_masks_require_every_component_in_order() -> None:
    with pytest.raises(
        ValueError,
        match=r"LatentState.active_masks component order \('video', 'audio'\).*\('video',\)",
    ):
        LatentState(
            {"video": torch.zeros(2, 3, 2), "audio": torch.zeros(2, 1, 2)},
            active_masks={"video": torch.ones(2, 3, 1, dtype=torch.bool)},
        )


def test_latent_state_masks_require_the_component_batch_size() -> None:
    with pytest.raises(
        ValueError,
        match=r"LatentState.active_masks\['video'\].*batch size 2.*\(3, 3, 1\)",
    ):
        LatentState(
            {"video": torch.zeros(2, 3, 2)},
            active_masks={"video": torch.ones(3, 3, 1, dtype=torch.bool)},
        )


def test_latent_state_masks_require_a_boolean_dtype() -> None:
    with pytest.raises(TypeError, match=r"LatentState.active_masks\['video'\].*bool.*int64"):
        LatentState(
            {"video": torch.zeros(2, 3, 2)},
            active_masks={"video": torch.ones(2, 3, 1, dtype=torch.int64)},
        )


def test_latent_state_masks_must_broadcast_to_their_component() -> None:
    with pytest.raises(
        ValueError,
        match=r"LatentState.active_masks\['video'\].*\(2, 3, 2\).*received.*\(2, 5, 1\)",
    ):
        LatentState(
            {"video": torch.zeros(2, 3, 2)},
            active_masks={"video": torch.ones(2, 5, 1, dtype=torch.bool)},
        )


def test_latent_state_without_masks_stays_maskless() -> None:
    state = LatentState({"video": torch.zeros(2, 3, 2)})

    assert state.active_masks is None
