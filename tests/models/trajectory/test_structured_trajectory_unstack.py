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

from typing import Dict, Tuple

import pytest
import torch

from flow_factory.samples.trajectory import (
    StructuredTrajectory,
    unstack_structured_trajectories,
)

COMPONENT_ORDER: Tuple[str, ...] = ("video", "audio")


def _schedule() -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    timesteps = torch.tensor([3.0, 2.0, 1.0])
    sigmas = torch.tensor([0.9, 0.5, 0.1])
    return {name: (timesteps, sigmas) for name in COMPONENT_ORDER}


def _state_index_maps() -> Dict[str, torch.Tensor]:
    return {name: torch.tensor([0, 1, 2]) for name in COMPONENT_ORDER}


def test_unstack_is_the_inverse_of_stack() -> None:
    """Round-tripping a batch through stack and unstack preserves every tensor."""
    states = {
        "video": torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4),
        "audio": torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2),
    }
    log_probs = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    log_prob_index_map = torch.tensor([0, 1])

    trajectories = unstack_structured_trajectories(
        component_order=COMPONENT_ORDER,
        states=states,
        schedule=_schedule(),
        state_index_maps=_state_index_maps(),
        log_probs=log_probs,
        log_prob_index_map=log_prob_index_map,
    )

    assert len(trajectories) == 2
    restacked = StructuredTrajectory.stack(trajectories)
    for name in COMPONENT_ORDER:
        assert torch.equal(restacked.components[name].states, states[name])
    assert torch.equal(restacked.log_probs, log_probs)


def test_unstack_keeps_the_log_prob_map_absent_when_there_are_no_log_probs() -> None:
    """A map without values would claim a transition layout the trajectory lacks."""
    states = {name: torch.zeros(1, 3, 2) for name in COMPONENT_ORDER}

    trajectories = unstack_structured_trajectories(
        component_order=COMPONENT_ORDER,
        states=states,
        schedule=_schedule(),
        state_index_maps=_state_index_maps(),
        log_probs=None,
        log_prob_index_map=torch.tensor([0, 1]),
    )

    assert trajectories[0].log_probs is None
    assert trajectories[0].log_prob_index_map is None


def test_unstack_rejects_a_mapping_that_breaks_component_order() -> None:
    """Component order is the authoritative contract every mapping must follow."""
    states = {name: torch.zeros(1, 3, 2) for name in COMPONENT_ORDER}
    reordered = {"audio": torch.tensor([0, 1, 2]), "video": torch.tensor([0, 1, 2])}

    with pytest.raises(ValueError, match="state_index_maps component order"):
        unstack_structured_trajectories(
            component_order=COMPONENT_ORDER,
            states=states,
            schedule=_schedule(),
            state_index_maps=reordered,
        )


def test_unstack_rejects_components_that_disagree_on_batch_size() -> None:
    """Splitting per sample requires one shared batch axis across components."""
    states = {"video": torch.zeros(2, 3, 2), "audio": torch.zeros(1, 3, 2)}

    with pytest.raises(ValueError, match="batch size"):
        unstack_structured_trajectories(
            component_order=COMPONENT_ORDER,
            states=states,
            schedule=_schedule(),
            state_index_maps=_state_index_maps(),
        )
