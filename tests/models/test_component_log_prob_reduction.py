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


def test_reduction_matches_a_single_scheduler_over_concatenated_components() -> None:
    video = torch.tensor([1.0, 3.0])
    audio = torch.tensor([5.0, 7.0])

    result = reduce_component_log_probs(
        {"video": video, "audio": audio},
        {"video": 192, "audio": 32},
    )

    torch.testing.assert_close(result, (video * 192 + audio * 32) / 224)


def test_reduction_supports_more_than_two_components() -> None:
    log_probs = {name: torch.tensor([float(index)]) for index, name in enumerate("abc")}
    dofs = {"a": 1, "b": 2, "c": 3}

    result = reduce_component_log_probs(log_probs, dofs)

    torch.testing.assert_close(result, torch.tensor([(0 * 1 + 1 * 2 + 2 * 3) / 6]))


@pytest.mark.parametrize("dof", [0, -4, True, 2.0])
def test_reduction_rejects_non_positive_int_degrees_of_freedom(dof: object) -> None:
    log_probs = {"video": torch.tensor([1.0]), "audio": torch.tensor([2.0])}

    with pytest.raises(ValueError, match=r"positive int.*'audio'"):
        reduce_component_log_probs(log_probs, {"video": 4, "audio": dof})


def test_reduction_rejects_component_key_mismatch() -> None:
    log_probs = {"video": torch.tensor([1.0]), "audio": torch.tensor([2.0])}

    with pytest.raises(ValueError, match=r"component order.*\('video', 'audio'\)"):
        reduce_component_log_probs(log_probs, {"video": 4})


def test_reduction_supports_stored_per_transition_log_probs() -> None:
    video = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    audio = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    result = reduce_component_log_probs(
        {"video": video, "audio": audio},
        {"video": 3, "audio": 1},
    )

    torch.testing.assert_close(result, (video * 3 + audio) / 4)


def test_reduction_rejects_mismatched_component_shapes() -> None:
    log_probs = {"video": torch.zeros(2), "audio": torch.zeros(3)}

    with pytest.raises(ValueError, match=r"matching.*'audio'"):
        reduce_component_log_probs(log_probs, {"video": 4, "audio": 5})
