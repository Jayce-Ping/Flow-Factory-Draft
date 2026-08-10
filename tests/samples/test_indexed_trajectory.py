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

from flow_factory.samples import (
    ComponentTimes,
    ComponentTrajectory,
    IndexedTrajectoryTensor,
    LatentState,
    MultiModalStepOutput,
    ReplayStep,
    StructuredTrajectory,
)


def _indexed(offset: float = 0.0) -> IndexedTrajectoryTensor:
    return IndexedTrajectoryTensor(
        values=torch.tensor([[1.0, 2.0], [3.0, 4.0]]) + offset,
        index_map=torch.tensor([-1, 0, 1]),
    )


def _component(offset: float = 0.0) -> ComponentTrajectory:
    return ComponentTrajectory(
        states=torch.tensor([[1.0], [2.0], [3.0]]) + offset,
        timesteps=torch.tensor([1000.0, 500.0, 0.0]),
        state_index_map=torch.tensor([0, 1, 2]),
    )


def _trajectory(offset: float = 0.0) -> StructuredTrajectory:
    return StructuredTrajectory(
        components={"latent": _component(offset)},
        log_probs=torch.tensor([0.1, 0.2]) + offset,
        log_prob_index_map=torch.tensor([0, 1]),
        component_log_probs={"latent": torch.tensor([0.1, 0.2]) + offset},
        callbacks={"velocity": {"latent": _indexed(offset)}},
    )


def test_indexed_trajectory_tensor_reads_collected_rollout_positions() -> None:
    indexed = _indexed()

    assert torch.equal(indexed.at(1), torch.tensor([1.0, 2.0]))
    assert torch.equal(indexed.at(2), torch.tensor([3.0, 4.0]))
    assert indexed.num_stored == 2


def test_indexed_trajectory_tensor_rejects_uncollected_sentinel_position() -> None:
    with pytest.raises(
        ValueError,
        match=r"velocity latent.*rollout position 0.*sentinel -1.*\[-1, 0, 1\]",
    ):
        _indexed().at(0, identifier="velocity latent")


def test_indexed_trajectory_tensor_rejects_unsigned_index_map() -> None:
    with pytest.raises(TypeError, match=r"signed integer.*index_map.*uint8"):
        IndexedTrajectoryTensor(
            values=torch.zeros(2, 3),
            index_map=torch.tensor([0, 1], dtype=torch.uint8),
        )


def test_indexed_trajectory_tensor_rejects_out_of_range_index_map() -> None:
    with pytest.raises(ValueError, match=r"2 stored.*\[0, 1, 5\]"):
        IndexedTrajectoryTensor(
            values=torch.zeros(2, 3),
            index_map=torch.tensor([0, 1, 5]),
        )


def test_indexed_trajectory_tensor_stack_reads_batched_positions() -> None:
    stacked = IndexedTrajectoryTensor.stack([_indexed(), _indexed(10.0)])

    assert stacked.batched is True
    assert tuple(stacked.values.shape) == (2, 2, 2)
    assert torch.equal(stacked.at(2), torch.tensor([[3.0, 4.0], [13.0, 14.0]]))


def test_indexed_trajectory_tensor_stack_requires_shared_index_map() -> None:
    other = IndexedTrajectoryTensor(
        values=torch.zeros(2, 2),
        index_map=torch.tensor([0, 1, -1]),
    )

    with pytest.raises(ValueError, match=r"index_map.*\[-1, 0, 1\].*\[0, 1, -1\].*sample index 1"):
        IndexedTrajectoryTensor.stack([_indexed(), other])


def test_indexed_trajectory_tensor_maps_every_tensor() -> None:
    indexed = _indexed()

    indexed.map_tensors(lambda tensor: tensor.to(torch.device("cpu")))

    assert indexed.values.device.type == "cpu"
    assert indexed.index_map.device.type == "cpu"


def test_structured_trajectory_stacks_component_log_probs_and_callbacks() -> None:
    stacked = StructuredTrajectory.stack([_trajectory(), _trajectory(10.0)])

    assert tuple(stacked.component_log_probs["latent"].shape) == (2, 2)
    assert torch.equal(
        stacked.component_log_probs["latent"],
        torch.tensor([[0.1, 0.2], [10.1, 10.2]]),
    )
    velocity = stacked.callbacks["velocity"]["latent"]
    assert velocity.batched is True
    assert torch.equal(velocity.at(2), torch.tensor([[3.0, 4.0], [13.0, 14.0]]))


def test_structured_trajectory_rejects_callback_component_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match=r"callbacks\['velocity'\].*\('latent',\).*\('audio',\)",
    ):
        StructuredTrajectory(
            components={"latent": _component()},
            callbacks={"velocity": {"audio": _indexed()}},
        )


def test_structured_trajectory_rejects_component_log_prob_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match=r"component_log_probs.*\('latent',\).*\('audio',\)",
    ):
        StructuredTrajectory(
            components={"latent": _component()},
            log_probs=torch.tensor([0.1, 0.2]),
            component_log_probs={"audio": torch.tensor([0.1, 0.2])},
        )


def test_structured_trajectory_moves_component_log_probs_and_callbacks() -> None:
    trajectory = _trajectory()

    trajectory.map_tensors(lambda tensor: tensor * 2)

    assert torch.equal(
        trajectory.component_log_probs["latent"],
        torch.tensor([0.2, 0.4]),
    )
    assert torch.equal(
        trajectory.callbacks["velocity"]["latent"].values,
        torch.tensor([[2.0, 4.0], [6.0, 8.0]]),
    )


def test_structured_trajectory_stack_requires_identical_callback_fields() -> None:
    other = StructuredTrajectory(
        components={"latent": _component(10.0)},
        log_probs=torch.tensor([10.1, 10.2]),
        log_prob_index_map=torch.tensor([0, 1]),
        component_log_probs={"latent": torch.tensor([10.1, 10.2])},
        callbacks={"next_latents_mean": {"latent": _indexed(10.0)}},
    )

    with pytest.raises(
        ValueError,
        match=r"callback field.*velocity.*next_latents_mean.*sample index 1",
    ):
        StructuredTrajectory.stack([_trajectory(), other])


def test_multi_modal_step_output_validates_component_statistic_mappings() -> None:
    output = MultiModalStepOutput(
        next_state_mean=LatentState({"latent": torch.zeros(2, 3)}),
        std_dev_t={"latent": torch.full((2, 1), 0.25)},
        dt={"latent": torch.full((2, 1), -0.5)},
        log_prob=torch.tensor([0.75, 0.5]),
        component_log_probs={"latent": torch.tensor([0.75, 0.5])},
    )

    assert tuple(output.std_dev_t) == ("latent",)
    assert tuple(output.component_log_probs) == ("latent",)

    with pytest.raises(TypeError, match=r"MultiModalStepOutput.std_dev_t\['latent'\].*float"):
        MultiModalStepOutput(std_dev_t={"latent": 0.25})


def _times(names: tuple) -> ComponentTimes:
    return ComponentTimes(
        timestep={name: torch.tensor([500.0, 500.0]) for name in names},
        next_timestep={name: torch.tensor(0) for name in names},
    )


def test_structured_trajectory_validates_component_order_without_joint_log_probs() -> None:
    with pytest.raises(
        ValueError,
        match=r"component_log_probs component order \('video', 'audio'\).*\('audio', 'video'\)",
    ):
        StructuredTrajectory(
            components={"video": _component(), "audio": _component(5.0)},
            component_log_probs={
                "audio": torch.tensor([0.1, 0.2]),
                "video": torch.tensor([0.3, 0.4]),
            },
        )


def test_structured_trajectory_requires_consistent_component_log_prob_lengths() -> None:
    with pytest.raises(
        ValueError,
        match=r"component_log_probs\['audio'\] shape \(2,\).*'video'.*\(3,\)",
    ):
        StructuredTrajectory(
            components={"video": _component(), "audio": _component(5.0)},
            component_log_probs={
                "video": torch.tensor([0.1, 0.2]),
                "audio": torch.tensor([0.3, 0.4, 0.5]),
            },
        )


def test_structured_trajectory_stack_reports_component_log_prob_shape_mismatch() -> None:
    first = StructuredTrajectory(
        components={"latent": _component()},
        component_log_probs={"latent": torch.tensor([0.1, 0.2])},
    )
    other = StructuredTrajectory(
        components={"latent": _component(10.0)},
        component_log_probs={"latent": torch.tensor([10.1, 10.2, 10.3])},
    )

    with pytest.raises(
        ValueError,
        match=r"component_log_probs\['latent'\].*\(2,\).*\(3,\).*sample index 1",
    ):
        StructuredTrajectory.stack([first, other])


def test_replay_step_requires_common_component_order() -> None:
    with pytest.raises(
        ValueError,
        match=r"ReplayStep.next_state component order \('video', 'audio'\).*\('audio', 'video'\)",
    ):
        ReplayStep(
            state=LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(2, 4)}),
            next_state=LatentState({"audio": torch.zeros(2, 4), "video": torch.zeros(2, 3)}),
            times=_times(("video", "audio")),
        )


def test_replay_step_requires_a_shared_batch_size() -> None:
    with pytest.raises(
        ValueError,
        match=r"ReplayStep.next_state\['audio'\].*batch size 2.*\(3, 4\)",
    ):
        ReplayStep(
            state=LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(2, 4)}),
            next_state=LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(3, 4)}),
            times=_times(("video", "audio")),
        )


def test_replay_step_requires_scalar_component_log_probabilities() -> None:
    with pytest.raises(
        ValueError,
        match=r"ReplayStep.component_log_probs\['latent'\].*\(2,\).*\(2, 3\)",
    ):
        ReplayStep(
            state=LatentState({"latent": torch.zeros(2, 3)}),
            next_state=LatentState({"latent": torch.zeros(2, 3)}),
            times=_times(("latent",)),
            log_prob=torch.zeros(2),
            component_log_probs={"latent": torch.zeros(2, 3)},
        )


def test_multi_modal_step_output_requires_common_component_order() -> None:
    with pytest.raises(
        ValueError,
        match=r"MultiModalStepOutput.std_dev_t component order "
        r"\('video', 'audio'\).*\('audio', 'video'\)",
    ):
        MultiModalStepOutput(
            next_state_mean=LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(2, 4)}),
            std_dev_t={"audio": torch.zeros(2, 1), "video": torch.zeros(2, 1)},
        )


def test_multi_modal_step_output_requires_a_shared_batch_size() -> None:
    with pytest.raises(
        ValueError,
        match=r"MultiModalStepOutput.std_dev_t\['latent'\].*batch size 2.*\(3, 1\)",
    ):
        MultiModalStepOutput(
            next_state_mean=LatentState({"latent": torch.zeros(2, 3)}),
            std_dev_t={"latent": torch.zeros(3, 1)},
        )


def test_multi_modal_step_output_requires_scalar_component_log_probabilities() -> None:
    with pytest.raises(
        ValueError,
        match=r"MultiModalStepOutput.component_log_probs\['latent'\].*\(2,\).*\(2, 3\)",
    ):
        MultiModalStepOutput(
            next_state_mean=LatentState({"latent": torch.zeros(2, 3)}),
            component_log_probs={"latent": torch.zeros(2, 3)},
        )
