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

from flow_factory.samples import BaseSample, ComponentTrajectory, StructuredTrajectory


def _component(offset: float, shape: tuple[int, ...] = (3, 2)) -> ComponentTrajectory:
    return ComponentTrajectory(
        states=torch.arange(0, torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(
            shape
        )
        + offset,
        timesteps=torch.tensor([1000.0, 500.0, 0.0]),
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
        state_index_map=torch.tensor([0, 1, 2]),
    )


def _trajectory(component_order: tuple[str, ...] = ("video", "audio")) -> StructuredTrajectory:
    available = {
        "video": _component(0.0),
        "audio": _component(10.0, shape=(3, 1)),
    }
    return StructuredTrajectory(
        components={name: available[name] for name in component_order},
        log_probs=torch.tensor([0.1, 0.2]),
        log_prob_index_map=torch.tensor([0, 1]),
    )


def test_component_trajectory_validates_schedule_and_state_indices() -> None:
    with pytest.raises(ValueError, match=r"timesteps.*3.*sigmas.*2"):
        ComponentTrajectory(
            states=torch.zeros(3, 2),
            timesteps=torch.tensor([1000.0, 500.0, 0.0]),
            sigmas=torch.tensor([1.0, 0.0]),
            state_index_map=torch.tensor([0, 1, 2]),
        )

    with pytest.raises(ValueError, match=r"state_index_map.*3.*received.*4"):
        ComponentTrajectory(
            states=torch.zeros(3, 2),
            timesteps=torch.tensor([1000.0, 500.0, 0.0]),
            state_index_map=torch.tensor([0, 1, 4]),
        )


def test_structured_trajectory_preserves_component_order_and_moves_tensors() -> None:
    trajectory = _trajectory()

    assert trajectory.component_names == ("video", "audio")
    moved = trajectory.to("meta")

    assert moved is trajectory
    assert trajectory.components["video"].states.device.type == "meta"
    assert trajectory.log_probs.device.type == "meta"


def test_sample_stack_collates_nested_trajectory_and_keeps_legacy_none() -> None:
    samples = [
        BaseSample(trajectory=_trajectory(), prompt="first"),
        BaseSample(
            trajectory=StructuredTrajectory(
                components={
                    "video": _component(100.0),
                    "audio": _component(110.0, shape=(3, 1)),
                },
                log_probs=torch.tensor([0.3, 0.4]),
                log_prob_index_map=torch.tensor([0, 1]),
            ),
            prompt="second",
        ),
    ]

    batch = BaseSample.stack(samples)

    assert isinstance(batch.trajectory, StructuredTrajectory)
    assert batch.trajectory.components["video"].states.shape == (2, 3, 2)
    assert batch.trajectory.components["audio"].states.shape == (2, 3, 1)
    assert batch.trajectory.log_probs.shape == (2, 2)
    assert BaseSample.stack([BaseSample(), BaseSample()]).trajectory is None


@pytest.mark.parametrize(
    ("second", "message"),
    [
        (_trajectory(("audio", "video")), r"component.*order.*video.*audio.*audio.*video"),
        (
            StructuredTrajectory(
                components={
                    "video": ComponentTrajectory(
                        states=torch.zeros(3, 2),
                        timesteps=torch.tensor([1000.0, 500.0, 0.0]),
                        state_index_map=torch.tensor([0, 0, 2]),
                    ),
                    "audio": _component(0.0, shape=(3, 1)),
                },
                log_probs=torch.zeros(2),
                log_prob_index_map=torch.tensor([0, 1]),
            ),
            r"video.*state_index_map.*expected.*0.*1.*2.*received.*0.*0.*2",
        ),
    ],
)
def test_sample_stack_rejects_mismatched_trajectory_metadata(
    second: StructuredTrajectory, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BaseSample.stack([BaseSample(trajectory=_trajectory()), BaseSample(trajectory=second)])
