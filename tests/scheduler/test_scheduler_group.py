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

from typing import Any, List, Tuple

import pytest

from flow_factory.scheduler import SchedulerGroup


class SchedulerFake:
    """Scheduler-like fake that records lifecycle dispatch."""

    def __init__(self, name: str, calls: List[Tuple[Any, ...]]) -> None:
        self.name = name
        self.calls = calls

    def step(self) -> None:
        """Provide the scheduler step surface."""

    def eval(self) -> None:
        """Record eval dispatch."""
        self.calls.append((self.name, "eval"))

    def train(self, mode: bool = True) -> None:
        """Record train dispatch."""
        self.calls.append((self.name, "train", mode))

    def rollout(self, mode: bool = True) -> None:
        """Record rollout dispatch."""
        self.calls.append((self.name, "rollout", mode))

    def set_seed(self, seed: int) -> None:
        """Record seed dispatch."""
        self.calls.append((self.name, "seed", seed))


def test_scheduler_group_exposes_immutable_ordered_mapping_and_primary() -> None:
    calls: List[Tuple[Any, ...]] = []
    video = SchedulerFake("video", calls)
    audio = SchedulerFake("audio", calls)
    group = SchedulerGroup({"video": video, "audio": audio}, primary_name="video")

    assert tuple(group) == ("video", "audio")
    assert group.names == ("video", "audio")
    assert group["audio"] is audio
    assert group.primary_name == "video"
    assert group.primary is video

    with pytest.raises(TypeError):
        group["video"] = audio


def test_scheduler_group_dispatches_modes_and_seed_in_declared_order() -> None:
    calls: List[Tuple[Any, ...]] = []
    group = SchedulerGroup(
        {
            "video": SchedulerFake("video", calls),
            "audio": SchedulerFake("audio", calls),
        },
        primary_name="video",
    )

    group.eval()
    group.train(mode=False)
    group.rollout(mode=True)
    group.set_seed(17)

    assert calls == [
        ("video", "eval"),
        ("audio", "eval"),
        ("video", "train", False),
        ("audio", "train", False),
        ("video", "rollout", True),
        ("audio", "rollout", True),
        ("video", "seed", 17),
        ("audio", "seed", 17),
    ]


@pytest.mark.parametrize(
    ("schedulers", "primary_name", "error", "message"),
    [
        ({}, "latent", ValueError, r"non-empty"),
        ({"": object()}, "", ValueError, r"component name.*non-empty"),
        ({"latent": object()}, "missing", ValueError, r"primary_name.*missing.*latent"),
        ({"latent": object()}, "latent", TypeError, r"scheduler-like.*latent.*object.*step"),
    ],
)
def test_scheduler_group_rejects_invalid_construction(
    schedulers: dict[str, object],
    primary_name: str,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        SchedulerGroup(schedulers, primary_name=primary_name)
