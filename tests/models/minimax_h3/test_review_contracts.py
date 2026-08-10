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

import inspect
from typing import Any, List

import pytest
import torch

import flow_factory.models.minimax_h3 as minimax_core
from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import BaseSample, ComponentTimes, LatentState
from flow_factory.scheduler import MiniMaxH3SDEScheduler, SDESchedulerOutput


class BridgeAdapter(BaseAdapter):
    """Minimal adapter exposing the real structured trajectory bridge."""

    trajectory_component_order = ("video", "audio")

    def load_pipeline(self) -> Any:
        """Remain unused in bridge-only tests."""
        raise NotImplementedError

    def decode_latents(self, latents: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return latents unchanged."""
        return latents

    def inference(self, **kwargs: Any) -> List[BaseSample]:
        """Return no generated samples."""
        return []

    def forward(self, **kwargs: Any) -> SDESchedulerOutput:
        """Remain unused in bridge-only tests."""
        raise NotImplementedError


def _schedule() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        "video": (
            torch.tensor([1000.0, 700.0, 300.0, 0.0]),
            torch.tensor([1.0, 0.7, 0.3, 0.0]),
        ),
        "audio": (
            torch.tensor([1000.0, 500.0, 200.0, 0.0]),
            torch.tensor([1.0, 0.5, 0.2, 0.0]),
        ),
    }


def _build_batch(
    state_map: torch.Tensor,
    callback_map: torch.Tensor,
) -> Any:
    stored_states = int(state_map.max().item()) + 1
    stored_callbacks = int(callback_map.max().item()) + 1
    trajectories = minimax_core.build_structured_trajectories(
        states={
            "video": torch.arange(stored_states * 2 * 96, dtype=torch.float32).reshape(
                1, stored_states, 2, 96
            ),
            "audio": torch.arange(stored_states * 3 * 32, dtype=torch.float32).reshape(
                1, stored_states, 3, 32
            ),
        },
        state_index_map=state_map,
        schedule=_schedule(),
        callbacks={
            "velocity": {
                "video": torch.ones(1, stored_callbacks, 2, 96),
                "audio": torch.ones(1, stored_callbacks, 3, 32) * 2,
            }
        },
        callback_index_map=callback_map,
    )
    return BaseSample.stack([BaseSample(trajectory=trajectories[0])])


def test_target_state_rejects_empty_batches_and_generated_rows() -> None:
    with pytest.raises(ValueError, match=r"non-empty batch"):
        minimax_core.validate_target_state(
            LatentState(
                {
                    "video": torch.zeros(0, 2, 96),
                    "audio": torch.zeros(0, 3, 32),
                }
            )
        )
    with pytest.raises(ValueError, match=r"video.*non-empty generated rows"):
        minimax_core.validate_target_state(
            LatentState(
                {
                    "video": torch.zeros(1, 0, 96),
                    "audio": torch.zeros(1, 3, 32),
                }
            )
        )


@pytest.mark.parametrize(
    ("video_shape", "audio_shape", "message"),
    [
        ((0, 2, 2, 96), (0, 2, 3, 32), r"non-empty batch"),
        ((1, 2, 0, 96), (1, 2, 3, 32), r"video.*non-empty generated rows"),
    ],
)
def test_trajectory_builder_rejects_empty_batches_and_rows(
    video_shape: tuple[int, ...],
    audio_shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        minimax_core.build_structured_trajectories(
            states={
                "video": torch.zeros(video_shape),
                "audio": torch.zeros(audio_shape),
            },
            state_index_map=torch.tensor([0, 1, -1, -1]),
            schedule=_schedule(),
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("sigma", torch.tensor([float("nan")]), r"sigma.*finite.*\[0, 1\]"),
        ("timestep", torch.tensor([250.0], dtype=torch.float64), r"dtype.*state"),
        ("next_sigma", torch.tensor([0.1]), r"next_sigma.*zero"),
        ("timestep", torch.tensor([251.0]), r"timestep.*sigma.*1000"),
    ],
)
def test_component_times_validation_is_complete(
    field: str,
    bad_value: torch.Tensor,
    message: str,
) -> None:
    state = LatentState(
        {
            "video": torch.zeros(1, 2, 96),
            "audio": torch.zeros(1, 3, 32),
        }
    )
    mappings = {
        "timestep": {"video": torch.tensor([250.0]), "audio": torch.tensor([250.0])},
        "next_timestep": {"video": torch.zeros(1), "audio": torch.zeros(1)},
        "sigma": {"video": torch.tensor([0.25]), "audio": torch.tensor([0.25])},
        "next_sigma": {"video": torch.zeros(1), "audio": torch.zeros(1)},
    }
    mappings[field]["video"] = bad_value
    times = ComponentTimes(**mappings)
    with pytest.raises((TypeError, ValueError), match=message):
        minimax_core.draw_forward_process_noise(
            state,
            times,
            generator=torch.Generator().manual_seed(1),
        )


@pytest.mark.parametrize(
    ("schedule", "message"),
    [
        (
            {
                "video": (torch.tensor([1000, 0]), torch.tensor([1.0, 0.0])),
                "audio": (torch.tensor([1000.0, 0.0]), torch.tensor([1.0, 0.0])),
            },
            r"video.*timesteps.*floating",
        ),
        (
            {
                "video": (
                    torch.tensor([1000.0, 400.0, 0.0]),
                    torch.tensor([1.0, 0.5, 0.0]),
                ),
                "audio": (
                    torch.tensor([1000.0, 500.0, 0.0]),
                    torch.tensor([1.0, 0.5, 0.0]),
                ),
            },
            r"video.*timesteps.*sigmas.*1000",
        ),
        (
            {
                "video": (
                    torch.tensor([1000.0, 700.0, 0.0]),
                    torch.tensor([1.0, 0.7, 0.0]),
                ),
                "audio": (
                    torch.tensor([1000.0, 500.0, 0.0]),
                    torch.tensor([1.0, 0.6, 0.0]),
                ),
            },
            r"audio.*timesteps.*sigmas.*1000",
        ),
    ],
)
def test_trajectory_schedule_validation_is_complete(
    schedule: dict[str, tuple[torch.Tensor, torch.Tensor]],
    message: str,
) -> None:
    schedule_length = len(schedule["video"][0])
    state_map = (
        torch.tensor([0, 1])
        if schedule_length == 2
        else torch.tensor([0, 1, -1])
    )
    with pytest.raises((TypeError, ValueError), match=message):
        minimax_core.build_structured_trajectories(
            states={
                "video": torch.zeros(1, 2, 2, 96),
                "audio": torch.zeros(1, 2, 3, 32),
            },
            state_index_map=state_map,
            schedule=schedule,
        )


def test_real_bridge_replays_dense_and_sparse_collected_positions() -> None:
    adapter = object.__new__(BridgeAdapter)
    dense = _build_batch(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2]))
    sparse = _build_batch(torch.tensor([0, 1, -1, 2]), torch.tensor([0, -1, 1]))

    dense_replay = adapter.get_replay_step(dense, 1)
    sparse_replay = adapter.get_replay_step(sparse, 0)
    dense_callback = adapter.get_replay_callback(dense, 1, "velocity")
    sparse_callback = adapter.get_replay_callback(sparse, 0, "velocity")

    assert dense_replay.state.component_names == ("video", "audio")
    assert sparse_replay.next_state.components["audio"].shape == (1, 3, 32)
    assert dense_callback.components["video"].shape == (1, 2, 96)
    assert sparse_callback.components["audio"].shape == (1, 3, 32)


def test_real_bridge_terminal_only_extracts_terminal_and_rejects_uncollected() -> None:
    adapter = object.__new__(BridgeAdapter)
    batch = _build_batch(torch.tensor([-1, -1, -1, 0]), torch.tensor([-1, -1, 0]))

    terminal = adapter.get_terminal_state(batch)
    assert terminal.components["video"].shape == (1, 2, 96)
    with pytest.raises(ValueError, match=r"uncollected.*rollout position 0|sentinel -1"):
        adapter.get_replay_step(batch, 0)
    with pytest.raises(ValueError, match=r"rollout position 0.*sentinel -1"):
        adapter.get_replay_callback(batch, 0, "velocity")


def test_new_public_apis_have_complete_google_docstrings() -> None:
    public_functions = [
        minimax_core.shift_sigma,
        minimax_core.inverse_shift_sigma,
        minimax_core.framework_sigma_to_model_time,
        minimax_core.model_time_to_framework_sigma,
        minimax_core.build_training_component_times,
        minimax_core.validate_target_state,
        minimax_core.draw_forward_process_noise,
        minimax_core.pack_video_latents,
        minimax_core.unpack_video_latents,
        minimax_core.pack_audio_latents,
        minimax_core.unpack_audio_latents,
        minimax_core.combine_component_log_probs,
        minimax_core.build_component_step_output,
        minimax_core.build_structured_trajectories,
        MiniMaxH3SDEScheduler.set_timesteps,
        MiniMaxH3SDEScheduler.scale_noise,
        MiniMaxH3SDEScheduler.step,
    ]
    incomplete = []
    for function in public_functions:
        docstring = inspect.getdoc(function) or ""
        if "Args:" not in docstring or "Returns:" not in docstring:
            incomplete.append(function.__qualname__)
    assert incomplete == []
