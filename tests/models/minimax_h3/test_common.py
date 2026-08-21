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

from flow_factory.models.minimax_h3 import (
    MINIMAX_H3_COMPONENT_ORDER,
    build_component_step_output,
    build_structured_trajectories,
    build_training_component_times,
    combine_component_log_probs,
    draw_forward_process_noise,
    framework_sigma_to_model_time,
    inverse_shift_sigma,
    pack_audio_latents,
    pack_video_latents,
    shift_sigma,
    unpack_audio_latents,
    unpack_video_latents,
    validate_target_state,
)
from flow_factory.samples import LatentState
from flow_factory.scheduler import SDESchedulerOutput

ORACLE_COMMIT = "huggingface/diffusers@f53d552036a0d1bd5570782a39cd40cfabf112bc"


def _state(batch_size: int = 2) -> LatentState:
    return LatentState(
        {
            "video": torch.zeros(batch_size, 3, 96),
            "audio": torch.ones(batch_size, 5, 32),
        }
    )


def test_shift_round_trip_and_primary_time_mapping() -> None:
    sigma = torch.tensor([1.0, 0.8, 0.25, 0.0])
    shifted = shift_sigma(sigma, 12.0)
    assert torch.allclose(inverse_shift_sigma(shifted, 12.0), sigma)
    primary_timestep = shifted * 1000

    times = build_training_component_times(primary_timestep, video_shift=12.0, audio_shift=3.0)
    expected_audio_sigma = shift_sigma(sigma, 3.0)
    assert tuple(times.timestep) == MINIMAX_H3_COMPONENT_ORDER
    assert torch.allclose(times.sigma["video"], shifted)
    assert torch.allclose(times.sigma["audio"], expected_audio_sigma)
    assert torch.allclose(
        times.timestep["audio"],
        expected_audio_sigma * 1000,
    )
    assert torch.equal(times.next_sigma["video"], torch.zeros_like(shifted))
    assert torch.equal(framework_sigma_to_model_time(shifted), 1 - shifted)


def test_target_state_validation_requires_order_width_dtype_and_device() -> None:
    validate_target_state(_state())
    with pytest.raises(ValueError, match=r"component order.*video.*audio"):
        validate_target_state(
            LatentState({"audio": torch.zeros(2, 5, 32), "video": torch.zeros(2, 3, 96)})
        )
    with pytest.raises(ValueError, match=r"video.*width 96.*95"):
        validate_target_state(
            LatentState({"video": torch.zeros(2, 3, 95), "audio": torch.zeros(2, 5, 32)})
        )
    with pytest.raises(ValueError, match=r"dtype.*video.*float32.*audio.*float64"):
        validate_target_state(
            LatentState(
                {
                    "video": torch.zeros(2, 3, 96),
                    "audio": torch.zeros(2, 5, 32, dtype=torch.float64),
                }
            )
        )


def test_forward_noising_uses_video_audio_draw_order_and_data_ward_sign() -> None:
    state = _state(batch_size=1)
    sigmas = torch.tensor([0.25])
    times = build_training_component_times(sigmas * 1000, video_shift=1.0, audio_shift=1.0)
    generator = torch.Generator().manual_seed(9)
    result = draw_forward_process_noise(state, times, generator=generator)

    replay = torch.Generator().manual_seed(9)
    video_noise = torch.randn(state.components["video"].shape, generator=replay)
    audio_noise = torch.randn(state.components["audio"].shape, generator=replay)
    for name, expected_noise in (("video", video_noise), ("audio", audio_noise)):
        clean = state.components[name]
        sigma = times.sigma[name].reshape(1, 1, 1)
        assert torch.equal(result.noise.components[name], expected_noise)
        assert torch.equal(result.state.components[name], (1 - sigma) * clean + sigma * expected_noise)
        assert torch.equal(result.target_velocity.components[name], clean - expected_noise)


def test_video_pack_unpack_exact_round_trip_and_geometry_validation() -> None:
    latents = torch.arange(2 * 24 * 3 * 4 * 6).reshape(2, 24, 3, 4, 6)
    packed = pack_video_latents(latents)
    assert packed.shape == (2, 3 * 2 * 3, 96)
    assert torch.equal(unpack_video_latents(packed, frames=3, height=4, width=6), latents)
    with pytest.raises(ValueError, match=r"height and width.*divisible by 2"):
        pack_video_latents(torch.zeros(1, 24, 2, 3, 4))
    with pytest.raises(ValueError, match=r"width 96"):
        unpack_video_latents(torch.zeros(1, 4, 95), frames=1, height=4, width=4)


def test_audio_pack_unpack_is_channel_major_and_exact() -> None:
    latents = torch.arange(2 * 2 * 32 * 4).reshape(2, 2, 32, 4)
    packed = pack_audio_latents(latents)
    assert torch.equal(packed[0, :4], latents[0, 0].transpose(0, 1))
    assert torch.equal(unpack_audio_latents(packed), latents)
    with pytest.raises(ValueError, match=r"\(B, 2, 32, F\)"):
        pack_audio_latents(torch.zeros(1, 1, 32, 4))


def test_joint_log_prob_weights_generated_scalar_dof() -> None:
    video = torch.tensor([1.0, 3.0])
    audio = torch.tensor([5.0, 7.0])
    result = combine_component_log_probs(video, audio, video_dof=192, audio_dof=32)
    assert torch.equal(result, (video * 192 + audio * 32) / 224)
    with pytest.raises(ValueError, match=r"positive.*video_dof"):
        combine_component_log_probs(video, audio, video_dof=0, audio_dof=32)


def test_component_step_output_keeps_heterogeneous_states_and_real_statistics() -> None:
    video_output = SDESchedulerOutput(
        next_latents=torch.zeros(2, 3, 96),
        next_latents_mean=torch.ones(2, 3, 96),
        std_dev_t=torch.full((2, 1, 1), 0.2),
        dt=torch.full((2, 1, 1), -0.1),
        log_prob=torch.tensor([1.0, 2.0]),
        velocity=torch.full((2, 3, 96), 3.0),
    )
    audio_output = SDESchedulerOutput(
        next_latents=torch.zeros(2, 5, 32),
        next_latents_mean=torch.ones(2, 5, 32),
        std_dev_t=torch.full((2, 1, 1), 0.4),
        dt=torch.full((2, 1, 1), -0.2),
        log_prob=torch.tensor([4.0, 5.0]),
        velocity=torch.full((2, 5, 32), 6.0),
    )
    output = build_component_step_output(video_output, audio_output)

    assert output.next_state.components["video"].shape == (2, 3, 96)
    assert output.next_state.components["audio"].shape == (2, 5, 32)
    assert output.std_dev_t["video"] is video_output.std_dev_t
    assert torch.equal(
        output.log_prob,
        combine_component_log_probs(
            video_output.log_prob,
            audio_output.log_prob,
            video_dof=3 * 96,
            audio_dof=5 * 32,
        ),
    )


def test_component_step_output_combines_log_probs_without_next_state() -> None:
    video_output = SDESchedulerOutput(
        log_prob=torch.tensor([1.0, 2.0]),
        velocity=torch.zeros(2, 3, 96),
    )
    audio_output = SDESchedulerOutput(
        log_prob=torch.tensor([4.0, 5.0]),
        velocity=torch.zeros(2, 5, 32),
    )

    output = build_component_step_output(video_output, audio_output)

    assert output.next_state is None
    assert torch.equal(
        output.log_prob,
        combine_component_log_probs(
            video_output.log_prob,
            audio_output.log_prob,
            video_dof=3 * 96,
            audio_dof=5 * 32,
        ),
    )


@pytest.mark.parametrize(
    "state_map",
    [
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([0, -1, 1, 2]),
        torch.tensor([-1, -1, -1, 0]),
    ],
)
def test_structured_trajectory_builder_supports_dense_sparse_and_terminal_maps(
    state_map: torch.Tensor,
) -> None:
    stored = int(state_map.max().item()) + 1
    video_states = torch.arange(2 * stored * 3 * 96).reshape(2, stored, 3, 96)
    audio_states = torch.arange(2 * stored * 5 * 32).reshape(2, stored, 5, 32)
    schedule = {
        "video": (torch.tensor([1000.0, 700.0, 300.0, 0.0]), torch.tensor([1.0, 0.7, 0.3, 0.0])),
        "audio": (torch.tensor([1000.0, 500.0, 200.0, 0.0]), torch.tensor([1.0, 0.5, 0.2, 0.0])),
    }
    callback_map = torch.tensor([-1, 0, 1])
    callbacks = {
        "velocity": {
            "video": torch.zeros(2, 2, 3, 96),
            "audio": torch.zeros(2, 2, 5, 32),
        }
    }
    trajectories = build_structured_trajectories(
        states={"video": video_states, "audio": audio_states},
        state_index_map=state_map,
        schedule=schedule,
        log_probs=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        component_log_probs={
            "video": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "audio": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        },
        log_prob_index_map=torch.tensor([0, -1, 1]),
        callbacks=callbacks,
        callback_index_map=callback_map,
    )

    stacked = type(trajectories[0]).stack(trajectories)
    assert torch.equal(stacked.components["video"].states, video_states)
    assert stacked.components["video"].state_index_map.tolist() == state_map.tolist()
    assert stacked.callback_fields == ("velocity",)


def test_structured_trajectory_builder_keeps_independent_component_maps() -> None:
    state_maps = {
        "video": torch.tensor([0, -1, 1, 2]),
        "audio": torch.tensor([0, 1, -1, 2]),
    }
    callback_maps = {
        "video": torch.tensor([-1, 0, 1]),
        "audio": torch.tensor([0, -1, 1]),
    }
    schedule = {
        "video": (
            torch.tensor([1000.0, 700.0, 300.0, 0.0]),
            torch.tensor([1.0, 0.7, 0.3, 0.0]),
        ),
        "audio": (
            torch.tensor([1000.0, 500.0, 200.0, 0.0]),
            torch.tensor([1.0, 0.5, 0.2, 0.0]),
        ),
    }
    trajectories = build_structured_trajectories(
        states={
            "video": torch.zeros(1, 3, 3, 96),
            "audio": torch.zeros(1, 3, 5, 32),
        },
        state_index_map=state_maps,
        schedule=schedule,
        callbacks={
            "velocity": {
                "video": torch.zeros(1, 2, 3, 96),
                "audio": torch.zeros(1, 2, 5, 32),
            }
        },
        callback_index_map=callback_maps,
    )

    trajectory = trajectories[0]
    assert torch.equal(trajectory.components["video"].state_index_map, state_maps["video"])
    assert torch.equal(trajectory.components["audio"].state_index_map, state_maps["audio"])
    assert torch.equal(
        trajectory.callbacks["velocity"]["video"].index_map,
        callback_maps["video"],
    )
    assert torch.equal(
        trajectory.callbacks["velocity"]["audio"].index_map,
        callback_maps["audio"],
    )
