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

"""Contract tests for the shared LTX2 structured trajectory builder."""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from ltx2_fakes import (
    AUDIO_SCHEDULER_OFFSET,
    AUDIO_SEQ_LEN,
    BATCH_SIZE,
    CHANNELS,
    VIDEO_SCHEDULER_OFFSET,
    VIDEO_SEQ_LEN,
    PipelineFake,
    SchedulerFake,
    TransformerFake,
    conditioning_mask,
)

from flow_factory.models.ltx2._common import (
    LTX2_STRUCTURED_CALLBACK_FIELDS,
    build_ltx2_full_component_schedule,
    build_ltx2_structured_trajectories,
)
from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter
from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter
from flow_factory.samples import StructuredTrajectory
from flow_factory.utils.noise_schedule import flow_match_sigma

TOTAL_SEQ_LEN = VIDEO_SEQ_LEN + AUDIO_SEQ_LEN
NUM_STEPS = 3
ROLLOUT_TIMESTEPS = torch.tensor([900.0, 600.0, 300.0])
ScheduleType = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


def _adapter(cls: type = LTX2_T2AV_Adapter) -> Any:
    log: List[Tuple[str, Any]] = []
    transformer = TransformerFake()
    adapter = object.__new__(cls)
    adapter.pipeline = PipelineFake(SchedulerFake(VIDEO_SCHEDULER_OFFSET, log), transformer)
    adapter.component_runtime = SimpleNamespace(get_component=lambda name: transformer)
    adapter.load_scheduler = lambda: SchedulerFake(AUDIO_SCHEDULER_OFFSET, log)
    adapter.scheduler_group = adapter.build_scheduler_group()
    return adapter


def _states(stored: int = NUM_STEPS + 1) -> torch.Tensor:
    return torch.arange(
        BATCH_SIZE * stored * TOTAL_SEQ_LEN * CHANNELS, dtype=torch.float32
    ).reshape(BATCH_SIZE, stored, TOTAL_SEQ_LEN, CHANNELS)


def _schedule() -> ScheduleType:
    return build_ltx2_full_component_schedule(_adapter(), ROLLOUT_TIMESTEPS)


def _log_probs(stored: int = NUM_STEPS) -> torch.Tensor:
    return torch.arange(BATCH_SIZE * stored, dtype=torch.float32).reshape(BATCH_SIZE, stored) / 10


def _callback(stored: int = NUM_STEPS) -> torch.Tensor:
    return (
        torch.arange(BATCH_SIZE * stored * TOTAL_SEQ_LEN * CHANNELS, dtype=torch.float32).reshape(
            BATCH_SIZE, stored, TOTAL_SEQ_LEN, CHANNELS
        )
        * -1
    )


def _build(
    adapter: Optional[Any] = None,
    *,
    states: Optional[torch.Tensor] = None,
    state_index_map: Optional[torch.Tensor] = None,
    video_seq_len: int = VIDEO_SEQ_LEN,
    schedule: Optional[ScheduleType] = None,
    log_probs: Optional[torch.Tensor] = None,
    component_log_probs: Optional[Dict[str, torch.Tensor]] = None,
    log_prob_index_map: Optional[torch.Tensor] = None,
    callbacks: Optional[Dict[str, Any]] = None,
    callback_index_map: Optional[torch.Tensor] = None,
    video_active_mask: Optional[torch.Tensor] = None,
) -> List[StructuredTrajectory]:
    return build_ltx2_structured_trajectories(
        _adapter() if adapter is None else adapter,
        states=_states() if states is None else states,
        state_index_map=(
            torch.arange(NUM_STEPS + 1) if state_index_map is None else state_index_map
        ),
        video_seq_len=video_seq_len,
        schedule=_schedule() if schedule is None else schedule,
        log_probs=log_probs,
        component_log_probs=component_log_probs,
        log_prob_index_map=log_prob_index_map,
        callbacks=callbacks,
        callback_index_map=callback_index_map,
        video_active_mask=video_active_mask,
    )


def test_the_full_component_schedule_appends_the_terminal_zero_position() -> None:
    schedule = build_ltx2_full_component_schedule(_adapter(), ROLLOUT_TIMESTEPS)

    assert tuple(schedule) == ("video", "audio")
    for name in ("video", "audio"):
        timesteps, sigmas = schedule[name]
        assert timesteps.shape == (NUM_STEPS + 1,)
        assert torch.equal(timesteps[:-1], ROLLOUT_TIMESTEPS)
        assert float(timesteps[-1]) == 0.0
        assert torch.equal(sigmas, flow_match_sigma(timesteps))
        assert float(sigmas[-1]) == 0.0


def test_the_twin_schedules_stay_numerically_equal_but_are_stored_independently() -> None:
    schedule = build_ltx2_full_component_schedule(_adapter(), ROLLOUT_TIMESTEPS)

    video_timesteps, video_sigmas = schedule["video"]
    audio_timesteps, audio_sigmas = schedule["audio"]
    assert torch.equal(video_timesteps, audio_timesteps)
    assert torch.equal(video_sigmas, audio_sigmas)
    assert video_timesteps is not audio_timesteps
    assert video_sigmas is not audio_sigmas


def test_the_full_component_schedule_rejects_a_batched_rollout_schedule() -> None:
    with pytest.raises(ValueError, match=r"rollout timesteps shaped \(T,\).*\(1, 3\)"):
        build_ltx2_full_component_schedule(_adapter(), ROLLOUT_TIMESTEPS.unsqueeze(0))


def test_a_dense_map_splits_every_collected_state_into_video_then_audio() -> None:
    states = _states()

    trajectories = _build(states=states)

    assert len(trajectories) == BATCH_SIZE
    for index, trajectory in enumerate(trajectories):
        assert trajectory.component_names == ("video", "audio")
        video = trajectory.components["video"]
        audio = trajectory.components["audio"]
        assert torch.equal(video.states, states[index, :, :VIDEO_SEQ_LEN])
        assert torch.equal(audio.states, states[index, :, VIDEO_SEQ_LEN:])
        assert video.states.shape == (NUM_STEPS + 1, VIDEO_SEQ_LEN, CHANNELS)
        assert audio.states.shape == (NUM_STEPS + 1, AUDIO_SEQ_LEN, CHANNELS)
        assert torch.equal(video.state_index_map, torch.arange(NUM_STEPS + 1))
        assert torch.equal(audio.state_index_map, video.state_index_map)
        assert video.active_mask is None and audio.active_mask is None
        assert trajectory.log_probs is None
        assert trajectory.callbacks is None


def test_every_component_carries_the_full_schedule_it_was_stepped_on() -> None:
    video_timesteps, video_sigmas = _schedule()["video"]

    trajectories = _build()

    for trajectory in trajectories:
        for name in ("video", "audio"):
            component = trajectory.components[name]
            assert torch.equal(component.timesteps, video_timesteps)
            assert torch.equal(component.sigmas, video_sigmas)


def test_a_terminal_only_map_keeps_the_uncollected_sentinel_positions() -> None:
    states = _states(stored=1)
    terminal_map = torch.tensor([-1, -1, -1, 0])

    trajectories = _build(states=states, state_index_map=terminal_map)

    for index, trajectory in enumerate(trajectories):
        for name, expected in (
            ("video", states[index, :, :VIDEO_SEQ_LEN]),
            ("audio", states[index, :, VIDEO_SEQ_LEN:]),
        ):
            component = trajectory.components[name]
            assert torch.equal(component.state_index_map, terminal_map)
            assert torch.equal(component.states, expected)
            assert component.states.shape[0] == 1


def test_joint_and_component_log_probs_share_one_sparse_map() -> None:
    joint = _log_probs(stored=2)
    components = {"video": joint + 1, "audio": joint + 2}
    sparse_map = torch.tensor([0, 1, -1])

    trajectories = _build(
        log_probs=joint,
        component_log_probs=components,
        log_prob_index_map=sparse_map,
    )

    for index, trajectory in enumerate(trajectories):
        assert torch.equal(trajectory.log_probs, joint[index])
        assert tuple(trajectory.component_log_probs) == ("video", "audio")
        assert torch.equal(trajectory.component_log_probs["video"], components["video"][index])
        assert torch.equal(trajectory.component_log_probs["audio"], components["audio"][index])
        assert torch.equal(trajectory.log_prob_index_map, sparse_map)


def test_a_joint_log_prob_may_be_stored_without_component_log_probs() -> None:
    joint = _log_probs()

    trajectories = _build(log_probs=joint, log_prob_index_map=torch.arange(NUM_STEPS))

    for index, trajectory in enumerate(trajectories):
        assert torch.equal(trajectory.log_probs, joint[index])
        assert trajectory.component_log_probs is None


def test_callback_fields_split_per_component_and_keep_the_callback_map() -> None:
    velocity = _callback(stored=2)
    callback_map = torch.tensor([0, -1, 1])

    trajectories = _build(
        callbacks={"velocity": velocity},
        callback_index_map=callback_map,
    )

    for index, trajectory in enumerate(trajectories):
        assert trajectory.callback_fields == ("velocity",)
        stored = trajectory.callbacks["velocity"]
        assert tuple(stored) == ("video", "audio")
        assert torch.equal(stored["video"].values, velocity[index, :, :VIDEO_SEQ_LEN])
        assert torch.equal(stored["audio"].values, velocity[index, :, VIDEO_SEQ_LEN:])
        assert torch.equal(stored["video"].index_map, callback_map)
        assert not stored["video"].batched
        assert torch.equal(stored["video"].at(0), velocity[index, 0, :VIDEO_SEQ_LEN])
        assert torch.equal(stored["audio"].at(2), velocity[index, 1, VIDEO_SEQ_LEN:])


def test_an_empty_callback_mapping_stores_no_callback_fields() -> None:
    trajectories = _build(callbacks={}, callback_index_map=torch.arange(NUM_STEPS))

    assert all(trajectory.callbacks is None for trajectory in trajectories)


def test_the_video_active_mask_marks_generated_tokens_and_audio_stays_full_active() -> None:
    mask = ~conditioning_mask().bool()

    trajectories = _build(
        adapter=_adapter(LTX2_I2AV_Adapter),
        video_active_mask=mask,
    )

    for index, trajectory in enumerate(trajectories):
        video_mask = trajectory.components["video"].active_mask
        audio_mask = trajectory.components["audio"].active_mask
        assert video_mask.dtype is torch.bool
        assert video_mask.shape == (VIDEO_SEQ_LEN, 1)
        assert torch.equal(video_mask.reshape(VIDEO_SEQ_LEN), mask[index])
        assert audio_mask.shape == (AUDIO_SEQ_LEN, 1)
        assert audio_mask.dtype is torch.bool
        assert bool(audio_mask.all())


def test_a_channel_broadcast_active_mask_is_accepted_unchanged() -> None:
    mask = (~conditioning_mask().bool()).unsqueeze(-1)

    trajectories = _build(adapter=_adapter(LTX2_I2AV_Adapter), video_active_mask=mask)

    for index, trajectory in enumerate(trajectories):
        assert torch.equal(trajectory.components["video"].active_mask, mask[index])


def test_stacking_the_built_trajectories_restores_the_batch_axis() -> None:
    joint = _log_probs()
    mask = ~conditioning_mask().bool()

    batched = StructuredTrajectory.stack(
        _build(
            adapter=_adapter(LTX2_I2AV_Adapter),
            log_probs=joint,
            component_log_probs={"video": joint, "audio": joint * 2},
            log_prob_index_map=torch.arange(NUM_STEPS),
            callbacks={"velocity": _callback()},
            callback_index_map=torch.arange(NUM_STEPS),
            video_active_mask=mask,
        )
    )

    video = batched.components["video"]
    assert video.states.shape == (BATCH_SIZE, NUM_STEPS + 1, VIDEO_SEQ_LEN, CHANNELS)
    assert video.timesteps.shape == (BATCH_SIZE, NUM_STEPS + 1)
    assert video.sigmas.shape == (BATCH_SIZE, NUM_STEPS + 1)
    assert video.active_mask.shape == (BATCH_SIZE, VIDEO_SEQ_LEN, 1)
    assert batched.log_probs.shape == (BATCH_SIZE, NUM_STEPS)
    assert batched.component_log_probs["audio"].shape == (BATCH_SIZE, NUM_STEPS)
    assert batched.callbacks["velocity"]["audio"].batched
    assert batched.callbacks["velocity"]["audio"].values.shape == (
        BATCH_SIZE,
        NUM_STEPS,
        AUDIO_SEQ_LEN,
        CHANNELS,
    )


def test_the_declared_callback_fields_cover_the_trainer_owned_latent_fields() -> None:
    assert LTX2_STRUCTURED_CALLBACK_FIELDS == ("next_latents", "next_latents_mean", "velocity")


def test_the_builder_rejects_states_without_the_concatenated_layout() -> None:
    with pytest.raises(ValueError, match=r"states shaped \(B, stored, V \+ A, C\).*\(2, 4, 15\)"):
        _build(states=_states()[..., 0])


def test_the_builder_rejects_non_tensor_states() -> None:
    with pytest.raises(TypeError, match=r"torch.Tensor states.*list"):
        _build(states=[1.0, 2.0])


def test_the_builder_rejects_a_video_split_outside_the_sequence() -> None:
    with pytest.raises(ValueError, match=r"video_seq_len in \[1, 14\].*received 15"):
        _build(video_seq_len=TOTAL_SEQ_LEN)


def test_the_builder_rejects_a_zero_video_split() -> None:
    with pytest.raises(ValueError, match=r"video_seq_len in \[1, 14\].*received 0"):
        _build(video_seq_len=0)


def test_the_builder_rejects_a_schedule_in_the_wrong_component_order() -> None:
    schedule = _schedule()
    reordered = {"audio": schedule["audio"], "video": schedule["video"]}

    with pytest.raises(ValueError, match=r"schedule component order \('video', 'audio'\)"):
        _build(schedule=reordered)


def test_the_builder_rejects_a_schedule_that_is_not_one_dimensional() -> None:
    schedule = _schedule()
    timesteps, sigmas = schedule["video"]
    schedule["video"] = (timesteps.unsqueeze(0), sigmas.unsqueeze(0))

    with pytest.raises(ValueError, match=r"component 'video' timesteps.*\(T \+ 1,\)"):
        _build(schedule=schedule)


def test_the_builder_rejects_a_sigma_schedule_of_a_different_length() -> None:
    schedule = _schedule()
    timesteps, sigmas = schedule["video"]
    schedule["video"] = (timesteps, sigmas[:-1])

    with pytest.raises(ValueError, match=r"component 'video' timesteps/sigmas"):
        _build(schedule=schedule)


@pytest.mark.parametrize("component", ["video", "audio"])
def test_the_builder_rejects_a_schedule_without_the_terminal_zero(component: str) -> None:
    schedule = _schedule()
    timesteps, sigmas = schedule[component]
    schedule[component] = (timesteps + 1.0, sigmas)

    with pytest.raises(ValueError, match=rf"component {component!r}.*terminal.*zero"):
        _build(schedule=schedule)


def test_the_builder_rejects_a_state_map_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match=r"state_index_map length 4.*received 3"):
        _build(state_index_map=torch.arange(NUM_STEPS))


def test_the_builder_rejects_an_unsigned_state_map() -> None:
    with pytest.raises(TypeError, match=r"signed integer.*state_index_map.*uint8"):
        _build(state_index_map=torch.arange(NUM_STEPS + 1, dtype=torch.uint8))


def test_the_builder_rejects_a_state_map_pointing_past_the_stored_states() -> None:
    with pytest.raises(ValueError, match=r"state_index_map values in \[-1, 0\]"):
        _build(states=_states(stored=1), state_index_map=torch.tensor([-1, -1, 0, 1]))


def test_the_builder_rejects_a_state_map_below_the_uncollected_sentinel() -> None:
    with pytest.raises(ValueError, match=r"state_index_map values in \[-1, 3\]"):
        _build(state_index_map=torch.tensor([-2, 1, 2, 3]))


def test_the_builder_rejects_log_probs_without_their_index_map() -> None:
    with pytest.raises(ValueError, match=r"log_prob_index_map alongside log_probs"):
        _build(log_probs=_log_probs())


def test_the_builder_rejects_component_log_probs_without_the_joint_log_prob() -> None:
    joint = _log_probs()

    with pytest.raises(ValueError, match=r"component_log_probs.*joint log_probs"):
        _build(
            component_log_probs={"video": joint, "audio": joint},
            log_prob_index_map=torch.arange(NUM_STEPS),
        )


def test_the_builder_rejects_component_log_probs_in_the_wrong_order() -> None:
    joint = _log_probs()

    with pytest.raises(ValueError, match=r"component_log_probs component order"):
        _build(
            log_probs=joint,
            component_log_probs={"audio": joint, "video": joint},
            log_prob_index_map=torch.arange(NUM_STEPS),
        )


def test_the_builder_rejects_a_component_log_prob_of_a_different_length() -> None:
    joint = _log_probs()

    with pytest.raises(ValueError, match=r"component_log_probs\['audio'\].*\(2, 3\)"):
        _build(
            log_probs=joint,
            component_log_probs={"video": joint, "audio": joint[:, :-1]},
            log_prob_index_map=torch.arange(NUM_STEPS),
        )


def test_the_builder_rejects_a_log_prob_map_pointing_past_the_stored_entries() -> None:
    with pytest.raises(ValueError, match=r"log_prob_index_map values in \[-1, 2\]"):
        _build(log_probs=_log_probs(), log_prob_index_map=torch.tensor([0, 1, 3]))


def test_the_builder_rejects_a_log_prob_map_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match=r"log_prob_index_map length 3.*received 4"):
        _build(log_probs=_log_probs(), log_prob_index_map=torch.arange(NUM_STEPS + 1))


def test_the_builder_rejects_callbacks_without_their_index_map() -> None:
    with pytest.raises(ValueError, match=r"callback_index_map alongside callback fields"):
        _build(callbacks={"velocity": _callback()})


def test_the_builder_rejects_a_callback_map_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match=r"callback_index_map length 3.*received 4"):
        _build(
            callbacks={"velocity": _callback()},
            callback_index_map=torch.arange(NUM_STEPS + 1),
        )


def test_the_builder_rejects_a_callback_that_does_not_match_the_state_layout() -> None:
    with pytest.raises(ValueError, match=r"callback 'velocity'.*\(B, stored, V \+ A, C\)"):
        _build(
            callbacks={"velocity": _callback()[:, :, :-1]},
            callback_index_map=torch.arange(NUM_STEPS),
        )


def test_the_builder_rejects_a_non_tensor_callback_result() -> None:
    with pytest.raises(TypeError, match=r"callback 'velocity'.*torch.Tensor.*list"):
        _build(
            callbacks={"velocity": [1.0, 2.0]},
            callback_index_map=torch.arange(NUM_STEPS),
        )


def test_the_builder_rejects_a_non_boolean_active_mask() -> None:
    with pytest.raises(TypeError, match=r"video_active_mask dtype torch.bool.*float32"):
        _build(adapter=_adapter(LTX2_I2AV_Adapter), video_active_mask=conditioning_mask())


def test_the_builder_rejects_an_active_mask_that_does_not_cover_the_video_tokens() -> None:
    mask = torch.ones(BATCH_SIZE, VIDEO_SEQ_LEN - 1, dtype=torch.bool)

    with pytest.raises(ValueError, match=r"video_active_mask shaped \(2, 12\) or \(2, 12, 1\)"):
        _build(adapter=_adapter(LTX2_I2AV_Adapter), video_active_mask=mask)


def test_the_builder_rejects_an_active_mask_without_a_generated_token() -> None:
    mask = torch.zeros(BATCH_SIZE, VIDEO_SEQ_LEN, dtype=torch.bool)

    with pytest.raises(ValueError, match=r"video_active_mask.*positive.*sample"):
        _build(adapter=_adapter(LTX2_I2AV_Adapter), video_active_mask=mask)


def test_builder_errors_name_the_calling_adapter() -> None:
    with pytest.raises(ValueError, match=r"LTX2_I2AV_Adapter"):
        _build(adapter=_adapter(LTX2_I2AV_Adapter), video_seq_len=0)
