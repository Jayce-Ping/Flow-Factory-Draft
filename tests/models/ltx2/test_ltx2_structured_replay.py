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

"""Collate and replay the structured trajectories an LTX2 rollout emits."""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
from ltx2_fakes import (
    AUDIO_SEQ_LEN,
    BATCH_SIZE,
    CHANNELS,
    FRAME_SEQ_LEN,
    GENERATED_VIDEO_NUMEL,
    VIDEO_SEQ_LEN,
)
from ltx2_inference_fakes import (
    NUM_INFERENCE_STEPS,
    SEED,
    condition_images,
    inference_adapter,
    inference_kwargs,
)

from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter, LTX2I2AVSample
from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter, LTX2Sample
from flow_factory.samples import LatentState, StructuredTrajectory
from flow_factory.utils.noise_schedule import flow_match_sigma

ADAPTERS = [LTX2_T2AV_Adapter, LTX2_I2AV_Adapter]
SAMPLE_TYPES = {LTX2_T2AV_Adapter: LTX2Sample, LTX2_I2AV_Adapter: LTX2I2AVSample}
# The last rollout transition leaves the SDE window, so it stores no log probability.
STOCHASTIC_STEPS = list(range(NUM_INFERENCE_STEPS - 1))
RETURN_FIELDS = ("next_latents", "next_latents_mean", "std_dev_t", "dt", "log_prob", "velocity")


def _rollout(cls: type) -> SimpleNamespace:
    """Run one rollout and collate its samples into a training batch."""
    kwargs = inference_kwargs()
    if cls is LTX2_I2AV_Adapter:
        kwargs["condition_images"] = condition_images()

    torch.manual_seed(SEED)
    adapter, _ = inference_adapter(cls)
    samples = adapter.inference(**kwargs)
    batch = SAMPLE_TYPES[cls].stack(samples)
    return SimpleNamespace(
        adapter=adapter,
        samples=samples,
        batch=batch,
        conditioned=cls is LTX2_I2AV_Adapter,
    )


@pytest.fixture(scope="module", params=ADAPTERS, ids=lambda cls: cls.__name__)
def rollout(request: Any) -> SimpleNamespace:
    """Return one collated rollout batch per LTX2 adapter."""
    return _rollout(request.param)


@pytest.fixture(scope="module")
def i2av_rollout() -> SimpleNamespace:
    """Return the conditioned rollout batch the I2AV-only expectations read."""
    return _rollout(LTX2_I2AV_Adapter)


def _legacy_forward(rollout: SimpleNamespace, state: LatentState, times: Any, **overrides: Any):
    """Run the unchanged concatenated forward on the same replayed inputs."""
    kwargs: Dict[str, Any] = {
        "connector_prompt_embeds": rollout.batch["connector_prompt_embeds"],
        "connector_audio_prompt_embeds": rollout.batch["connector_audio_prompt_embeds"],
        "connector_attention_mask": rollout.batch["connector_attention_mask"],
        "video_seq_len": VIDEO_SEQ_LEN,
        "guidance_scale": 1.0,
        "height": rollout.batch["height"],
        "width": rollout.batch["width"],
        "num_frames": rollout.batch["num_frames"],
        "frame_rate": rollout.batch["frame_rate"],
    }
    if rollout.conditioned:
        kwargs["conditioning_mask"] = rollout.batch["conditioning_mask"]
    kwargs.update(overrides)
    return rollout.adapter.forward(
        t=times.timestep["video"],
        t_next=times.next_timestep["video"],
        latents=torch.cat([state.components["video"], state.components["audio"]], dim=1),
        return_kwargs=list(RETURN_FIELDS),
        **kwargs,
    )


def test_collating_rollout_samples_yields_one_batched_structured_trajectory(
    rollout: SimpleNamespace,
) -> None:
    trajectory = rollout.batch["trajectory"]

    assert isinstance(trajectory, StructuredTrajectory)
    assert trajectory.component_names == ("video", "audio")
    assert trajectory.components["video"].states.shape == (
        BATCH_SIZE,
        NUM_INFERENCE_STEPS + 1,
        VIDEO_SEQ_LEN,
        CHANNELS,
    )
    assert trajectory.components["audio"].states.shape == (
        BATCH_SIZE,
        NUM_INFERENCE_STEPS + 1,
        AUDIO_SEQ_LEN,
        CHANNELS,
    )
    assert trajectory.log_probs.shape == (BATCH_SIZE, NUM_INFERENCE_STEPS - 1)
    assert trajectory.callbacks["velocity"]["video"].batched is True
    for index, sample in enumerate(rollout.samples):
        assert torch.equal(
            trajectory.components["video"].states[index],
            sample.trajectory.components["video"].states,
        )


def test_the_terminal_state_reads_each_components_last_stored_state(
    rollout: SimpleNamespace,
) -> None:
    trajectory = rollout.batch["trajectory"]

    terminal = rollout.adapter.get_terminal_state(rollout.batch)

    assert terminal.component_names == ("video", "audio")
    for name in ("video", "audio"):
        assert torch.equal(terminal.components[name], trajectory.components[name].states[:, -1])
    assert (terminal.active_masks is not None) is rollout.conditioned


@pytest.mark.parametrize("step_index", STOCHASTIC_STEPS)
def test_each_replay_step_reads_the_stored_states_and_component_schedules(
    rollout: SimpleNamespace, step_index: int
) -> None:
    trajectory = rollout.batch["trajectory"]

    step = rollout.adapter.get_replay_step(rollout.batch, step_index)

    for name in ("video", "audio"):
        component = trajectory.components[name]
        assert torch.equal(step.state.components[name], component.states[:, step_index])
        assert torch.equal(step.next_state.components[name], component.states[:, step_index + 1])
        assert torch.equal(step.times.timestep[name], component.timesteps[:, step_index])
        assert torch.equal(step.times.next_timestep[name], component.timesteps[:, step_index + 1])
        assert torch.equal(
            step.times.sigma[name], flow_match_sigma(component.timesteps[:, step_index])
        )


def test_the_component_schedules_end_at_the_shared_terminal_time(
    rollout: SimpleNamespace,
) -> None:
    trajectory = rollout.batch["trajectory"]

    for name in ("video", "audio"):
        timesteps = trajectory.components[name].timesteps
        assert timesteps.shape == (BATCH_SIZE, NUM_INFERENCE_STEPS + 1)
        assert not bool(timesteps[:, -1].any())
        assert bool((timesteps[:, :-1].diff(dim=-1) < 0).all())


@pytest.mark.parametrize("step_index", STOCHASTIC_STEPS)
def test_each_stochastic_replay_step_carries_the_stored_log_probabilities(
    rollout: SimpleNamespace, step_index: int
) -> None:
    trajectory = rollout.batch["trajectory"]

    step = rollout.adapter.get_replay_step(rollout.batch, step_index)

    assert torch.equal(step.log_prob, trajectory.log_probs[:, step_index])
    for name in ("video", "audio"):
        assert torch.equal(
            step.component_log_probs[name],
            trajectory.component_log_probs[name][:, step_index],
        )


def test_the_deterministic_final_transition_reports_its_uncollected_log_probability(
    rollout: SimpleNamespace,
) -> None:
    with pytest.raises(ValueError, match=r"log_prob_index_map.*uncollected.*-1"):
        rollout.adapter.get_replay_step(rollout.batch, NUM_INFERENCE_STEPS - 1)


@pytest.mark.parametrize("field", ["next_latents_mean", "velocity"])
@pytest.mark.parametrize("step_index", range(NUM_INFERENCE_STEPS))
def test_each_replay_callback_reads_the_stored_component_split(
    rollout: SimpleNamespace, field: str, step_index: int
) -> None:
    stored = rollout.batch["trajectory"].callbacks[field]

    callback = rollout.adapter.get_replay_callback(rollout.batch, step_index, field)

    assert callback.component_names == ("video", "audio")
    for name in ("video", "audio"):
        assert torch.equal(callback.components[name], stored[name].values[:, step_index])
    assert (callback.active_masks is not None) is rollout.conditioned


def test_an_unstored_replay_callback_names_the_stored_fields(
    rollout: SimpleNamespace,
) -> None:
    with pytest.raises(ValueError, match=r"'std_dev_t'.*next_latents_mean.*velocity"):
        rollout.adapter.get_replay_callback(rollout.batch, 0, "std_dev_t")


@pytest.mark.parametrize("step_index", STOCHASTIC_STEPS)
def test_replaying_a_step_reproduces_the_legacy_concatenated_forward(
    rollout: SimpleNamespace, step_index: int
) -> None:
    step = rollout.adapter.get_replay_step(rollout.batch, step_index)

    output = rollout.adapter.forward_state(
        batch=rollout.batch,
        state=step.state,
        times=step.times,
        next_state=step.next_state,
        compute_log_prob=True,
        return_fields=RETURN_FIELDS,
        guidance_scale=1.0,
    )
    legacy = _legacy_forward(
        rollout,
        step.state,
        step.times,
        next_latents=torch.cat(
            [step.next_state.components["video"], step.next_state.components["audio"]],
            dim=1,
        ),
        compute_log_prob=True,
    )

    for field, legacy_values in (
        ("next_state", legacy.next_latents),
        ("next_state_mean", legacy.next_latents_mean),
        ("velocity", legacy.velocity),
    ):
        state = getattr(output, field)
        assert torch.equal(
            torch.cat([state.components["video"], state.components["audio"]], dim=1),
            legacy_values,
        )
    assert torch.equal(output.log_prob, legacy.log_prob)
    assert torch.equal(output.std_dev_t["video"].reshape(-1), legacy.std_dev_t.reshape(-1))
    assert torch.equal(output.dt["video"].reshape(-1), legacy.dt.reshape(-1))


@pytest.mark.parametrize("step_index", STOCHASTIC_STEPS)
def test_replaying_a_stored_transition_recovers_its_rollout_log_probability(
    rollout: SimpleNamespace, step_index: int
) -> None:
    step = rollout.adapter.get_replay_step(rollout.batch, step_index)

    output = rollout.adapter.forward_state(
        batch=rollout.batch,
        state=step.state,
        times=step.times,
        next_state=step.next_state,
        compute_log_prob=True,
        return_fields=("next_latents", "log_prob"),
        guidance_scale=1.0,
    )

    assert torch.equal(output.log_prob, step.log_prob)
    for name in ("video", "audio"):
        assert torch.equal(output.component_log_probs[name], step.component_log_probs[name])


def test_the_i2av_replay_masks_mark_only_the_generated_video_tokens(
    i2av_rollout: SimpleNamespace,
) -> None:
    step = i2av_rollout.adapter.get_replay_step(i2av_rollout.batch, 0)

    active = i2av_rollout.adapter.get_state_active_numel(step.state)

    assert active == {"video": GENERATED_VIDEO_NUMEL, "audio": AUDIO_SEQ_LEN * CHANNELS}
    assert torch.equal(
        step.state.active_masks["video"].reshape(BATCH_SIZE, VIDEO_SEQ_LEN),
        ~i2av_rollout.batch["conditioning_mask"].bool(),
    )


def test_the_i2av_conditioning_tokens_never_move_through_a_replayed_step(
    i2av_rollout: SimpleNamespace,
) -> None:
    for step_index in STOCHASTIC_STEPS:
        step = i2av_rollout.adapter.get_replay_step(i2av_rollout.batch, step_index)
        fixed = step.state.components["video"][:, :FRAME_SEQ_LEN]

        output = i2av_rollout.adapter.forward_state(
            batch=i2av_rollout.batch,
            state=step.state,
            times=step.times,
            compute_log_prob=False,
            return_fields=("next_latents", "next_latents_mean"),
            guidance_scale=1.0,
        )

        assert torch.equal(step.next_state.components["video"][:, :FRAME_SEQ_LEN], fixed)
        for field in ("next_state", "next_state_mean"):
            replayed = getattr(output, field).components["video"]
            assert torch.equal(replayed[:, :FRAME_SEQ_LEN], fixed)
            assert not torch.equal(
                replayed[:, FRAME_SEQ_LEN:], step.state.components["video"][:, FRAME_SEQ_LEN:]
            )


def test_the_i2av_conditioning_tokens_never_move_through_forward_process_noising(
    i2av_rollout: SimpleNamespace,
) -> None:
    step = i2av_rollout.adapter.get_replay_step(i2av_rollout.batch, 0)
    clean = step.state

    noised = i2av_rollout.adapter.add_forward_process_noise(
        clean, step.times, generator=torch.Generator().manual_seed(3)
    )

    fixed = clean.components["video"][:, :FRAME_SEQ_LEN]
    assert torch.equal(noised.state.components["video"][:, :FRAME_SEQ_LEN], fixed)
    assert not torch.equal(
        noised.state.components["video"][:, FRAME_SEQ_LEN:],
        clean.components["video"][:, FRAME_SEQ_LEN:],
    )
    assert not bool(noised.target_velocity.components["video"][:, :FRAME_SEQ_LEN].any())


def test_the_i2av_reducers_ignore_the_conditioning_tokens_of_a_replayed_state(
    i2av_rollout: SimpleNamespace,
) -> None:
    step = i2av_rollout.adapter.get_replay_step(i2av_rollout.batch, 0)
    video = torch.zeros(BATCH_SIZE, VIDEO_SEQ_LEN, CHANNELS)
    video[:, :FRAME_SEQ_LEN] = 500.0
    video[:, FRAME_SEQ_LEN:] = 3.0
    audio = torch.full((BATCH_SIZE, AUDIO_SEQ_LEN, CHANNELS), 5.0)
    values = {"video": video, "audio": audio}

    reduced = i2av_rollout.adapter.reduce_component_latent_values(values, state=step.state)
    combined = i2av_rollout.adapter.reduce_latent_values(values, state=step.state)

    audio_numel = AUDIO_SEQ_LEN * CHANNELS
    expected = (3.0 * GENERATED_VIDEO_NUMEL + 5.0 * audio_numel) / (
        GENERATED_VIDEO_NUMEL + audio_numel
    )
    assert torch.equal(reduced["video"], torch.full((BATCH_SIZE,), 3.0))
    assert torch.allclose(combined, torch.full((BATCH_SIZE,), expected))


def test_the_t2av_replay_carries_no_active_masks() -> None:
    rollout = _rollout(LTX2_T2AV_Adapter)

    step = rollout.adapter.get_replay_step(rollout.batch, 0)

    assert step.state.active_masks is None
    assert step.next_state.active_masks is None
    assert rollout.adapter.get_state_active_numel(step.state) == {
        "video": VIDEO_SEQ_LEN * CHANNELS,
        "audio": AUDIO_SEQ_LEN * CHANNELS,
    }


def _terminal_only_batch(cls: type) -> Any:
    kwargs = inference_kwargs(trajectory_indices=[-1])
    if cls is LTX2_I2AV_Adapter:
        kwargs["condition_images"] = condition_images()
    torch.manual_seed(SEED)
    adapter, _ = inference_adapter(cls)
    return adapter, SAMPLE_TYPES[cls].stack(adapter.inference(**kwargs))


@pytest.mark.parametrize("cls", ADAPTERS, ids=lambda cls: cls.__name__)
def test_a_terminal_only_rollout_still_replays_its_terminal_state(cls: type) -> None:
    adapter, batch = _terminal_only_batch(cls)

    terminal = adapter.get_terminal_state(batch)

    assert terminal.components["video"].shape == (BATCH_SIZE, VIDEO_SEQ_LEN, CHANNELS)
    assert terminal.components["audio"].shape == (BATCH_SIZE, AUDIO_SEQ_LEN, CHANNELS)


@pytest.mark.parametrize("cls", ADAPTERS, ids=lambda cls: cls.__name__)
def test_a_terminal_only_rollout_reports_the_uncollected_replay_positions(cls: type) -> None:
    adapter, batch = _terminal_only_batch(cls)

    with pytest.raises(ValueError, match=r"state_index_map.*uncollected.*-1"):
        adapter.get_replay_step(batch, 0)


def _replay_states(rollout: SimpleNamespace) -> List[LatentState]:
    return [
        rollout.adapter.get_replay_step(rollout.batch, step_index).state
        for step_index in STOCHASTIC_STEPS
    ]


def test_the_i2av_conditioning_tokens_are_identical_at_every_replay_position(
    i2av_rollout: SimpleNamespace,
) -> None:
    fixed = [state.components["video"][:, :FRAME_SEQ_LEN] for state in _replay_states(i2av_rollout)]

    for values in fixed[1:]:
        assert torch.equal(values, fixed[0])
