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

"""Full T2AV/I2AV rollout parity against the pre-Task-4B legacy loop oracle."""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

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
from ltx2_inference_oracle import run_legacy_rollout

from flow_factory.models.ltx2._common import combine_modality_log_prob
from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter
from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter
from flow_factory.samples import StructuredTrajectory
from flow_factory.utils.noise_schedule import flow_match_sigma

ADAPTERS = [LTX2_T2AV_Adapter, LTX2_I2AV_Adapter]
LATENT_CALLBACK = "next_latents_mean"
STATISTIC_CALLBACK = "std_dev_t"
CAPTURED_CALLBACK = "noise_level"


def _kwargs(cls: type, **overrides: Any) -> Dict[str, Any]:
    kwargs = inference_kwargs(**overrides)
    if cls is LTX2_I2AV_Adapter:
        kwargs["condition_images"] = condition_images()
    return kwargs


def _run_pair(
    cls: type, *, explicit_generator_seed: Optional[int] = None, **overrides: Any
) -> SimpleNamespace:
    """Run the legacy oracle and the structured rollout from the same RNG seed."""
    kwargs = _kwargs(cls, **overrides)
    conditioned = cls is LTX2_I2AV_Adapter

    torch.manual_seed(SEED)
    oracle_adapter, oracle_log = inference_adapter(cls)
    oracle_generator = (
        None
        if explicit_generator_seed is None
        else torch.Generator().manual_seed(explicit_generator_seed)
    )
    oracle = run_legacy_rollout(
        oracle_adapter,
        conditioned=conditioned,
        generator=oracle_generator,
        **kwargs,
    )
    oracle_generator_rng = None if oracle_generator is None else oracle_generator.get_state()
    oracle_generator_draw = (
        None if oracle_generator is None else torch.randn(5, generator=oracle_generator)
    )
    oracle_rng = torch.get_rng_state()
    oracle_draw = torch.randn(5)

    torch.manual_seed(SEED)
    adapter, log = inference_adapter(cls)
    generator = (
        None
        if explicit_generator_seed is None
        else torch.Generator().manual_seed(explicit_generator_seed)
    )
    samples = adapter.inference(generator=generator, **kwargs)
    generator_rng = None if generator is None else generator.get_state()
    generator_draw = None if generator is None else torch.randn(5, generator=generator)
    rng = torch.get_rng_state()
    draw = torch.randn(5)

    return SimpleNamespace(
        conditioned=conditioned,
        oracle=oracle,
        oracle_adapter=oracle_adapter,
        oracle_log=oracle_log,
        oracle_generator_rng=oracle_generator_rng,
        oracle_generator_draw=oracle_generator_draw,
        oracle_rng=oracle_rng,
        oracle_draw=oracle_draw,
        samples=samples,
        adapter=adapter,
        log=log,
        generator_rng=generator_rng,
        generator_draw=generator_draw,
        rng=rng,
        draw=draw,
    )


@pytest.fixture(scope="module", params=ADAPTERS, ids=lambda cls: cls.__name__)
def rollout(request: Any) -> SimpleNamespace:
    """Return one legacy/structured rollout pair per LTX2 adapter."""
    return _run_pair(request.param)


@pytest.fixture(scope="module")
def i2av_rollout() -> SimpleNamespace:
    """Return the conditioned rollout pair the I2AV-only expectations read."""
    return _run_pair(LTX2_I2AV_Adapter)


def _stacked(rollout: SimpleNamespace) -> StructuredTrajectory:
    return StructuredTrajectory.stack([sample.trajectory for sample in rollout.samples])


def _dispatch_steps(log: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    return [entry for entry in log if entry[0] == "step"]


def test_the_structured_rollout_reproduces_the_legacy_scheduler_dispatch_log(
    rollout: SimpleNamespace,
) -> None:
    assert rollout.log == rollout.oracle_log
    assert len(_dispatch_steps(rollout.log)) == 2 * NUM_INFERENCE_STEPS
    assert len(rollout.adapter.pipeline.transformer.calls) == len(
        rollout.oracle_adapter.pipeline.transformer.calls
    )


def test_the_structured_rollout_consumes_the_same_rng_stream(
    rollout: SimpleNamespace,
) -> None:
    assert torch.equal(rollout.rng, rollout.oracle_rng)
    assert torch.equal(rollout.draw, rollout.oracle_draw)


@pytest.mark.parametrize("cls", ADAPTERS, ids=lambda cls: cls.__name__)
def test_public_inference_preserves_explicit_generator_and_decode_parity(cls: type) -> None:
    pair = _run_pair(
        cls,
        explicit_generator_seed=SEED + 1,
        decode_timestep=0.35,
        decode_noise_scale=0.2,
    )

    for index, sample in enumerate(pair.samples):
        assert torch.equal(sample.video, pair.oracle["video"][index])
        assert torch.equal(sample.audio, pair.oracle["audio"][index].reshape(1, -1))
    trajectory = _stacked(pair)
    terminal_index = int(trajectory.components["video"].state_index_map[-1].item())
    final_latents = torch.cat(
        [
            trajectory.components["video"].states[:, terminal_index],
            trajectory.components["audio"].states[:, terminal_index],
        ],
        dim=1,
    )
    assert torch.equal(final_latents, pair.oracle["final_latents"])
    assert torch.equal(pair.generator_rng, pair.oracle_generator_rng)
    assert torch.equal(pair.generator_draw, pair.oracle_generator_draw)
    assert torch.equal(pair.rng, pair.oracle_rng)
    assert torch.equal(pair.draw, pair.oracle_draw)
    assert pair.log == pair.oracle_log


def test_the_structured_rollout_decodes_the_legacy_media(rollout: SimpleNamespace) -> None:
    for index, sample in enumerate(rollout.samples):
        assert torch.equal(sample.video, rollout.oracle["video"][index])
        assert torch.equal(sample.audio, rollout.oracle["audio"][index].reshape(1, -1))
    assert (
        rollout.adapter.pipeline.vae.decode_calls
        == rollout.oracle_adapter.pipeline.vae.decode_calls
    )
    assert rollout.adapter.pipeline.freed_hooks == 1


def test_the_terminal_component_states_recombine_into_the_legacy_final_latents(
    rollout: SimpleNamespace,
) -> None:
    trajectory = _stacked(rollout)
    video = trajectory.components["video"]
    audio = trajectory.components["audio"]
    terminal = int(video.state_index_map[-1].item())

    recombined = torch.cat([video.states[:, terminal], audio.states[:, terminal]], dim=1)
    assert torch.equal(recombined, rollout.oracle["final_latents"])


def test_every_collected_state_matches_the_legacy_concatenated_collection(
    rollout: SimpleNamespace,
) -> None:
    trajectory = _stacked(rollout)
    states = torch.stack(rollout.oracle["collected_states"], dim=1)
    video_seq_len = rollout.oracle["video_seq_len"]

    assert torch.equal(trajectory.components["video"].states, states[:, :, :video_seq_len])
    assert torch.equal(trajectory.components["audio"].states, states[:, :, video_seq_len:])
    for name in ("video", "audio"):
        assert torch.equal(
            trajectory.components[name].state_index_map, rollout.oracle["state_index_map"]
        )


def test_the_component_schedules_extend_the_legacy_timesteps_with_the_terminal_zero(
    rollout: SimpleNamespace,
) -> None:
    trajectory = _stacked(rollout)
    legacy_timesteps = rollout.oracle["timesteps"]

    for name in ("video", "audio"):
        component = trajectory.components[name]
        assert component.timesteps.shape == (BATCH_SIZE, NUM_INFERENCE_STEPS + 1)
        for sample_index in range(BATCH_SIZE):
            assert torch.equal(component.timesteps[sample_index, :-1], legacy_timesteps)
            assert float(component.timesteps[sample_index, -1]) == 0.0
            assert torch.equal(
                component.sigmas[sample_index],
                flow_match_sigma(component.timesteps[sample_index]),
            )


def test_the_video_and_audio_schedules_are_stored_as_independent_tensors(
    rollout: SimpleNamespace,
) -> None:
    for sample in rollout.samples:
        video = sample.trajectory.components["video"]
        audio = sample.trajectory.components["audio"]
        assert torch.equal(video.timesteps, audio.timesteps)
        assert video.timesteps is not audio.timesteps
        assert video.sigmas is not audio.sigmas


def test_the_joint_log_probs_match_the_legacy_sparse_collection(
    rollout: SimpleNamespace,
) -> None:
    trajectory = _stacked(rollout)
    legacy = torch.stack(rollout.oracle["log_probs"], dim=1)

    assert torch.equal(trajectory.log_probs, legacy)
    # The final transition leaves the SDE window, so it stores no log probability.
    # The legacy loop published a dense identity map here even though only the
    # first two transitions were collected; the structured map is signed instead.
    assert torch.equal(trajectory.log_prob_index_map, torch.tensor([0, 1, -1]))
    assert trajectory.log_probs.shape == (BATCH_SIZE, NUM_INFERENCE_STEPS - 1)


def test_the_component_log_probs_recombine_into_the_stored_joint_log_prob(
    rollout: SimpleNamespace,
) -> None:
    trajectory = _stacked(rollout)
    n_video = GENERATED_VIDEO_NUMEL if rollout.conditioned else VIDEO_SEQ_LEN * CHANNELS

    recombined = combine_modality_log_prob(
        trajectory.component_log_probs["video"],
        trajectory.component_log_probs["audio"],
        n_video=n_video,
        n_audio=AUDIO_SEQ_LEN * CHANNELS,
    )
    assert torch.equal(recombined, trajectory.log_probs)
    assert not torch.equal(
        trajectory.component_log_probs["video"],
        trajectory.component_log_probs["audio"],
    )


@pytest.mark.parametrize("field", [LATENT_CALLBACK, "velocity"])
def test_the_latent_callbacks_split_the_legacy_callback_collection(
    rollout: SimpleNamespace, field: str
) -> None:
    trajectory = _stacked(rollout)
    legacy = rollout.oracle["callbacks"][field]
    video_seq_len = rollout.oracle["video_seq_len"]

    stored = trajectory.callbacks[field]
    assert torch.equal(stored["video"].values, legacy[:, :, :video_seq_len])
    assert torch.equal(stored["audio"].values, legacy[:, :, video_seq_len:])
    for name in ("video", "audio"):
        assert torch.equal(stored[name].index_map, rollout.oracle["callback_index_map"])


def test_only_the_latent_callback_fields_become_structured(rollout: SimpleNamespace) -> None:
    trajectory = _stacked(rollout)

    assert trajectory.callback_fields == (LATENT_CALLBACK, "velocity")


def test_the_non_latent_callbacks_stay_legacy_extra_kwargs(
    rollout: SimpleNamespace,
) -> None:
    # ``noise_level`` is captured through the legacy callback API; this test
    # intentionally does not define new indexing semantics for custom callbacks.
    legacy_statistic = rollout.oracle["callbacks"][STATISTIC_CALLBACK]
    legacy_captured = rollout.oracle["callbacks"][CAPTURED_CALLBACK]

    for index, sample in enumerate(rollout.samples):
        stored = sample.extra_kwargs[STATISTIC_CALLBACK]
        assert torch.equal(stored.reshape(-1), legacy_statistic[index].reshape(-1))
        assert sample.extra_kwargs[CAPTURED_CALLBACK] == legacy_captured[index]
        assert LATENT_CALLBACK not in sample.extra_kwargs
        assert "velocity" not in sample.extra_kwargs
        assert torch.equal(
            sample.extra_kwargs["callback_index_map"], rollout.oracle["callback_index_map"]
        )
        assert sample.extra_kwargs["duration_s"] == rollout.oracle["duration_s"]


def test_the_scheduler_statistic_callback_keeps_the_packed_component_rank(
    rollout: SimpleNamespace,
) -> None:
    # I2AV steps the video scheduler on unpacked frames, so the legacy statistic
    # carried a rank-5 layout that no longer described the packed latents it
    # accompanied. Both adapters now publish the packed per-sample rank.
    for sample in rollout.samples:
        assert sample.extra_kwargs[STATISTIC_CALLBACK].shape == (NUM_INFERENCE_STEPS, 1, 1)


def test_the_legacy_trajectory_fields_stay_unset_when_a_trajectory_is_stored(
    rollout: SimpleNamespace,
) -> None:
    for sample in rollout.samples:
        assert isinstance(sample.trajectory, StructuredTrajectory)
        assert sample.timesteps is None
        assert sample.all_latents is None
        assert sample.latent_index_map is None
        assert sample.log_probs is None
        assert sample.log_prob_index_map is None


def test_the_rollout_preserves_the_legacy_sample_metadata(rollout: SimpleNamespace) -> None:
    kwargs = _kwargs(type(rollout.adapter))

    for index, sample in enumerate(rollout.samples):
        assert sample.height == kwargs["height"]
        assert sample.width == kwargs["width"]
        assert sample.num_frames == kwargs["num_frames"]
        assert sample.frame_rate == kwargs["frame_rate"]
        assert sample.video_seq_len == rollout.oracle["video_seq_len"]
        assert sample.prompt == kwargs["prompt"][index]
        assert torch.equal(sample.prompt_ids, kwargs["prompt_ids"][index])
        assert torch.equal(sample.connector_prompt_embeds, kwargs["connector_prompt_embeds"][index])
        assert sample.audio_sample_rate == int(
            rollout.adapter.pipeline.vocoder.config.output_sampling_rate
        )


@pytest.mark.parametrize("cls", ADAPTERS, ids=lambda cls: cls.__name__)
def test_disabled_collection_returns_evaluation_only_samples(cls: type) -> None:
    torch.manual_seed(SEED)
    adapter, _ = inference_adapter(cls)

    samples = adapter.inference(**_kwargs(cls, trajectory_indices=None, compute_log_prob=False))

    for sample in samples:
        assert sample.trajectory is None
        assert sample.all_latents is None
        assert sample.log_probs is None
        assert sample.video is not None
        assert sample.audio is not None
        assert "callback_index_map" not in sample.extra_kwargs


@pytest.mark.parametrize("cls", ADAPTERS, ids=lambda cls: cls.__name__)
def test_a_terminal_only_request_stores_one_state_per_component(cls: type) -> None:
    torch.manual_seed(SEED)
    adapter, _ = inference_adapter(cls)

    samples = adapter.inference(**_kwargs(cls, trajectory_indices=[-1]))

    expected_map = torch.tensor([-1] * NUM_INFERENCE_STEPS + [0])
    for sample in samples:
        for name in ("video", "audio"):
            component = sample.trajectory.components[name]
            assert component.states.shape[0] == 1
            assert torch.equal(component.state_index_map, expected_map)
        assert sample.trajectory.log_probs is None
        assert sample.trajectory.callbacks is None


def test_the_i2av_rollout_stores_the_inverse_conditioning_mask(
    i2av_rollout: SimpleNamespace,
) -> None:
    legacy_mask = i2av_rollout.oracle["conditioning_mask"]

    for index, sample in enumerate(i2av_rollout.samples):
        expected = ~legacy_mask[index].bool()
        video_mask = sample.trajectory.components["video"].active_mask
        assert video_mask.dtype is torch.bool
        assert torch.equal(video_mask.reshape(VIDEO_SEQ_LEN), expected)
        assert int(video_mask.sum().item()) == VIDEO_SEQ_LEN - FRAME_SEQ_LEN
        assert bool(sample.trajectory.components["audio"].active_mask.all())
        assert torch.equal(sample.conditioning_mask, legacy_mask[index])


def test_the_t2av_rollout_stores_no_active_mask() -> None:
    pair = _run_pair(LTX2_T2AV_Adapter)

    for sample in pair.samples:
        for name in ("video", "audio"):
            assert sample.trajectory.components[name].active_mask is None


def test_the_i2av_conditioning_tokens_never_move_across_the_rollout(
    i2av_rollout: SimpleNamespace,
) -> None:
    for sample in i2av_rollout.samples:
        states = sample.trajectory.components["video"].states
        fixed = states[:, :FRAME_SEQ_LEN]
        assert torch.equal(fixed, fixed[:1].expand_as(fixed))
        moving = states[:, FRAME_SEQ_LEN:]
        assert not torch.equal(moving[0], moving[-1])
