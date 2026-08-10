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

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
import torch
from diffusers.utils.torch_utils import randn_tensor

from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter, LTX2I2AVSample
from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter, LTX2Sample
from flow_factory.samples import ComponentTimes, LatentState, MultiModalStepOutput
from flow_factory.scheduler import SDESchedulerOutput
from flow_factory.utils.noise_schedule import flow_match_sigma

BATCH_SIZE = 2
CHANNELS = 2
HEIGHT = 64
WIDTH = 96
NUM_FRAMES = 9
FRAME_RATE = 24.0
LATENT_F = 2
LATENT_H = 2
LATENT_W = 3
FRAME_SEQ_LEN = LATENT_H * LATENT_W
VIDEO_SEQ_LEN = LATENT_F * FRAME_SEQ_LEN
AUDIO_SEQ_LEN = 3
TEXT_SEQ_LEN = 4
TEXT_DIM = 8
TIMESTEP = 500.0
# The conditioning frame occupies the first packed frame; only the remaining
# frame is stepped, so I2AV's stochastic video DOF is one frame of tokens.
GENERATED_VIDEO_NUMEL = (VIDEO_SEQ_LEN - FRAME_SEQ_LEN) * CHANNELS


class SchedulerFake:
    """Deterministic scheduler twin recording every dispatch it receives."""

    def __init__(self, offset: float, log: List[Tuple[str, Any]]) -> None:
        self.offset = offset
        self.log = log
        self.steps: List[Dict[str, Any]] = []

    def step(
        self,
        *,
        velocity: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
        timestep_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        compute_log_prob: bool = True,
        return_dict: bool = True,
        return_kwargs: Any = (),
        noise_level: Optional[float] = None,
    ) -> SDESchedulerOutput:
        """Return an affine transition whose statistics identify this twin."""
        self.steps.append({"latents": latents, "velocity": velocity, "timestep": timestep})
        batch_size = latents.shape[0]
        broadcast = (batch_size,) + (1,) * (latents.ndim - 1)
        return SDESchedulerOutput(
            next_latents=latents + self.offset * velocity,
            next_latents_mean=latents + 0.5 * self.offset * velocity,
            std_dev_t=torch.full(broadcast, self.offset),
            dt=torch.full(broadcast, -self.offset),
            log_prob=(
                velocity.reshape(batch_size, -1).mean(dim=1) * self.offset
                if compute_log_prob
                else None
            ),
            velocity=velocity if "velocity" in tuple(return_kwargs) else None,
        )

    def eval(self) -> None:
        """Record one eval dispatch."""
        self.log.append(("eval", self.offset))

    def train(self, mode: bool = True) -> None:
        """Record one train dispatch."""
        self.log.append(("train", self.offset))

    def rollout(self, mode: bool = True) -> None:
        """Record one rollout dispatch."""
        self.log.append(("rollout", self.offset))

    def set_seed(self, seed: int) -> None:
        """Record one seed dispatch."""
        self.log.append(("set_seed", self.offset))


class TransformerFake:
    """Joint video/audio transformer returning affine velocity predictions."""

    dtype = torch.float32

    def __init__(self) -> None:
        self.rope = SimpleNamespace(prepare_video_coords=self._video_coords)
        self.audio_rope = SimpleNamespace(prepare_audio_coords=self._audio_coords)

    def _video_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        fps: Optional[float] = None,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, 3, num_frames * height * width, device=device)

    def _audio_coords(
        self, batch_size: int, audio_num_frames: int, device: torch.device
    ) -> torch.Tensor:
        return torch.zeros(batch_size, 1, audio_num_frames, device=device)

    @contextmanager
    def cache_context(self, name: str) -> Iterator[None]:
        """Provide the diffusers cache-context interface."""
        yield

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-modality velocity predictions."""
        return hidden_states * 0.5 + 1.0, audio_hidden_states * -0.25 + 2.0


class PipelineFake:
    """LTX2 pipeline stand-in exposing the geometry the forward pass reads."""

    vae_spatial_compression_ratio = 32
    vae_temporal_compression_ratio = 8
    transformer_spatial_patch_size = 1
    transformer_temporal_patch_size = 1
    audio_sampling_rate = 16384
    audio_hop_length = 256
    audio_vae_temporal_compression_ratio = 8

    def __init__(self, scheduler: SchedulerFake, transformer: TransformerFake) -> None:
        self.scheduler = scheduler
        self.transformer = transformer

    def _unpack_latents(
        self,
        latents: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int,
        patch_size_t: int,
    ) -> torch.Tensor:
        batch_size, seq_len, channels = latents.shape
        if seq_len != num_frames * height * width:
            raise ValueError(
                f"expected packed sequence {num_frames * height * width}, received {seq_len}"
            )
        return latents.reshape(batch_size, num_frames, height, width, channels).permute(
            0, 4, 1, 2, 3
        )

    def _pack_latents(
        self, latents: torch.Tensor, patch_size: int, patch_size_t: int
    ) -> torch.Tensor:
        batch_size, channels = latents.shape[0], latents.shape[1]
        return latents.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, channels)


def _adapter(cls: type, dispatch_log: Optional[List[Tuple[str, Any]]] = None) -> Any:
    log: List[Tuple[str, Any]] = [] if dispatch_log is None else dispatch_log
    transformer = TransformerFake()
    adapter = object.__new__(cls)
    adapter.pipeline = PipelineFake(SchedulerFake(0.5, log), transformer)
    adapter.component_runtime = SimpleNamespace(get_component=lambda name: transformer)
    adapter.load_scheduler = lambda: SchedulerFake(0.25, log)
    adapter.scheduler_group = adapter.build_scheduler_group()
    return adapter


def _video_latents() -> torch.Tensor:
    return (
        torch.arange(BATCH_SIZE * VIDEO_SEQ_LEN * CHANNELS, dtype=torch.float32).reshape(
            BATCH_SIZE, VIDEO_SEQ_LEN, CHANNELS
        )
        / 10.0
    )


def _audio_latents() -> torch.Tensor:
    return (
        torch.arange(BATCH_SIZE * AUDIO_SEQ_LEN * CHANNELS, dtype=torch.float32).reshape(
            BATCH_SIZE, AUDIO_SEQ_LEN, CHANNELS
        )
        / 5.0
    )


def _conditioning_mask() -> torch.Tensor:
    mask = torch.zeros(BATCH_SIZE, VIDEO_SEQ_LEN)
    mask[:, :FRAME_SEQ_LEN] = 1.0
    return mask


def _state(*, masked: bool = False) -> LatentState:
    components = {"video": _video_latents(), "audio": _audio_latents()}
    if not masked:
        return LatentState(components)
    return LatentState(
        components,
        active_masks={
            "video": (~_conditioning_mask().bool()).unsqueeze(-1),
            "audio": torch.ones(BATCH_SIZE, AUDIO_SEQ_LEN, 1, dtype=torch.bool),
        },
    )


def _times(*, audio_timestep: Optional[float] = None) -> ComponentTimes:
    timestep = torch.full((BATCH_SIZE,), TIMESTEP)
    audio = timestep if audio_timestep is None else torch.full((BATCH_SIZE,), audio_timestep)
    return ComponentTimes(
        timestep={"video": timestep, "audio": audio},
        next_timestep={"video": torch.zeros(BATCH_SIZE), "audio": torch.zeros(BATCH_SIZE)},
    )


def _sample_kwargs() -> Dict[str, Any]:
    return {
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "frame_rate": FRAME_RATE,
        "video_seq_len": VIDEO_SEQ_LEN,
        "connector_prompt_embeds": torch.zeros(TEXT_SEQ_LEN, TEXT_DIM),
        "connector_audio_prompt_embeds": torch.zeros(TEXT_SEQ_LEN, TEXT_DIM),
        "connector_attention_mask": torch.ones(TEXT_SEQ_LEN),
    }


def _t2av_batch(video_seq_len: int = VIDEO_SEQ_LEN) -> Any:
    kwargs = _sample_kwargs()
    kwargs["video_seq_len"] = video_seq_len
    return LTX2Sample.stack([LTX2Sample(**kwargs) for _ in range(BATCH_SIZE)])


def _i2av_batch(*, with_conditioning_mask: bool = True) -> Any:
    mask = _conditioning_mask()
    samples = [
        LTX2I2AVSample(
            conditioning_mask=mask[index] if with_conditioning_mask else None,
            **_sample_kwargs(),
        )
        for index in range(BATCH_SIZE)
    ]
    return LTX2I2AVSample.stack(samples)


def _forward_kwargs() -> Dict[str, Any]:
    return {
        "connector_prompt_embeds": torch.zeros(BATCH_SIZE, TEXT_SEQ_LEN, TEXT_DIM),
        "connector_audio_prompt_embeds": torch.zeros(BATCH_SIZE, TEXT_SEQ_LEN, TEXT_DIM),
        "connector_attention_mask": torch.ones(BATCH_SIZE, TEXT_SEQ_LEN),
        "guidance_scale": 1.0,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "frame_rate": FRAME_RATE,
    }


@pytest.mark.parametrize("cls", [LTX2_T2AV_Adapter, LTX2_I2AV_Adapter])
def test_ltx2_adapters_declare_the_video_audio_component_order(cls: type) -> None:
    assert cls.trajectory_component_order == ("video", "audio")


@pytest.mark.parametrize("cls", [LTX2_T2AV_Adapter, LTX2_I2AV_Adapter])
def test_scheduler_group_pairs_twin_instances_behind_the_video_primary(cls: type) -> None:
    adapter = _adapter(cls)

    assert adapter.scheduler_group.names == ("video", "audio")
    assert adapter.scheduler_group.primary_name == "video"
    assert adapter.scheduler_group.primary is adapter.pipeline.scheduler
    assert adapter.scheduler_group["audio"] is adapter.audio_scheduler
    assert adapter.audio_scheduler is not adapter.scheduler


@pytest.mark.parametrize("cls", [LTX2_T2AV_Adapter, LTX2_I2AV_Adapter])
def test_scheduler_group_dispatches_video_before_audio(cls: type) -> None:
    dispatch_log: List[Tuple[str, Any]] = []
    adapter = _adapter(cls, dispatch_log)

    adapter.set_trajectory_seed(7)
    adapter.scheduler_group.eval()

    assert dispatch_log == [
        ("set_seed", 0.5),
        ("set_seed", 0.25),
        ("eval", 0.5),
        ("eval", 0.25),
    ]


@pytest.mark.parametrize("cls", [LTX2_T2AV_Adapter, LTX2_I2AV_Adapter])
def test_training_component_times_mirror_the_primary_coordinate(cls: type) -> None:
    adapter = _adapter(cls)
    primary = torch.tensor([1000.0, 250.0])
    rng_state = torch.get_rng_state()

    times = adapter.build_training_component_times(primary)

    assert torch.equal(torch.get_rng_state(), rng_state)
    assert tuple(times.timestep) == ("video", "audio")
    assert tuple(times.sigma) == ("video", "audio")
    assert torch.equal(times.timestep["video"], primary)
    assert torch.equal(times.timestep["audio"], primary)
    assert torch.equal(times.sigma["video"], flow_match_sigma(primary))
    assert torch.equal(times.sigma["audio"], flow_match_sigma(primary))
    assert torch.equal(times.next_timestep["audio"], torch.zeros_like(primary))
    assert torch.equal(times.next_sigma["video"], torch.zeros_like(primary))


@pytest.mark.parametrize("cls", [LTX2_T2AV_Adapter, LTX2_I2AV_Adapter])
def test_training_component_times_reject_a_non_batched_coordinate(cls: type) -> None:
    adapter = _adapter(cls)

    with pytest.raises(ValueError, match=r"primary_timesteps.*\(B,\).*\(2, 1\)"):
        adapter.build_training_component_times(torch.zeros(2, 1))


@pytest.mark.parametrize("cls", [LTX2_T2AV_Adapter, LTX2_I2AV_Adapter])
def test_forward_process_noise_draws_video_before_audio(cls: type) -> None:
    adapter = _adapter(cls)
    clean = _state()
    sigma = torch.ones(BATCH_SIZE)
    times = ComponentTimes(
        timestep={"video": sigma * 1000, "audio": sigma * 1000},
        next_timestep={"video": torch.zeros(BATCH_SIZE), "audio": torch.zeros(BATCH_SIZE)},
        sigma={"video": sigma, "audio": sigma},
        next_sigma={"video": torch.zeros(BATCH_SIZE), "audio": torch.zeros(BATCH_SIZE)},
    )

    noised = adapter.add_forward_process_noise(
        clean, times, generator=torch.Generator().manual_seed(11)
    )

    expected = torch.Generator().manual_seed(11)
    expected_video = randn_tensor(
        (BATCH_SIZE, VIDEO_SEQ_LEN, CHANNELS), generator=expected, dtype=torch.float32
    )
    expected_audio = randn_tensor(
        (BATCH_SIZE, AUDIO_SEQ_LEN, CHANNELS), generator=expected, dtype=torch.float32
    )
    assert torch.equal(noised.noise.components["video"], expected_video)
    assert torch.equal(noised.noise.components["audio"], expected_audio)
    assert tuple(noised.state.components) == ("video", "audio")


def test_t2av_forward_state_matches_the_legacy_concatenated_forward() -> None:
    adapter = _adapter(LTX2_T2AV_Adapter)
    state = _state()
    times = _times()

    output = adapter.forward_state(
        batch=_t2av_batch(),
        state=state,
        times=times,
        compute_log_prob=True,
        return_fields=("next_latents", "next_latents_mean", "std_dev_t", "dt", "log_prob"),
        guidance_scale=1.0,
    )
    legacy = adapter.forward(
        t=times.timestep["video"],
        t_next=times.next_timestep["video"],
        latents=torch.cat([state.components["video"], state.components["audio"]], dim=1),
        video_seq_len=VIDEO_SEQ_LEN,
        compute_log_prob=True,
        return_kwargs=["next_latents", "next_latents_mean", "std_dev_t", "dt", "log_prob"],
        **_forward_kwargs(),
    )

    assert isinstance(output, MultiModalStepOutput)
    assert output.next_state.component_names == ("video", "audio")
    assert torch.equal(
        torch.cat(
            [output.next_state.components["video"], output.next_state.components["audio"]],
            dim=1,
        ),
        legacy.next_latents,
    )
    assert torch.equal(
        torch.cat(
            [
                output.next_state_mean.components["video"],
                output.next_state_mean.components["audio"],
            ],
            dim=1,
        ),
        legacy.next_latents_mean,
    )
    assert torch.equal(output.log_prob, legacy.log_prob)
    assert torch.equal(output.std_dev_t["video"], legacy.std_dev_t)
    assert torch.equal(output.dt["video"], legacy.dt)


def test_t2av_component_log_probs_combine_into_the_legacy_joint_value() -> None:
    adapter = _adapter(LTX2_T2AV_Adapter)
    state = _state()

    output = adapter.forward_state(
        batch=_t2av_batch(),
        state=state,
        times=_times(),
        compute_log_prob=True,
        return_fields=("next_latents", "log_prob"),
        guidance_scale=1.0,
    )

    video_numel = VIDEO_SEQ_LEN * CHANNELS
    audio_numel = AUDIO_SEQ_LEN * CHANNELS
    expected = (
        output.component_log_probs["video"] * video_numel
        + output.component_log_probs["audio"] * audio_numel
    ) / (video_numel + audio_numel)
    assert torch.equal(output.log_prob, expected)
    assert not torch.equal(output.component_log_probs["video"], output.component_log_probs["audio"])


def test_t2av_forward_state_carries_component_velocity_and_masks() -> None:
    adapter = _adapter(LTX2_T2AV_Adapter)
    state = _state(masked=True)

    output = adapter.forward_state(
        batch=_t2av_batch(),
        state=state,
        times=_times(),
        compute_log_prob=False,
        return_fields=("next_latents", "velocity"),
        guidance_scale=1.0,
    )

    assert output.velocity.component_names == ("video", "audio")
    assert output.velocity.components["video"].shape == (BATCH_SIZE, VIDEO_SEQ_LEN, CHANNELS)
    assert tuple(output.next_state.active_masks) == ("video", "audio")
    assert torch.equal(output.next_state.active_masks["video"], state.active_masks["video"])


def test_t2av_forward_state_rejects_decoupled_component_timesteps() -> None:
    adapter = _adapter(LTX2_T2AV_Adapter)

    with pytest.raises(ValueError, match=r"timestep.*'video'.*'audio'.*equal"):
        adapter.forward_state(
            batch=_t2av_batch(),
            state=_state(),
            times=_times(audio_timestep=250.0),
            guidance_scale=1.0,
        )


def test_t2av_forward_state_rejects_a_video_sequence_length_mismatch() -> None:
    adapter = _adapter(LTX2_T2AV_Adapter)

    with pytest.raises(ValueError, match=r"video_seq_len.*11.*video component.*12"):
        adapter.forward_state(
            batch=_t2av_batch(video_seq_len=11),
            state=_state(),
            times=_times(),
            guidance_scale=1.0,
        )


def test_i2av_forward_state_preserves_the_conditioning_frame() -> None:
    adapter = _adapter(LTX2_I2AV_Adapter)
    state = _state(masked=True)

    output = adapter.forward_state(
        batch=_i2av_batch(),
        state=state,
        times=_times(),
        compute_log_prob=True,
        return_fields=("next_latents", "log_prob"),
        guidance_scale=1.0,
    )

    video_next = output.next_state.components["video"]
    assert torch.equal(video_next[:, :FRAME_SEQ_LEN], state.components["video"][:, :FRAME_SEQ_LEN])
    assert not torch.equal(
        video_next[:, FRAME_SEQ_LEN:], state.components["video"][:, FRAME_SEQ_LEN:]
    )
    assert tuple(output.next_state.active_masks) == ("video", "audio")


def test_i2av_component_statistics_broadcast_onto_the_packed_video_latents() -> None:
    adapter = _adapter(LTX2_I2AV_Adapter)

    output = adapter.forward_state(
        batch=_i2av_batch(),
        state=_state(masked=True),
        times=_times(),
        compute_log_prob=False,
        return_fields=("next_latents", "std_dev_t", "dt"),
        guidance_scale=1.0,
    )

    video_shape = (BATCH_SIZE, VIDEO_SEQ_LEN, CHANNELS)
    for statistic in (output.std_dev_t["video"], output.dt["video"]):
        assert statistic.shape == (BATCH_SIZE, 1, 1)
        assert torch.broadcast_shapes(statistic.shape, video_shape) == video_shape


def test_i2av_forward_state_requires_the_batch_conditioning_mask() -> None:
    adapter = _adapter(LTX2_I2AV_Adapter)

    with pytest.raises(ValueError, match=r"conditioning_mask"):
        adapter.forward_state(
            batch=_i2av_batch(with_conditioning_mask=False),
            state=_state(masked=True),
            times=_times(),
            guidance_scale=1.0,
        )


def test_i2av_forward_state_rejects_a_mask_that_disagrees_with_the_conditioning_mask() -> None:
    adapter = _adapter(LTX2_I2AV_Adapter)
    state = LatentState(
        {"video": _video_latents(), "audio": _audio_latents()},
        active_masks={
            "video": torch.ones(BATCH_SIZE, VIDEO_SEQ_LEN, 1, dtype=torch.bool),
            "audio": torch.ones(BATCH_SIZE, AUDIO_SEQ_LEN, 1, dtype=torch.bool),
        },
    )

    with pytest.raises(ValueError, match=r"active_masks\['video'\].*conditioning_mask"):
        adapter.forward_state(
            batch=_i2av_batch(),
            state=state,
            times=_times(),
            guidance_scale=1.0,
        )


def test_i2av_forward_state_rejects_a_partially_inactive_audio_mask() -> None:
    adapter = _adapter(LTX2_I2AV_Adapter)
    audio_mask = torch.ones(BATCH_SIZE, AUDIO_SEQ_LEN, 1, dtype=torch.bool)
    audio_mask[:, 0] = False
    state = LatentState(
        {"video": _video_latents(), "audio": _audio_latents()},
        active_masks={
            "video": (~_conditioning_mask().bool()).unsqueeze(-1),
            "audio": audio_mask,
        },
    )

    with pytest.raises(ValueError, match=r"active_masks\['audio'\].*all active"):
        adapter.forward_state(
            batch=_i2av_batch(),
            state=state,
            times=_times(),
            guidance_scale=1.0,
        )


def test_i2av_active_numel_counts_only_the_generated_video_tokens() -> None:
    adapter = _adapter(LTX2_I2AV_Adapter)

    active = adapter.get_state_active_numel(_state(masked=True))

    assert active == {"video": GENERATED_VIDEO_NUMEL, "audio": AUDIO_SEQ_LEN * CHANNELS}


def test_i2av_reducers_ignore_the_conditioning_frame() -> None:
    adapter = _adapter(LTX2_I2AV_Adapter)
    state = _state(masked=True)
    video = torch.zeros(BATCH_SIZE, VIDEO_SEQ_LEN, CHANNELS)
    video[:, :FRAME_SEQ_LEN] = 1000.0
    video[:, FRAME_SEQ_LEN:] = 2.0
    audio = torch.full((BATCH_SIZE, AUDIO_SEQ_LEN, CHANNELS), 4.0)

    reduced = adapter.reduce_component_latent_values({"video": video, "audio": audio}, state=state)
    global_reduced = adapter.reduce_latent_values({"video": video, "audio": audio}, state=state)

    assert torch.equal(reduced["video"], torch.full((BATCH_SIZE,), 2.0))
    expected_global = (2.0 * GENERATED_VIDEO_NUMEL + 4.0 * AUDIO_SEQ_LEN * CHANNELS) / (
        GENERATED_VIDEO_NUMEL + AUDIO_SEQ_LEN * CHANNELS
    )
    assert torch.allclose(global_reduced, torch.full((BATCH_SIZE,), expected_global))


@pytest.mark.parametrize("cls", [LTX2_T2AV_Adapter, LTX2_I2AV_Adapter])
def test_legacy_concatenated_forward_stays_numerically_unchanged(cls: type) -> None:
    adapter = _adapter(cls)
    video = _video_latents()
    audio = _audio_latents()
    extra = {"conditioning_mask": _conditioning_mask()} if cls is LTX2_I2AV_Adapter else {}

    output = adapter.forward(
        t=torch.full((BATCH_SIZE,), TIMESTEP),
        t_next=torch.zeros(BATCH_SIZE),
        latents=torch.cat([video, audio], dim=1),
        video_seq_len=VIDEO_SEQ_LEN,
        compute_log_prob=True,
        return_kwargs=["next_latents", "log_prob", "velocity"],
        **extra,
        **_forward_kwargs(),
    )

    sigma = TIMESTEP / 1000
    video_velocity = (video - (video - (video * 0.5 + 1.0) * sigma)) / sigma
    audio_velocity = (audio - (audio - (audio * -0.25 + 2.0) * sigma)) / sigma
    expected_video = video + 0.5 * video_velocity
    if cls is LTX2_I2AV_Adapter:
        expected_video = torch.cat(
            [video[:, :FRAME_SEQ_LEN], expected_video[:, FRAME_SEQ_LEN:]], dim=1
        )
    expected_audio = audio + 0.25 * audio_velocity

    assert isinstance(output, SDESchedulerOutput)
    assert torch.allclose(output.next_latents, torch.cat([expected_video, expected_audio], dim=1))
    assert torch.allclose(output.velocity, torch.cat([video_velocity, audio_velocity], dim=1))
    assert output.log_prob.shape == (BATCH_SIZE,)
