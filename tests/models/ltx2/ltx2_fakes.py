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

"""Tiny LTX2 scheduler/transformer/pipeline fakes shared by the LTX2 tests.

The golden-oracle generator imports this module against the pre-Task-4A source
tree, so it must only use names that already existed there: keep the imports to
``torch`` and ``flow_factory.scheduler.SDESchedulerOutput``.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from flow_factory.scheduler import SDESchedulerOutput

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

VIDEO_SCHEDULER_OFFSET = 0.5
AUDIO_SCHEDULER_OFFSET = 0.25


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
        self.log.append(("step", self.offset))
        self.steps.append(
            {
                "latents": latents,
                "velocity": velocity,
                "timestep": timestep,
                "timestep_next": timestep_next,
                "next_latents": next_latents,
                "compute_log_prob": compute_log_prob,
                "return_kwargs": tuple(sorted(return_kwargs)),
                "noise_level": noise_level,
            }
        )
        batch_size = latents.shape[0]
        broadcast = (batch_size,) + (1,) * (latents.ndim - 1)
        requested = tuple(return_kwargs)
        return SDESchedulerOutput(
            next_latents=latents + self.offset * velocity,
            next_latents_mean=(
                latents + 0.5 * self.offset * velocity if "next_latents_mean" in requested else None
            ),
            std_dev_t=torch.full(broadcast, self.offset) if "std_dev_t" in requested else None,
            dt=torch.full(broadcast, -self.offset) if "dt" in requested else None,
            log_prob=(
                velocity.reshape(batch_size, -1).mean(dim=1) * self.offset
                if compute_log_prob
                else None
            ),
            velocity=velocity if "velocity" in requested else None,
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

    def __init__(self, noise_scale: float = 0.0) -> None:
        self.noise_scale = noise_scale
        self.calls: List[Dict[str, Any]] = []
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
        """Return per-modality velocity predictions.

        A positive ``noise_scale`` draws from the global RNG so a test can pin the
        number and order of random draws the forward pass performs.
        """
        self.calls.append({"hidden_states": hidden_states.shape})
        video = hidden_states * 0.5 + 1.0
        audio = audio_hidden_states * -0.25 + 2.0
        if self.noise_scale:
            video = video + self.noise_scale * torch.randn(hidden_states.shape)
            audio = audio + self.noise_scale * torch.randn(audio_hidden_states.shape)
        return video, audio


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


def video_latents() -> torch.Tensor:
    """Return the shared deterministic packed video latents."""
    return (
        torch.arange(BATCH_SIZE * VIDEO_SEQ_LEN * CHANNELS, dtype=torch.float32).reshape(
            BATCH_SIZE, VIDEO_SEQ_LEN, CHANNELS
        )
        / 10.0
    )


def audio_latents() -> torch.Tensor:
    """Return the shared deterministic packed audio latents."""
    return (
        torch.arange(BATCH_SIZE * AUDIO_SEQ_LEN * CHANNELS, dtype=torch.float32).reshape(
            BATCH_SIZE, AUDIO_SEQ_LEN, CHANNELS
        )
        / 5.0
    )


def conditioning_mask() -> torch.Tensor:
    """Return the I2AV conditioning mask pinning the first packed frame."""
    mask = torch.zeros(BATCH_SIZE, VIDEO_SEQ_LEN)
    mask[:, :FRAME_SEQ_LEN] = 1.0
    return mask


def forward_conditioning_kwargs() -> Dict[str, Any]:
    """Return the model-conditioning arguments the legacy forward reads."""
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
