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

# src/flow_factory/models/ltx2/_common.py
"""Shared helpers for the LTX2 (T2AV / I2AV) adapters."""
from __future__ import annotations

import torch


def combine_modality_log_prob(
    video_log_prob: torch.Tensor,
    audio_log_prob: torch.Tensor,
    n_video: int,
    n_audio: int,
) -> torch.Tensor:
    """Element-weighted mean of the per-step video/audio log-probs.

    LTX2 steps video and audio with two scheduler instances, each returning a
    per-sample log_prob already meaned over its own latent dims. Weighting by the
    element counts ``n_video`` / ``n_audio`` reproduces the mean that a single
    scheduler over the concatenated ``[video|audio]`` latent would produce (mean
    over all latent dims), so the joint log_prob keeps the same scale as the
    original video-only path.

    Args:
        video_log_prob: Per-sample video log_prob, shape ``(B,)``.
        audio_log_prob: Per-sample audio log_prob, shape ``(B,)``.
        n_video: Number of video latent elements per sample (the tensor passed to
            the video ``step()``).
        n_audio: Number of audio latent elements per sample.

    Returns:
        Per-sample joint log_prob, shape ``(B,)``.
    """
    total = n_video + n_audio
    return (video_log_prob * n_video + audio_log_prob * n_audio) / total
