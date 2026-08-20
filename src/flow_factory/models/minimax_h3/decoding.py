# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decode target-only MiniMax H3 packed rows with pinned frozen blocks."""

from typing import Any, Mapping, Tuple

import torch

from ...samples import LatentState
from ._common import validate_target_state
from .blocks import run_h3_blocks
from .dependency import require_minimax_h3_support


def decode_h3_targets(
    pipeline: Any,
    target_state: LatentState,
    geometry: Mapping[str, int],
    *,
    output_type: str = "pil",
    workflow: str = "t2va",
) -> Tuple[Any, torch.Tensor, int]:
    """Decode B=1 target rows into video and stereo audio.

    Args:
        pipeline: Adapter-owned pipeline with materialized frozen VAEs.
        target_state: Generated target-only packed rows.
        geometry: Cached latent geometry.
        output_type: Upstream video output type.
        workflow: Workflow identifier used in diagnostics.

    Returns:
        Video output, stereo audio tensor, and sample rate.
    """
    validate_target_state(target_state)
    batch_size = target_state.components["video"].shape[0]
    if batch_size != 1:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} decode requires B=1, received B={batch_size}"
        )
    required = ("num_latent_frames", "latent_height", "latent_width", "num_audio_latents")
    missing = tuple(field for field in required if field not in geometry)
    if missing:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} decode geometry missing fields={missing}"
        )
    symbols = require_minimax_h3_support()
    values = {
        **geometry,
        "latents": target_state.components["video"].squeeze(0),
        "audio_latents": target_state.components["audio"].squeeze(0),
        "num_condition_video_rows": 0,
        "num_condition_audio_rows": 0,
        "output_type": output_type,
    }
    outputs = run_h3_blocks(
        pipeline,
        [
            symbols.AfterDenoiseStep(),
            symbols.VideoDecodeStep(),
            symbols.AudioDecodeStep(),
        ],
        values,
        requested_outputs=("videos", "audio", "sampling_rate"),
        workflow=workflow,
    )
    audio = outputs["audio"]
    sampling_rate = outputs["sampling_rate"]
    if not isinstance(audio, torch.Tensor) or audio.ndim != 3 or audio.shape[:2] != (1, 2):
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} decoded audio expected shape (1,2,samples), "
            f"received {getattr(audio, 'shape', None)}"
        )
    if not isinstance(sampling_rate, int) or isinstance(sampling_rate, bool) or sampling_rate <= 0:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} sampling_rate expected positive int, "
            f"received {sampling_rate!r}"
        )
    return outputs["videos"], audio, sampling_rate
