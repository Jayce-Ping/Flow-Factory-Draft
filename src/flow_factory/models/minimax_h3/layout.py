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

"""Build independent H3 component schedules and packed-row timestep plans."""

from dataclasses import dataclass
from typing import Any, List, Mapping, Tuple

import torch

from ._common import framework_sigma_to_model_time


@dataclass(frozen=True)
class H3SchedulePlan:
    """Hold full component schedules and one row plan per transition."""

    schedules: Mapping[str, Tuple[torch.Tensor, torch.Tensor]]
    model_times: Mapping[str, torch.Tensor]
    row_timestep_plan: List[Tuple[torch.Tensor, torch.Tensor]]


def build_row_timesteps(
    layout: Mapping[str, Any],
    video_time: float,
    audio_time: float,
    keyframe_noise_aug: float,
    *,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign H3 clean time to every packed sequence row.

    Args:
        layout: Packed row indices and condition counts.
        video_time: Generated-video clean time.
        audio_time: Generated-audio clean time.
        keyframe_noise_aug: Visual condition clean-time floor.
        device: Execution device for returned tensors.

    Returns:
        Sorted unique times and per-row inverse indices.
    """
    video_indices, audio_indices, text_indices, video_count, audio_count = _validate_layout(layout)
    sequence_length = int(video_indices.numel() + audio_indices.numel() + text_indices.numel())
    row_times = torch.full((sequence_length,), float(video_time), dtype=torch.float32)
    row_times[video_indices[:video_count]] = max(float(video_time), float(keyframe_noise_aug))
    row_times[audio_indices[audio_count:]] = float(audio_time)
    row_times[audio_indices[:audio_count]] = 1.0
    unique, inverse = torch.unique(row_times, sorted=True, return_inverse=True)
    return unique.to(device), inverse.to(device)


def build_h3_schedule_plan(
    video_scheduler: Any,
    audio_scheduler: Any,
    num_inference_steps: int,
    layout: Mapping[str, Any],
    device: torch.device,
    *,
    keyframe_noise_aug: float = 0.999,
) -> H3SchedulePlan:
    """Set two N-transition schedulers and build all row-time plans.

    Args:
        video_scheduler: Resolved Flow-Factory H3 video scheduler.
        audio_scheduler: Resolved Flow-Factory H3 audio scheduler.
        num_inference_steps: Exact number of transitions.
        layout: Packed row layout.
        device: Transformer execution device.
        keyframe_noise_aug: Visual conditioning clean-time floor.

    Returns:
        Independent full schedules, clean times, and row plans.
    """
    if (
        not isinstance(num_inference_steps, int)
        or isinstance(num_inference_steps, bool)
        or num_inference_steps <= 0
    ):
        raise ValueError(
            "MiniMax H3 field='num_inference_steps' expected positive transition count, "
            f"received {num_inference_steps!r}"
        )
    for component, scheduler in (("video", video_scheduler), ("audio", audio_scheduler)):
        scheduler.set_timesteps(num_inference_steps, device=device)
        if (
            len(scheduler.timesteps) != num_inference_steps
            or len(scheduler.sigmas) != num_inference_steps + 1
        ):
            raise ValueError(
                f"MiniMax H3 component={component!r} expected N={num_inference_steps} transitions "
                f"and N+1 sigma points, received {len(scheduler.timesteps)} and {len(scheduler.sigmas)}"
            )
    schedules = {
        "video": (video_scheduler.sigmas * 1000, video_scheduler.sigmas),
        "audio": (audio_scheduler.sigmas * 1000, audio_scheduler.sigmas),
    }
    model_times = {
        component: framework_sigma_to_model_time(sigmas)
        for component, (_, sigmas) in schedules.items()
    }
    row_plan = [
        build_row_timesteps(
            layout,
            float(model_times["video"][index]),
            float(model_times["audio"][index]),
            keyframe_noise_aug,
            device=device,
        )
        for index in range(num_inference_steps)
    ]
    return H3SchedulePlan(schedules=schedules, model_times=model_times, row_timestep_plan=row_plan)


def _validate_layout(
    layout: Mapping[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    required = (
        "video_indices",
        "audio_indices",
        "text_indices",
        "num_condition_video_rows",
        "num_condition_audio_rows",
    )
    missing = tuple(field for field in required if field not in layout)
    if missing:
        raise ValueError(f"MiniMax H3 layout missing required fields={missing}")
    indices = []
    for field in required[:3]:
        value = layout[field]
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype != torch.long:
            raise ValueError(
                f"MiniMax H3 layout field={field!r} expected one-dimensional torch.long, "
                f"received {type(value).__name__}/{getattr(value, 'shape', None)}/"
                f"{getattr(value, 'dtype', None)}"
            )
        indices.append(value.cpu())
    counts = []
    for field, index_values in zip(required[3:], indices[:2]):
        count = layout[field]
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or not 0 <= count <= len(index_values)
        ):
            raise ValueError(
                f"MiniMax H3 layout field={field!r} expected int in [0,{len(index_values)}], "
                f"received {count!r}"
            )
        counts.append(count)
    all_indices = torch.cat(indices)
    expected = torch.arange(all_indices.numel())
    if not torch.equal(torch.sort(all_indices).values, expected):
        raise ValueError(
            "MiniMax H3 packed layout expected each sequence row exactly once, "
            f"received indices={all_indices.tolist()}"
        )
    return indices[0], indices[1], indices[2], counts[0], counts[1]
