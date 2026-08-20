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

"""Execute frozen pinned MiniMax H3 modular blocks over shared state."""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from ...samples import LatentState
from .dependency import require_minimax_h3_support

WORKFLOWS = ("t2va", "fl2va", "ref2va")
LAYOUT_FIELDS = (
    "position_ids",
    "token_tags",
    "video_indices",
    "audio_indices",
    "text_indices",
    "num_condition_video_rows",
    "num_condition_audio_rows",
)
GEOMETRY_FIELDS = (
    "height",
    "width",
    "num_frames",
    "num_latent_frames",
    "latent_height",
    "latent_width",
    "num_audio_latents",
)
ENCODE_COMMON_FIELDS = (
    "prompt_embeds",
    "text_token_tags",
    *GEOMETRY_FIELDS,
    *LAYOUT_FIELDS,
)
ENCODE_WORKFLOW_FIELDS = {
    "t2va": (*ENCODE_COMMON_FIELDS, "keyframe_anchors"),
    "fl2va": (
        *ENCODE_COMMON_FIELDS,
        "keyframes",
        "keyframe_anchors",
        "condition_latents",
    ),
    "ref2va": (
        *ENCODE_COMMON_FIELDS,
        "normalized_references",
        "condition_latents",
        "audio_condition_latents",
    ),
}


def run_h3_blocks(
    pipeline: Any,
    blocks: Sequence[Any],
    values: Mapping[str, Any],
    *,
    requested_outputs: Sequence[str],
    workflow: str = "shared",
) -> Dict[str, Any]:
    """Run ordered upstream blocks and select promised state fields.

    Args:
        pipeline: Materialized adapter-owned modular pipeline.
        blocks: Ordered instantiated upstream blocks.
        values: Initial state values.
        requested_outputs: Fields promised to the caller.
        workflow: Workflow identifier used in diagnostics.

    Returns:
        Requested values preserving upstream object identity.
    """
    symbols = require_minimax_h3_support()
    state = symbols.PipelineState(values=dict(values))
    for block in blocks:
        result = block(pipeline, state)
        if isinstance(result, tuple) and len(result) == 2:
            pipeline, state = result
        else:
            state = result
    selected = {}
    for field in requested_outputs:
        if field not in state.values:
            raise ValueError(
                f"MiniMax H3 workflow={workflow!r} promised field={field!r}, "
                f"but block sequence produced fields={tuple(state.values)}"
            )
        selected[field] = state.values[field]
    return selected


def encode_h3_workflow_inputs(
    pipeline: Any,
    values: Mapping[str, Any],
    *,
    workflow: str,
) -> Dict[str, Any]:
    """Encode one H3 workflow with pinned frozen upstream blocks.

    Args:
        pipeline: Adapter-owned modular pipeline with frozen components.
        values: Raw workflow inputs.
        workflow: One of ``t2va``, ``fl2va``, or ``ref2va``.

    Returns:
        Cache-safe encoded and layout values.
    """
    symbols = require_minimax_h3_support()
    _validate_workflow(workflow)
    block_types = {
        "t2va": (
            symbols.TextEncoderStep,
            symbols.NoKeyframeAnchorsStep,
            symbols.PrepareLayoutStep,
        ),
        "fl2va": (
            symbols.ResizeStep,
            symbols.FL2VATextEncoderStep,
            symbols.KeyframeEncoderStep,
            symbols.PrepareLayoutStep,
        ),
        "ref2va": (
            symbols.RefSetupStep,
            symbols.Ref2VATextEncoderStep,
            symbols.ReferenceEncoderStep,
            symbols.RefPrepareLayoutStep,
        ),
    }[workflow]
    return run_h3_blocks(
        pipeline,
        [block_type() for block_type in block_types],
        values,
        requested_outputs=ENCODE_WORKFLOW_FIELDS[workflow],
        workflow=workflow,
    )


def prepare_h3_rollout_state(
    pipeline: Any,
    cached_values: Mapping[str, Any],
    *,
    workflow: str,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.Tensor] = None,
    audio_latents: Optional[torch.Tensor] = None,
) -> Tuple[LatentState, Dict[str, torch.Tensor]]:
    """Prepare target rows and immutable condition prefixes with pinned blocks.

    Args:
        pipeline: Adapter-owned modular pipeline.
        cached_values: Cached encoding and layout fields.
        workflow: H3 workflow identifier.
        generator: Shared ordered random generator.
        latents: Optional pre-generated video noise.
        audio_latents: Optional pre-generated audio noise.

    Returns:
        Target-only state and immutable video/audio prefixes.
    """
    symbols = require_minimax_h3_support()
    _validate_workflow(workflow)
    values = dict(cached_values)
    values.update(generator=generator, latents=latents, audio_latents=audio_latents)
    block_types = []
    if workflow != "t2va":
        block_types.append(symbols.PrepareConditionLatentsStep)
    block_types.append(symbols.PrepareLatentsStep)
    if workflow == "fl2va":
        block_types.append(symbols.FL2VAPrepareLatentsStep)
    elif workflow == "ref2va":
        block_types.append(symbols.Ref2VAPrepareLatentsStep)
    output = run_h3_blocks(
        pipeline,
        [block_type() for block_type in block_types],
        values,
        requested_outputs=("latents", "audio_latents"),
        workflow=workflow,
    )
    video_rows = _as_batched_rows(output["latents"], 96, workflow, "latents")
    audio_rows = _as_batched_rows(output["audio_latents"], 32, workflow, "audio_latents")
    video_count = _condition_count(cached_values, workflow, "video")
    audio_count = _condition_count(cached_values, workflow, "audio")
    _validate_rollout_row_counts(
        cached_values,
        workflow,
        "video",
        full_row_count=video_rows.shape[1],
        condition_row_count=video_count,
    )
    _validate_rollout_row_counts(
        cached_values,
        workflow,
        "audio",
        full_row_count=audio_rows.shape[1],
        condition_row_count=audio_count,
    )
    prefixes = {
        "video": video_rows[:, :video_count],
        "audio": audio_rows[:, :audio_count],
    }
    targets = LatentState(
        {
            "video": video_rows[:, video_count:],
            "audio": audio_rows[:, audio_count:],
        }
    )
    return targets, prefixes


def _validate_workflow(workflow: str) -> None:
    if workflow not in WORKFLOWS:
        raise ValueError(f"expected MiniMax H3 workflow in {WORKFLOWS}, received {workflow!r}")


def _condition_count(values: Mapping[str, Any], workflow: str, component: str) -> int:
    field = f"num_condition_{component}_rows"
    count = values.get(field, 0)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} field={field!r} expected non-negative int, "
            f"received {count!r}"
        )
    return count


def _validate_rollout_row_counts(
    values: Mapping[str, Any],
    workflow: str,
    component: str,
    *,
    full_row_count: int,
    condition_row_count: int,
) -> None:
    index_field = f"{component}_indices"
    indices = values.get(index_field)
    if not isinstance(indices, torch.Tensor) or indices.ndim != 1:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} component={component!r} field={index_field!r} "
            f"actual={getattr(indices, 'shape', type(indices).__name__)} "
            "expected=one-dimensional Tensor"
        )
    expected_full_rows = indices.numel()
    if full_row_count != expected_full_rows:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} component={component!r} "
            f"field='full_row_count' actual={full_row_count} expected={expected_full_rows}"
        )
    actual_target_rows = max(full_row_count - condition_row_count, 0)
    expected_target_rows = expected_full_rows - condition_row_count
    if actual_target_rows != expected_target_rows or expected_target_rows <= 0:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} component={component!r} "
            f"field='target_row_count' actual={actual_target_rows} "
            f"expected={expected_target_rows}"
        )


def _as_batched_rows(rows: torch.Tensor, width: int, workflow: str, field: str) -> torch.Tensor:
    if not isinstance(rows, torch.Tensor):
        raise TypeError(
            f"MiniMax H3 workflow={workflow!r} field={field!r} expected Tensor, "
            f"received {type(rows).__name__}"
        )
    if rows.ndim == 2:
        rows = rows.unsqueeze(0)
    if rows.ndim != 3 or rows.shape[0] != 1 or rows.shape[-1] != width:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} field={field!r} expected shape (B=1,N,{width}), "
            f"received {tuple(rows.shape)}"
        )
    return rows
