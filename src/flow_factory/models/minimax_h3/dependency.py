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

"""Provide the single optional-import boundary for MiniMax H3 support."""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Tuple, Type

import torch

MINIMAX_H3_DIFFUSERS_MIN_VERSION = "0.40.0"
MINIMAX_H3_INSTALL = f"pip install 'diffusers>={MINIMAX_H3_DIFFUSERS_MIN_VERSION}'"
_WORKFLOWS = ("t2va", "fl2va", "ref2va")
_WORKFLOW_TRIGGERS = {
    "t2va": {"prompt": True},
    "fl2va": (
        {"prompt": True, "image": True},
        {"prompt": True, "last_image": True},
    ),
    "ref2va": {"prompt": True, "references": True},
}
_BLOCK_FIELDS: Tuple[str, ...] = (
    "ResizeStep",
    "RefSetupStep",
    "TextEncoderStep",
    "FL2VATextEncoderStep",
    "Ref2VATextEncoderStep",
    "NoKeyframeAnchorsStep",
    "KeyframeEncoderStep",
    "ReferenceEncoderStep",
    "PrepareLayoutStep",
    "RefPrepareLayoutStep",
    "PrepareConditionLatentsStep",
    "PrepareLatentsStep",
    "FL2VAPrepareLatentsStep",
    "Ref2VAPrepareLatentsStep",
    "SetTimestepsStep",
    "AfterDenoiseStep",
    "VideoDecodeStep",
    "AudioDecodeStep",
)
_REFERENCE_FIELDS: Tuple[str, ...] = ("ImageReference", "VideoReference", "AudioReference")


@dataclass(frozen=True)
class MiniMaxH3Symbols:
    """Hold all upstream classes used by the shared H3 core."""

    ModularPipeline: Type[Any]
    MiniMaxH3ModularPipeline: Type[Any]
    MiniMaxH3Blocks: Type[Any]
    MiniMaxH3AttnProcessor: Type[Any]
    dispatch_attention_fn: Callable[..., torch.Tensor]
    PipelineState: Type[Any]
    ResizeStep: Type[Any]
    RefSetupStep: Type[Any]
    TextEncoderStep: Type[Any]
    FL2VATextEncoderStep: Type[Any]
    Ref2VATextEncoderStep: Type[Any]
    NoKeyframeAnchorsStep: Type[Any]
    KeyframeEncoderStep: Type[Any]
    ReferenceEncoderStep: Type[Any]
    PrepareLayoutStep: Type[Any]
    RefPrepareLayoutStep: Type[Any]
    PrepareConditionLatentsStep: Type[Any]
    PrepareLatentsStep: Type[Any]
    FL2VAPrepareLatentsStep: Type[Any]
    Ref2VAPrepareLatentsStep: Type[Any]
    SetTimestepsStep: Type[Any]
    AfterDenoiseStep: Type[Any]
    VideoDecodeStep: Type[Any]
    AudioDecodeStep: Type[Any]
    ImageReference: Type[Any]
    VideoReference: Type[Any]
    AudioReference: Type[Any]


try:
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AttnProcessor
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3FL2VAPrepareLatentsStep,
        MiniMaxH3NoKeyframeAnchorsStep,
        MiniMaxH3PrepareConditionLatentsStep,
        MiniMaxH3PrepareLatentsStep,
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3Ref2VAPrepareLatentsStep,
        MiniMaxH3Ref2VAPrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
    )
    from diffusers.modular_pipelines.minimax_h3.before_encoder import (
        MiniMaxH3Ref2VASetupStep,
        MiniMaxH3ResizeStep,
    )
    from diffusers.modular_pipelines.minimax_h3.decoders import (
        MiniMaxH3AfterDenoiseStep,
        MiniMaxH3AudioDecodeStep,
        MiniMaxH3VideoDecodeStep,
    )
    from diffusers.modular_pipelines.minimax_h3.encoders import (
        MiniMaxH3FL2VATextEncoderStep,
        MiniMaxH3KeyframeVaeEncoderStep,
        MiniMaxH3Ref2VAReferenceEncoderStep,
        MiniMaxH3Ref2VATextEncoderStep,
        MiniMaxH3TextEncoderStep,
    )
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
    from diffusers.modular_pipelines.minimax_h3.references import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    from diffusers import ModularPipeline

    _SYMBOLS = MiniMaxH3Symbols(
        ModularPipeline=ModularPipeline,
        MiniMaxH3ModularPipeline=MiniMaxH3ModularPipeline,
        MiniMaxH3Blocks=MiniMaxH3Blocks,
        MiniMaxH3AttnProcessor=MiniMaxH3AttnProcessor,
        dispatch_attention_fn=dispatch_attention_fn,
        PipelineState=PipelineState,
        ResizeStep=MiniMaxH3ResizeStep,
        RefSetupStep=MiniMaxH3Ref2VASetupStep,
        TextEncoderStep=MiniMaxH3TextEncoderStep,
        FL2VATextEncoderStep=MiniMaxH3FL2VATextEncoderStep,
        Ref2VATextEncoderStep=MiniMaxH3Ref2VATextEncoderStep,
        NoKeyframeAnchorsStep=MiniMaxH3NoKeyframeAnchorsStep,
        KeyframeEncoderStep=MiniMaxH3KeyframeVaeEncoderStep,
        ReferenceEncoderStep=MiniMaxH3Ref2VAReferenceEncoderStep,
        PrepareLayoutStep=MiniMaxH3PrepareLayoutStep,
        RefPrepareLayoutStep=MiniMaxH3Ref2VAPrepareLayoutStep,
        PrepareConditionLatentsStep=MiniMaxH3PrepareConditionLatentsStep,
        PrepareLatentsStep=MiniMaxH3PrepareLatentsStep,
        FL2VAPrepareLatentsStep=MiniMaxH3FL2VAPrepareLatentsStep,
        Ref2VAPrepareLatentsStep=MiniMaxH3Ref2VAPrepareLatentsStep,
        SetTimestepsStep=MiniMaxH3SetTimestepsStep,
        AfterDenoiseStep=MiniMaxH3AfterDenoiseStep,
        VideoDecodeStep=MiniMaxH3VideoDecodeStep,
        AudioDecodeStep=MiniMaxH3AudioDecodeStep,
        ImageReference=MiniMaxH3ImageReference,
        VideoReference=MiniMaxH3VideoReference,
        AudioReference=MiniMaxH3AudioReference,
    )
    _IMPORT_ERROR = None
except ImportError as import_error:
    _SYMBOLS = None
    _IMPORT_ERROR = import_error


def require_minimax_h3_support() -> MiniMaxH3Symbols:
    """Return H3 symbols or raise one actionable feature-probe error.

    Returns:
        Immutable bundle of required upstream symbols.
    """
    if _SYMBOLS is None:
        raise _feature_probe_error(
            "required symbols could not be imported", _IMPORT_ERROR
        ) from _IMPORT_ERROR
    try:
        _probe_symbol_bundle(_SYMBOLS)
    except (AttributeError, TypeError, ValueError) as probe_error:
        raise _feature_probe_error(str(probe_error), probe_error) from probe_error
    return _SYMBOLS


def _probe_symbol_bundle(symbols: MiniMaxH3Symbols) -> None:
    processor = symbols.MiniMaxH3AttnProcessor()
    if not callable(processor):
        raise TypeError("MiniMaxH3AttnProcessor instance must be callable")
    if not callable(symbols.dispatch_attention_fn):
        raise TypeError("dispatch_attention_fn must be callable")
    state_values = {"probe": object()}
    try:
        state = symbols.PipelineState(values=state_values)
    except TypeError as state_error:
        raise TypeError(
            "PipelineState must support construction as PipelineState(values=...)"
        ) from state_error
    if not hasattr(state, "values") or state.values != state_values:
        raise ValueError(
            "PipelineState must support PipelineState(values=...) and preserve a readable values mapping"
        )

    workflow_blocks_class = symbols.MiniMaxH3Blocks
    workflow_map = getattr(workflow_blocks_class, "_workflow_map", None)
    if not isinstance(workflow_map, dict):
        raise ValueError("MiniMaxH3Blocks._workflow_map must declare t2va, fl2va, and ref2va")
    for workflow in _WORKFLOWS:
        actual_triggers = workflow_map.get(workflow)
        expected_triggers = _WORKFLOW_TRIGGERS[workflow]
        if actual_triggers != expected_triggers:
            raise ValueError(
                "MiniMaxH3Blocks._workflow_map trigger mismatch for "
                f"{workflow}: expected {expected_triggers!r}, received {actual_triggers!r}"
            )

    for field in _BLOCK_FIELDS:
        block_class = getattr(symbols, field)
        if not callable(block_class):
            raise TypeError(f"{field} expected a callable block class, received {block_class!r}")
        block = block_class()
        if not callable(block):
            raise TypeError(f"{field} instance must be callable as block(pipeline, state)")
        _probe_block_call_shape(block, field)

    for field in _REFERENCE_FIELDS:
        reference_class = getattr(symbols, field)
        if not callable(reference_class):
            raise TypeError(
                f"{field} expected a callable reference class, received {reference_class!r}"
            )

    build_row_timesteps = getattr(symbols.SetTimestepsStep, "build_row_timesteps", None)
    if not callable(build_row_timesteps):
        raise TypeError("SetTimestepsStep.build_row_timesteps expected a callable API")
    row_plan = build_row_timesteps(
        torch.tensor([2]),
        torch.tensor([1]),
        0,
        0,
        1,
        0.2,
        0.4,
        0.999,
        1.0,
    )
    if (
        not isinstance(row_plan, tuple)
        or len(row_plan) != 2
        or not all(isinstance(value, torch.Tensor) for value in row_plan)
    ):
        raise ValueError(
            "SetTimestepsStep.build_row_timesteps expected to return two torch.Tensor values"
        )


def _probe_block_call_shape(block: Any, field: str) -> None:
    try:
        signature = inspect.signature(block)
    except (TypeError, ValueError):
        return
    parameters = tuple(signature.parameters.values())
    if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    required_keyword_only = tuple(
        parameter
        for parameter in parameters
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
    )
    required_positional = tuple(
        parameter for parameter in positional if parameter.default is inspect.Parameter.empty
    )
    if len(positional) < 2 or len(required_positional) > 2 or required_keyword_only:
        raise TypeError(f"{field} must support the callable API block(pipeline, state)")


def _feature_probe_error(detail: str, cause: Any) -> ImportError:
    cause_text = "" if cause is None else f"; cause={type(cause).__name__}: {cause}"
    return ImportError(
        "MiniMax H3 feature probe failed: "
        f"{detail}{cause_text}. MiniMax H3 requires "
        f"diffusers>={MINIMAX_H3_DIFFUSERS_MIN_VERSION}. Install with: {MINIMAX_H3_INSTALL}"
    )
