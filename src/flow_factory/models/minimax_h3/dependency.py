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

"""Provide the single optional-import boundary for pinned MiniMax H3 support."""

import inspect
from dataclasses import dataclass
from typing import Any, Tuple, Type

import torch

MINIMAX_H3_DIFFUSERS_COMMIT = "f53d552036a0d1bd5570782a39cd40cfabf112bc"
MINIMAX_H3_INSTALL = (
    "pip install 'diffusers @ "
    "git+https://github.com/huggingface/diffusers.git@"
    f"{MINIMAX_H3_DIFFUSERS_COMMIT}'"
)
_WORKFLOWS = ("t2va", "fl2va", "ref2va")
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
    """Hold all pinned upstream classes used by the shared H3 core."""

    MiniMaxH3ModularPipeline: Type[Any]
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
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
    from diffusers.modular_pipelines.minimax_h3.references import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    _SYMBOLS = MiniMaxH3Symbols(
        MiniMaxH3ModularPipeline=MiniMaxH3ModularPipeline,
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
    """Return pinned H3 symbols or raise one actionable feature-probe error.

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

    pipeline_class = symbols.MiniMaxH3ModularPipeline
    workflow_map = getattr(pipeline_class, "_workflow_map", None)
    if not isinstance(workflow_map, dict) or any(
        workflow not in workflow_map for workflow in _WORKFLOWS
    ):
        raise ValueError(
            "MiniMaxH3ModularPipeline._workflow_map must declare t2va, fl2va, and ref2va"
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
        f"{detail}{cause_text}. Required exact diffusers commit "
        f"{MINIMAX_H3_DIFFUSERS_COMMIT}. Install with: {MINIMAX_H3_INSTALL}"
    )
