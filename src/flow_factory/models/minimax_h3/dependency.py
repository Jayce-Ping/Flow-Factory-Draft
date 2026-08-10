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

from dataclasses import dataclass
from typing import Any, Type

MINIMAX_H3_DIFFUSERS_COMMIT = "f53d552036a0d1bd5570782a39cd40cfabf112bc"
MINIMAX_H3_INSTALL = (
    "pip install 'diffusers @ "
    "git+https://github.com/huggingface/diffusers.git@"
    f"{MINIMAX_H3_DIFFUSERS_COMMIT}'"
)


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
        raise ImportError(
            "MiniMax H3 support requires all t2va/fl2va/ref2va modular block APIs from "
            f"diffusers commit {MINIMAX_H3_DIFFUSERS_COMMIT}. Install with: {MINIMAX_H3_INSTALL}"
        ) from _IMPORT_ERROR
    return _SYMBOLS
