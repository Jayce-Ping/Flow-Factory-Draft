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

"""Strict public schema for version-two supervised offline dataset records.

This module owns only the JSON boundary and its model-agnostic normalized
representation. It deliberately does not parse legacy online-generation records,
decode media, or attach loader-owned identities such as source ids and row ids.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Literal, Mapping, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

MediaType = Literal["image", "video", "audio"]


class _StrictFrozenModel(BaseModel):
    """Base configuration shared by every public V2 schema object."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class _MediaRefBase(_StrictFrozenModel):
    """Shared path contract for the exact-key media variants."""

    path: str
    slot: str | None = None

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("media path must be a non-empty string")
        return value

    @field_validator("slot")
    @classmethod
    def _validate_slot(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("media slot must be a non-empty string when provided")
        return value


class ImageRef(_MediaRefBase):
    """Image asset with no rate fields."""

    type: Literal["image"]


class VideoRef(_MediaRefBase):
    """Video asset with an optional source-frame-rate override."""

    type: Literal["video"]
    fps: float | None = Field(default=None, gt=0)


class AudioRef(_MediaRefBase):
    """Audio asset with an optional source-sample-rate override."""

    type: Literal["audio"]
    sample_rate: int | None = Field(default=None, gt=0)


MediaRef = Annotated[
    Union[ImageRef, VideoRef, AudioRef],
    Field(discriminator="type"),
]


class InputSpec(_StrictFrozenModel):
    """Model input shared by the supervised offline algorithm families."""

    prompt: str
    negative_prompt: str | None = None
    media: List[MediaRef] = Field(default_factory=list)


class OutputCandidateSpec(_StrictFrozenModel):
    """One generated-output candidate, potentially containing several modalities."""

    media: List[MediaRef] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_input_only_slots(self) -> "OutputCandidateSpec":
        slotted_indices = tuple(
            index for index, media in enumerate(self.media) if media.slot is not None
        )
        if slotted_indices:
            raise ValueError(
                "media slot is input-only and cannot appear in an output candidate; "
                f"indices={slotted_indices!r}"
            )
        return self


class DemonstrationSpec(_StrictFrozenModel):
    """A single supervised target without naming a training algorithm."""

    type: Literal["demonstration"]
    target: OutputCandidateSpec


class PreferenceSpec(_StrictFrozenModel):
    """A pairwise preference sharing one record-level input."""

    type: Literal["preference"]
    chosen: OutputCandidateSpec
    rejected: OutputCandidateSpec


SupervisionSpec = Annotated[
    Union[DemonstrationSpec, PreferenceSpec],
    Field(discriminator="type"),
]


class DatasetRecordV2(_StrictFrozenModel):
    """Strict V2 JSONL record for demonstration or preference data."""

    schema_version: Literal[2]
    input: InputSpec
    supervision: SupervisionSpec
    metadata: Dict[str, JsonValue] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MediaAsset:
    """Normalized media reference with a dataset-root-resolved path."""

    type: MediaType
    path: str
    fps: float | None = None
    sample_rate: int | None = None
    slot: str | None = None


@dataclass(frozen=True, slots=True)
class NormalizedModelInput:
    """Immutable normalized generation conditions preserving media order."""

    prompt: str
    negative_prompt: str | None
    media: tuple[MediaAsset, ...]


@dataclass(frozen=True, slots=True)
class NormalizedOutputCandidate:
    """Immutable ordered output-media candidate."""

    media: tuple[MediaAsset, ...]


@dataclass(frozen=True, slots=True)
class DemonstrationSupervision:
    """Normalized demonstration supervision."""

    target: NormalizedOutputCandidate


@dataclass(frozen=True, slots=True)
class PreferenceSupervision:
    """Normalized pairwise preference supervision."""

    chosen: NormalizedOutputCandidate
    rejected: NormalizedOutputCandidate


NormalizedSupervision = Union[DemonstrationSupervision, PreferenceSupervision]


@dataclass(frozen=True, slots=True)
class NormalizedDatasetRecord:
    """Immutable supervised schema facts, excluding loader-owned identities."""

    model_input: NormalizedModelInput
    supervision: NormalizedSupervision
    metadata_json: str
    schema_version: int = 2


def normalize_v2_record(
    record: Mapping[str, Any] | DatasetRecordV2,
    *,
    dataset_dir: str | os.PathLike[str],
) -> NormalizedDatasetRecord:
    """Validate and normalize one public V2 record.

    Relative media paths are joined to the expanded dataset directory. Absolute
    paths remain absolute. Neither path form is required to exist at this schema
    boundary; decoding owns that check later.

    Args:
        record: Raw JSON-compatible mapping or an already validated V2 record.
        dataset_dir: Dataset root used for relative media paths.

    Returns:
        Immutable normalized schema facts with canonical JSON metadata.
    """
    parsed = (
        record if isinstance(record, DatasetRecordV2) else DatasetRecordV2.model_validate(record)
    )
    normalized_input = NormalizedModelInput(
        prompt=parsed.input.prompt,
        negative_prompt=parsed.input.negative_prompt,
        media=tuple(_normalize_media(media, dataset_dir) for media in parsed.input.media),
    )

    supervision: NormalizedSupervision
    if isinstance(parsed.supervision, DemonstrationSpec):
        supervision = DemonstrationSupervision(
            target=_normalize_candidate(parsed.supervision.target, dataset_dir)
        )
    else:
        supervision = PreferenceSupervision(
            chosen=_normalize_candidate(parsed.supervision.chosen, dataset_dir),
            rejected=_normalize_candidate(parsed.supervision.rejected, dataset_dir),
        )

    metadata_json = json.dumps(
        parsed.metadata,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return NormalizedDatasetRecord(
        model_input=normalized_input,
        supervision=supervision,
        metadata_json=metadata_json,
    )


def _normalize_candidate(
    candidate: OutputCandidateSpec,
    dataset_dir: str | os.PathLike[str],
) -> NormalizedOutputCandidate:
    return NormalizedOutputCandidate(
        media=tuple(_normalize_media(media, dataset_dir) for media in candidate.media)
    )


def _normalize_media(
    media: MediaRef,
    dataset_dir: str | os.PathLike[str],
) -> MediaAsset:
    expanded_path = os.path.expanduser(media.path)
    if os.path.isabs(expanded_path):
        path = os.path.normpath(expanded_path)
    else:
        expanded_dataset_dir = os.path.expanduser(os.fspath(dataset_dir))
        path = os.path.normpath(os.path.join(expanded_dataset_dir, expanded_path))
    return MediaAsset(
        type=media.type,
        path=path,
        fps=media.fps if isinstance(media, VideoRef) else None,
        sample_rate=media.sample_rate if isinstance(media, AudioRef) else None,
        slot=media.slot,
    )


__all__ = [
    "DatasetRecordV2",
    "AudioRef",
    "DemonstrationSpec",
    "DemonstrationSupervision",
    "ImageRef",
    "InputSpec",
    "MediaAsset",
    "MediaRef",
    "MediaType",
    "NormalizedDatasetRecord",
    "NormalizedModelInput",
    "NormalizedOutputCandidate",
    "NormalizedSupervision",
    "OutputCandidateSpec",
    "PreferenceSpec",
    "PreferenceSupervision",
    "SupervisionSpec",
    "VideoRef",
    "normalize_v2_record",
]
