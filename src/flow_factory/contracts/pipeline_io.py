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

"""Dependency-neutral declarations for model pipeline inputs and outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class MediaType(str, Enum):
    """Media modalities understood by pipeline I/O contracts."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class RateRequirement(str, Enum):
    """Declare whether a modality-specific rate field is accepted or required."""

    NOT_APPLICABLE = "not_applicable"
    OPTIONAL = "optional"
    REQUIRED = "required"


class InputMediaBinding(str, Enum):
    """Describe how input media are bound to model-facing arguments."""

    GROUPED_BY_TYPE = "grouped_by_type"
    ORDERED_REFERENCES = "ordered_references"


class InputMediaOrder(str, Enum):
    """Describe which input-media ordering carries semantic meaning."""

    INSENSITIVE = "insensitive"
    WITHIN_TYPE = "within_type"
    GLOBAL = "global"


class NegativePromptPolicy(str, Enum):
    """Declare whether a pipeline accepts a negative prompt."""

    UNSUPPORTED = "unsupported"
    OPTIONAL = "optional"
    REQUIRED = "required"


class GeometrySource(str, Enum):
    """Declare where output geometry is resolved for one pipeline operation."""

    CONFIGURED = "configured"
    INPUT_MEDIA = "input_media"
    OUTPUT_MEDIA = "output_media"
    PRIMARY_OUTPUT_MEDIA = "primary_output_media"


class BatchCapability(str, Enum):
    """Describe the media-layout uniformity supported by a pipeline operation."""

    UNIFORM = "uniform"
    RAGGED = "ragged"
    SINGLE_SAMPLE = "single_sample"


@runtime_checkable
class DecodedMediaLike(Protocol):
    """Structural boundary for decoded media without importing a dataset type."""

    @property
    def type(self) -> str:
        """Return the public media type discriminator."""
        ...

    @property
    def payload(self) -> Any:
        """Return the decoded CPU-side media payload."""
        ...

    @property
    def fps(self) -> float | None:
        """Return source frames per second when applicable."""
        ...

    @property
    def sample_rate(self) -> int | None:
        """Return source samples per second when applicable."""
        ...


@runtime_checkable
class InputMediaLike(Protocol):
    """Structural media reference accepted by input-contract validation."""

    @property
    def type(self) -> str:
        """Return the public media type discriminator."""
        ...

    @property
    def fps(self) -> float | None:
        """Return an optional source frame rate."""
        ...

    @property
    def sample_rate(self) -> int | None:
        """Return an optional source sample rate."""
        ...


@runtime_checkable
class ModelInputLike(Protocol):
    """Structural normalized input consumed by a pipeline I/O contract."""

    @property
    def prompt(self) -> str:
        """Return the positive text prompt."""
        ...

    @property
    def negative_prompt(self) -> str | None:
        """Return the optional negative text prompt."""
        ...

    @property
    def media(self) -> tuple[InputMediaLike, ...]:
        """Return input media in public record order."""
        ...


@dataclass(frozen=True, slots=True)
class MediaFormat:
    """Declare one media modality and its rate-metadata requirements."""

    type: MediaType
    fps: RateRequirement
    sample_rate: RateRequirement

    def __post_init__(self) -> None:
        """Validate strict field types and modality-specific rate coherence."""
        _require_enum(self.type, MediaType, "type")
        _require_enum(self.fps, RateRequirement, "fps")
        _require_enum(self.sample_rate, RateRequirement, "sample_rate")

        if self.type is MediaType.IMAGE:
            if (
                self.fps is not RateRequirement.NOT_APPLICABLE
                or self.sample_rate is not RateRequirement.NOT_APPLICABLE
            ):
                raise ValueError("image media cannot declare fps or sample_rate requirements")
            return
        if self.type is MediaType.VIDEO:
            if self.fps is RateRequirement.NOT_APPLICABLE:
                raise ValueError("video media must declare fps as optional or required")
            if self.sample_rate is not RateRequirement.NOT_APPLICABLE:
                raise ValueError("video media cannot declare a sample_rate requirement")
            return
        if self.fps is not RateRequirement.NOT_APPLICABLE:
            raise ValueError("audio media cannot declare an fps requirement")
        if self.sample_rate is RateRequirement.NOT_APPLICABLE:
            raise ValueError("audio media must declare sample_rate as optional or required")


@dataclass(frozen=True, slots=True)
class InputMediaRule:
    """Declare the accepted count for one input media format."""

    format: MediaFormat
    min_count: int
    max_count: int | None

    def __post_init__(self) -> None:
        """Validate strict cardinality types and bounds."""
        _require_instance(self.format, MediaFormat, "format")
        _require_non_negative_int(self.min_count, "min_count")
        if self.max_count is not None:
            _require_non_negative_int(self.max_count, "max_count")
            if self.max_count == 0:
                raise ValueError("max_count=0 is not canonical; omit the media rule instead")
            if self.max_count < self.min_count:
                raise ValueError(
                    f"expected max_count >= min_count, received "
                    f"min_count={self.min_count} and max_count={self.max_count}"
                )


@dataclass(frozen=True, slots=True)
class InputMediaSpec:
    """Declare input-media types, counts, ordering, and argument binding."""

    rules: tuple[InputMediaRule, ...]
    binding: InputMediaBinding
    order: InputMediaOrder

    def __post_init__(self) -> None:
        """Validate a canonical and coherent input-media declaration."""
        _require_tuple(self.rules, InputMediaRule, "rules")
        _require_enum(self.binding, InputMediaBinding, "binding")
        _require_enum(self.order, InputMediaOrder, "order")

        media_types = tuple(rule.format.type for rule in self.rules)
        if len(set(media_types)) != len(media_types):
            raise ValueError("input media rules must contain each media type at most once")
        canonical_media_types = tuple(
            media_type for media_type in MediaType if media_type in media_types
        )
        if media_types != canonical_media_types:
            raise ValueError(
                "input media rules must use canonical type order "
                f"{canonical_media_types}, received {media_types}"
            )
        if not self.rules:
            if self.binding is not InputMediaBinding.GROUPED_BY_TYPE:
                raise ValueError("media-free inputs must use grouped_by_type binding")
            if self.order is not InputMediaOrder.INSENSITIVE:
                raise ValueError("media-free inputs must use insensitive ordering")
            return
        if self.binding is InputMediaBinding.ORDERED_REFERENCES:
            if self.order is not InputMediaOrder.GLOBAL:
                raise ValueError("ordered_references binding requires global input ordering")
        elif self.order is InputMediaOrder.GLOBAL:
            raise ValueError("global input ordering requires ordered_references binding")


@dataclass(frozen=True, slots=True)
class OutputMediaSequence:
    """Declare the exact ordered media sequence produced by a pipeline."""

    items: tuple[MediaFormat, ...]

    def __post_init__(self) -> None:
        """Validate a non-empty, immutable output-media sequence."""
        _require_tuple(self.items, MediaFormat, "items")
        if not self.items:
            raise ValueError("output media sequence must contain at least one item")


@dataclass(frozen=True, slots=True)
class PipelineIOContract:
    """Declare model-agnostic pipeline input and decoded-output semantics."""

    input_media: InputMediaSpec
    negative_prompt: NegativePromptPolicy
    output_media: OutputMediaSequence
    geometry_source: GeometrySource
    batch_capability: BatchCapability

    def __post_init__(self) -> None:
        """Validate strict types for every pipeline I/O declaration."""
        _require_instance(self.input_media, InputMediaSpec, "input_media")
        _require_enum(self.negative_prompt, NegativePromptPolicy, "negative_prompt")
        _require_instance(self.output_media, OutputMediaSequence, "output_media")
        _require_enum(self.geometry_source, GeometrySource, "geometry_source")
        _require_enum(self.batch_capability, BatchCapability, "batch_capability")
        if self.geometry_source is GeometrySource.INPUT_MEDIA and not any(
            rule.min_count > 0 for rule in self.input_media.rules
        ):
            raise ValueError(
                "input_media geometry requires at least one input media rule with min_count > 0"
            )


def validate_pipeline_model_input(
    model_input: ModelInputLike,
    contract: PipelineIOContract,
) -> None:
    """Validate one normalized input against model-declared pipeline semantics.

    The function depends only on structural protocols, so the dataset schema and
    model adapter remain independent. Callers must run it before preprocessing;
    otherwise an adapter may silently ignore unsupported conditioning media.
    """
    _require_instance(contract, PipelineIOContract, "contract")
    if not isinstance(model_input, ModelInputLike):
        raise TypeError(
            "expected model_input to implement ModelInputLike, received "
            f"{type(model_input).__name__}: {model_input!r}"
        )
    if type(model_input.prompt) is not str:
        raise TypeError(
            "expected model_input.prompt to be str, received "
            f"{type(model_input.prompt).__name__}: {model_input.prompt!r}"
        )
    negative_prompt = model_input.negative_prompt
    if negative_prompt is not None and type(negative_prompt) is not str:
        raise TypeError(
            "expected model_input.negative_prompt to be str or None, received "
            f"{type(negative_prompt).__name__}: {negative_prompt!r}"
        )
    if contract.negative_prompt is NegativePromptPolicy.UNSUPPORTED and negative_prompt is not None:
        raise ValueError("pipeline does not support negative_prompt")
    if contract.negative_prompt is NegativePromptPolicy.REQUIRED and negative_prompt is None:
        raise ValueError("pipeline requires negative_prompt")

    media = model_input.media
    if type(media) is not tuple:
        raise TypeError(
            "expected model_input.media to be tuple, received " f"{type(media).__name__}: {media!r}"
        )
    rules_by_type = {rule.format.type.value: rule for rule in contract.input_media.rules}
    counts = {media_type: 0 for media_type in rules_by_type}
    for index, item in enumerate(media):
        if not isinstance(item, InputMediaLike):
            raise TypeError(
                f"expected model_input.media[{index}] to implement InputMediaLike, "
                f"received {type(item).__name__}: {item!r}"
            )
        media_type = item.type
        if type(media_type) is not str:
            raise TypeError(
                f"expected model_input.media[{index}].type to be str, received "
                f"{type(media_type).__name__}: {media_type!r}"
            )
        rule = rules_by_type.get(media_type)
        if rule is None:
            raise ValueError(
                f"pipeline does not accept input media type {media_type!r} at index {index}; "
                f"accepted types={tuple(rules_by_type)!r}"
            )
        counts[media_type] += 1
        _validate_input_rate(item.fps, rule.format.fps, "fps", index)
        _validate_input_rate(
            item.sample_rate,
            rule.format.sample_rate,
            "sample_rate",
            index,
        )

    for media_type, rule in rules_by_type.items():
        count = counts[media_type]
        if count < rule.min_count:
            raise ValueError(
                f"pipeline requires at least {rule.min_count} input {media_type!r} item(s), "
                f"received {count}"
            )
        if rule.max_count is not None and count > rule.max_count:
            raise ValueError(
                f"pipeline accepts at most {rule.max_count} input {media_type!r} item(s), "
                f"received {count}"
            )


def _validate_input_rate(
    value: object,
    requirement: RateRequirement,
    rate_name: str,
    media_index: int,
) -> None:
    """Validate one normalized rate against its declared requirement."""
    if requirement is RateRequirement.NOT_APPLICABLE:
        if value is not None:
            raise ValueError(
                f"pipeline input media[{media_index}] does not accept {rate_name}, "
                f"received {value!r}"
            )
        return
    if value is None:
        if requirement is RateRequirement.REQUIRED:
            raise ValueError(f"pipeline input media[{media_index}] requires {rate_name}")
        return
    if rate_name == "fps":
        if type(value) is not float or not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"pipeline input media[{media_index}] requires finite positive fps, "
                f"received {value!r}"
            )
        return
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"pipeline input media[{media_index}] requires positive integer sample_rate, "
            f"received {value!r}"
        )


def _require_enum(value: object, enum_type: type[Enum], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(
            f"expected {field_name} to be {enum_type.__name__}, received "
            f"{type(value).__name__}: {value!r}"
        )


def _require_instance(value: object, expected_type: type[object], field_name: str) -> None:
    if type(value) is not expected_type:
        raise TypeError(
            f"expected {field_name} to be {expected_type.__name__}, received "
            f"{type(value).__name__}: {value!r}"
        )


def _require_tuple(value: object, item_type: type[object], field_name: str) -> None:
    if type(value) is not tuple:
        raise TypeError(
            f"expected {field_name} to be tuple, received {type(value).__name__}: {value!r}"
        )
    for index, item in enumerate(value):
        _require_instance(item, item_type, f"{field_name}[{index}]")


def _require_non_negative_int(value: object, field_name: str) -> None:
    if type(value) is not int:
        raise TypeError(
            f"expected {field_name} to be int, received {type(value).__name__}: {value!r}"
        )
    if value < 0:
        raise ValueError(f"expected {field_name} >= 0, received {value}")


__all__ = [
    "BatchCapability",
    "DecodedMediaLike",
    "GeometrySource",
    "InputMediaBinding",
    "InputMediaLike",
    "InputMediaOrder",
    "InputMediaRule",
    "InputMediaSpec",
    "MediaFormat",
    "MediaType",
    "ModelInputLike",
    "NegativePromptPolicy",
    "OutputMediaSequence",
    "PipelineIOContract",
    "RateRequirement",
    "validate_pipeline_model_input",
]
