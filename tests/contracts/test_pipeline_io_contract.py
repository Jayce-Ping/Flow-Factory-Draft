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

"""Tests for dependency-neutral pipeline I/O declarations."""

from dataclasses import FrozenInstanceError, dataclass

import pytest

from flow_factory.contracts import (
    BatchCapability,
    DecodedMediaLike,
    GeometrySource,
    InputMediaBinding,
    InputMediaOrder,
    InputMediaRule,
    InputMediaSpec,
    MediaFormat,
    MediaType,
    NegativePromptPolicy,
    OutputMediaSequence,
    PipelineIOContract,
    RateRequirement,
)

IMAGE_FORMAT = MediaFormat(
    type=MediaType.IMAGE,
    fps=RateRequirement.NOT_APPLICABLE,
    sample_rate=RateRequirement.NOT_APPLICABLE,
)


def _text_to_image_contract(
    negative_prompt: NegativePromptPolicy = NegativePromptPolicy.OPTIONAL,
) -> PipelineIOContract:
    return PipelineIOContract(
        input_media=InputMediaSpec(
            rules=(),
            binding=InputMediaBinding.GROUPED_BY_TYPE,
            order=InputMediaOrder.INSENSITIVE,
        ),
        negative_prompt=negative_prompt,
        output_media=OutputMediaSequence(items=(IMAGE_FORMAT,)),
        geometry_source=GeometrySource.CONFIGURED,
        batch_capability=BatchCapability.UNIFORM,
    )


def test_contract_distinguishes_sd35_and_flux1_negative_prompt_support() -> None:
    """Shared T2I media shapes do not hide model-specific text input policy."""
    sd35 = _text_to_image_contract()
    flux1 = _text_to_image_contract(NegativePromptPolicy.UNSUPPORTED)

    assert sd35 != flux1
    assert sd35.input_media.rules == ()
    assert sd35.negative_prompt is NegativePromptPolicy.OPTIONAL
    assert flux1.negative_prompt is NegativePromptPolicy.UNSUPPORTED
    assert tuple(item.type for item in sd35.output_media.items) == (MediaType.IMAGE,)


def test_contract_represents_flux1_kontext_grouped_single_image_input() -> None:
    """Kontext adds exactly one grouped image without changing output semantics."""
    contract = PipelineIOContract(
        input_media=InputMediaSpec(
            rules=(InputMediaRule(format=IMAGE_FORMAT, min_count=1, max_count=1),),
            binding=InputMediaBinding.GROUPED_BY_TYPE,
            order=InputMediaOrder.INSENSITIVE,
        ),
        negative_prompt=NegativePromptPolicy.UNSUPPORTED,
        output_media=OutputMediaSequence(items=(IMAGE_FORMAT,)),
        geometry_source=GeometrySource.CONFIGURED,
        batch_capability=BatchCapability.UNIFORM,
    )

    assert contract.input_media.rules[0].min_count == 1
    assert contract.input_media.rules[0].max_count == 1
    assert contract.geometry_source is GeometrySource.CONFIGURED


def test_contract_represents_ordered_multimodal_input_and_exact_av_output() -> None:
    """Future ordered-reference and aligned AV pipelines remain expressible."""
    video = MediaFormat(
        type=MediaType.VIDEO,
        fps=RateRequirement.REQUIRED,
        sample_rate=RateRequirement.NOT_APPLICABLE,
    )
    audio = MediaFormat(
        type=MediaType.AUDIO,
        fps=RateRequirement.NOT_APPLICABLE,
        sample_rate=RateRequirement.REQUIRED,
    )
    contract = PipelineIOContract(
        input_media=InputMediaSpec(
            rules=(
                InputMediaRule(format=IMAGE_FORMAT, min_count=0, max_count=None),
                InputMediaRule(format=video, min_count=0, max_count=None),
                InputMediaRule(format=audio, min_count=0, max_count=None),
            ),
            binding=InputMediaBinding.ORDERED_REFERENCES,
            order=InputMediaOrder.GLOBAL,
        ),
        negative_prompt=NegativePromptPolicy.UNSUPPORTED,
        output_media=OutputMediaSequence(items=(video, audio)),
        geometry_source=GeometrySource.PRIMARY_OUTPUT_MEDIA,
        batch_capability=BatchCapability.RAGGED,
    )

    assert tuple(item.type for item in contract.output_media.items) == (
        MediaType.VIDEO,
        MediaType.AUDIO,
    )
    assert contract.output_media.items[0].fps is RateRequirement.REQUIRED
    assert contract.output_media.items[1].sample_rate is RateRequirement.REQUIRED


def test_pipeline_contract_and_nested_values_are_frozen_and_hashable() -> None:
    """The declaration is deeply immutable without a serialization framework."""
    contract = _text_to_image_contract()

    with pytest.raises(FrozenInstanceError):
        contract.geometry_source = GeometrySource.OUTPUT_MEDIA  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        contract.input_media.order = InputMediaOrder.GLOBAL  # type: ignore[misc]
    assert hash(contract)


@dataclass
class _DecodedFixture:
    type: str
    payload: object
    fps: float | None
    sample_rate: int | None


def test_decoded_media_protocol_is_structural_and_serialization_independent() -> None:
    """Dataset-owned decoded objects need no contract inheritance or conversion."""
    media = _DecodedFixture(type="image", payload=object(), fps=None, sample_rate=None)

    assert isinstance(media, DecodedMediaLike)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {
                "type": "image",
                "fps": RateRequirement.NOT_APPLICABLE,
                "sample_rate": RateRequirement.NOT_APPLICABLE,
            },
            "expected type to be MediaType",
        ),
        (
            {
                "type": MediaType.IMAGE,
                "fps": "not_applicable",
                "sample_rate": RateRequirement.NOT_APPLICABLE,
            },
            "expected fps to be RateRequirement",
        ),
    ],
)
def test_media_format_rejects_raw_enum_values(kwargs: dict[str, object], match: str) -> None:
    """Public constructors do not coerce strings into contract enums."""
    with pytest.raises(TypeError, match=match):
        MediaFormat(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("count", [True, 1.0, "1"])
def test_input_media_rule_rejects_coercible_count_types(count: object) -> None:
    """Cardinality values must be exact integers and never bools or strings."""
    with pytest.raises(TypeError, match="expected min_count to be int"):
        InputMediaRule(format=IMAGE_FORMAT, min_count=count, max_count=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {
                "type": MediaType.IMAGE,
                "fps": RateRequirement.OPTIONAL,
                "sample_rate": RateRequirement.NOT_APPLICABLE,
            },
            "image media cannot declare fps or sample_rate requirements",
        ),
        (
            {
                "type": MediaType.VIDEO,
                "fps": RateRequirement.OPTIONAL,
                "sample_rate": RateRequirement.OPTIONAL,
            },
            "video media cannot declare a sample_rate requirement",
        ),
        (
            {
                "type": MediaType.AUDIO,
                "fps": RateRequirement.OPTIONAL,
                "sample_rate": RateRequirement.OPTIONAL,
            },
            "audio media cannot declare an fps requirement",
        ),
    ],
)
def test_media_format_rejects_rates_from_another_modality(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Each modality rejects rate fields belonging to another modality."""
    with pytest.raises(ValueError, match=match):
        MediaFormat(**kwargs)  # type: ignore[arg-type]


def test_media_format_requires_an_applicable_rate_policy_for_video_and_audio() -> None:
    """Rate-bearing modalities cannot leave their native rate unspecified."""
    with pytest.raises(ValueError, match="video media must declare fps"):
        MediaFormat(
            type=MediaType.VIDEO,
            fps=RateRequirement.NOT_APPLICABLE,
            sample_rate=RateRequirement.NOT_APPLICABLE,
        )
    with pytest.raises(ValueError, match="audio media must declare sample_rate"):
        MediaFormat(
            type=MediaType.AUDIO,
            fps=RateRequirement.NOT_APPLICABLE,
            sample_rate=RateRequirement.NOT_APPLICABLE,
        )


def test_input_media_spec_requires_tuple_and_unique_media_types() -> None:
    """Input rules are immutable and unambiguous by media type."""
    rule = InputMediaRule(format=IMAGE_FORMAT, min_count=0, max_count=1)
    with pytest.raises(TypeError, match="expected rules to be tuple"):
        InputMediaSpec(
            rules=[rule],  # type: ignore[arg-type]
            binding=InputMediaBinding.GROUPED_BY_TYPE,
            order=InputMediaOrder.INSENSITIVE,
        )
    with pytest.raises(ValueError, match="each media type at most once"):
        InputMediaSpec(
            rules=(rule, rule),
            binding=InputMediaBinding.GROUPED_BY_TYPE,
            order=InputMediaOrder.WITHIN_TYPE,
        )


def test_input_media_rules_require_canonical_type_order() -> None:
    """Equivalent grouped declarations have one stable ordering and hash."""
    video = MediaFormat(
        type=MediaType.VIDEO,
        fps=RateRequirement.OPTIONAL,
        sample_rate=RateRequirement.NOT_APPLICABLE,
    )

    with pytest.raises(ValueError, match="canonical type order"):
        InputMediaSpec(
            rules=(
                InputMediaRule(format=video, min_count=0, max_count=1),
                InputMediaRule(format=IMAGE_FORMAT, min_count=0, max_count=1),
            ),
            binding=InputMediaBinding.GROUPED_BY_TYPE,
            order=InputMediaOrder.WITHIN_TYPE,
        )


def test_input_media_rule_rejects_noncanonical_or_inverted_bounds() -> None:
    """A rule must accept at least one item and keep its count interval ordered."""
    with pytest.raises(ValueError, match="max_count=0 is not canonical"):
        InputMediaRule(format=IMAGE_FORMAT, min_count=0, max_count=0)
    with pytest.raises(ValueError, match="max_count >= min_count"):
        InputMediaRule(format=IMAGE_FORMAT, min_count=2, max_count=1)


@pytest.mark.parametrize(
    "binding,order,match",
    [
        (
            InputMediaBinding.ORDERED_REFERENCES,
            InputMediaOrder.WITHIN_TYPE,
            "ordered_references binding requires global",
        ),
        (
            InputMediaBinding.GROUPED_BY_TYPE,
            InputMediaOrder.GLOBAL,
            "global input ordering requires ordered_references",
        ),
    ],
)
def test_input_binding_and_order_must_be_coherent(
    binding: InputMediaBinding,
    order: InputMediaOrder,
    match: str,
) -> None:
    """Grouped arguments cannot silently lose global reference ordering."""
    with pytest.raises(ValueError, match=match):
        InputMediaSpec(
            rules=(InputMediaRule(format=IMAGE_FORMAT, min_count=0, max_count=None),),
            binding=binding,
            order=order,
        )


def test_media_free_input_has_one_canonical_binding_and_order() -> None:
    """Prompt-only pipelines reject meaningless reference binding declarations."""
    with pytest.raises(ValueError, match="media-free inputs must use grouped_by_type"):
        InputMediaSpec(
            rules=(),
            binding=InputMediaBinding.ORDERED_REFERENCES,
            order=InputMediaOrder.GLOBAL,
        )
    with pytest.raises(ValueError, match="media-free inputs must use insensitive"):
        InputMediaSpec(
            rules=(),
            binding=InputMediaBinding.GROUPED_BY_TYPE,
            order=InputMediaOrder.WITHIN_TYPE,
        )


def test_output_media_sequence_is_non_empty_and_strictly_tuple_typed() -> None:
    """An exact output sequence cannot be missing or represented by a mutable list."""
    with pytest.raises(ValueError, match="at least one item"):
        OutputMediaSequence(items=())
    with pytest.raises(TypeError, match="expected items to be tuple"):
        OutputMediaSequence(items=[IMAGE_FORMAT])  # type: ignore[arg-type]


def test_pipeline_contract_rejects_raw_policy_values_and_algorithm_shape_fields() -> None:
    """The I/O contract stays strict and excludes trajectory or latent layout concerns."""
    with pytest.raises(TypeError, match="expected negative_prompt to be NegativePromptPolicy"):
        PipelineIOContract(
            input_media=_text_to_image_contract().input_media,
            negative_prompt="optional",  # type: ignore[arg-type]
            output_media=OutputMediaSequence(items=(IMAGE_FORMAT,)),
            geometry_source=GeometrySource.CONFIGURED,
            batch_capability=BatchCapability.UNIFORM,
        )

    contract_fields = PipelineIOContract.__dataclass_fields__
    assert "trajectory_component_order" not in contract_fields
    assert "latent_axis" not in contract_fields
    assert "algorithm" not in contract_fields


def test_input_media_geometry_requires_a_guaranteed_input() -> None:
    """A conditional geometry source cannot rely on an optional-only input layout."""
    with pytest.raises(ValueError, match="requires at least one input media rule"):
        PipelineIOContract(
            input_media=InputMediaSpec(
                rules=(InputMediaRule(format=IMAGE_FORMAT, min_count=0, max_count=1),),
                binding=InputMediaBinding.GROUPED_BY_TYPE,
                order=InputMediaOrder.INSENSITIVE,
            ),
            negative_prompt=NegativePromptPolicy.OPTIONAL,
            output_media=OutputMediaSequence(items=(IMAGE_FORMAT,)),
            geometry_source=GeometrySource.INPUT_MEDIA,
            batch_capability=BatchCapability.UNIFORM,
        )
