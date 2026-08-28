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

"""Test reusable adapter pipeline-contract constructors."""

from flow_factory.contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaBinding,
    InputMediaOrder,
    InputMediaRule,
    MediaType,
    NegativePromptPolicy,
    RateRequirement,
)
from flow_factory.models.pipeline_contracts import (
    AUDIO_FORMAT_REQUIRED_RATE,
    IMAGE_FORMAT,
    VIDEO_FORMAT_OPTIONAL_FPS,
    audio_video_output_contract,
)


def test_audio_video_output_contract_defaults_to_exact_required_av_sequence() -> None:
    contract = audio_video_output_contract(
        negative_prompt=NegativePromptPolicy.UNSUPPORTED,
    )

    assert contract.input_media.rules == ()
    assert contract.input_media.binding is InputMediaBinding.GROUPED_BY_TYPE
    assert contract.input_media.order is InputMediaOrder.INSENSITIVE
    assert tuple(item.type for item in contract.output_media.items) == (
        MediaType.VIDEO,
        MediaType.AUDIO,
    )
    assert contract.output_media.items[0].fps is RateRequirement.REQUIRED
    assert contract.output_media.items[1].sample_rate is RateRequirement.REQUIRED
    assert contract.geometry_source is GeometrySource.OUTPUT_MEDIA
    assert contract.batch_capability is BatchCapability.SINGLE_SAMPLE


def test_audio_video_output_contract_preserves_grouped_image_bounds() -> None:
    contract = audio_video_output_contract(
        negative_prompt=NegativePromptPolicy.UNSUPPORTED,
        input_rules=(InputMediaRule(format=IMAGE_FORMAT, min_count=1, max_count=2),),
        input_order=InputMediaOrder.WITHIN_TYPE,
        geometry_source=GeometrySource.CONFIGURED,
    )

    assert contract.input_media.rules[0].min_count == 1
    assert contract.input_media.rules[0].max_count == 2
    assert contract.input_media.binding is InputMediaBinding.GROUPED_BY_TYPE
    assert contract.input_media.order is InputMediaOrder.WITHIN_TYPE
    assert contract.geometry_source is GeometrySource.CONFIGURED


def test_audio_video_output_contract_preserves_ordered_reference_rules() -> None:
    contract = audio_video_output_contract(
        negative_prompt=NegativePromptPolicy.UNSUPPORTED,
        input_rules=(
            InputMediaRule(format=IMAGE_FORMAT, min_count=0, max_count=12),
            InputMediaRule(format=VIDEO_FORMAT_OPTIONAL_FPS, min_count=0, max_count=12),
            InputMediaRule(format=AUDIO_FORMAT_REQUIRED_RATE, min_count=0, max_count=12),
        ),
        input_binding=InputMediaBinding.ORDERED_REFERENCES,
        input_order=InputMediaOrder.GLOBAL,
        output_fps=RateRequirement.OPTIONAL,
        output_sample_rate=RateRequirement.OPTIONAL,
        batch_capability=BatchCapability.RAGGED,
    )

    assert tuple(rule.format.type for rule in contract.input_media.rules) == (
        MediaType.IMAGE,
        MediaType.VIDEO,
        MediaType.AUDIO,
    )
    assert contract.input_media.binding is InputMediaBinding.ORDERED_REFERENCES
    assert contract.input_media.order is InputMediaOrder.GLOBAL
    assert contract.output_media.items[0].fps is RateRequirement.OPTIONAL
    assert contract.output_media.items[1].sample_rate is RateRequirement.OPTIONAL
    assert contract.batch_capability is BatchCapability.RAGGED
