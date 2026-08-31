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

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from flow_factory.data_utils.schema import (
    DatasetRecordV2,
    DemonstrationSpec,
    DemonstrationSupervision,
    PreferenceSupervision,
    normalize_v2_record,
)


def _demonstration_record(**overrides: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "schema_version": 2,
        "input": {"prompt": "A hill at sunset.", "media": []},
        "supervision": {
            "type": "demonstration",
            "target": {"media": [{"type": "image", "path": "target.png"}]},
        },
    }
    record.update(overrides)
    return record


def test_supervised_boundary_is_strict_and_normalized_record_is_frozen(tmp_path: Path) -> None:
    raw = _demonstration_record(metadata={"z": [2, 1], "a": {"text": "月亮"}})
    parsed = DatasetRecordV2.model_validate(raw)
    normalized = normalize_v2_record(parsed, dataset_dir=tmp_path / "dataset")

    assert isinstance(parsed.supervision, DemonstrationSpec)
    assert normalized.model_input.prompt == "A hill at sunset."
    assert normalized.model_input.media == ()
    assert isinstance(normalized.supervision, DemonstrationSupervision)
    assert normalized.metadata_json == '{"a":{"text":"月亮"},"z":[2,1]}'

    with pytest.raises(ValidationError):
        parsed.schema_version = 1
    with pytest.raises(FrozenInstanceError):
        normalized.metadata_json = "{}"


def test_supervision_is_required_at_the_public_v2_boundary() -> None:
    raw = _demonstration_record()
    raw.pop("supervision")

    with pytest.raises(ValidationError):
        DatasetRecordV2.model_validate(raw)


def test_demonstration_normalization_preserves_media_order_and_resolves_paths(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    absolute_target = tmp_path / "absolute" / "target.png"
    raw = _demonstration_record(
        input={
            "prompt": "Use the references in order.",
            "negative_prompt": "blurry",
            "media": [
                {"type": "image", "path": "images/first.png"},
                {"type": "video", "path": "videos/motion.mp4", "fps": 24},
                {"type": "audio", "path": "audios/voice.wav", "sample_rate": 16000},
            ],
        },
        supervision={
            "type": "demonstration",
            "target": {
                "media": [
                    {"type": "image", "path": str(absolute_target)},
                ]
            },
        },
    )

    normalized = normalize_v2_record(raw, dataset_dir=dataset_dir)

    assert [media.type for media in normalized.model_input.media] == [
        "image",
        "video",
        "audio",
    ]
    assert [media.path for media in normalized.model_input.media] == [
        str(dataset_dir / "images" / "first.png"),
        str(dataset_dir / "videos" / "motion.mp4"),
        str(dataset_dir / "audios" / "voice.wav"),
    ]
    assert normalized.model_input.media[1].fps == 24.0
    assert normalized.model_input.media[2].sample_rate == 16000
    assert isinstance(normalized.supervision, DemonstrationSupervision)
    assert normalized.supervision.target.media[0].path == str(absolute_target)


def test_input_media_slot_is_normalized_but_output_media_rejects_slots(
    tmp_path: Path,
) -> None:
    raw = _demonstration_record(
        input={
            "prompt": "End on this frame.",
            "media": [
                {
                    "type": "image",
                    "path": "ending.png",
                    "slot": "last_frame",
                }
            ],
        }
    )

    normalized = normalize_v2_record(raw, dataset_dir=tmp_path)

    assert normalized.model_input.media[0].slot == "last_frame"

    raw["supervision"]["target"]["media"][0]["slot"] = "last_frame"
    with pytest.raises(ValidationError, match="slot is input-only"):
        DatasetRecordV2.model_validate(raw)


def test_preference_normalization_keeps_both_arms_under_one_input(tmp_path: Path) -> None:
    raw = _demonstration_record(
        supervision={
            "type": "preference",
            "chosen": {
                "media": [
                    {"type": "video", "path": "chosen/result.mp4", "fps": 24},
                    {"type": "audio", "path": "chosen/result.mp4"},
                ]
            },
            "rejected": {
                "media": [
                    {"type": "video", "path": "rejected/result.mp4"},
                    {"type": "audio", "path": "rejected/result.mp4", "sample_rate": 48000},
                ]
            },
        }
    )

    normalized = normalize_v2_record(raw, dataset_dir=tmp_path / "dataset")

    assert isinstance(normalized.supervision, PreferenceSupervision)
    assert [media.type for media in normalized.supervision.chosen.media] == ["video", "audio"]
    assert [media.type for media in normalized.supervision.rejected.media] == ["video", "audio"]
    assert normalized.supervision.chosen.media[0].path == str(
        tmp_path / "dataset" / "chosen" / "result.mp4"
    )
    assert normalized.supervision.chosen.media[0].fps == 24.0
    assert normalized.supervision.rejected.media[1].path == str(
        tmp_path / "dataset" / "rejected" / "result.mp4"
    )
    assert normalized.supervision.rejected.media[1].sample_rate == 48000


def test_normalization_expands_dataset_root_and_normalizes_absolute_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    absolute_with_parent = str(tmp_path / "absolute" / ".." / "target.png")
    raw = _demonstration_record(
        input={
            "prompt": "normalize paths",
            "media": [
                {"type": "image", "path": "images/input.png"},
                {"type": "image", "path": absolute_with_parent},
            ],
        }
    )

    normalized = normalize_v2_record(raw, dataset_dir="~/dataset")

    assert normalized.model_input.media[0].path == str(
        tmp_path / "dataset" / "images" / "input.png"
    )
    assert normalized.model_input.media[1].path == str(tmp_path / "target.png")


@pytest.mark.parametrize(
    "media",
    [
        {"type": "image", "path": "image.png", "media_type": "image"},
        {"type": "image", "path": "image.png", "unknown": True},
    ],
)
def test_v2_media_rejects_unknown_keys(media: Dict[str, Any]) -> None:
    raw = _demonstration_record(input={"prompt": "strict", "media": [media]})

    with pytest.raises(ValidationError):
        DatasetRecordV2.model_validate(raw)


@pytest.mark.parametrize(
    "override",
    [
        {"unknown": True},
        {"input": {"prompt": "strict", "media": [], "unknown": True}},
        {
            "supervision": {
                "type": "demonstration",
                "target": {
                    "media": [{"type": "image", "path": "target.png"}],
                    "unknown": True,
                },
            }
        },
    ],
)
def test_v2_rejects_unknown_keys_at_every_public_level(override: Dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        DatasetRecordV2.model_validate(_demonstration_record(**override))


@pytest.mark.parametrize("path", ["", "   "])
def test_media_path_must_be_non_empty(path: str) -> None:
    raw = _demonstration_record(
        input={"prompt": "invalid path", "media": [{"type": "image", "path": path}]}
    )

    with pytest.raises(ValidationError):
        DatasetRecordV2.model_validate(raw)


@pytest.mark.parametrize(
    "media",
    [
        {"type": "video", "path": "clip.mp4", "fps": 0},
        {"type": "video", "path": "clip.mp4", "fps": float("inf")},
        {"type": "video", "path": "clip.mp4", "fps": "24"},
        {"type": "video", "path": "clip.mp4", "fps": True},
        {"type": "image", "path": "image.png", "fps": 24},
        {"type": "image", "path": "image.png", "fps": None},
        {"type": "audio", "path": "audio.wav", "fps": 24},
        {"type": "audio", "path": "audio.wav", "fps": None},
        {"type": "audio", "path": "audio.wav", "sample_rate": 0},
        {"type": "audio", "path": "audio.wav", "sample_rate": 16000.0},
        {"type": "audio", "path": "audio.wav", "sample_rate": "16000"},
        {"type": "audio", "path": "audio.wav", "sample_rate": True},
        {"type": "image", "path": "image.png", "sample_rate": 16000},
        {"type": "image", "path": "image.png", "sample_rate": None},
        {"type": "video", "path": "clip.mp4", "sample_rate": 16000},
        {"type": "video", "path": "clip.mp4", "sample_rate": None},
    ],
)
def test_media_rates_are_strict_positive_and_type_specific(media: Dict[str, Any]) -> None:
    raw = _demonstration_record(input={"prompt": "invalid rate", "media": [media]})

    with pytest.raises(ValidationError):
        DatasetRecordV2.model_validate(raw)


def test_video_fps_and_audio_sample_rate_are_optional_source_overrides() -> None:
    raw = _demonstration_record(
        input={
            "prompt": "source rates",
            "media": [
                {"type": "video", "path": "clip.mp4"},
                {"type": "audio", "path": "audio.wav"},
            ],
        }
    )

    parsed = DatasetRecordV2.model_validate(raw)

    assert parsed.input.media[0].fps is None
    assert parsed.input.media[1].sample_rate is None


@pytest.mark.parametrize("schema_version", [1, "2", True])
def test_schema_version_is_strictly_integer_two(schema_version: Any) -> None:
    with pytest.raises(ValidationError):
        DatasetRecordV2.model_validate(_demonstration_record(schema_version=schema_version))


@pytest.mark.parametrize("supervision_type", ["sft", "offline-dpo", "unknown"])
def test_supervision_uses_semantic_discriminator(supervision_type: str) -> None:
    raw = _demonstration_record(
        supervision={
            "type": supervision_type,
            "target": {"media": [{"type": "image", "path": "target.png"}]},
        }
    )

    with pytest.raises(ValidationError):
        DatasetRecordV2.model_validate(raw)


@pytest.mark.parametrize(
    "supervision",
    [
        {
            "type": "demonstration",
            "target": {"media": [{"type": "image", "path": "target.png"}]},
            "chosen": {"media": [{"type": "image", "path": "chosen.png"}]},
            "rejected": {"media": [{"type": "image", "path": "rejected.png"}]},
        },
        {
            "type": "preference",
            "chosen": {"media": [{"type": "image", "path": "chosen.png"}]},
            "rejected": {"media": [{"type": "image", "path": "rejected.png"}]},
            "target": {"media": [{"type": "image", "path": "target.png"}]},
        },
        {
            "type": "preference",
            "chosen": {"media": [{"type": "image", "path": "chosen.png"}]},
        },
    ],
)
def test_supervision_branches_cannot_be_mixed_or_incomplete(
    supervision: Dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        DatasetRecordV2.model_validate(_demonstration_record(supervision=supervision))


def test_output_candidate_requires_at_least_one_media_item() -> None:
    raw = _demonstration_record(supervision={"type": "demonstration", "target": {"media": []}})

    with pytest.raises(ValidationError):
        DatasetRecordV2.model_validate(raw)


@pytest.mark.parametrize("bad_value", [object(), float("nan"), float("inf")])
def test_metadata_accepts_only_finite_json_values(bad_value: Any) -> None:
    with pytest.raises(ValidationError):
        DatasetRecordV2.model_validate(_demonstration_record(metadata={"bad": bad_value}))
