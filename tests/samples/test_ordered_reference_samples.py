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

import json

import pytest

from flow_factory.samples import (
    I2AVSample,
    MiniMaxH3FL2VASample,
    MiniMaxH3Ref2VASample,
    MiniMaxH3T2VASample,
    OrderedReferenceConditionSample,
    Ref2AVSample,
    T2AVSample,
)
from flow_factory.samples.references import canonicalize_reference_manifest

REFERENCES = [
    {"kind": "image", "path": "subject.png"},
    {
        "kind": "video",
        "path": "motion.mp4",
        "fps": 29.97,
        "audio_path": "soundtrack.wav",
        "sample_rate": 48000,
    },
    {"kind": "audio", "path": "voice.wav", "sample_rate": 44100},
]


def test_canonical_manifest_preserves_order_and_rate_metadata() -> None:
    manifest = canonicalize_reference_manifest(REFERENCES, row_index=4)

    assert json.loads(manifest) == REFERENCES
    assert manifest == json.dumps(
        REFERENCES, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    assert canonicalize_reference_manifest(list(reversed(REFERENCES)), row_index=4) != manifest


@pytest.mark.parametrize(
    ("references", "match"),
    [
        ([{"kind": "depth", "path": "x"}], r"row 2.*reference 0.*kind.*depth"),
        ([{"kind": "image", "path": ""}], r"row 2.*reference 0.*path.*non-empty"),
        ([{"kind": "video", "path": "x", "fps": 0}], r"row 2.*reference 0.*fps.*positive"),
        (
            [{"kind": "audio", "path": "x", "sample_rate": 44100, "extra": 1}],
            r"row 2.*reference 0.*unknown keys.*extra",
        ),
        (
            [{"kind": "video", "path": "x", "sample_rate": 44100}],
            r"row 2.*reference 0.*sample_rate.*audio_path",
        ),
    ],
)
def test_manifest_validation_reports_row_and_reference(references: list[dict], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        canonicalize_reference_manifest(references, row_index=2)


def test_manifest_rejects_audio_only_with_row_and_reference_context() -> None:
    with pytest.raises(
        ValueError,
        match=r"row 7.*reference 0.*at least one image or video.*audio",
    ):
        canonicalize_reference_manifest(
            [{"kind": "audio", "path": "voice.wav", "sample_rate": 44100}],
            row_index=7,
        )


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    ("entry", "rate_name"),
    [
        ({"kind": "video", "path": "motion.mp4"}, "fps"),
        ({"kind": "audio", "path": "voice.wav"}, "sample_rate"),
    ],
)
def test_manifest_rejects_nonfinite_rates_with_context(
    entry: dict, rate_name: str, rate: float
) -> None:
    references = [
        {"kind": "image", "path": "subject.png"},
        {**entry, rate_name: rate},
    ]

    with pytest.raises(
        ValueError,
        match=rf"row 9.*reference 1.*{rate_name}.*finite positive.*{rate!r}",
    ):
        canonicalize_reference_manifest(references, row_index=9)


def test_model_samples_follow_exact_two_layer_hierarchy() -> None:
    assert MiniMaxH3T2VASample.__bases__ == (T2AVSample,)
    assert MiniMaxH3FL2VASample.__bases__ == (I2AVSample,)
    assert MiniMaxH3Ref2VASample.__bases__ == (Ref2AVSample,)
    assert Ref2AVSample.__bases__ == (OrderedReferenceConditionSample,)


def test_reference_manifest_changes_identity_and_stacks_as_list() -> None:
    manifest = canonicalize_reference_manifest(REFERENCES, row_index=0)
    reordered = canonicalize_reference_manifest(list(reversed(REFERENCES)), row_index=1)
    first = MiniMaxH3Ref2VASample(prompt="animate", reference_manifest=manifest)
    second = MiniMaxH3Ref2VASample(prompt="animate", reference_manifest=reordered)

    assert first.unique_id != second.unique_id
    assert "reference_manifest" not in MiniMaxH3Ref2VASample.shared_fields()
    assert MiniMaxH3Ref2VASample.stack([first, second]).reference_manifest == [
        manifest,
        reordered,
    ]
