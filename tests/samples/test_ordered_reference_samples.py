# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Cover the generic ordered heterogeneous reference contract."""

import json

import pytest

from flow_factory.samples import Ref2AVSample
from flow_factory.samples.references import (
    canonicalize_reference_manifest,
    parse_reference_manifest,
)


def test_reference_manifest_preserves_order_and_canonicalizes_keys() -> None:
    references = [
        {"path": "style.png", "kind": "image"},
        {
            "kind": "video",
            "path": "motion.mp4",
            "fps": 24,
            "audio_path": "sound.wav",
            "sample_rate": 48000,
        },
        {"kind": "audio", "path": "ambience.wav", "sample_rate": 32000},
    ]

    manifest = canonicalize_reference_manifest(references, row_index=3)

    assert [entry["kind"] for entry in parse_reference_manifest(manifest, 3)] == [
        "image",
        "video",
        "audio",
    ]
    assert manifest == json.dumps(
        references,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def test_reference_sample_identity_includes_ordered_manifest() -> None:
    first = Ref2AVSample(
        prompt="animate",
        reference_manifest=canonicalize_reference_manifest(
            [{"kind": "image", "path": "first.png"}],
            0,
        ),
    )
    second = Ref2AVSample(
        prompt="animate",
        reference_manifest=canonicalize_reference_manifest(
            [{"kind": "image", "path": "second.png"}],
            0,
        ),
    )

    assert first.unique_id != second.unique_id


@pytest.mark.parametrize(
    "references,match",
    [
        ([], "non-empty"),
        ([{"kind": "audio", "path": "only.wav"}], "image or video"),
        ([{"kind": "image", "path": ""}], "non-empty string"),
        ([{"kind": "image", "path": "x.png", "fps": 24}], "unknown keys"),
        ([{"kind": "video", "path": "x.mp4", "fps": float("nan")}], "finite positive"),
        (
            [{"kind": "video", "path": "x.mp4", "sample_rate": 32000}],
            "requires audio_path",
        ),
    ],
)
def test_reference_manifest_rejects_invalid_entries(references: object, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        canonicalize_reference_manifest(references, row_index=7)
