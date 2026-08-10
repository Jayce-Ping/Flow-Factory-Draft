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
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from PIL import Image

from flow_factory.data_utils.dataset import GeneralDataset

REFERENCES = [
    {"kind": "image", "path": "subject.png"},
    {"kind": "video", "path": "motion.mp4", "fps": 29.97},
    {"kind": "audio", "path": "voice.wav", "sample_rate": 44100},
]


class OrderedPreprocessor:
    supports_ordered_references = True

    def __init__(self) -> None:
        self.received: List[List[Dict[str, Any]]] = []

    def preprocess(
        self,
        prompt: List[str],
        references: List[List[Dict[str, Any]]],
        workflow: str,
        width: int,
    ) -> Dict[str, Any]:
        self.received = references
        assert workflow == "ref2va"
        assert width == 768
        return {"encoded": torch.tensor([[len(references[0])]], dtype=torch.float32)}


def _write_jsonl(dataset_dir: Path, references: List[Dict[str, Any]]) -> None:
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.jsonl").write_text(
        json.dumps({"prompt": "animate", "references": references}) + "\n",
        encoding="utf-8",
    )


def test_ordered_references_round_trip_arrow_without_media_objects(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_jsonl(dataset_dir, REFERENCES)
    monkeypatch.setattr(
        "flow_factory.data_utils.dataset._load_ordered_reference",
        lambda entry, data_root: {
            **entry,
            "media": Image.new("RGB", (2, 2)) if entry["kind"] == "image" else torch.ones(1),
        },
    )
    preprocessor = OrderedPreprocessor()
    dataset = GeneralDataset(
        dataset_dir=str(dataset_dir),
        cache_dir=str(tmp_path / "cache"),
        preprocess_func=preprocessor.preprocess,
        preprocess_kwargs={"workflow": "ref2va", "width": 768},
        preprocessing_batch_size=1,
        force_reprocess=True,
    )

    assert len(preprocessor.received) == 1
    assert [entry["kind"] for entry in preprocessor.received[0]] == ["image", "video", "audio"]
    assert preprocessor.received[0][1]["fps"] == 29.97
    assert preprocessor.received[0][2]["sample_rate"] == 44100
    row = dataset[0]
    assert json.loads(row["reference_manifest"]) == REFERENCES
    assert all(value is not None for value in row.values())
    assert not any(isinstance(value, Image.Image) for value in row.values())
    collated = GeneralDataset.collate_fn([row])
    assert collated["reference_manifest"] == [row["reference_manifest"]]
    assert collated["encoded"].shape == (1, 1)


def test_ordered_reference_source_and_semantic_kwargs_change_cache_fingerprint(
    tmp_path: Path, monkeypatch
) -> None:
    first_dir = tmp_path / "same-name-a" / "dataset"
    second_dir = tmp_path / "same-name-b" / "dataset"
    _write_jsonl(first_dir, REFERENCES)
    reordered = list(reversed(REFERENCES))
    _write_jsonl(second_dir, reordered)
    preprocessor = OrderedPreprocessor()
    monkeypatch.setattr(
        "flow_factory.data_utils.dataset._load_ordered_reference",
        lambda entry, data_root: {**entry, "media": torch.ones(1)},
    )

    first = GeneralDataset(
        dataset_dir=str(first_dir),
        cache_dir=str(tmp_path / "cache"),
        preprocess_func=preprocessor.preprocess,
        preprocess_kwargs={"workflow": "ref2va", "width": 768},
        preprocessing_batch_size=1,
    )
    second = GeneralDataset(
        dataset_dir=str(second_dir),
        cache_dir=str(tmp_path / "cache"),
        preprocess_func=preprocessor.preprocess,
        preprocess_kwargs={"workflow": "ref2va", "width": 768},
        preprocessing_batch_size=1,
    )
    different_geometry = GeneralDataset.compute_cache_path(
        dataset_dir=str(first_dir),
        split="train",
        cache_dir=str(tmp_path / "cache"),
        max_dataset_size=None,
        preprocess_func=preprocessor.preprocess,
        preprocess_kwargs={"workflow": "ref2va", "width": 1024},
        extra_hash_strs=[first._ordered_reference_source_hash],
    )

    assert first.merged_cache_path != second.merged_cache_path
    assert first.merged_cache_path != different_geometry


def test_generic_media_preprocessing_does_not_enable_ordered_references(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_jsonl(dataset_dir, REFERENCES)

    def generic(prompt: List[str], images=None, videos=None, audios=None) -> Dict[str, Any]:
        assert images is None
        assert videos is None
        assert audios is None
        return {"encoded": torch.ones(len(prompt), 1)}

    dataset = GeneralDataset(
        dataset_dir=str(dataset_dir),
        cache_dir=str(tmp_path / "cache"),
        preprocess_func=generic,
        preprocessing_batch_size=1,
        force_reprocess=True,
    )

    assert "reference_manifest" not in dataset.processed_dataset.column_names
    assert dataset[0]["metadata"]["references"] == REFERENCES


def test_ordered_reference_preprocessing_rejects_outer_batch_larger_than_one(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    rows = [
        {"prompt": "first", "references": REFERENCES},
        {"prompt": "second", "references": REFERENCES},
    ]
    (dataset_dir / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    preprocessor = OrderedPreprocessor()
    monkeypatch.setattr(
        "flow_factory.data_utils.dataset._load_ordered_reference",
        lambda entry, data_root: {**entry, "media": torch.ones(1)},
    )
    with pytest.raises(ValueError, match=r"ordered-reference.*expected B=1.*received B=2"):
        GeneralDataset(
            dataset_dir=str(dataset_dir),
            cache_dir=str(tmp_path / "cache"),
            preprocess_func=preprocessor.preprocess,
            preprocess_kwargs={"workflow": "ref2va", "width": 768},
            preprocessing_batch_size=2,
            force_reprocess=True,
        )
