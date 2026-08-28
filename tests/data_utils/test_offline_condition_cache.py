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

import gc
import json
import weakref
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from datasets import Dataset as HFDataset
from PIL import Image

from flow_factory.data_utils.dataset import GeneralDataset
from flow_factory.data_utils.offline_condition_cache import (
    build_offline_condition_cache,
    compute_offline_condition_source_hash,
    project_offline_condition_dataset,
)
from flow_factory.data_utils.offline_dataset import (
    OFFLINE_CONDITION_ID_COLUMN,
    compute_offline_condition_id,
)
from flow_factory.data_utils.schema import NormalizedDatasetRecord, normalize_v2_record


class CountingPreprocessor:
    def __init__(self) -> None:
        self.calls = 0
        self.received_kwargs: Dict[str, Any] = {}

    def preprocess(self, prompt: List[str], **kwargs: Any) -> Dict[str, torch.Tensor]:
        self.calls += 1
        self.received_kwargs = kwargs
        return {"prompt_embeds": torch.ones(len(prompt), 2)}


class GroupedPreprocessor:
    def __init__(self) -> None:
        self.images: Any = None
        self.videos: Any = None
        self.audios: Any = None

    def preprocess(
        self,
        prompt: List[str],
        images: Any = None,
        videos: Any = None,
        audios: Any = None,
    ) -> Dict[str, torch.Tensor]:
        self.images = images
        self.videos = videos
        self.audios = audios
        return {"encoded": torch.ones(len(prompt), 1)}


class OrderedPreprocessor:
    supports_ordered_references = True

    def __init__(self) -> None:
        self.references: Any = None

    def preprocess(
        self,
        prompt: List[str],
        references: List[List[Dict[str, Any]]],
    ) -> Dict[str, torch.Tensor]:
        self.references = references
        return {"encoded": torch.ones(len(prompt), 1)}


class CollisionPreprocessor:
    def preprocess(self, prompt: List[str]) -> Dict[str, List[str]]:
        return {OFFLINE_CONDITION_ID_COLUMN: ["adapter-owned"] * len(prompt)}


def _demonstration_record(
    dataset_dir: Path,
    *,
    input_media: List[Dict[str, Any]] | None = None,
    target_path: str = "target.png",
    metadata: Dict[str, Any] | None = None,
    negative_prompt: str | None = None,
) -> NormalizedDatasetRecord:
    return normalize_v2_record(
        {
            "schema_version": 2,
            "input": {
                "prompt": "condition prompt",
                "negative_prompt": negative_prompt,
                "media": input_media or [],
            },
            "supervision": {
                "type": "demonstration",
                "target": {"media": [{"type": "image", "path": target_path}]},
            },
            "metadata": metadata or {},
        },
        dataset_dir=dataset_dir,
    )


def _preference_record(dataset_dir: Path) -> NormalizedDatasetRecord:
    return normalize_v2_record(
        {
            "schema_version": 2,
            "input": {"prompt": "preference condition", "media": []},
            "supervision": {
                "type": "preference",
                "chosen": {"media": [{"type": "image", "path": "chosen-private.png"}]},
                "rejected": {"media": [{"type": "image", "path": "rejected-private.png"}]},
            },
            "metadata": {"annotator": "private"},
        },
        dataset_dir=dataset_dir,
    )


def test_target_and_metadata_changes_reuse_the_input_only_cache(tmp_path: Path) -> None:
    first = _demonstration_record(
        tmp_path,
        target_path="first-target-does-not-exist.png",
        metadata={"revision": 1},
    )
    second = _demonstration_record(
        tmp_path,
        target_path="second-target-does-not-exist.png",
        metadata={"revision": 2},
    )
    preprocessor = CountingPreprocessor()

    first_cache = build_offline_condition_cache(
        [first],
        source_name="demo-source",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=preprocessor.preprocess,
        preprocessing_batch_size=1,
    )
    second_cache = build_offline_condition_cache(
        [second],
        source_name="demo-source",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=preprocessor.preprocess,
        preprocessing_batch_size=1,
    )

    expected_condition_id = compute_offline_condition_id(
        first,
        index=0,
        source_name="demo-source",
    )
    assert preprocessor.calls == 1
    assert first_cache._fingerprint == second_cache._fingerprint
    assert first_cache.cache_files == second_cache.cache_files
    assert second_cache[0][OFFLINE_CONDITION_ID_COLUMN] == expected_condition_id
    assert "metadata" not in second_cache.column_names
    assert "metadata" not in second_cache[0]
    assert OFFLINE_CONDITION_ID_COLUMN not in preprocessor.received_kwargs
    assert "first-target-does-not-exist.png" not in repr(second_cache[0])
    assert "second-target-does-not-exist.png" not in repr(second_cache[0])


def test_condition_cache_does_not_retain_the_bound_preprocessor(tmp_path: Path) -> None:
    preprocessor = CountingPreprocessor()
    preprocessor_ref = weakref.ref(preprocessor)

    condition_cache = build_offline_condition_cache(
        [_demonstration_record(tmp_path)],
        source_name="lifecycle",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=preprocessor.preprocess,
        preprocessing_batch_size=1,
    )
    del preprocessor
    gc.collect()

    assert isinstance(condition_cache, HFDataset)
    assert not hasattr(condition_cache, "_preprocess_func")
    assert preprocessor_ref() is None


def test_projection_contains_only_input_and_identity_columns(tmp_path: Path) -> None:
    record = _demonstration_record(
        tmp_path,
        input_media=[{"type": "image", "path": "reference.png"}],
        target_path="private-target.png",
        metadata={"private": "metadata"},
        negative_prompt="bad quality",
    )

    projected = project_offline_condition_dataset(
        [record],
        source_name="source",
        ordered_references=False,
    )

    assert set(projected.column_names) == {
        "prompt",
        "negative_prompt",
        "images",
        OFFLINE_CONDITION_ID_COLUMN,
    }
    assert projected[0]["images"] == [str(tmp_path / "reference.png")]
    assert "private-target.png" not in repr(projected[0])


def test_preference_arms_never_enter_the_condition_projection(tmp_path: Path) -> None:
    projected = project_offline_condition_dataset(
        [_preference_record(tmp_path)],
        source_name="preference",
        ordered_references=False,
    )

    assert set(projected.column_names) == {"prompt", OFFLINE_CONDITION_ID_COLUMN}
    assert "chosen-private.png" not in repr(projected[0])
    assert "rejected-private.png" not in repr(projected[0])
    assert "annotator" not in repr(projected[0])


def test_condition_source_hash_is_order_sensitive() -> None:
    assert compute_offline_condition_source_hash(["first", "second"]) != (
        compute_offline_condition_source_hash(["second", "first"])
    )


def test_grouped_projection_preserves_per_modality_order_and_rate_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, color in (("first.png", (255, 0, 0)), ("second.png", (0, 255, 0))):
        Image.new("RGB", (2, 2), color=color).save(tmp_path / name)
    video_calls: List[tuple[str, float | None]] = []
    audio_calls: List[tuple[str, int | None]] = []

    def fake_load_video(path: str, fps: float | None = None) -> List[Image.Image]:
        video_calls.append((path, fps))
        value = 10 if path.endswith("first.mp4") else 20
        return [Image.new("RGB", (2, 2), color=(value, value, value))]

    def fake_load_audio(path: str, sample_rate: int | None = None) -> torch.Tensor:
        audio_calls.append((path, sample_rate))
        return torch.ones(1, 4)

    monkeypatch.setattr("flow_factory.data_utils.dataset.load_video_frames", fake_load_video)
    monkeypatch.setattr("flow_factory.data_utils.dataset.load_audio", fake_load_audio)
    record = _demonstration_record(
        tmp_path,
        input_media=[
            {"type": "image", "path": "first.png"},
            {"type": "video", "path": "first.mp4", "fps": 12.5},
            {"type": "audio", "path": "voice.wav", "sample_rate": 16000},
            {"type": "image", "path": "second.png"},
            {"type": "video", "path": "second.mp4"},
        ],
    )
    preprocessor = GroupedPreprocessor()

    build_offline_condition_cache(
        [record],
        source_name="grouped",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=preprocessor.preprocess,
        force_reprocess=True,
        preprocessing_batch_size=1,
    )

    assert [image.getpixel((0, 0)) for image in preprocessor.images[0]] == [
        (255, 0, 0),
        (0, 255, 0),
    ]
    assert video_calls == [
        (str(tmp_path / "first.mp4"), 12.5),
        (str(tmp_path / "second.mp4"), None),
    ]
    assert audio_calls == [(str(tmp_path / "voice.wav"), 16000)]
    assert len(preprocessor.videos[0]) == 2
    assert len(preprocessor.audios[0]) == 1


def test_ordered_heterogeneous_references_cross_arrow_as_canonical_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(tmp_path / "image.png")
    monkeypatch.setattr(
        "flow_factory.data_utils.dataset._decode_ordered_video",
        lambda path: (np.zeros((1, 2, 2, 3), dtype=np.uint8), 30.0, None, None),
    )
    monkeypatch.setattr(
        "flow_factory.data_utils.dataset._decode_ordered_audio",
        lambda path: (torch.zeros(1, 8), 22050),
    )
    record = _demonstration_record(
        tmp_path,
        input_media=[
            {"type": "image", "path": "image.png"},
            {"type": "video", "path": "video.mp4", "fps": 24.0},
            {"type": "audio", "path": "audio.wav", "sample_rate": 44100},
        ],
    )
    projected = project_offline_condition_dataset(
        [record],
        source_name="ordered",
        ordered_references=True,
    )
    raw_manifest = projected[0]["references"]
    raw_references = json.loads(raw_manifest)

    assert isinstance(raw_manifest, str)
    assert [set(reference) for reference in raw_references] == [
        {"kind", "path"},
        {"kind", "path", "fps"},
        {"kind", "path", "sample_rate"},
    ]
    assert [reference["kind"] for reference in raw_references] == [
        "image",
        "video",
        "audio",
    ]

    preprocessor = OrderedPreprocessor()
    cache = build_offline_condition_cache(
        [record],
        source_name="ordered",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=preprocessor.preprocess,
        force_reprocess=True,
    )
    loaded = preprocessor.references[0]

    assert set(loaded[0]) == {"kind", "path", "media"}
    assert set(loaded[1]) == {"kind", "path", "fps", "frames"}
    assert set(loaded[2]) == {"kind", "path", "sample_rate", "media"}
    assert cache[0][OFFLINE_CONDITION_ID_COLUMN] == compute_offline_condition_id(
        record,
        index=0,
        source_name="ordered",
    )


def test_preprocess_cannot_overwrite_reserved_condition_identity(tmp_path: Path) -> None:
    record = _demonstration_record(tmp_path)

    with pytest.raises(ValueError, match="collides with passthrough columns"):
        build_offline_condition_cache(
            [record],
            source_name="collision",
            dataset_dir=tmp_path,
            cache_dir=tmp_path / "cache",
            preprocess_func=CollisionPreprocessor().preprocess,
            force_reprocess=True,
            preprocessing_batch_size=1,
        )


def test_in_memory_general_dataset_requires_explicit_source_identity(tmp_path: Path) -> None:
    raw_dataset = HFDataset.from_dict({"prompt": ["hello"]})

    with pytest.raises(ValueError, match="source_hash_override is required"):
        GeneralDataset(
            dataset_dir=str(tmp_path),
            raw_dataset=raw_dataset,
            enable_preprocess=False,
        )


def test_file_backed_online_dataset_keeps_legacy_loading_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "datasets.config.HF_DATASETS_CACHE",
        str(tmp_path / "huggingface-cache"),
    )
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"prompt": "online prompt", "label": 7}) + "\n",
        encoding="utf-8",
    )
    preprocessor = CountingPreprocessor()

    dataset = GeneralDataset(
        dataset_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        preprocess_func=preprocessor.preprocess,
        force_reprocess=True,
        preprocessing_batch_size=1,
    )

    assert dataset[0]["prompt"] == "online prompt"
    assert dataset[0]["metadata"] == {"label": 7}
    assert preprocessor.calls == 1
