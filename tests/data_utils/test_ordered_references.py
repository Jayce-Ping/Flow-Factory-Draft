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
import wave
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
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
        height: int = 512,
        duration: float = 2.0,
        num_frames: int = 49,
        model_name_or_path: str = "MiniMaxAI/MiniMax-H3",
    ) -> Dict[str, Any]:
        self.received = references
        assert workflow == "ref2va"
        return {"encoded": torch.tensor([[len(references[0])]], dtype=torch.float32)}


def _write_jsonl(dataset_dir: Path, references: List[Dict[str, Any]]) -> None:
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.jsonl").write_text(
        json.dumps({"prompt": "animate", "references": references}) + "\n",
        encoding="utf-8",
    )


def _write_wav(path: Path, sample_rate: int) -> None:
    samples = (np.sin(np.linspace(0, np.pi * 4, sample_rate // 20)) * 12000).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def test_ordered_references_round_trip_real_media_and_merged_cache(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    references = [
        {"kind": "image", "path": "subject.png"},
        {"kind": "audio", "path": "voice.wav", "sample_rate": 22050},
    ]
    _write_jsonl(dataset_dir, references)
    Image.new("RGB", (4, 3), color=(12, 34, 56)).save(dataset_dir / "subject.png")
    _write_wav(dataset_dir / "voice.wav", sample_rate=22050)
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
    assert [entry["kind"] for entry in preprocessor.received[0]] == ["image", "audio"]
    assert preprocessor.received[0][0]["media"].size == (4, 3)
    assert preprocessor.received[0][1]["sample_rate"] == 22050
    assert preprocessor.received[0][1]["media"].shape[0] == 1
    row = dataset[0]
    assert json.loads(row["reference_manifest"]) == references
    assert all(value is not None for value in row.values())
    assert not any(isinstance(value, Image.Image) for value in row.values())
    merged_path = tmp_path / "merged"
    dataset.processed_dataset.save_to_disk(str(merged_path))
    reloaded = GeneralDataset.load_merged(str(merged_path))
    reloaded_row = reloaded[0]
    collated = GeneralDataset.collate_fn([reloaded_row])
    assert collated["reference_manifest"] == [row["reference_manifest"]]
    assert collated["encoded"].shape == (1, 1)


def test_video_reference_preserves_frames_fps_embedded_audio_and_rate(tmp_path: Path) -> None:
    av = pytest.importorskip("av", reason="PyAV is required by pinned MiniMax H3 video decode")
    dataset_dir = tmp_path / "dataset"
    references = [{"kind": "video", "path": "motion.mp4"}]
    _write_jsonl(dataset_dir, references)
    video_path = dataset_dir / "motion.mp4"

    try:
        with av.open(str(video_path), mode="w") as container:
            video_stream = container.add_stream("mpeg4", rate=12)
            video_stream.width = 8
            video_stream.height = 6
            video_stream.pix_fmt = "yuv420p"
            audio_stream = container.add_stream("aac", rate=16000)
            audio_stream.layout = "mono"
            for value in (20, 80, 140):
                frame = av.VideoFrame.from_ndarray(
                    np.full((6, 8, 3), value, dtype=np.uint8), format="rgb24"
                )
                for packet in video_stream.encode(frame):
                    container.mux(packet)
            waveform = np.zeros((1, 1600), dtype=np.int16)
            audio_frame = av.AudioFrame.from_ndarray(waveform, format="s16", layout="mono")
            audio_frame.sample_rate = 16000
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
            for stream in (video_stream, audio_stream):
                for packet in stream.encode():
                    container.mux(packet)
    except (ValueError, OSError, av.error.FFmpegError) as error:
        pytest.skip(f"installed PyAV lacks required MP4/AAC fixture codecs: {error}")

    preprocessor = OrderedPreprocessor()
    GeneralDataset(
        dataset_dir=str(dataset_dir),
        cache_dir=str(tmp_path / "cache"),
        preprocess_func=preprocessor.preprocess,
        preprocess_kwargs={"workflow": "ref2va", "width": 768},
        preprocessing_batch_size=1,
        force_reprocess=True,
    )

    video = preprocessor.received[0][0]
    assert video["frames"].shape == (3, 6, 8, 3)
    assert video["fps"] == pytest.approx(12.0)
    assert video["audio"].shape[0] == 1
    assert video["sample_rate"] == 16000


def test_reference_decode_error_has_row_reference_and_cause(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    rows = [
        {"prompt": "valid", "references": [{"kind": "image", "path": "valid.png"}]},
        {"prompt": "missing", "references": [{"kind": "image", "path": "missing.png"}]},
    ]
    (dataset_dir / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    Image.new("RGB", (2, 2)).save(dataset_dir / "valid.png")
    preprocessor = OrderedPreprocessor()

    with pytest.raises(
        ValueError,
        match=r"row 1.*reference 0.*image.*missing\.png",
    ) as caught:
        GeneralDataset(
            dataset_dir=str(dataset_dir),
            cache_dir=str(tmp_path / "cache"),
            preprocess_func=preprocessor.preprocess,
            preprocess_kwargs={"workflow": "ref2va", "width": 768},
            preprocessing_batch_size=1,
            force_reprocess=True,
        )

    assert isinstance(caught.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("workflow", "other-workflow"),
        ("duration", 3.0),
        ("num_frames", 73),
        ("model_name_or_path", "other/model"),
        ("width", 1024),
        ("height", 768),
    ],
)
def test_semantic_preprocess_argument_changes_cache_fingerprint(
    tmp_path: Path, field: str, changed: Any
) -> None:
    dataset_dir = tmp_path / f"dataset-{field}"
    _write_jsonl(dataset_dir, REFERENCES)
    preprocessor = OrderedPreprocessor()
    source = GeneralDataset(
        dataset_dir=str(dataset_dir),
        preprocess_func=preprocessor.preprocess,
        enable_preprocess=False,
    )
    kwargs = {
        "workflow": "ref2va",
        "duration": 2.0,
        "num_frames": 49,
        "model_name_or_path": "MiniMaxAI/MiniMax-H3",
        "width": 768,
        "height": 512,
    }
    changed_kwargs = {**kwargs, field: changed}

    def fingerprint(preprocess_kwargs: Dict[str, Any]) -> str:
        return GeneralDataset.compute_cache_path(
            dataset_dir=str(dataset_dir),
            split="train",
            cache_dir=str(tmp_path / "cache"),
            max_dataset_size=None,
            preprocess_func=preprocessor.preprocess,
            preprocess_kwargs=preprocess_kwargs,
            extra_hash_strs=[source._ordered_reference_source_hash],
        )

    assert fingerprint(kwargs) != fingerprint(changed_kwargs)


@pytest.mark.parametrize(
    "changed_references",
    [
        [
            {"kind": "image", "path": "subject.png"},
            {"kind": "video", "path": "motion.mp4", "fps": 24.0},
            {"kind": "audio", "path": "voice.wav", "sample_rate": 44100},
        ],
        [
            {"kind": "image", "path": "subject.png"},
            {"kind": "video", "path": "motion.mp4", "fps": 29.97},
            {"kind": "audio", "path": "voice.wav", "sample_rate": 48000},
        ],
        list(reversed(REFERENCES)),
    ],
)
def test_manifest_rates_and_order_change_source_fingerprint(
    tmp_path: Path, changed_references: List[Dict[str, Any]]
) -> None:
    first_dir = tmp_path / "first" / "dataset"
    second_dir = tmp_path / "second" / "dataset"
    _write_jsonl(first_dir, REFERENCES)
    _write_jsonl(second_dir, changed_references)
    preprocessor = OrderedPreprocessor()

    first = GeneralDataset(
        dataset_dir=str(first_dir),
        preprocess_func=preprocessor.preprocess,
        enable_preprocess=False,
    )
    second = GeneralDataset(
        dataset_dir=str(second_dir),
        preprocess_func=preprocessor.preprocess,
        enable_preprocess=False,
    )

    assert first._ordered_reference_source_hash != second._ordered_reference_source_hash


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
    tmp_path: Path,
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
    with pytest.raises(ValueError, match=r"ordered-reference.*expected B=1.*received B=2"):
        GeneralDataset(
            dataset_dir=str(dataset_dir),
            cache_dir=str(tmp_path / "cache"),
            preprocess_func=preprocessor.preprocess,
            preprocess_kwargs={"workflow": "ref2va", "width": 768},
            preprocessing_batch_size=2,
            force_reprocess=True,
        )
