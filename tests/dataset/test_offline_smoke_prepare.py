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
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import pytest
from PIL import Image

from dataset.offline_smoke.prepare import PENDING_REVISION, prepare_dataset
from dataset.offline_smoke.profiles import DATASET_REPO_IDS
from dataset.offline_smoke.validate import validate_dataset

PINNED_SHA = "a" * 40


def test_prepare_cli_supports_documented_module_execution() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "dataset.offline_smoke.prepare", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--algorithm" in result.stdout


def _write_lock(path: Path, *, pending: bool = False) -> None:
    revision = PENDING_REVISION if pending else PINNED_SHA
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "sft": {
                        "repo_id": DATASET_REPO_IDS["sft"],
                        "revision": revision,
                        "supervision_type": "demonstration",
                    },
                    "offline-dpo": {
                        "repo_id": DATASET_REPO_IDS["offline-dpo"],
                        "revision": revision,
                        "supervision_type": "preference",
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _record(algorithm: str, *, target: str, rejected: str) -> Dict[str, Any]:
    supervision: Dict[str, Any]
    if algorithm == "sft":
        supervision = {
            "type": "demonstration",
            "target": {"media": [{"type": "image", "path": target}]},
        }
    else:
        supervision = {
            "type": "preference",
            "chosen": {"media": [{"type": "image", "path": target}]},
            "rejected": {"media": [{"type": "image", "path": rejected}]},
        }
    return {
        "schema_version": 2,
        "input": {"prompt": "A deterministic smoke fixture.", "media": []},
        "supervision": supervision,
        "metadata": {"usage_tier": "smoke_only"},
    }


def _populate_snapshot(
    local_dir: Path,
    *,
    algorithm: str,
    profile_dir: str = "sd35-t2i",
    row_count: int = 4,
    target_path: str = "../../media/target.png",
    identical_pair: bool = False,
) -> None:
    media_dir = local_dir / "media"
    manifest_dir = local_dir / "profiles" / profile_dir
    media_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(media_dir / "target.png")
    rejected_color = (10, 20, 30) if identical_pair else (30, 20, 10)
    Image.new("RGB", (8, 8), color=rejected_color).save(media_dir / "rejected.png")
    rows = [
        _record(
            algorithm,
            target=target_path,
            rejected="../../media/rejected.png",
        )
        for _ in range(row_count)
    ]
    (manifest_dir / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.parametrize("algorithm", ["sft", "offline-dpo"])
def test_prepare_uses_independent_pinned_repo_and_exact_rank_geometry(
    tmp_path: Path,
    algorithm: str,
) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path)
    calls = []

    def download(**kwargs: Any) -> str:
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        _populate_snapshot(local_dir, algorithm=algorithm)
        return str(local_dir)

    prepared = prepare_dataset(
        algorithm=algorithm,
        profile_name="sd35-t2i",
        world_size=2,
        output_root=tmp_path / "prepared",
        offline=True,
        lock_path=lock_path,
        download_fn=download,
    )

    assert calls == [
        {
            "repo_id": DATASET_REPO_IDS[algorithm],
            "repo_type": "dataset",
            "revision": PINNED_SHA,
            "local_dir": calls[0]["local_dir"],
            "local_files_only": True,
        }
    ]
    rows = [json.loads(line) for line in (prepared / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert all(".." not in media["path"] for row in rows for media in _all_media(row))
    assert (prepared / "media" / "target.png").is_file()
    assert json.loads((prepared / "materialization.json").read_text())["revision"] == PINNED_SHA
    summary = validate_dataset(
        prepared,
        algorithm=algorithm,
        profile_name="sd35-t2i",
        expected_rows=4,
    )
    assert summary["row_count"] == 4


def test_pending_revision_fails_before_download(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, pending=True)

    def unexpected_download(**kwargs: Any) -> str:
        raise AssertionError(f"download must not run: {kwargs}")

    with pytest.raises(ValueError, match="publication is still pending"):
        prepare_dataset(
            algorithm="sft",
            profile_name="sd35-t2i",
            world_size=1,
            output_root=tmp_path / "prepared",
            lock_path=lock_path,
            download_fn=unexpected_download,
        )


def test_canonical_profile_name_fails_before_download(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path)

    def unexpected_download(**kwargs: Any) -> str:
        raise AssertionError(f"download must not run: {kwargs}")

    with pytest.raises(ValueError, match="published runtime alias"):
        prepare_dataset(
            algorithm="sft",
            profile_name="text_to_image",
            world_size=1,
            output_root=tmp_path / "prepared",
            lock_path=lock_path,
            download_fn=unexpected_download,
        )


def test_repeat_requires_explicit_opt_in(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path)

    def download(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        _populate_snapshot(local_dir, algorithm="sft", row_count=2)
        return str(local_dir)

    kwargs = {
        "algorithm": "sft",
        "profile_name": "sd35-t2i",
        "world_size": 2,
        "output_root": tmp_path / "prepared",
        "lock_path": lock_path,
        "download_fn": download,
    }
    with pytest.raises(ValueError, match="--allow-repeat"):
        prepare_dataset(**kwargs)
    prepared = prepare_dataset(**kwargs, allow_repeat=True)
    assert len((prepared / "train.jsonl").read_text().splitlines()) == 4


@pytest.mark.parametrize("failure", ["escape", "identical_pair"])
def test_invalid_snapshot_never_publishes_partial_output(tmp_path: Path, failure: str) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path)

    def download(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        target_path = "../../../escape.png" if failure == "escape" else "../../media/target.png"
        _populate_snapshot(
            local_dir,
            algorithm="offline-dpo",
            target_path=target_path,
            identical_pair=failure == "identical_pair",
        )
        if failure == "escape":
            Image.new("RGB", (8, 8)).save(local_dir.parent / "escape.png")
        return str(local_dir)

    with pytest.raises(ValueError, match="escapes|byte-identical"):
        prepare_dataset(
            algorithm="offline-dpo",
            profile_name="sd35-t2i",
            world_size=2,
            output_root=tmp_path / "prepared",
            lock_path=lock_path,
            download_fn=download,
        )
    assert not (tmp_path / "prepared" / "offline-dpo" / "sd35-t2i").exists()


def _all_media(row: Mapping[str, Any]):
    yield from row["input"].get("media", ())
    for name in ("target", "chosen", "rejected"):
        candidate = row["supervision"].get(name)
        if candidate is not None:
            yield from candidate["media"]
