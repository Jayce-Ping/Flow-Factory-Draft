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

"""Download and atomically materialize one pinned offline-smoke profile."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Mapping, Sequence

from huggingface_hub import snapshot_download

from flow_factory.data_utils.schema import DatasetRecordV2

from .profiles import DATASET_REPO_IDS, GPU_ALIAS_TO_PROFILE
from .validate import validate_dataset

Algorithm = Literal["sft", "offline-dpo"]
SnapshotDownload = Callable[..., str]
DEFAULT_LOCK_PATH = Path(__file__).with_name("datasets.lock.json")
DEFAULT_OUTPUT_ROOT = Path("dataset/_prepared_offline_smoke")
PENDING_REVISION = "PENDING_INITIAL_UPLOAD"
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class DatasetLock:
    """One immutable Hub dataset selection."""

    repo_id: str
    revision: str
    supervision_type: str


def load_dataset_lock(
    algorithm: Algorithm, lock_path: str | Path = DEFAULT_LOCK_PATH
) -> DatasetLock:
    """Load and verify one independent repository lock.

    Args:
        algorithm: Offline supervision family to resolve.
        lock_path: JSON lock file containing immutable Hub revisions.

    Returns:
        Validated repository selection for the requested algorithm.

    Raises:
        ValueError: If the algorithm, lock schema, repository, supervision, or revision is invalid.
    """
    if algorithm not in DATASET_REPO_IDS:
        raise ValueError(f"unsupported offline smoke algorithm: {algorithm!r}")
    path = Path(lock_path).expanduser()
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("datasets"), dict):
        raise ValueError(f"invalid offline smoke lock schema: {path}")
    entry = payload["datasets"].get(algorithm)
    if not isinstance(entry, dict):
        raise ValueError(f"offline smoke lock has no {algorithm!r} entry: {path}")
    lock = DatasetLock(
        repo_id=entry.get("repo_id"),
        revision=entry.get("revision"),
        supervision_type=entry.get("supervision_type"),
    )
    expected_supervision = "demonstration" if algorithm == "sft" else "preference"
    if lock.repo_id != DATASET_REPO_IDS[algorithm]:
        raise ValueError(
            f"locked repo_id {lock.repo_id!r} disagrees with profiles.py "
            f"{DATASET_REPO_IDS[algorithm]!r}"
        )
    if lock.supervision_type != expected_supervision:
        raise ValueError(
            f"locked supervision {lock.supervision_type!r} must be {expected_supervision!r}"
        )
    if not isinstance(lock.revision, str) or not _COMMIT_SHA.fullmatch(lock.revision):
        detail = (
            "initial publication is still pending"
            if lock.revision == PENDING_REVISION
            else "not a commit SHA"
        )
        raise ValueError(
            f"offline smoke revision for {algorithm!r} is {detail}: {lock.revision!r}; "
            "publish the dataset and pin its 40-character commit SHA"
        )
    return lock


def prepare_dataset(
    *,
    algorithm: Algorithm,
    profile_name: str,
    world_size: int,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    per_device_batch_size: int = 1,
    batches_per_rank: int = 2,
    allow_repeat: bool = False,
    offline: bool = False,
    lock_path: str | Path = DEFAULT_LOCK_PATH,
    download_fn: SnapshotDownload | None = None,
) -> Path:
    """Materialize exactly the records required by one distributed smoke run.

    Args:
        algorithm: ``sft`` or ``offline-dpo`` supervision family.
        profile_name: Model-specific runtime alias published by the dataset repository.
        world_size: Number of distributed ranks in the target run.
        output_root: Root directory for ready-to-use datasets.
        per_device_batch_size: Number of records consumed by each rank per batch.
        batches_per_rank: Exact number of rank-local batches to materialize.
        allow_repeat: Whether to cycle the base fixture when more rows are requested.
        offline: Whether Hub access must use already downloaded local files only.
        lock_path: JSON lock file containing immutable Hub revisions.
        download_fn: Optional snapshot downloader used by tests.

    Returns:
        Path to the atomically published, self-contained profile directory.

    Raises:
        FileExistsError: If the requested output profile already exists.
        ValueError: If sizing, lock data, source records, or media violate their contracts.
    """
    for name, value in (
        ("world_size", world_size),
        ("per_device_batch_size", per_device_batch_size),
        ("batches_per_rank", batches_per_rank),
    ):
        _require_positive_int(value, name)
    if profile_name not in GPU_ALIAS_TO_PROFILE:
        aliases = ", ".join(GPU_ALIAS_TO_PROFILE)
        raise ValueError(
            f"unknown published runtime alias {profile_name!r}; choose one of: {aliases}. "
            "Canonical task profile names describe contracts but are not Hub directories."
        )
    source_profile = profile_name
    requested_rows = world_size * per_device_batch_size * batches_per_rank
    lock = load_dataset_lock(algorithm, lock_path)
    root = Path(output_root).expanduser().resolve()
    target = root / algorithm / source_profile
    if target.exists():
        raise FileExistsError(f"offline smoke output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    download = snapshot_download if download_fn is None else download_fn
    with tempfile.TemporaryDirectory(prefix=".offline-smoke-download-", dir=root) as download_dir:
        snapshot = Path(
            download(
                repo_id=lock.repo_id,
                repo_type="dataset",
                revision=lock.revision,
                local_dir=download_dir,
                local_files_only=offline,
            )
        ).resolve()
        source_manifest = snapshot / "profiles" / source_profile / "train.jsonl"
        rows = _load_rows(source_manifest, supervision_type=lock.supervision_type)
        if requested_rows > len(rows) and not allow_repeat:
            raise ValueError(
                f"profile {source_profile!r} has {len(rows)} base rows but {requested_rows} "
                "are required; pass --allow-repeat to cycle smoke-only records explicitly"
            )
        selected = [rows[index % len(rows)] for index in range(requested_rows)]
        staging = Path(tempfile.mkdtemp(prefix=f".{source_profile}-", dir=target.parent))
        try:
            with (staging / "train.jsonl").open("w", encoding="utf-8") as handle:
                for row in selected:
                    rewritten = _materialize_row(row, source_manifest.parent, snapshot, staging)
                    handle.write(
                        json.dumps(
                            rewritten, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                        )
                        + "\n"
                    )
            summary = validate_dataset(
                staging,
                algorithm=algorithm,
                profile_name=profile_name,
                expected_rows=requested_rows,
            )
            materialization = {
                "algorithm_profile": profile_name,
                "allow_repeat": allow_repeat,
                "repo_id": lock.repo_id,
                "revision": lock.revision,
                "row_count": requested_rows,
                "source_profile": source_profile,
                "validation": summary,
            }
            (staging / "materialization.json").write_text(
                json.dumps(materialization, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(staging, target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return target


def _load_rows(path: Path, *, supervision_type: str) -> list[DatasetRecordV2]:
    if not path.is_file():
        raise FileNotFoundError(f"pinned snapshot has no ready-to-use profile manifest: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {path}:{line_number}")
            parsed = DatasetRecordV2.model_validate_json(line)
            if parsed.supervision.type != supervision_type:
                raise ValueError(
                    f"{path}:{line_number} has {parsed.supervision.type!r} supervision, "
                    f"expected {supervision_type!r}"
                )
            rows.append(parsed)
    if not rows:
        raise ValueError(f"offline smoke profile manifest is empty: {path}")
    return rows


def _materialize_row(
    record: DatasetRecordV2,
    manifest_dir: Path,
    snapshot_root: Path,
    staging: Path,
) -> Dict[str, Any]:
    payload = record.model_dump(mode="json", exclude_none=True)
    for media in _media_dicts(payload):
        raw_path = Path(media["path"])
        if raw_path.is_absolute():
            raise ValueError(f"Hub smoke manifests cannot use absolute media paths: {raw_path}")
        source = (manifest_dir / raw_path).resolve(strict=True)
        if not source.is_relative_to(snapshot_root) or not source.is_file():
            raise ValueError(f"Hub smoke media path escapes its pinned snapshot: {raw_path}")
        relative = source.relative_to(snapshot_root)
        if not relative.parts or relative.parts[0] != "media":
            raise ValueError(f"Hub smoke media must live under repo-root media/: {raw_path}")
        destination = staging / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        media["path"] = relative.as_posix()
    return payload


def _media_dicts(payload: Mapping[str, Any]):
    yield from payload["input"].get("media", ())
    supervision = payload["supervision"]
    for candidate_name in ("target", "chosen", "rejected"):
        candidate = supervision.get(candidate_name)
        if candidate is not None:
            yield from candidate["media"]


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=tuple(DATASET_REPO_IDS), required=True)
    parser.add_argument("--profile", choices=tuple(GPU_ALIAS_TO_PROFILE), required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--batches-per-rank", type=int, default=2)
    parser.add_argument("--allow-repeat", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare one pinned dataset profile from command-line arguments.

    Args:
        argv: Optional argument sequence. Uses ``sys.argv`` when omitted.

    Returns:
        Process exit code zero after successful materialization.
    """
    args = _build_parser().parse_args(argv)
    path = prepare_dataset(
        algorithm=args.algorithm,
        profile_name=args.profile,
        world_size=args.world_size,
        output_root=args.output_root,
        per_device_batch_size=args.per_device_batch_size,
        batches_per_rank=args.batches_per_rank,
        allow_repeat=args.allow_repeat,
        offline=args.offline,
        lock_path=args.lock_path,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
