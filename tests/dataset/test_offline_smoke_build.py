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

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import av
import numpy as np
import pytest
from PIL import Image

from dataset.offline_smoke.build_mini import RUNTIME_ALIASES, build
from dataset.offline_smoke.profiles import GPU_ALIAS_TO_PROFILE
from dataset.offline_smoke.validate import validate_dataset
from flow_factory.contracts import (
    validate_pipeline_model_input,
    validate_pipeline_output_candidate,
)
from flow_factory.data_utils.offline_dataset import load_offline_manifest
from flow_factory.models.registry import get_model_adapter_class

ROWS = 3
AV_ALIASES = (
    "ltx2-t2av",
    "ltx2-i2av",
    "h3-t2va",
    "h3-fl2va",
    "h3-ref2va",
)


@pytest.fixture(scope="module")
def built_repos(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("offline-smoke-build")
    sft_root, dpo_root = build(
        root / "staging",
        seed=20260830,
        replace=False,
        records_per_alias=ROWS,
    )
    return root, sft_root, dpo_root


def test_all_runtime_aliases_pass_public_validation(
    built_repos: tuple[Path, Path, Path],
) -> None:
    root, sft_root, dpo_root = built_repos
    for algorithm, repo, supervision_type in (
        ("sft", sft_root, "demonstration"),
        ("offline-dpo", dpo_root, "preference"),
    ):
        manifest = json.loads((repo / "dataset_manifest.json").read_text(encoding="utf-8"))
        assert manifest["condition_endpoint_check"] == {
            "metric": "decoded_rgb_max_absolute_difference",
            "tolerance": 0,
        }
        profile_names = {path.name for path in (repo / "profiles").iterdir()}
        assert profile_names == set(RUNTIME_ALIASES)
        for alias in RUNTIME_ALIASES:
            rows = _rows(repo, alias)
            assert len(rows) == ROWS
            assert {row["supervision"]["type"] for row in rows} == {supervision_type}
            materialized = _materialize(repo, alias, root / "materialized" / algorithm / alias)
            summary = validate_dataset(
                materialized,
                algorithm=algorithm,
                profile_name=alias,
                expected_rows=ROWS,
            )
            assert summary["row_count"] == ROWS


def test_runtime_geometries_and_av_order(
    built_repos: tuple[Path, Path, Path],
) -> None:
    _, sft_root, dpo_root = built_repos
    expected_frames = {"wan-t2v": 5, "ltx2-t2av": 9, "h3-t2va": 124}
    for alias, count in expected_frames.items():
        row = _rows(sft_root, alias)[0]
        video = _resolve(sft_root, alias, row["supervision"]["target"]["media"][0])
        assert len(_decode_video(video)) == count

    for repo, candidate_names in ((sft_root, ("target",)), (dpo_root, ("chosen", "rejected"))):
        for alias in AV_ALIASES:
            for row in _rows(repo, alias):
                for name in candidate_names:
                    media = row["supervision"][name]["media"]
                    assert [item["type"] for item in media] == ["video", "audio"]


def test_generated_rows_remain_legal_subsets_of_real_adapter_contracts(
    built_repos: tuple[Path, Path, Path],
) -> None:
    _, sft_root, _ = built_repos
    for alias in RUNTIME_ALIASES:
        if alias == "bagel-mri2i":
            # The Bagel contract is covered behind its optional-kernel seam in
            # tests/models/test_bagel_output_codec.py.
            continue
        profile = GPU_ALIAS_TO_PROFILE[alias]
        case = next(case for case in profile.gpu_cases if case.alias == alias)
        adapter_type = get_model_adapter_class(case.model_type)
        contract = adapter_type.pipeline_io_contract
        manifest = sft_root / "profiles" / alias / "train.jsonl"
        records = load_offline_manifest(
            manifest,
            supervision_type="demonstration",
            dataset_dir=manifest.parent,
        )
        for record in records:
            validate_pipeline_model_input(record.model_input, contract)
            validate_pipeline_output_candidate(record.supervision.target.media, contract)


def test_h3_sparse_frame_slots_and_dpo_endpoints(
    built_repos: tuple[Path, Path, Path],
) -> None:
    _, _, dpo_root = built_repos
    rows = _rows(dpo_root, "h3-fl2va")
    assert [[item["slot"] for item in row["input"]["media"]] for row in rows] == [
        ["first_frame"],
        ["last_frame"],
        ["first_frame", "last_frame"],
    ]

    for alias in ("wan-i2v-first", "wan-flf2v", "ltx2-i2av", "h3-fl2va"):
        for row in _rows(dpo_root, alias):
            chosen = _candidate_video(dpo_root, alias, row, "chosen")
            rejected = _candidate_video(dpo_root, alias, row, "rejected")
            for condition in row["input"]["media"]:
                slot = condition.get("slot")
                if slot not in {"first_frame", "last_frame"}:
                    continue
                endpoint = 0 if slot == "first_frame" else -1
                expected = np.asarray(
                    Image.open(_resolve(dpo_root, alias, condition)).convert("RGB")
                )
                assert np.array_equal(chosen[endpoint], expected)
                assert np.array_equal(rejected[endpoint], expected)


def test_dpo_arms_are_content_distinct(
    built_repos: tuple[Path, Path, Path],
) -> None:
    _, _, dpo_root = built_repos
    digest_cache: dict[Path, str] = {}
    for alias in RUNTIME_ALIASES:
        for row in _rows(dpo_root, alias):
            chosen = _signature(dpo_root, alias, row["supervision"]["chosen"], digest_cache)
            rejected = _signature(
                dpo_root,
                alias,
                row["supervision"]["rejected"],
                digest_cache,
            )
            assert chosen != rejected


def _rows(repo: Path, alias: str) -> list[dict[str, Any]]:
    path = repo / "profiles" / alias / "train.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _all_media(row: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    yield from row["input"].get("media", ())
    for name in ("target", "chosen", "rejected"):
        candidate = row["supervision"].get(name)
        if candidate is not None:
            yield from candidate["media"]


def _resolve(repo: Path, alias: str, media: Mapping[str, Any]) -> Path:
    return (repo / "profiles" / alias / media["path"]).resolve()


def _materialize(repo: Path, alias: str, destination: Path) -> Path:
    rows = _rows(repo, alias)
    destination.mkdir(parents=True)
    repo = repo.resolve()
    for row in rows:
        for media in _all_media(row):
            source = _resolve(repo, alias, media)
            relative = source.relative_to(repo)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(source, target)
            media["path"] = relative.as_posix()
    (destination / "train.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return destination


def _decode_video(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def _candidate_video(
    repo: Path,
    alias: str,
    row: Mapping[str, Any],
    name: str,
) -> list[np.ndarray]:
    media = next(item for item in row["supervision"][name]["media"] if item["type"] == "video")
    return _decode_video(_resolve(repo, alias, media))


def _signature(
    repo: Path,
    alias: str,
    candidate: Mapping[str, Any],
    cache: dict[Path, str],
) -> tuple[tuple[str, str], ...]:
    values = []
    for media in candidate["media"]:
        path = _resolve(repo, alias, media)
        if path not in cache:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            cache[path] = digest.hexdigest()
        values.append((media["type"], cache[path]))
    return tuple(values)
