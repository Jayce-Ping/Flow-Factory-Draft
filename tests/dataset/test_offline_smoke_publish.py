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
from types import SimpleNamespace

import pytest

from dataset.offline_smoke import publish
from dataset.offline_smoke.profiles import SFT_REPO_ID


def _staging_tree(
    root: Path,
    *,
    repository_id: str = SFT_REPO_ID,
    supervision_type: str = "demonstration",
) -> Path:
    for directory in ("media", "profiles/sd35-t2i"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for filename in ("README.md", "LICENSE", "provenance.jsonl"):
        (root / filename).write_text("fixture\n", encoding="utf-8")
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "flow_factory_schema_version": 2,
                "repository_id": repository_id,
                "supervision_type": supervision_type,
                "runtime_aliases": ["sd35-t2i"],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_wrong_supervision_fails_before_any_hub_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _staging_tree(
        tmp_path / "dpo-staging",
        repository_id="Jayce-Ping/Flow-Factory-Offline-DPO-Smoke",
        supervision_type="preference",
    )

    def unexpected_api() -> None:
        raise AssertionError("HfApi must not be constructed for mismatched staging")

    monkeypatch.setattr(publish, "HfApi", unexpected_api)
    with pytest.raises(ValueError, match="requires 'demonstration' supervision"):
        publish.publish_staging_tree(
            algorithm="sft",
            staging_dir=staging,
            commit_message="must not publish",
            confirm_public=True,
        )


def test_valid_staging_binds_default_repo_and_returns_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _staging_tree(tmp_path / "sft-staging")
    calls = []

    class FakeApi:
        def create_repo(self, **kwargs) -> None:
            calls.append(("create_repo", kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload_folder", kwargs))
            return SimpleNamespace(oid="b" * 40)

    monkeypatch.setattr(publish, "HfApi", FakeApi)
    revision = publish.publish_staging_tree(
        algorithm="sft",
        staging_dir=staging,
        commit_message="publish test fixture",
        confirm_public=True,
    )

    assert revision == "b" * 40
    assert [name for name, _ in calls] == ["create_repo", "upload_folder"]
    assert all(kwargs["repo_id"] == SFT_REPO_ID for _, kwargs in calls)
