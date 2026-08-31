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

"""Publish one validated offline-smoke staging tree to the Hugging Face Hub.

This command is intentionally separate from dataset construction. Building and
validating fixtures are local, reversible operations; publication creates public
external state and therefore requires the explicit ``--confirm-public`` flag.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal, Sequence

from huggingface_hub import HfApi

from .profiles import OFFLINE_DPO_REPO_ID, SFT_REPO_ID

Algorithm = Literal["sft", "offline-dpo"]
_SUPERVISION_BY_ALGORITHM = {"sft": "demonstration", "offline-dpo": "preference"}


def default_repo_id(algorithm: Algorithm) -> str:
    """Return the public repository owned by one supervision family.

    Args:
        algorithm: Offline supervision family.

    Returns:
        Canonical public Hugging Face dataset repository ID.

    Raises:
        ValueError: If the algorithm is unsupported.
    """
    if algorithm == "sft":
        return SFT_REPO_ID
    if algorithm == "offline-dpo":
        return OFFLINE_DPO_REPO_ID
    raise ValueError(f"unsupported offline smoke algorithm: {algorithm!r}")


def validate_staging_tree(
    staging_dir: Path,
    *,
    algorithm: Algorithm,
    destination_repo_id: str,
) -> None:
    """Reject mismatched or unsafe publication trees before any Hub mutation.

    Args:
        staging_dir: Local self-contained repository tree to publish.
        algorithm: Supervision family selected by the publication command.
        destination_repo_id: Hub dataset repository that would receive the tree.

    Returns:
        None.

    Raises:
        ValueError: If identity, supervision, structure, or paths are invalid.
    """
    staging_dir = staging_dir.resolve()
    if not staging_dir.is_dir():
        raise ValueError(f"staging directory does not exist: {staging_dir}")
    for relative_path in (
        "README.md",
        "LICENSE",
        "dataset_manifest.json",
        "media",
        "profiles",
        "provenance.jsonl",
    ):
        if not (staging_dir / relative_path).exists():
            raise ValueError(
                f"staging directory is incomplete: missing {relative_path!r} under {staging_dir}"
            )

    with (staging_dir / "dataset_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    canonical_repo_id = default_repo_id(algorithm)
    expected_supervision = _SUPERVISION_BY_ALGORITHM[algorithm]
    if manifest.get("schema_version") != 1 or manifest.get("flow_factory_schema_version") != 2:
        raise ValueError("staging dataset manifest has an unsupported schema version")
    if manifest.get("supervision_type") != expected_supervision:
        raise ValueError(
            f"{algorithm!r} publication requires {expected_supervision!r} supervision, "
            f"found {manifest.get('supervision_type')!r}"
        )
    if (
        destination_repo_id == canonical_repo_id
        and manifest.get("repository_id") != canonical_repo_id
    ):
        raise ValueError(
            f"canonical destination {canonical_repo_id!r} requires a matching staging repository_id, "
            f"found {manifest.get('repository_id')!r}"
        )
    runtime_aliases = manifest.get("runtime_aliases")
    if (
        not isinstance(runtime_aliases, list)
        or not runtime_aliases
        or any(not isinstance(alias, str) or not alias for alias in runtime_aliases)
        or len(runtime_aliases) != len(set(runtime_aliases))
    ):
        raise ValueError("staging dataset manifest must declare unique non-empty runtime aliases")
    profile_aliases = {path.name for path in (staging_dir / "profiles").iterdir() if path.is_dir()}
    if profile_aliases != set(runtime_aliases):
        raise ValueError("staging profile directories disagree with manifest runtime_aliases")

    for path in staging_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"public staging trees cannot contain symlinks: {path}")
        if path.is_file() and not path.resolve().is_relative_to(staging_dir):
            raise ValueError(f"staging file escapes publication root: {path}")


def publish_staging_tree(
    *,
    algorithm: Algorithm,
    staging_dir: Path,
    repo_id: str | None = None,
    commit_message: str,
    confirm_public: bool,
) -> str:
    """Create or update one public dataset repository.

    Args:
        algorithm: Offline supervision family being published.
        staging_dir: Validated local repository tree.
        repo_id: Optional noncanonical destination repository override.
        commit_message: Hub commit message.
        confirm_public: Explicit acknowledgement of public external state.

    Returns:
        Immutable Hub commit SHA returned by the upload.

    Raises:
        ValueError: If public confirmation or staging identity is invalid.
    """
    if not confirm_public:
        raise ValueError("public dataset publication requires --confirm-public")
    resolved_repo_id = default_repo_id(algorithm) if repo_id is None else repo_id
    validate_staging_tree(
        staging_dir,
        algorithm=algorithm,
        destination_repo_id=resolved_repo_id,
    )
    api = HfApi()
    api.create_repo(
        repo_id=resolved_repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=resolved_repo_id,
        repo_type="dataset",
        folder_path=str(staging_dir.resolve()),
        commit_message=commit_message,
    )
    return commit.oid


def _build_parser() -> argparse.ArgumentParser:
    """Build the publication-only command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("sft", "offline-dpo"), required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument(
        "--commit-message",
        default="Publish deterministic Flow-Factory offline smoke fixtures",
    )
    parser.add_argument(
        "--confirm-public",
        action="store_true",
        help="Acknowledge that this command creates or updates a public Hub dataset.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Publish one staging tree and print a machine-readable result.

    Args:
        argv: Optional argument sequence. Uses ``sys.argv`` when omitted.

    Returns:
        Process exit code zero after a successful Hub commit.
    """
    args = _build_parser().parse_args(argv)
    commit_sha = publish_staging_tree(
        algorithm=args.algorithm,
        staging_dir=args.staging_dir,
        repo_id=args.repo_id,
        commit_message=args.commit_message,
        confirm_public=args.confirm_public,
    )
    print(
        json.dumps(
            {
                "algorithm": args.algorithm,
                "repo_id": args.repo_id or default_repo_id(args.algorithm),
                "revision": commit_sha,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
