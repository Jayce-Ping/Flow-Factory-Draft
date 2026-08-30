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

"""Thin validation boundary for materialized offline-smoke datasets.

The model adapter remains authoritative for encoded geometry and condition/output
semantics. This module only applies the existing public V2 and pipeline contracts,
checks that media stay inside the dataset root, uniquely decodes referenced files,
and rejects byte-identical offline-DPO candidates.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Tuple

from flow_factory.contracts import (
    validate_pipeline_model_input,
    validate_pipeline_output_candidate,
)
from flow_factory.data_utils.offline_dataset import (
    DEFAULT_MEDIA_DECODERS,
    load_offline_manifest,
)
from flow_factory.data_utils.schema import (
    DemonstrationSupervision,
    MediaAsset,
    NormalizedDatasetRecord,
    NormalizedOutputCandidate,
    PreferenceSupervision,
)

from .profiles import get_profile

Algorithm = Literal["sft", "offline-dpo"]
_SUPERVISION_BY_ALGORITHM: Mapping[Algorithm, str] = {
    "sft": "demonstration",
    "offline-dpo": "preference",
}
_HASH_CHUNK_SIZE = 1024 * 1024


def validate_dataset(
    dataset_dir: str | Path,
    *,
    algorithm: Algorithm,
    profile_name: str,
    expected_rows: int | None = None,
) -> Dict[str, Any]:
    """Validate one self-contained materialized dataset without loading a model.

    Args:
        dataset_dir: Directory containing ``train.jsonl`` and local media.
        algorithm: Supervision family required by the trainer.
        profile_name: Canonical task profile or model-specific runtime alias.
        expected_rows: Optional exact row-count requirement.

    Returns:
        Validation summary with profile identity and unique media hashes.

    Raises:
        FileNotFoundError: If the dataset or a referenced media file is missing.
        TypeError: If a record violates a typed schema or contract field.
        ValueError: If records, paths, media, or preference arms are invalid.
    """
    supervision_type = _require_algorithm(algorithm)
    if expected_rows is not None:
        _require_positive_int(expected_rows, "expected_rows")
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"offline smoke dataset directory does not exist: {root}")

    profile = get_profile(profile_name)
    records = load_offline_manifest(
        root / "train.jsonl",
        supervision_type=supervision_type,
        dataset_dir=root,
    )
    if expected_rows is not None and len(records) != expected_rows:
        raise ValueError(
            f"offline smoke dataset must contain exactly {expected_rows} rows, "
            f"found {len(records)}"
        )

    digest_cache: Dict[str, Tuple[Path, str]] = {}
    decode_cache: Dict[Tuple[str, Path], object] = {}
    for row_index, record in enumerate(records):
        _validate_contract(record, profile.contract, row_index=row_index)
        for asset in _record_media(record):
            cached = digest_cache.get(asset.path)
            if cached is None:
                path = _require_contained_file(asset.path, root=root)
                cached = (path, _sha256_file(path))
                digest_cache[asset.path] = cached
            else:
                path = cached[0]
            cache_key = (asset.type, path)
            if cache_key not in decode_cache:
                decode_cache[cache_key] = _decode_media(asset, context=f"row {row_index}")
        if isinstance(record.supervision, PreferenceSupervision):
            _require_distinct_candidates(
                record.supervision.chosen,
                record.supervision.rejected,
                digest_cache=digest_cache,
                context=f"row {row_index}",
            )

    return {
        "algorithm": algorithm,
        "requested_profile": profile_name,
        "canonical_profile": profile.profile_id,
        "row_count": len(records),
        "media_file_count": len(digest_cache),
        "media_sha256": {
            path.relative_to(root).as_posix(): digest for path, digest in digest_cache.values()
        },
    }


def _require_algorithm(algorithm: str) -> str:
    if algorithm not in _SUPERVISION_BY_ALGORITHM:
        raise ValueError(
            f"unsupported offline smoke algorithm {algorithm!r}; "
            f"expected one of {tuple(_SUPERVISION_BY_ALGORITHM)!r}"
        )
    return _SUPERVISION_BY_ALGORITHM[algorithm]


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer >= 1, got {type(value).__name__}: {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def _validate_contract(record, contract, *, row_index: int) -> None:
    try:
        validate_pipeline_model_input(record.model_input, contract)
        for _, candidate in _record_candidates(record):
            validate_pipeline_output_candidate(candidate.media, contract)
    except (TypeError, ValueError) as exc:
        raise type(exc)(
            f"offline smoke row {row_index} violates its profile contract: {exc}"
        ) from exc


def _record_candidates(
    record: NormalizedDatasetRecord,
) -> Tuple[Tuple[str, NormalizedOutputCandidate], ...]:
    supervision = record.supervision
    if isinstance(supervision, DemonstrationSupervision):
        return (("target", supervision.target),)
    if isinstance(supervision, PreferenceSupervision):
        return (("chosen", supervision.chosen), ("rejected", supervision.rejected))
    raise TypeError(f"unsupported normalized supervision: {type(supervision).__name__}")


def _record_media(record: NormalizedDatasetRecord):
    yield from record.model_input.media
    for _, candidate in _record_candidates(record):
        yield from candidate.media


def _require_contained_file(path_value: str, *, root: Path) -> Path:
    path = Path(path_value)
    try:
        lexical_relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"offline smoke media path escapes dataset root: {path}") from exc
    cursor = root
    for part in lexical_relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"offline smoke media paths cannot traverse symlinks: {cursor}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"offline smoke media file does not exist: {path}") from exc
    if not resolved.is_relative_to(root):
        raise ValueError(f"offline smoke media path escapes dataset root: {path}")
    if not resolved.is_file():
        raise ValueError(f"offline smoke media path is not a regular file: {path}")
    return resolved


def _decode_media(asset: MediaAsset, *, context: str) -> object:
    decoder = DEFAULT_MEDIA_DECODERS.get(asset.type)
    if decoder is None:
        raise ValueError(f"{context} has no decoder for media type {asset.type!r}")
    try:
        return decoder(asset)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise type(exc)(f"{context} failed to decode {asset.type} {asset.path!r}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_distinct_candidates(
    chosen: NormalizedOutputCandidate,
    rejected: NormalizedOutputCandidate,
    *,
    digest_cache: Mapping[str, Tuple[Path, str]],
    context: str,
) -> None:
    def signature(candidate: NormalizedOutputCandidate) -> Tuple[Tuple[str, str], ...]:
        return tuple(
            (
                asset.type,
                digest_cache[asset.path][1],
            )
            for asset in candidate.media
        )

    if signature(chosen) == signature(rejected):
        raise ValueError(f"{context} has byte-identical chosen and rejected candidates")
