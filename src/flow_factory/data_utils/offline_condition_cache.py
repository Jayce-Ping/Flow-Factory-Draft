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

"""Project V2 inputs into model condition preprocessing and caching.

Offline supervision has a deliberately separate lifecycle from input
conditions. This module exposes only ``NormalizedDatasetRecord.model_input`` to
``GeneralDataset``. Target, chosen, rejected, and record metadata are never
inserted into the raw Arrow table and therefore cannot leak into an adapter or
invalidate an input-condition cache.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

from datasets import Dataset as HFDataset

from ..samples.references import canonicalize_reference_manifest
from .dataset import (
    METADATA_COLUMN,
    GeneralDataset,
    PreprocessCallable,
    _supports_ordered_references,
)
from .offline_dataset import (
    OFFLINE_CONDITION_ID_COLUMN,
    compute_offline_condition_id,
)
from .schema import MediaAsset, NormalizedDatasetRecord

_CONDITION_SOURCE_FORMAT = "flow-factory-offline-condition-v1"


def project_offline_condition_dataset(
    records: Sequence[NormalizedDatasetRecord],
    *,
    source_name: str,
    ordered_references: bool,
    _media_digest_cache: MutableMapping[str, str] | None = None,
) -> HFDataset:
    """Build an input-only raw dataset for ``GeneralDataset`` preprocessing.

    Grouped adapters receive ``prompt`` plus per-modality ``images``, ``videos``,
    and ``audios`` columns. Ordered-reference adapters receive one canonical JSON
    string per row instead of an Arrow list-of-struct column. The string is
    restored to validated legacy ``kind`` entries only at the adapter preprocess
    boundary, avoiding Arrow's heterogeneous-struct null-key expansion.
    """
    if not isinstance(ordered_references, bool):
        raise TypeError(
            "ordered_references must be a bool, " f"got {type(ordered_references).__name__}"
        )
    stable_records = tuple(records)
    if not stable_records:
        raise ValueError("offline condition projection requires at least one record")
    for index, record in enumerate(stable_records):
        if not isinstance(record, NormalizedDatasetRecord):
            raise TypeError(
                "offline condition projection accepts normalized V2 records only, "
                f"got {type(record).__name__} at index {index}"
            )

    media_digest_cache: MutableMapping[str, str] = (
        {} if _media_digest_cache is None else _media_digest_cache
    )
    condition_ids = [
        compute_offline_condition_id(
            record,
            index=index,
            source_name=source_name,
            _media_digest_cache=media_digest_cache,
        )
        for index, record in enumerate(stable_records)
    ]
    columns: Dict[str, List[Any]] = {
        "prompt": [record.model_input.prompt for record in stable_records],
        OFFLINE_CONDITION_ID_COLUMN: condition_ids,
    }

    negative_prompts = [record.model_input.negative_prompt for record in stable_records]
    if any(value is not None for value in negative_prompts):
        # Adapter tokenizers consume a homogeneous text batch. In a mixed V2
        # batch, an omitted optional negative prompt is semantically the empty
        # prompt, not a tokenizer-level ``None`` value.
        columns["negative_prompt"] = [
            value if value is not None else "" for value in negative_prompts
        ]

    if ordered_references:
        columns["references"] = [
            canonicalize_reference_manifest(
                [_to_legacy_reference(asset) for asset in record.model_input.media],
                row_index=index,
            )
            for index, record in enumerate(stable_records)
        ]
    else:
        grouped_columns = {
            "images": [
                [asset.path for asset in record.model_input.media if asset.type == "image"]
                for record in stable_records
            ],
            "videos": [
                [
                    _to_grouped_rate_spec(asset, rate_name="fps")
                    for asset in record.model_input.media
                    if asset.type == "video"
                ]
                for record in stable_records
            ],
            "audios": [
                [
                    _to_grouped_rate_spec(asset, rate_name="sample_rate")
                    for asset in record.model_input.media
                    if asset.type == "audio"
                ]
                for record in stable_records
            ],
        }
        columns.update(
            {column_name: values for column_name, values in grouped_columns.items() if any(values)}
        )

    return HFDataset.from_dict(columns)


def compute_offline_condition_source_hash(condition_ids: Sequence[str]) -> str:
    """Hash ordered input identities for an input-only cache fingerprint."""
    stable_ids = tuple(condition_ids)
    if not stable_ids:
        raise ValueError("offline condition source hash requires at least one condition id")
    for index, condition_id in enumerate(stable_ids):
        if not isinstance(condition_id, str) or not condition_id:
            raise ValueError(
                "offline condition ids must be non-empty strings, "
                f"got {condition_id!r} at index {index}"
            )
    payload = json.dumps(
        {
            "format": _CONDITION_SOURCE_FORMAT,
            "condition_ids": stable_ids,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_offline_condition_cache(
    records: Sequence[NormalizedDatasetRecord],
    *,
    source_name: str,
    dataset_dir: str | os.PathLike[str],
    preprocess_func: PreprocessCallable,
    preprocess_kwargs: Mapping[str, Any] | None = None,
    cache_dir: str | os.PathLike[str] = "~/.cache/flow_factory/datasets",
    force_reprocess: bool = False,
    preprocessing_batch_size: int | None = None,
    extra_hash_strs: Sequence[str] | None = None,
    target_arrow_path: str | os.PathLike[str] | None = None,
) -> HFDataset:
    """Preprocess and cache only offline input conditions.

    The default Arrow target is derived from the same input-only merged-cache
    fingerprint used by ``GeneralDataset``. Rebuilding from records whose target
    or metadata changed therefore loads the existing cache without invoking the
    adapter again. Callers orchestrating distributed preprocessing may provide a
    rank-specific ``target_arrow_path`` instead.
    """
    if not callable(preprocess_func):
        raise TypeError(
            "offline condition cache requires a callable preprocess_func, "
            f"got {type(preprocess_func).__name__}"
        )
    ordered_references = _supports_ordered_references(preprocess_func)
    raw_dataset = project_offline_condition_dataset(
        records,
        source_name=source_name,
        ordered_references=ordered_references,
    )
    condition_ids = tuple(raw_dataset[OFFLINE_CONDITION_ID_COLUMN])
    source_hash = compute_offline_condition_source_hash(condition_ids)
    normalized_dataset_dir = os.path.expanduser(os.fspath(dataset_dir))
    normalized_cache_dir = os.path.expanduser(os.fspath(cache_dir))
    normalized_preprocess_kwargs = dict(preprocess_kwargs or {})
    normalized_extra_hash_strs = list(extra_hash_strs or ())

    if preprocessing_batch_size is None:
        preprocessing_batch_size = 1 if ordered_references else 16
    if (
        not isinstance(preprocessing_batch_size, int)
        or isinstance(preprocessing_batch_size, bool)
        or preprocessing_batch_size <= 0
    ):
        raise ValueError(
            "preprocessing_batch_size must be a positive integer, "
            f"got {preprocessing_batch_size!r}"
        )

    normalized_target_arrow_path: str
    if target_arrow_path is None:
        merged_cache_path = GeneralDataset.compute_cache_path(
            dataset_dir=normalized_dataset_dir,
            split="train",
            cache_dir=normalized_cache_dir,
            max_dataset_size=None,
            preprocess_func=preprocess_func,
            preprocess_kwargs=normalized_preprocess_kwargs,
            extra_hash_strs=normalized_extra_hash_strs,
            source_hash_override=source_hash,
        )
        normalized_target_arrow_path = f"{merged_cache_path}.arrow"
    else:
        normalized_target_arrow_path = os.path.expanduser(os.fspath(target_arrow_path))

    dataset_builder = GeneralDataset(
        dataset_dir=normalized_dataset_dir,
        split="train",
        cache_dir=normalized_cache_dir,
        preprocess_func=preprocess_func,
        preprocess_kwargs=normalized_preprocess_kwargs,
        preprocessing_batch_size=preprocessing_batch_size,
        force_reprocess=force_reprocess,
        extra_hash_strs=normalized_extra_hash_strs,
        image_dir=normalized_dataset_dir,
        video_dir=normalized_dataset_dir,
        audio_dir=normalized_dataset_dir,
        target_arrow_path=normalized_target_arrow_path,
        raw_dataset=raw_dataset,
        source_hash_override=source_hash,
        passthrough_columns=(OFFLINE_CONDITION_ID_COLUMN,),
    )
    condition_cache = dataset_builder.processed_dataset
    if METADATA_COLUMN in condition_cache.column_names:
        condition_cache = condition_cache.remove_columns(METADATA_COLUMN)
    return condition_cache


def _to_legacy_reference(asset: MediaAsset) -> Dict[str, Any]:
    reference: Dict[str, Any] = {"kind": asset.type, "path": asset.path}
    if asset.type == "video" and asset.fps is not None:
        reference["fps"] = asset.fps
    elif asset.type == "audio" and asset.sample_rate is not None:
        reference["sample_rate"] = asset.sample_rate
    return reference


def _to_grouped_rate_spec(asset: MediaAsset, *, rate_name: str) -> Dict[str, Any]:
    spec: Dict[str, Any] = {"path": asset.path}
    rate = getattr(asset, rate_name)
    if rate is not None:
        spec[rate_name] = rate
    return spec


__all__ = [
    "build_offline_condition_cache",
    "compute_offline_condition_source_hash",
    "project_offline_condition_dataset",
]
