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

"""Validate and serialize ordered heterogeneous reference manifests."""

import json
from typing import Any, Dict, List

_REFERENCE_KEYS = {
    "image": frozenset({"kind", "path"}),
    "video": frozenset({"kind", "path", "fps", "audio_path", "sample_rate"}),
    "audio": frozenset({"kind", "path", "sample_rate"}),
}


def canonicalize_reference_manifest(references: Any, row_index: int) -> str:
    """Validate references and return their canonical JSON representation.

    Args:
        references: Ordered JSON-compatible reference entries.
        row_index: Dataset row index used in validation errors.

    Returns:
        Compact canonical JSON preserving the input array order.
    """
    if not isinstance(references, list):
        raise TypeError(
            "expected references to be a non-empty list "
            f"at row {row_index}, got {type(references).__name__}: {references!r}"
        )
    if not references:
        raise ValueError(f"expected references to be non-empty at row {row_index}, got []")

    validated: List[Dict[str, Any]] = []
    for reference_index, entry in enumerate(references):
        validated.append(_validate_reference_entry(entry, row_index, reference_index))
    return json.dumps(validated, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def parse_reference_manifest(manifest: str, row_index: int) -> List[Dict[str, Any]]:
    """Parse and validate a canonical reference manifest.

    Args:
        manifest: JSON array string.
        row_index: Dataset row index used in validation errors.

    Returns:
        Ordered validated reference dictionaries.
    """
    if not isinstance(manifest, str):
        raise TypeError(
            f"expected reference manifest string at row {row_index}, "
            f"got {type(manifest).__name__}: {manifest!r}"
        )
    references = json.loads(manifest)
    canonical = canonicalize_reference_manifest(references, row_index=row_index)
    return json.loads(canonical)


def _validate_reference_entry(entry: Any, row_index: int, reference_index: int) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        raise TypeError(
            f"expected object at row {row_index}, reference {reference_index}, "
            f"got {type(entry).__name__}: {entry!r}"
        )
    kind = entry.get("kind")
    if not isinstance(kind, str) or kind not in _REFERENCE_KEYS:
        raise ValueError(
            f"at row {row_index}, reference {reference_index}, expected kind in "
            f"{tuple(_REFERENCE_KEYS)}, got {kind!r}"
        )
    path = entry.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            f"at row {row_index}, reference {reference_index}, expected path to be "
            f"a non-empty string, got {path!r}"
        )

    unknown_keys = set(entry) - _REFERENCE_KEYS[kind]
    if unknown_keys:
        raise ValueError(
            f"at row {row_index}, reference {reference_index}, unknown keys "
            f"{sorted(unknown_keys)} for kind {kind!r}"
        )
    for rate_name in ("fps", "sample_rate"):
        if rate_name in entry:
            rate = entry[rate_name]
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
                raise ValueError(
                    f"at row {row_index}, reference {reference_index}, expected "
                    f"{rate_name} to be positive numeric, got {rate!r}"
                )
    if kind == "video":
        audio_path = entry.get("audio_path")
        if audio_path is not None and (not isinstance(audio_path, str) or not audio_path):
            raise ValueError(
                f"expected non-empty audio_path string at row {row_index}, "
                f"reference {reference_index}, got {audio_path!r}"
            )
        if "sample_rate" in entry and "audio_path" not in entry:
            raise ValueError(
                f"at row {row_index}, reference {reference_index}, sample_rate "
                f"{entry['sample_rate']!r} requires audio_path"
            )
    return dict(entry)
