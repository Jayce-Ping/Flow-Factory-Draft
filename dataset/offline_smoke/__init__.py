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

"""Build declarations for Flow-Factory's public offline smoke datasets."""

from .profiles import (
    CANONICAL_PROFILES,
    DATASET_REPO_IDS,
    GPU_ALIAS_TO_PROFILE,
    MAIN_GPU_ALIASES,
    OFFLINE_DPO_REPO_ID,
    SFT_REPO_ID,
    SUPPLEMENTAL_GPU_ALIASES,
    GPUSmokeCase,
    OfflineSmokeProfile,
    SmokeGeometry,
    get_profile,
    output_media_types,
)

__all__ = [
    "CANONICAL_PROFILES",
    "DATASET_REPO_IDS",
    "GPUSmokeCase",
    "GPU_ALIAS_TO_PROFILE",
    "MAIN_GPU_ALIASES",
    "OFFLINE_DPO_REPO_ID",
    "OfflineSmokeProfile",
    "SFT_REPO_ID",
    "SUPPLEMENTAL_GPU_ALIASES",
    "SmokeGeometry",
    "get_profile",
    "output_media_types",
]
