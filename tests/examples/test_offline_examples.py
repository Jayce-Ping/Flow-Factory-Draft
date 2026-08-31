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

from pathlib import Path

import pytest

from flow_factory.data_utils.offline_dataset import OfflineSupervisionType, load_offline_manifest
from flow_factory.data_utils.schema import DemonstrationSupervision
from flow_factory.hparams import Arguments

ROOT = Path(__file__).resolve().parents[2]
OFFLINE_EXAMPLES = (
    (
        "examples/sft/lora/sd3_5/default.yaml",
        "dataset/sft_sd3_5",
        "demonstration",
    ),
    (
        "examples/offline_dpo/lora/sd3_5/default.yaml",
        "dataset/offline_dpo_sd3_5",
        "preference",
    ),
)


@pytest.mark.parametrize(
    ("config_path", "expected_dataset_dir", "supervision_type"),
    OFFLINE_EXAMPLES,
)
def test_offline_examples_use_repository_dataset_root(
    config_path: str,
    expected_dataset_dir: str,
    supervision_type: OfflineSupervisionType,
) -> None:
    """Keep checked-in offline fixtures under the repository dataset root."""
    config = Arguments.load_from_yaml(str(ROOT / config_path))

    assert len(config.data_args.datasets) == 1
    configured_dataset_dir = Path(config.data_args.datasets[0].dataset_dir)
    assert configured_dataset_dir == Path(expected_dataset_dir)
    assert configured_dataset_dir.parts[0] == "dataset"

    dataset_dir = ROOT / configured_dataset_dir
    records = load_offline_manifest(
        dataset_dir / "train.jsonl",
        supervision_type=supervision_type,
    )
    assert len(records) == 2

    assets_dir = (ROOT / "assets").resolve()
    for record in records:
        supervision = record.supervision
        candidates = (
            (supervision.target,)
            if isinstance(supervision, DemonstrationSupervision)
            else (supervision.chosen, supervision.rejected)
        )
        for candidate in candidates:
            for media in candidate.media:
                media_path = Path(media.path).resolve()
                assert media_path.is_relative_to(assets_dir)
                assert media_path.is_file()
