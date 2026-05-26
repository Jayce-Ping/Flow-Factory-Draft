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

# src/flow_factory/hparams/eval_dataset_args.py
"""
Evaluation Dataset Arguments Configuration.

Supports multiple evaluation datasets, each with independent data paths.
"""
from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from typing import Any, Optional, List

from .abc import ArgABC


@dataclass
class EvalDatasetArguments(ArgABC):
    """
    Configuration for a single evaluation dataset.

    Each evaluation dataset defines an independent data source for evaluation.
    Reward models are mapped to eval datasets via the ``datasets`` field in
    ``RewardArguments``.

    Attributes:
        name: Unique identifier for this eval dataset (used in metric keys
            and reward routing, e.g. ``eval/{name}/reward_*``).
        dataset_dir: Path to dataset folder containing the split file
            (e.g. ``test.jsonl`` or ``train.jsonl``).
        split: Which split file to use (default: ``"test"``).
        image_dir: Override image root directory for this dataset.
        video_dir: Override video root directory for this dataset.
        audio_dir: Override audio root directory for this dataset.
        max_dataset_size: Limit number of samples for this eval dataset.

    YAML Configuration Example:
        ```yaml
        eval_datasets:
          - name: "geneval"
            dataset_dir: "dataset/geneval"
          - name: "pickscore"
            dataset_dir: "dataset/pickscore"
            max_dataset_size: 500
        ```
    """

    name: str = field(
        default="default",
        metadata={"help": "Unique name for this eval dataset (used in metric keys)."},
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to dataset folder containing the split file."},
    )
    split: str = field(
        default="test",
        metadata={"help": "Which split to use for evaluation (e.g. 'test', 'train')."},
    )
    image_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Override image root directory for this eval dataset."},
    )
    video_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Override video root directory for this eval dataset."},
    )
    audio_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Override audio root directory for this eval dataset."},
    )
    max_dataset_size: Optional[int] = field(
        default=None,
        metadata={"help": "Limit number of samples for this eval dataset."},
    )

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        return self.__str__()
