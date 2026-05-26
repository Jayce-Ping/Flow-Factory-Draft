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

Supports multiple evaluation datasets, each with independent data paths
and optional per-dataset eval generation overrides.
"""
from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from typing import Any, Optional, List, Tuple, Union

from .abc import ArgABC


@dataclass
class EvalDatasetArguments(ArgABC):
    """
    Configuration for a single evaluation dataset.

    Each evaluation dataset defines an independent data source for evaluation.
    Reward models are mapped to eval datasets via the ``datasets`` field in
    ``RewardArguments``.

    Fields marked as "override" default to ``None``, meaning the value from
    the shared ``EvaluationArguments`` (the ``eval:`` YAML section) is used.
    When set, they override the shared value for this specific dataset only.

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
        resolution: Override resolution for this dataset.
        num_inference_steps: Override inference steps for this dataset.
        guidance_scale: Override guidance scale for this dataset.

    YAML Configuration Example:
        ```yaml
        eval_datasets:
          - name: "geneval"
            dataset_dir: "dataset/geneval"
            num_inference_steps: 28    # override shared eval setting
            guidance_scale: 5.0        # override shared eval setting
          - name: "pickscore"
            dataset_dir: "dataset/pickscore"
            max_dataset_size: 500
            # inherits shared eval.num_inference_steps, eval.guidance_scale
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
    # --- Data path overrides ---
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
    # --- Eval generation overrides (None = inherit from shared eval section) ---
    resolution: Optional[Union[int, Tuple[int, int], List[int]]] = field(
        default=None,
        metadata={"help": "Override resolution for this eval dataset. None inherits from eval section."},
    )
    num_inference_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Override number of inference steps. None inherits from eval section."},
    )
    guidance_scale: Optional[float] = field(
        default=None,
        metadata={"help": "Override guidance scale. None inherits from eval section."},
    )

    def get_merged_eval_kwargs(self, base_eval_args) -> dict[str, Any]:
        """
        Merge per-dataset overrides with shared EvaluationArguments.

        Returns a dict of eval kwargs suitable for passing to ``sample_batch()``.
        Per-dataset fields that are not None override the corresponding field
        from ``base_eval_args``.

        Args:
            base_eval_args: The shared ``EvaluationArguments`` instance.

        Returns:
            Dict of merged eval generation kwargs.
        """
        # Start with all shared eval args
        merged = dict(base_eval_args)

        # Override with per-dataset values (only non-None)
        if self.resolution is not None:
            merged['resolution'] = self.resolution
            # Resolve height/width from resolution
            if isinstance(self.resolution, int):
                merged['height'] = self.resolution
                merged['width'] = self.resolution
            elif isinstance(self.resolution, (list, tuple)) and len(self.resolution) >= 2:
                merged['height'] = self.resolution[0]
                merged['width'] = self.resolution[1]
        if self.num_inference_steps is not None:
            merged['num_inference_steps'] = self.num_inference_steps
        if self.guidance_scale is not None:
            merged['guidance_scale'] = self.guidance_scale

        return merged

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        return self.__str__()
