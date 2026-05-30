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

# src/flow_factory/hparams/dataset_args.py
"""
Unified dataset configuration.

A single :class:`DatasetArguments` describes one on-disk dataset folder
(typically containing both ``train.jsonl`` and ``test.jsonl`` plus shared
``images/`` / ``videos/`` / ``audios/`` subdirectories).  The same entry
opts into training and/or evaluation by populating the ``train`` /
``eval`` sub-blocks (both default to ``None`` = "not used for that split").

This replaces the earlier asymmetric design where training and eval
datasets were configured via separate top-level lists; the new schema
matches the on-disk reality (one folder = both splits) and removes
duplicated path/media-root configuration.

YAML example::

    data:
      datasets:
        - name: geneval
          dataset_dir: dataset/geneval
          train: { weight: 1.0 }
          eval:  { num_inference_steps: 28, guidance_scale: 5.0 }
        - name: pickscore
          dataset_dir: dataset/pickscore
          train: { weight: 3.0, max_dataset_size: 5000 }
          eval:  null            # not used for eval

Reward routing references each dataset by ``name`` via
``RewardArguments.datasets``.  The ``__source__`` carried on every sample
is exactly this name.
"""
from __future__ import annotations

import re
import yaml
from dataclasses import dataclass, field
from typing import Any, ClassVar, List, Optional, Tuple, Union

from .abc import ArgABC


# Names appear in metric keys (``train/source/{name}/...``), in cache
# fingerprints (``train_source:{name}``), and as the routing key for
# ``RewardArguments.datasets`` — so we constrain the alphabet to
# something safe across all three contexts.
_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_\-.]*$")


@dataclass
class DatasetTrainSpec(ArgABC):
    """Per-dataset training participation.

    Attributes:
        enabled: When False, the dataset is excluded from training even
            when this block is present.
        split: Which JSONL/TXT split file to load (typically ``"train"``).
        weight: Mixing weight for this source.  Must be ``> 0``.  Sources
            with larger weights surface more often in the per-step
            schedule.  Uniform mixing = all weights equal.
        max_dataset_size: Per-source cap on number of training samples
            (None = inherit ``DataArguments.max_dataset_size``).
    """

    enabled: bool = field(
        default=True,
        metadata={"help": "Set False to keep this dataset configured but exclude it from training."},
    )
    split: str = field(
        default="train",
        metadata={"help": "Split file for training (default: 'train' -> train.jsonl / train.txt)."},
    )
    weight: float = field(
        default=1.0,
        metadata={"help": "Mixing weight (must be > 0). Per-step source selection probability."},
    )
    max_dataset_size: Optional[int] = field(
        default=None,
        metadata={"help": "Cap on training samples for this source (None = inherit DataArguments.max_dataset_size)."},
    )

    def __post_init__(self) -> None:
        # Concrete weight validation lives in
        # ``Arguments._validate_dataset_routing`` so we can raise with
        # the full per-source context. Here we only normalise the type.
        self.weight = float(self.weight)

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class DatasetEvalSpec(ArgABC):
    """Per-dataset evaluation participation.

    Mirrors the override surface of the legacy
    :class:`flow_factory.hparams.eval_dataset_args.EvalDatasetArguments`,
    minus ``name`` / ``dataset_dir`` / media-root fields (those live on
    the parent :class:`DatasetArguments`).

    Attributes:
        enabled: When False, the dataset is excluded from evaluation
            even when this block is present.
        split: Which JSONL/TXT split file to load (typically ``"test"``).
        max_dataset_size: Per-source cap on number of eval samples.
        resolution / num_inference_steps / guidance_scale: Optional
            overrides for this dataset; ``None`` inherits from the
            shared ``EvaluationArguments``.
    """

    enabled: bool = field(
        default=True,
        metadata={"help": "Set False to keep this dataset configured but exclude it from evaluation."},
    )
    split: str = field(
        default="test",
        metadata={"help": "Split file for evaluation (default: 'test' -> test.jsonl / test.txt)."},
    )
    max_dataset_size: Optional[int] = field(
        default=None,
        metadata={"help": "Cap on eval samples for this source."},
    )
    resolution: Optional[Union[int, Tuple[int, int], List[int]]] = field(
        default=None,
        metadata={"help": "Override eval resolution for this dataset. None inherits shared eval.resolution."},
    )
    num_inference_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Override eval inference steps for this dataset."},
    )
    guidance_scale: Optional[float] = field(
        default=None,
        metadata={"help": "Override eval guidance scale for this dataset."},
    )

    # Scalar fields that override the corresponding ``EvaluationArguments``
    # value when not None. Mirrors the data-driven ClassVar pattern used in
    # the legacy ``EvalDatasetArguments``.
    _EVAL_OVERRIDE_FIELDS: ClassVar[tuple] = ("num_inference_steps", "guidance_scale")

    def get_merged_eval_kwargs(self, base_eval_args) -> dict[str, Any]:
        """Merge per-dataset overrides with shared :class:`EvaluationArguments`.

        Returns a dict of eval kwargs suitable for passing to
        ``BaseTrainer.sample_batch``.  Per-dataset fields that are not
        ``None`` override the corresponding field from ``base_eval_args``.

        Args:
            base_eval_args: The shared ``EvaluationArguments`` instance.

        Returns:
            Dict of merged eval generation kwargs (resolution expanded to
            ``height``/``width`` when overridden).
        """
        merged = dict(base_eval_args)

        # Scalar overrides (data-driven via ClassVar)
        for name in self._EVAL_OVERRIDE_FIELDS:
            val = getattr(self, name, None)
            if val is not None:
                merged[name] = val

        # Resolution requires special handling (expands to height/width)
        if self.resolution is not None:
            merged['resolution'] = self.resolution
            if isinstance(self.resolution, int):
                merged['height'] = self.resolution
                merged['width'] = self.resolution
            elif isinstance(self.resolution, (list, tuple)) and len(self.resolution) >= 2:
                merged['height'] = self.resolution[0]
                merged['width'] = self.resolution[1]

        return merged

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class DatasetArguments(ArgABC):
    """Configuration for a single on-disk dataset folder.

    The same entry can opt into training (via the ``train`` sub-block)
    and/or evaluation (via the ``eval`` sub-block).  When both are set,
    ``train.jsonl`` and ``test.jsonl`` in the same ``dataset_dir`` are
    used; shared ``image_dir`` / ``video_dir`` / ``audio_dir`` live at
    the top level and apply to both splits.

    Attributes:
        name: Unique dataset identifier.  Used as ``__source__`` on every
            sample, in cache fingerprints, in metric keys
            (``train/source/{name}/...`` and ``eval/{name}/...``), and
            as the routing key for ``RewardArguments.datasets``.  Must
            match ``^[A-Za-z0-9_][A-Za-z0-9_\\-.]*$``.
        dataset_dir: Folder containing the split JSONL/TXT files plus
            (by default) ``images/`` / ``videos/`` / ``audios/`` media
            subdirs.
        image_dir / video_dir / audio_dir: Optional per-dataset media
            root overrides.  None falls back to ``{dataset_dir}/<media>``
            (or to ``DataArguments.{image,video,audio}_dir`` when set).
        train: Training participation (None = not used for training).
        eval: Eval participation (None = not used for evaluation).

    YAML example::

        data:
          datasets:
            - name: geneval
              dataset_dir: dataset/geneval
              train: { weight: 1.0 }
              eval:  { num_inference_steps: 28, guidance_scale: 5.0 }
            - name: pickscore
              dataset_dir: dataset/pickscore
              train: { weight: 3.0, max_dataset_size: 5000 }
              eval:  null
    """

    name: str = field(
        default="default",
        metadata={"help": "Unique dataset name (used in metric keys, cache fingerprints, reward routing)."},
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Folder with train.jsonl / test.jsonl plus shared media subdirs."},
    )
    image_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Override image root (default: <dataset_dir>/images)."},
    )
    video_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Override video root."},
    )
    audio_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Override audio root."},
    )
    train: Optional[DatasetTrainSpec] = field(
        default=None,
        metadata={"help": "Training participation (None = not used for training)."},
    )
    eval: Optional[DatasetEvalSpec] = field(
        default=None,
        metadata={"help": "Eval participation (None = not used for eval)."},
    )

    # Stable monotonic integer assigned by `Arguments._assign_source_ids`
    # based on this entry's position in `data.datasets`.  Used as the
    # transport-friendly form of `source` everywhere a string would
    # otherwise be hot-path overhead (per-sample tensor for cross-rank
    # gather, set-membership in `RewardArguments._datasets_resolved`).
    # `None` until alignment runs.
    source_id: Optional[int] = field(default=None, repr=True)

    # Fields that, when not None, override DataArguments-level paths for
    # the per-source GeneralDataset construction.  Drives the
    # data-driven override pattern.
    _DATA_OVERRIDE_FIELDS: ClassVar[tuple] = ("image_dir", "video_dir", "audio_dir")

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name or ""):
            raise ValueError(
                f"DatasetArguments.name={self.name!r} must match {_NAME_RE.pattern!r}. "
                "It is used in metric keys, cache fingerprints, and reward routing."
            )

        # Coerce nested dicts -> dataclasses (ArgABC.from_dict does NOT recurse).
        if isinstance(self.train, dict):
            self.train = DatasetTrainSpec.from_dict(self.train)
        if isinstance(self.eval, dict):
            self.eval = DatasetEvalSpec.from_dict(self.eval)

        # A dataset that opts into neither train nor eval would be a config-file
        # no-op; surface the mistake early.
        if not self.is_training_source and not self.is_eval_source:
            raise ValueError(
                f"DatasetArguments(name={self.name!r}) declares neither `train` nor `eval` "
                "participation. Set at least one of `train: {...}` or `eval: {...}` "
                "(or remove this entry)."
            )

    @property
    def is_training_source(self) -> bool:
        """True iff this dataset participates in training."""
        return self.train is not None and self.train.enabled

    @property
    def is_eval_source(self) -> bool:
        """True iff this dataset participates in evaluation."""
        return self.eval is not None and self.eval.enabled

    def get_dataset_overrides(self) -> dict[str, Any]:
        """Return per-source media-root overrides for ``GeneralDataset.__init__``.

        Returns only the non-None overrides plus ``dataset_dir``, ready
        for ``base_kwargs.update(d.get_dataset_overrides())`` without
        clobbering ``DataArguments`` defaults.
        """
        out: dict[str, Any] = {"dataset_dir": self.dataset_dir}
        for f_name in self._DATA_OVERRIDE_FIELDS:
            v = getattr(self, f_name, None)
            if v is not None:
                out[f_name] = v
        return out

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        return self.__str__()
