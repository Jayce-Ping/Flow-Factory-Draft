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

"""Training arguments for DiffusionOPD (on-policy distillation).

DiffusionOPD distills several task-specialized teachers into a single
student along the student's own rollout trajectories. Each teacher is a
LoRA checkpoint routed to one or more training datasets (``data.datasets``
entries) via :class:`TeacherConfig.applicable_datasets`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from ..abc import ArgABC
from ._base import TrainingArguments


@dataclass
class TeacherConfig(ArgABC):
    """Configuration for a single DiffusionOPD teacher.

    Attributes:
        path: Path or HuggingFace repo id of the teacher LoRA checkpoint.
            Must share the student's LoRA architecture (target modules;
            rank/alpha compatible) since it is loaded into the student's
            active adapter slot.
        name: Unique teacher identifier (used for the named-parameter
            snapshot and log keys). Defaults to ``opd_teacher_{i}``.
        applicable_datasets: Dataset names (matching ``data.datasets[*].name``)
            whose student rollouts are distilled from this teacher's per-step
            KL. Required and non-empty. NOTE: the schema intentionally permits
            several teachers to list the same dataset (so a future
            multi-teacher/ensemble trainer can reuse this config), but the
            current ``DiffusionOPDTrainer`` distills exactly one teacher per
            dataset and raises if a dataset is claimed by more than one teacher.
        guidance_scale: Optional per-teacher CFG override for the teacher
            forward. ``None`` falls back to the trainer-global student
            ``guidance_scale`` (e.g. a GenEval teacher queried at its
            no-CFG training distribution while the student rolls out at a
            higher scale).
    """

    path: str = field(
        metadata={"help": "Path or HF repo id of the teacher LoRA checkpoint."},
    )
    name: Optional[str] = field(
        default=None,
        metadata={"help": "Unique teacher name (defaults to opd_teacher_{i})."},
    )
    applicable_datasets: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Dataset names this teacher is distilled on (matches data.datasets[*].name)."
        },
    )
    guidance_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "Per-teacher CFG override for the teacher forward (None = student guidance_scale)."
        },
    )


@dataclass
class DiffusionOPDTrainingArguments(TrainingArguments):
    r"""Training arguments for the DiffusionOPD distillation trainer.

    Supports both ODE and SDE dynamics: the per-step loss is
    ``kl_div_j = 0.5 * ||mu_S - mu_T||^2 / denom`` where ``denom`` is the
    scheduler's transition variance (``1`` for ODE), so no extra config is
    needed to switch dynamics — only ``scheduler.dynamics_type``.
    """

    teachers: List[TeacherConfig] = field(
        default_factory=list,
        metadata={"help": "List of teacher configs; each maps a LoRA checkpoint -> dataset(s)."},
    )
    teacher_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={
            "help": "Device to store teacher LoRA snapshots ('cuda' = fast swaps, 'cpu' = lower VRAM)."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        # Coerce dict entries -> TeacherConfig (ArgABC.from_dict does not recurse
        # into nested list-of-ArgABC fields).
        coerced: List[TeacherConfig] = []
        for i, teacher in enumerate(self.teachers):
            if isinstance(teacher, TeacherConfig):
                coerced.append(teacher)
            elif isinstance(teacher, dict):
                coerced.append(TeacherConfig.from_dict(teacher))
            else:
                raise TypeError(
                    f"train.teachers[{i}] must be a dict or TeacherConfig, "
                    f"got {type(teacher).__name__}."
                )
        self.teachers = coerced

        if not self.teachers:
            raise ValueError(
                "DiffusionOPDTrainingArguments requires at least one teacher; "
                "got an empty `train.teachers` list."
            )

        seen_names: set[str] = set()
        for i, teacher in enumerate(self.teachers):
            if not teacher.path:
                raise ValueError(f"train.teachers[{i}] is missing a `path`.")
            if not teacher.applicable_datasets:
                raise ValueError(
                    f"DiffusionOPD requires each teacher to specify non-empty `applicable_datasets`; "
                    f"teacher[{i}] (path={teacher.path!r}) has none. Each teacher must declare "
                    "which `data.datasets[*].name` dataset(s) it is distilled on."
                )
            if teacher.name is None:
                teacher.name = f"opd_teacher_{i}"
            if teacher.name in seen_names:
                raise ValueError(
                    f"Duplicate teacher name {teacher.name!r}; teacher names must be unique."
                )
            seen_names.add(teacher.name)
            if teacher.guidance_scale is not None:
                teacher.guidance_scale = float(teacher.guidance_scale)

    def get_preprocess_guidance_scale(self) -> float:
        """Encode negative prompts if the student OR any teacher uses CFG > 1.

        DiffusionOPD rolls out / forwards at the student ``guidance_scale``
        but may query a teacher at its own (possibly higher) scale, so the
        preprocessing stage must cover the max of both.
        """
        scales = [self.guidance_scale]
        scales += [
            teacher.guidance_scale
            for teacher in self.teachers
            if teacher.guidance_scale is not None
        ]
        return max(scales)
