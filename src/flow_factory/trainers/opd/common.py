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

# src/flow_factory/trainers/opd/common.py
"""Teacher-loading helper for the DiffusionOPD trainer.

The trainer's 2-pass design (teacher targets pre-computed in a no_grad pass,
then a student-only gradient loop) removes the reference implementation's hot
swap machinery, so this module needs exactly one helper: :func:`load_teachers`,
which loads each teacher LoRA checkpoint into a named-parameter snapshot using
the adapter primitives already in :mod:`flow_factory.models.abc`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

import torch

from ...utils.logger_utils import setup_logger

if TYPE_CHECKING:
    from ...models.abc import BaseAdapter

logger = setup_logger(__name__, rank_zero_only=True)


def load_teachers(
    adapter: "BaseAdapter",
    teacher_paths: List[str],
    teacher_param_device: str,
    teacher_names: Optional[List[Optional[str]]] = None,
) -> List[str]:
    """Load each teacher LoRA checkpoint into a named-parameter snapshot.

    For every teacher the live student LoRA tensors are snapshotted, the
    teacher checkpoint is loaded into the active adapter slot via
    :meth:`BaseAdapter._load_lora` (clobbering the student weights), captured
    into a named snapshot via :meth:`BaseAdapter.add_named_parameters`, and the
    student weights are then restored. Swap a teacher in at run time with
    ``with adapter.use_named_parameters(name): ...``.

    Because ``_load_lora`` loads into the student's active ``"default"`` adapter
    slot, every teacher checkpoint MUST share the student's LoRA architecture
    (same ``target_components`` / target modules and rank-compatible weights).
    Incompatible checkpoints raise a clear error pointing at this constraint.

    Args:
        adapter: Active :class:`BaseAdapter` in LoRA finetune mode with the
            student adapter already attached.
        teacher_paths: Local checkpoint paths or HF Hub repo ids
            (``owner/repo[/subfolder][@revision]``, optional ``hf://`` prefix),
            resolved via :meth:`BaseAdapter._resolve_checkpoint_path`. Must be
            non-empty.
        teacher_param_device: ``'cpu'`` (low VRAM, H2D copy per swap) or
            ``'cuda'`` (on-device, LoRA-sized VRAM per teacher).
        teacher_names: Optional snapshot names (one per path). A ``None`` entry
            (or short list) falls back to ``'opd_teacher_{i}'``.

    Returns:
        Snapshot names in the same order as ``teacher_paths`` -- the lookup
        keys for :meth:`BaseAdapter.use_named_parameters`.

    Raises:
        ValueError: ``teacher_paths`` is empty, the adapter is not in LoRA
            mode, or it exposes no trainable LoRA components.
        RuntimeError: a teacher checkpoint is incompatible with the student's
            LoRA architecture.
    """
    if not teacher_paths:
        raise ValueError(
            f"DiffusionOPD requires at least one teacher LoRA path; got teacher_paths={teacher_paths!r}."
        )
    if adapter.model_args.finetune_type != "lora":
        raise ValueError(
            "load_teachers requires the adapter to be in 'lora' finetune mode "
            f"(teacher LoRAs load into the student's adapter slot), but "
            f"model_args.finetune_type={adapter.model_args.finetune_type!r}."
        )

    target_components: List[str] = [
        comp for comp, mods in adapter.target_module_map.items() if mods
    ]
    if not target_components:
        raise ValueError(
            "Adapter has no trainable LoRA components; expected at least one entry with "
            f"non-empty modules in target_module_map={adapter.target_module_map!r}."
        )

    names: List[str] = []
    for i, path in enumerate(teacher_paths):
        name = (
            teacher_names[i]
            if teacher_names and i < len(teacher_names) and teacher_names[i]
            else f"opd_teacher_{i}"
        )
        _load_one_teacher(adapter, name, path, target_components, teacher_param_device)
        names.append(name)

    logger.info(
        f"Loaded {len(names)} DiffusionOPD teacher(s): {names} (device={teacher_param_device!r})."
    )
    return names


def _load_one_teacher(
    adapter: "BaseAdapter",
    name: str,
    lora_path: str,
    target_components: List[str],
    device: str,
) -> None:
    """Load one teacher LoRA into snapshot ``name``, restoring student weights.

    The student LoRA tensors are the live ``nn.Parameter`` objects; ``_load_lora``
    mutates them in place, so we keep detached clones and copy them back in a
    ``finally`` block (even if loading raised). Loading errors are surfaced with
    the LoRA-architecture constraint that almost always causes them.
    """
    # Resolve HF Hub specs / validate local layout before touching weights.
    lora_path = adapter._resolve_checkpoint_path(lora_path)
    if len(target_components) > 1:
        for comp in target_components:
            sub = os.path.join(lora_path, comp)
            if not os.path.exists(sub):
                raise FileNotFoundError(
                    f"Multi-component LoRA layout requires per-component subdirectories; "
                    f"missing {sub!r} for component {comp!r} under teacher path {lora_path!r}."
                )

    live_params = adapter._get_component_parameters(target_components)
    if not live_params:
        raise ValueError(
            f"No trainable LoRA parameters found on components {target_components!r}; "
            "ensure the student LoRA adapter is attached before loading teachers."
        )
    saved_data = [p.detach().clone() for p in live_params]

    try:
        adapter._load_lora(lora_path)
        adapter.add_named_parameters(
            name=name,
            target_components=target_components,
            device=device,
            overwrite=True,
        )
    except (RuntimeError, ValueError, KeyError, TypeError) as e:
        # Almost always a LoRA-architecture mismatch (rank/alpha/target modules)
        # between the teacher checkpoint and the student adapter slot.
        raise RuntimeError(
            f"Failed to load teacher LoRA {name!r} from {lora_path!r}. Teacher checkpoints "
            f"must share the student's LoRA architecture (target_components={target_components}, "
            "matching target modules and compatible rank/alpha), since they load into the "
            "student's active adapter slot. Verify the teacher was trained with the same "
            f"LoRA config as the student. Original error: {e}"
        ) from e
    finally:
        # Always restore the student weights, even if loading/snapshotting raised.
        with torch.no_grad():
            for live, saved in zip(live_params, saved_data):
                live.data.copy_(saved.to(live.device))
