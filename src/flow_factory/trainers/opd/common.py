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
"""Shared target-space math and teacher loading for DiffusionOPD.

Target projection is shared by the teacher and student passes; the per-sample
loss consumes their projected outputs. Teacher loading stores each teacher LoRA
checkpoint in a named-parameter snapshot using the adapter primitives in
:mod:`flow_factory.models.abc`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

import torch

from ...utils.base import to_broadcast_tensor
from ...utils.logger_utils import setup_logger
from ...utils.noise_schedule import flow_match_sigma

if TYPE_CHECKING:
    from ...models.abc import BaseAdapter

logger = setup_logger(__name__, rank_zero_only=True)


def validate_loss_target_for_dynamics(loss_target: str, dynamics_type: str) -> None:
    """Validate that the target is defined for the scheduler dynamics.

    Args:
        loss_target: Configured target space (``xt``, ``v``, or ``x0``).
        dynamics_type: Active scheduler dynamics.

    Raises:
        ValueError: ``v`` or ``x0`` is requested for non-ODE dynamics.
    """
    if loss_target in ("v", "x0") and dynamics_type != "ODE":
        raise ValueError(
            "DiffusionOPD velocity-derived targets require ODE dynamics: "
            f"received loss_target={loss_target!r} with dynamics_type={dynamics_type!r}. "
            "Use scheduler.dynamics_type='ODE' or set train.loss_target='xt'."
        )


def _require_matching_target(
    value: Optional[torch.Tensor],
    *,
    name: str,
    loss_target: str,
    latents: torch.Tensor,
) -> torch.Tensor:
    """Validate and return one model output used as a target."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"Expected `{name}` to be a torch.Tensor for loss_target={loss_target!r}, "
            f"got {type(value).__name__}: {value!r}."
        )
    if value.shape != latents.shape:
        raise ValueError(
            "DiffusionOPD target projection requires the same shape for `latents` "
            f"and `{name}`, got latents={tuple(latents.shape)} and "
            f"{name}={tuple(value.shape)}."
        )
    return value


def project_distillation_target(
    *,
    loss_target: str,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    next_latents_mean: Optional[torch.Tensor],
    velocity: Optional[torch.Tensor],
) -> torch.Tensor:
    """Project a scheduler step output into the configured target space.

    Args:
        loss_target: Target space (``xt``, ``v``, or ``x0``).
        latents: Current noisy latent state.
        timestep: Current scheduler-scale timestep.
        next_latents_mean: Predicted one-step transition mean.
        velocity: Predicted flow velocity.

    Returns:
        The prediction represented in the configured target space.

    Raises:
        TypeError: A required input is not a tensor.
        ValueError: The target is unsupported or a model output shape differs
            from ``latents``.
    """
    if not isinstance(latents, torch.Tensor):
        raise TypeError(
            "Expected `latents` to be a torch.Tensor when projecting a DiffusionOPD "
            f"target, got {type(latents).__name__}: {latents!r}."
        )

    if loss_target == "xt":
        return _require_matching_target(
            next_latents_mean,
            name="next_latents_mean",
            loss_target=loss_target,
            latents=latents,
        )
    if loss_target not in ("v", "x0"):
        raise ValueError(
            f"DiffusionOPD loss_target must be one of ('xt', 'v', 'x0'), got {loss_target!r}."
        )

    velocity = _require_matching_target(
        velocity,
        name="velocity",
        loss_target=loss_target,
        latents=latents,
    )
    if loss_target == "v":
        return velocity

    if not isinstance(timestep, torch.Tensor):
        raise TypeError(
            "Expected `timestep` to be a torch.Tensor for loss_target='x0', "
            f"got {type(timestep).__name__}: {timestep!r}."
        )
    latents_float = latents.float()
    sigma = to_broadcast_tensor(flow_match_sigma(timestep.float()), latents_float)
    return latents_float - sigma * velocity.float()


def compute_per_sample_distillation_loss(
    student_target: torch.Tensor,
    teacher_target: torch.Tensor,
    *,
    self_normalize: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute target-space MSE with optional detached self-normalization.

    Args:
        student_target: Student prediction in the configured target space.
        teacher_target: Detached teacher prediction in the same target space.
        self_normalize: Whether to divide by detached mean absolute error.
        eps: Positive denominator floor added after self-normalization.

    Returns:
        Per-sample loss reduced over all non-batch dimensions.

    Raises:
        TypeError: Targets are not tensors or ``self_normalize`` is not bool.
        ValueError: Target shapes are invalid or ``eps`` is not positive.
    """
    if not isinstance(student_target, torch.Tensor) or not isinstance(teacher_target, torch.Tensor):
        raise TypeError(
            "Expected `student_target` and `teacher_target` to be torch.Tensor values, "
            f"got {type(student_target).__name__} and {type(teacher_target).__name__}."
        )
    if student_target.shape != teacher_target.shape:
        raise ValueError(
            "DiffusionOPD loss requires matching shapes for `student_target` and "
            f"`teacher_target`, got {tuple(student_target.shape)} and "
            f"{tuple(teacher_target.shape)}."
        )
    if student_target.ndim < 2:
        raise ValueError(
            "DiffusionOPD loss requires target tensors with a batch dimension and at "
            f"least one feature dimension, got shape {tuple(student_target.shape)}."
        )
    if not isinstance(self_normalize, bool):
        raise TypeError(
            "Expected `self_normalize` to be a bool, "
            f"got {type(self_normalize).__name__}: {self_normalize!r}."
        )
    if eps <= 0:
        raise ValueError(f"Expected `eps` to be positive, got {eps!r}.")

    error = student_target.float() - teacher_target.float()
    per_sample_mse = error.square().flatten(1).mean(dim=1)
    if not self_normalize:
        return per_sample_mse
    scale = error.abs().flatten(1).mean(dim=1).detach()
    return per_sample_mse / (scale + eps)


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
