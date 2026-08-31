"""Model-level selective gradient-checkpointing utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Callable

import torch
from torch.utils.checkpoint import checkpoint

from ..hparams.gradient_checkpointing import GradientCheckpointingSpec

CheckpointUnit = tuple[str, torch.nn.Module]


def discover_gradient_checkpointing_units(module: torch.nn.Module) -> list[CheckpointUnit]:
    """Return repeated model blocks in deterministic module-registration order."""
    repeated_blocks = tuple(getattr(module, "_repeated_blocks", None) or ())
    if not repeated_blocks:
        repeated_blocks = tuple(getattr(module, "_no_split_modules", None) or ())
    if not repeated_blocks:
        raise ValueError(
            f"selective gradient checkpointing expected {type(module).__name__} to "
            "declare `_repeated_blocks` or `_no_split_modules`"
        )
    units = [
        (name, child)
        for name, child in module.named_modules()
        if name and type(child).__name__ in repeated_blocks
    ]
    if not units:
        raise ValueError(
            f"selective gradient checkpointing found no blocks in "
            f"{type(module).__name__}; expected classes={repeated_blocks}"
        )
    return units


def select_gradient_checkpointing_units(
    spec: GradientCheckpointingSpec,
    units: Sequence[CheckpointUnit],
) -> list[CheckpointUnit]:
    """Resolve one selective policy against ordered checkpoint units."""
    unit_count = len(units)
    if unit_count < 1:
        raise ValueError("expected at least one gradient-checkpointing unit")
    if spec.mode == "full":
        return list(units)
    if spec.mode == "none":
        return []
    if spec.mode == "every_n":
        indices = range(0, unit_count, spec.every_n)
    elif spec.mode == "fraction":
        selected_count = max(1, math.ceil(unit_count * spec.fraction))
        if selected_count == 1:
            indices = (unit_count // 2,)
        else:
            indices = tuple(
                round(index * (unit_count - 1) / (selected_count - 1))
                for index in range(selected_count)
            )
    elif spec.mode == "layers":
        invalid = [index for index in spec.layers if index >= unit_count]
        if invalid:
            raise ValueError(
                f"checkpoint layer indices {invalid} exceed unit_count={unit_count}; "
                f"available={[name for name, _ in units]}"
            )
        indices = spec.layers
    else:
        raise ValueError(f"unsupported gradient checkpointing mode={spec.mode!r}")
    return [units[index] for index in indices]


def selective_gradient_checkpointing_function(
    selected_units: Sequence[CheckpointUnit],
) -> Callable[..., Any]:
    """Build the Diffusers callback that checkpoints only selected module identities."""
    selected_ids = frozenset(id(module) for _, module in selected_units)

    def checkpoint_selected(module: torch.nn.Module, *args: Any) -> Any:
        if id(module) not in selected_ids:
            return module(*args)
        return checkpoint(module.__call__, *args, use_reentrant=False)

    return checkpoint_selected
