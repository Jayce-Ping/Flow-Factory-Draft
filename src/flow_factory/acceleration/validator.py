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

# src/flow_factory/acceleration/validator.py
"""Paradigm-gated safety validation for accelerators (fail-fast, ``constraints.md`` #26).

The correctness contract (``constraints.md`` #7 + #20a):

* A ``lossy`` accelerator changes ``noise_pred`` and cannot be replicated in the
  Stage-6 training forward (which needs full gradient through every block). It is
  therefore only valid in the ``rollout`` slot, and only for **decoupled** /
  **distillation** algorithms — for **coupled** algorithms (GRPO / GRPO-Guard /
  DPPO) the rollout log-prob becomes the PPO "old log-prob" and an approximated
  rollout would bias the importance ratio -> silently wrong gradients.
* A ``lossless`` accelerator is numerically ~identical and, because it mutates the
  shared transformer used by both ``inference()`` and ``forward()``, is safe in
  any slot for any algorithm.
"""

from typing import Optional

from .abc import BaseAccelerator
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)

# RL paradigms that tolerate a lossy (distribution-shifting) rollout: the rollout
# trajectory's per-step log-prob does not enter the training loss.
_LOSSY_SAFE_PARADIGMS = frozenset({"decoupled", "distillation"})


def validate_accelerator(
    accelerator: BaseAccelerator,
    *,
    slot: str,
    paradigm: Optional[str],
    trainer_name: str,
) -> None:
    """Validate one accelerator against its config slot and the trainer paradigm.

    Args:
        accelerator: The constructed accelerator instance.
        slot: The config slot it was placed in — ``"shared"`` (applied to both
            rollout and training via :meth:`~BaseAccelerator.setup`) or
            ``"rollout"`` (applied only during Stage 3 via
            :meth:`~BaseAccelerator.rollout_context`).
        paradigm: The trainer's RL paradigm
            (``"coupled"`` / ``"decoupled"`` / ``"distillation"``), or ``None`` if
            the trainer did not declare one.
        trainer_name: Trainer class name, for error messages.

    Raises:
        ValueError: If the accelerator is unsafe for the given slot / paradigm.
    """
    name = type(accelerator).__name__

    if slot not in ("shared", "rollout"):
        raise ValueError(
            f"Unknown acceleration slot {slot!r} for '{name}'; expected 'shared' or 'rollout'."
        )

    # The `shared` slot applies to BOTH rollout and the training forward, so only
    # lossless accelerators may live there.
    if slot == "shared":
        if accelerator.safety != "lossless":
            raise ValueError(
                f"Accelerator '{name}' (safety='{accelerator.safety}') is configured under "
                "`acceleration.shared_accelerator`, but the shared slot is applied to BOTH "
                "rollout and the training forward — only lossless accelerators are allowed there. "
                "Move a lossy accelerator to `acceleration.rollout_accelerator`."
            )
        if accelerator.stage != "both":
            raise ValueError(
                f"Accelerator '{name}' (stage='{accelerator.stage}') cannot occupy the shared "
                "slot; the shared slot requires a stage='both' accelerator."
            )
        return

    # slot == "rollout": lossless is always fine; lossy is gated on paradigm.
    if accelerator.safety == "lossy":
        if paradigm is None:
            raise ValueError(
                f"Trainer '{trainer_name}' did not declare a `paradigm`, so a lossy rollout "
                f"accelerator ('{name}') cannot be validated for correctness. Set the trainer's "
                "`paradigm` class attribute to one of 'coupled' / 'decoupled' / 'distillation'."
            )
        if paradigm not in _LOSSY_SAFE_PARADIGMS:
            raise ValueError(
                f"Lossy rollout accelerator '{name}' is unsafe for the '{paradigm}' trainer "
                f"'{trainer_name}'. Lossy acceleration changes `noise_pred` during rollout but "
                "cannot be replicated in the training forward; for coupled algorithms "
                "(GRPO / GRPO-Guard / DPPO) this biases the PPO importance ratio and silently "
                f"corrupts gradients. Allowed paradigms: {sorted(_LOSSY_SAFE_PARADIGMS)}. Use a "
                "lossless accelerator (e.g. 'torch_compile') instead, or switch to a decoupled "
                "algorithm (NFT / AWM / DGPO / DPO / CRD)."
            )
        logger.warning(
            "Lossy rollout accelerator '%s' enabled for '%s' (paradigm='%s'). This shifts the "
            "generated-sample distribution; monitor the reward mean/std for regression.",
            name,
            trainer_name,
            paradigm,
        )
