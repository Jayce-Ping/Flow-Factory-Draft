"""Prepared-backend and checkpoint contracts for multi-role trainers."""

from .backend import (
    MultiRoleBackendValidationMixin,
    configure_deepspeed_micro_batch_size,
    validate_supported_distributed_plan,
)
from .checkpointing import MultiRoleCheckpointingMixin

__all__ = [
    "MultiRoleBackendValidationMixin",
    "MultiRoleCheckpointingMixin",
    "configure_deepspeed_micro_batch_size",
    "validate_supported_distributed_plan",
]
