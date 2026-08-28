"""Prepared-backend and checkpoint contracts for multi-role trainers."""

from .backend import (
    MultiRoleBackendValidationMixin,
    configure_deepspeed_micro_batch_size,
    validate_supported_distributed_plan,
)
from .checkpointing import MULTIROLE_RUNTIME_CHILD_NAME, MultiRoleCheckpointingMixin

__all__ = [
    "MultiRoleBackendValidationMixin",
    "MultiRoleCheckpointingMixin",
    "MULTIROLE_RUNTIME_CHILD_NAME",
    "configure_deepspeed_micro_batch_size",
    "validate_supported_distributed_plan",
]
