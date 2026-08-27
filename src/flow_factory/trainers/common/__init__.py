"""Shared, algorithm-independent trainer primitives."""

from .dpo_objective import dpo_objective
from .forward_kwargs import (
    reference_forward_kwargs,
    replay_forward_kwargs,
    training_forward_kwargs,
)
from .offline_batch import bind_output_forward_context, move_condition_to_device
from .replay_batching import move_and_stack_samples
from .runtime_state import TrainerRuntimeState
from .sample_prefetch import iter_prefetched_batches
from .state_validation import (
    require_component_sigmas,
    require_latent_state,
    require_velocity_state,
    state_batch_size,
)

__all__ = [
    "bind_output_forward_context",
    "dpo_objective",
    "iter_prefetched_batches",
    "move_and_stack_samples",
    "move_condition_to_device",
    "reference_forward_kwargs",
    "replay_forward_kwargs",
    "require_component_sigmas",
    "require_latent_state",
    "require_velocity_state",
    "state_batch_size",
    "TrainerRuntimeState",
    "training_forward_kwargs",
]
