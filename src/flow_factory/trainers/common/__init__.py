"""Shared, algorithm-independent trainer primitives."""

from .forward_kwargs import (
    reference_forward_kwargs,
    replay_forward_kwargs,
    training_forward_kwargs,
)
from .replay_batching import move_and_stack_samples
from .sample_prefetch import iter_prefetched_batches
from .state_validation import (
    require_component_sigmas,
    require_latent_state,
    require_velocity_state,
    state_batch_size,
)

__all__ = [
    "iter_prefetched_batches",
    "move_and_stack_samples",
    "reference_forward_kwargs",
    "replay_forward_kwargs",
    "require_component_sigmas",
    "require_latent_state",
    "require_velocity_state",
    "state_batch_size",
    "training_forward_kwargs",
]
