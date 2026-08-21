"""Runtime primitives for forward-process-decoupled trainers."""

from .runtime import (
    DecoupledReplayBatch,
    iter_decoupled_replay_batches,
    iter_decoupled_steps,
)

__all__ = [
    "DecoupledReplayBatch",
    "iter_decoupled_replay_batches",
    "iter_decoupled_steps",
]
