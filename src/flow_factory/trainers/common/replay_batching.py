"""Sample movement and collation for replay-oriented trainer loops."""

from collections.abc import Sequence

import torch

from ...samples import BaseSample, StackedSampleBatch


def move_and_stack_samples(
    samples: Sequence[BaseSample],
    device: str | torch.device,
    *,
    non_blocking: bool = False,
) -> StackedSampleBatch:
    """Move one micro-batch to ``device`` and collate it exactly once."""
    moved = [sample.to(device, non_blocking=non_blocking) for sample in samples]
    return BaseSample.stack(moved)
