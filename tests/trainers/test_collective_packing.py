from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flow_factory.advantage import AdvantageProcessor
from flow_factory.samples import BaseSample
from flow_factory.utils.dist import gather_aligned_floating_tensors


class GatherRecorder:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls.append(tensor.detach().clone())
        return torch.cat((tensor, tensor), dim=0)


def test_aligned_floating_tensors_use_one_gather_and_round_trip_named_columns():
    accelerator = GatherRecorder()

    gathered = gather_aligned_floating_tensors(
        accelerator,
        {
            "reward": torch.tensor([3.0, 4.0], dtype=torch.float64),
            "advantage": torch.tensor([1.0, 2.0], dtype=torch.float32),
        },
    )

    assert len(accelerator.calls) == 1
    assert accelerator.calls[0].shape == (2, 12)
    assert accelerator.calls[0].dtype == torch.uint8
    torch.testing.assert_close(
        gathered["advantage"],
        torch.tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        gathered["reward"],
        torch.tensor([3.0, 4.0, 3.0, 4.0], dtype=torch.float64),
    )


def test_aligned_floating_tensors_reject_misaligned_or_nonfloating_fields():
    accelerator = GatherRecorder()

    with pytest.raises(ValueError, match="expected tensor 'reward' shape"):
        gather_aligned_floating_tensors(
            accelerator,
            {
                "advantage": torch.ones(2),
                "reward": torch.ones(3),
            },
        )
    with pytest.raises(TypeError, match="floating dtype"):
        gather_aligned_floating_tensors(
            accelerator,
            {
                "advantage": torch.ones(2),
                "reward": torch.ones(2, dtype=torch.int64),
            },
        )


def test_zero_std_ratio_rides_the_existing_batched_stats_reduction():
    reductions: list[torch.Tensor] = []

    def reduce(tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        assert reduction == "sum"
        reductions.append(tensor.detach().clone())
        return tensor

    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=reduce,
    )
    processor = AdvantageProcessor(
        accelerator=accelerator,
        reward_weights={"ocr": {"default": 1.0}},
        group_size=2,
        sampler_type="group_contiguous",
    )
    rewards = np.array([1.0, 1.0, 2.0, 2.0])
    group_indices = np.array([0, 0, 1, 1])

    metrics = processor._build_weighted_sum_log_data(
        gathered_rewards={"ocr": rewards},
        group_indices=group_indices,
        aggregated_rewards=rewards,
        advantages=np.zeros(4),
        samples=[BaseSample(prompt=str(index)) for index in range(4)],
    )

    assert len(reductions) == 1
    assert metrics["train/reward_zero_std_ratio"] == 1.0
