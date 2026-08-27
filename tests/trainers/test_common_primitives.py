from types import SimpleNamespace

import pytest
import torch

from flow_factory.samples import BaseSample, StackedSampleBatch
from flow_factory.trainers.common.forward_kwargs import (
    reference_forward_kwargs,
    replay_forward_kwargs,
    training_forward_kwargs,
)
from flow_factory.trainers.common.replay_batching import move_and_stack_samples
from flow_factory.trainers.common.sample_prefetch import iter_prefetched_batches


def test_forward_kwargs_preserve_batch_precedence_and_explicit_reference_overrides():
    trainer = SimpleNamespace(training_args={"guidance_scale": 3.0, "resolution": 512, "seed": 42})
    batch = StackedSampleBatch(
        [
            BaseSample(
                prompt="prompt",
                extra_kwargs={"guidance_scale": 1.0},
            )
        ]
    )

    expected = {"resolution": 512, "seed": 42}
    assert training_forward_kwargs(trainer, batch) == expected
    assert replay_forward_kwargs(trainer, batch) == expected
    assert reference_forward_kwargs(trainer, batch, guidance_scale=4.0) == {
        **expected,
        "guidance_scale": 4.0,
    }


def test_forward_kwargs_accept_plain_condition_mapping():
    """Offline conditions need mapping semantics without a rollout sample wrapper."""
    trainer = SimpleNamespace(training_args={"guidance_scale": 3.0, "height": 512})
    condition = {"guidance_scale": 1.0, "prompt_embeds": torch.ones(2, 4)}

    assert training_forward_kwargs(trainer, condition) == {"height": 512}


def test_forward_kwargs_reject_non_mapping_condition():
    trainer = SimpleNamespace(training_args={"height": 512})

    with pytest.raises(TypeError, match="conditioning mapping"):
        training_forward_kwargs(trainer, ["not", "a", "mapping"])


def test_move_and_stack_samples_keeps_moved_sources_on_batch():
    samples = [
        BaseSample(prompt="a", prompt_embeds=torch.tensor([1.0])),
        BaseSample(prompt="b", prompt_embeds=torch.tensor([2.0])),
    ]

    batch = move_and_stack_samples(samples, torch.device("cpu"))

    assert isinstance(batch, StackedSampleBatch)
    assert batch.samples == samples
    assert batch["prompt"] == ["a", "b"]
    torch.testing.assert_close(batch["prompt_embeds"], torch.tensor([[1.0], [2.0]]))


def test_cpu_prefetch_path_chunks_and_stacks_once_per_micro_batch():
    samples = [
        BaseSample(prompt=str(index), prompt_embeds=torch.tensor([float(index)]))
        for index in range(3)
    ]

    batches = list(
        iter_prefetched_batches(
            samples,
            2,
            device=torch.device("cpu"),
            offload_samples_to_cpu=False,
        )
    )

    assert [len(batch.samples) for batch in batches] == [2, 1]
    assert batches[0]["prompt"] == ["0", "1"]
    assert batches[1]["prompt"] == ["2"]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_prefetch_rejects_non_positive_batch_size(batch_size):
    with pytest.raises(ValueError, match="expected per_device_batch_size >= 1"):
        list(
            iter_prefetched_batches(
                [],
                batch_size,
                device=torch.device("cpu"),
                offload_samples_to_cpu=False,
            )
        )
