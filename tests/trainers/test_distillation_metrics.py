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

"""Cover the distillation metric buffer that feeds DMD, TDM, and TDM-R1 logging."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator

import pytest
import torch

from flow_factory.trainers.distillation.distillation_runtime import (
    generate_one_rollout_batch,
    pop_distillation_metrics,
    record_distillation_metric,
    record_state_statistics,
    run_distillation_training_step,
    run_role_phase,
)


class SingleRankAccelerator:
    """Reduce over one rank, which leaves every buffered value unchanged."""

    device = torch.device("cpu")

    def reduce(self, tensor: torch.Tensor, reduction: str) -> torch.Tensor:
        """Return the tensor untouched, as a one-rank group reduction would."""
        # Accelerate maps every reduction other than "max" onto a sum, so the buffer
        # may only ask for the two it really implements.
        if reduction not in {"sum", "max"}:
            raise ValueError(f"expected a sum or max reduction, received {reduction!r}")
        return tensor


class DivergentAccelerator(SingleRankAccelerator):
    """Simulate a peer rank that buffered a larger metric set."""

    def reduce(self, tensor: torch.Tensor, reduction: str) -> torch.Tensor:
        """Raise the group maximum above this rank's own signature."""
        if reduction == "max":
            return tensor + 1.0
        return tensor


@contextmanager
def _null_context() -> Iterator[None]:
    """Stand in for the trainer's sampling context."""
    yield


def _trainer(accelerator: Any = None) -> SimpleNamespace:
    """Build the minimal trainer surface the metric buffer touches."""
    return SimpleNamespace(accelerator=accelerator or SingleRankAccelerator())


def _state(**components: torch.Tensor) -> SimpleNamespace:
    """Build a latent state exposing only the components mapping."""
    return SimpleNamespace(components=components)


def test_repeated_records_are_averaged_not_overwritten() -> None:
    """A role phase records once per microbatch, so the log must be their mean."""
    trainer = _trainer()

    record_distillation_metric(trainer, "train/fake_loss", 1.0)
    record_distillation_metric(trainer, "train/fake_loss", 3.0)

    assert pop_distillation_metrics(trainer) == {"train/fake_loss": 2.0}


def test_popping_clears_the_buffer_so_epochs_do_not_bleed_together() -> None:
    """A metric left in the buffer would keep being averaged into later epochs."""
    trainer = _trainer()
    record_distillation_metric(trainer, "train/fake_loss", 1.0)

    pop_distillation_metrics(trainer)

    assert pop_distillation_metrics(trainer) == {}


def test_tensor_values_are_detached_from_the_graph() -> None:
    """Buffering a live loss tensor would pin its whole graph until the epoch ends."""
    trainer = _trainer()
    loss = (torch.ones(1, requires_grad=True) * 2).sum()

    record_distillation_metric(trainer, "train/generator_loss", loss)

    assert pop_distillation_metrics(trainer) == {"train/generator_loss": 2.0}


def test_state_statistics_pool_every_component() -> None:
    """A multi-component state must report one mean and std over all of its latents."""
    trainer = _trainer()

    record_state_statistics(
        trainer,
        "train/x0_gen",
        _state(image=torch.tensor([1.0, 3.0]), text=torch.tensor([5.0, 7.0])),
    )

    metrics = pop_distillation_metrics(trainer)
    assert metrics["train/x0_gen_mean"] == pytest.approx(4.0)
    # Population rather than sample std: the values are pooled from per-rank sums, where
    # a Bessel correction has no single N to apply.
    pooled = torch.tensor([1.0, 3.0, 5.0, 7.0])
    assert metrics["train/x0_gen_std"] == pytest.approx(float(pooled.std(unbiased=False)))


def test_diverging_metric_sets_raise_instead_of_deadlocking() -> None:
    """The reduction would hang on rank-dependent keys, which is undiagnosable at scale."""
    trainer = _trainer(DivergentAccelerator())
    record_distillation_metric(trainer, "train/fake_loss", 1.0)

    with pytest.raises(RuntimeError, match="same distillation metrics"):
        pop_distillation_metrics(trainer)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1.0", None])
def test_non_finite_scalars_are_rejected_at_the_call_site(value: Any) -> None:
    """A NaN loss silently averaged into the log hides the step that produced it."""
    with pytest.raises((TypeError, ValueError)):
        record_distillation_metric(_trainer(), "train/fake_loss", value)


def test_non_scalar_tensors_are_rejected() -> None:
    """Recording a per-sample tensor would silently mean-reduce a batch dimension."""
    with pytest.raises(ValueError, match="received a tensor of shape"):
        record_distillation_metric(_trainer(), "train/fake_loss", torch.ones(4))


def test_metrics_are_summed_across_ranks_before_averaging() -> None:
    """Two ranks recording different losses must log the mean of both, not one of them."""

    class TwoRankAccelerator(SingleRankAccelerator):
        def reduce(self, tensor: torch.Tensor, reduction: str) -> torch.Tensor:
            if reduction == "sum":
                # The peer rank buffered the same key twice with value 5.
                return tensor + torch.tensor([10.0, 2.0], dtype=torch.float64)
            return tensor

    trainer = _trainer(TwoRankAccelerator())
    record_distillation_metric(trainer, "train/fake_loss", 1.0)
    record_distillation_metric(trainer, "train/fake_loss", 3.0)

    metrics: Dict[str, float] = pop_distillation_metrics(trainer)
    assert metrics["train/fake_loss"] == pytest.approx((1.0 + 3.0 + 10.0) / 4.0)


def test_an_epoch_logs_what_its_roles_recorded() -> None:
    """Metrics recorded during optimize must reach the logger against the epoch's step."""
    logged: list = []

    def optimize(microbatches: Any) -> None:
        del microbatches
        record_distillation_metric(trainer, "train/generator_loss", 7.0)

    trainer = SimpleNamespace(
        accelerator=SingleRankAccelerator(),
        training_args=SimpleNamespace(gradient_accumulation_steps=2),
        epoch=3,
        step=11,
        show_progress_bar=False,
        sampling_context=lambda: _null_context(),
        sample=lambda: ["sample"],
        _prepare_training_feedback=lambda samples: None,
        optimize=optimize,
        log_data=lambda data, step: logged.append((data, step)),
    )

    run_distillation_training_step(trainer)

    assert logged == [({"train/generator_loss": 7.0}, 11)]


def test_role_variant_stays_active_through_backward_recomputation() -> None:
    events: list[tuple[str, str]] = []

    class Adapter:
        active = "generator"

        @contextmanager
        def use_component_variant(self, role_name: str) -> Iterator[None]:
            previous = self.active
            self.active = role_name
            try:
                yield
            finally:
                self.active = previous

    class Coordinator:
        roles = {"surrogate": SimpleNamespace(last_grad_norm=None)}

        @contextmanager
        def phase(self, role_name: str) -> Iterator[None]:
            events.append(("phase", role_name))
            yield

        @contextmanager
        def microbatch(self) -> Iterator[None]:
            yield

        def backward(self, loss: torch.Tensor) -> None:
            events.append(("backward", adapter.active))
            loss.backward()

    adapter = Adapter()
    trainer = SimpleNamespace(
        adapter=adapter,
        role_optimization=Coordinator(),
        epoch=0,
        show_progress_bar=False,
        _finish_role_microbatch=lambda: None,
    )

    run_role_phase(
        trainer,
        "surrogate",
        [["sample"]],
        lambda batch: torch.ones((), requires_grad=True),
    )

    assert events == [("phase", "surrogate"), ("backward", "surrogate")]
    assert adapter.active == "generator"


def test_each_rollout_scores_only_the_batch_it_just_generated() -> None:
    """A buffer carried between iterations mismatches reward and sample counts.

    The reward processor packs rewards beside per-sample ids before gathering, so a
    stale batch in the buffer surfaces as a shape error deep in the advantage code
    rather than anywhere near the rollout that caused it.
    """

    class RecordingBuffer:
        def __init__(self) -> None:
            self.samples: list = []
            self.clears = 0

        def clear(self) -> None:
            self.clears += 1
            self.samples = []

        def add_samples(self, samples: list) -> None:
            self.samples.extend(samples)

    buffer = RecordingBuffer()
    trainer = SimpleNamespace(
        dataloader=[["prompt"]],
        adapter=SimpleNamespace(rollout=lambda: None),
        _rollout_acceleration=_null_context,
        autocast=_null_context,
        sample_batch=lambda batch, **kwargs: (
            kwargs["reward_buffer"].add_samples(["sample"]) or ["sample"]
        ),
    )

    for _ in range(2):
        generate_one_rollout_batch(trainer, reward_buffer=buffer, algorithm_name="TDM-R1")

    assert buffer.clears == 2
    assert buffer.samples == ["sample"]


def test_an_epoch_that_records_nothing_logs_nothing() -> None:
    """An empty log call would still stamp a step and clutter the run's history."""
    logged: list = []
    trainer = SimpleNamespace(
        accelerator=SingleRankAccelerator(),
        training_args=SimpleNamespace(gradient_accumulation_steps=1),
        epoch=0,
        step=0,
        show_progress_bar=False,
        sampling_context=lambda: _null_context(),
        sample=lambda: ["sample"],
        _prepare_training_feedback=lambda samples: None,
        optimize=lambda microbatches: None,
        log_data=lambda data, step: logged.append((data, step)),
    )

    run_distillation_training_step(trainer)

    assert logged == []
