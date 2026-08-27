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

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

import pytest
import torch

from flow_factory.contracts import OFFLINE_EXECUTION_CONTRACT, MediaType
from flow_factory.data_utils.offline_dataset import (
    DecodedMedia,
    DemonstrationOutputBatch,
    OfflineBatch,
    PreferenceOutputBatch,
)
from flow_factory.data_utils.schema import NormalizedModelInput
from flow_factory.models.output_state import (
    EncodedOutputState,
    GeometrySignature,
    MediaGeometrySignature,
)
from flow_factory.samples import ComponentTimes, LatentState, MultiModalStepOutput, NoisedState
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.execution import TrainingProgress
from flow_factory.trainers.offline.sft import SFTTrainer


class _TrainingArguments(dict):
    """Minimal mapping/attribute hybrid used by shared forward helpers."""

    def __init__(self, num_train_timesteps: int = 2) -> None:
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.weighting_scheme = "uniform"
        self.timestep_range = (0.0, 0.99)
        self.time_shift = 1.0
        self.logit_mean = 0.0
        self.logit_std = 1.0


class _Accelerator:
    """Record one Accelerate accumulation decision per dataloader microbatch."""

    def __init__(self, sync_schedule: list[bool]) -> None:
        self.device = torch.device("cpu")
        self._sync_schedule = iter(sync_schedule)
        self.sync_gradients = False
        self.accumulate_calls = 0
        self.backward_losses: list[torch.Tensor] = []
        self.prepare_calls = 0

    @contextmanager
    def accumulate(self, model_bundle: Any) -> Iterator[None]:
        """Set the next synchronization boundary and record the prepared root."""
        assert model_bundle is _MODEL_BUNDLE
        self.accumulate_calls += 1
        self.sync_gradients = next(self._sync_schedule)
        yield

    def backward(self, loss: torch.Tensor) -> None:
        """Record the exact scalar passed to the single backward call."""
        self.backward_losses.append(loss.detach().clone())

    def prepare(self, *args: Any, **kwargs: Any) -> Any:
        """Reject attempts to reshape an already-distributed offline loader."""
        del args, kwargs
        self.prepare_calls += 1
        raise AssertionError("offline train dataloader must not use accelerator.prepare")


class _Adapter:
    """Exercise the typed output codec and structured forward-process seams."""

    trajectory_component_order = ("latent",)

    def __init__(self, *, fail_forward_call: int | None = None) -> None:
        self.training = False
        self.train_calls = 0
        self.encode_calls = 0
        self.forward_calls = 0
        self.fail_forward_call = fail_forward_call
        self.encode_conditions: list[Mapping[str, Any]] = []
        self.forward_batches: list[Mapping[str, Any]] = []

    def train(self) -> None:
        """Return trainable components to train mode after evaluation."""
        self.training = True
        self.train_calls += 1

    def encode_output_state(
        self,
        media_batch: tuple[tuple[DecodedMedia, ...], ...],
        condition: Mapping[str, Any],
    ) -> EncodedOutputState:
        """Represent one on-the-fly VAE pass with output-owned forward context."""
        self.encode_calls += 1
        self.encode_conditions.append(condition)
        batch_size = len(media_batch)
        signature = GeometrySignature(
            media=(
                MediaGeometrySignature(
                    type=MediaType.IMAGE,
                    height=8,
                    width=8,
                ),
            )
        )
        return EncodedOutputState(
            clean_state=LatentState({"latent": torch.zeros(batch_size, 1)}),
            forward_context={"output_ids": torch.arange(batch_size).unsqueeze(1)},
            decode_context={},
            geometry_signatures=tuple(signature for _ in range(batch_size)),
        )

    def build_training_component_times(
        self,
        primary_timesteps: torch.Tensor,
        *,
        batch: Mapping[str, Any],
    ) -> ComponentTimes:
        """Preserve the sampled coordinate for the fake velocity prediction."""
        assert batch["prompt_embeds"].shape[0] == primary_timesteps.shape[0]
        zeros = torch.zeros_like(primary_timesteps)
        return ComponentTimes(
            timestep={"latent": primary_timesteps},
            next_timestep={"latent": zeros},
            sigma={"latent": zeros},
            next_sigma={"latent": zeros},
        )

    def add_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        *,
        generator: torch.Generator | None = None,
    ) -> NoisedState:
        """Return a zero target so sampled coordinates determine known losses."""
        del times, generator
        target = LatentState({"latent": torch.zeros_like(clean_state.components["latent"])})
        return NoisedState(state=clean_state, target_velocity=target, noise=target)

    def forward_state(
        self,
        *,
        batch: Mapping[str, Any],
        state: LatentState,
        times: ComponentTimes,
        **kwargs: Any,
    ) -> MultiModalStepOutput:
        """Return a coordinate-valued velocity and record the complete model batch."""
        del kwargs
        self.forward_calls += 1
        if self.forward_calls == self.fail_forward_call:
            raise RuntimeError("injected SFT forward failure")
        assert self.training
        self.forward_batches.append(batch)
        value = times.timestep["latent"].reshape(-1, 1).expand_as(state.components["latent"])
        return MultiModalStepOutput(velocity=LatentState({"latent": value}))

    def reduce_latent_values(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        state: LatentState,
    ) -> torch.Tensor:
        """Apply the standard one-component non-batch mean reduction."""
        del state
        return values["latent"].flatten(1).mean(dim=1)


_MODEL_BUNDLE = object()


def _batch(batch_size: int = 2) -> OfflineBatch:
    target_media = tuple(
        (
            DecodedMedia(
                type="image",
                path=f"target-{index}.png",
                payload=torch.zeros(3, 8, 8),
            ),
        )
        for index in range(batch_size)
    )
    return OfflineBatch(
        condition={
            "prompt_embeds": torch.zeros(batch_size, 3),
            "nested": {"attention_mask": torch.ones(batch_size, 2)},
        },
        condition_ids=tuple(f"condition-{index}" for index in range(batch_size)),
        record_ids=tuple(f"record-{index}" for index in range(batch_size)),
        sources=tuple("source" for _ in range(batch_size)),
        source_ids=torch.zeros(batch_size, dtype=torch.long),
        model_inputs=tuple(
            NormalizedModelInput(prompt="prompt", negative_prompt=None, media=())
            for _ in range(batch_size)
        ),
        supervision_type="demonstration",
        output=DemonstrationOutputBatch(target_media=target_media),
        metadata_json=tuple("{}" for _ in range(batch_size)),
    )


def _trainer(
    accelerator: _Accelerator,
    adapter: _Adapter,
) -> tuple[SFTTrainer, list[dict[str, list[torch.Tensor]]]]:
    trainer = object.__new__(SFTTrainer)
    trainer.accelerator = accelerator
    trainer.adapter = adapter
    trainer.training_args = _TrainingArguments()
    trainer.model_bundle = _MODEL_BUNDLE
    trainer.autocast = nullcontext
    trainer.progress = TrainingProgress()
    trainer._loss_info = defaultdict(list)
    optimizer_windows: list[dict[str, list[torch.Tensor]]] = []

    def apply_optimizer_step(
        loss_info: dict[str, list[torch.Tensor]],
    ) -> dict[str, list[torch.Tensor]]:
        optimizer_windows.append({key: list(values) for key, values in loss_info.items()})
        trainer.step += 1
        return defaultdict(list)

    trainer._apply_optimizer_step = apply_optimizer_step
    return trainer, optimizer_windows


def test_sft_declares_the_offline_decoupled_execution_contract() -> None:
    """SFT uses the finite dataset driver without inheriting an online trainer."""
    assert SFTTrainer.__bases__ == (BaseTrainer,)
    assert SFTTrainer.execution_contract is OFFLINE_EXECUTION_CONTRACT
    assert SFTTrainer.paradigm == "decoupled"


def test_sft_builds_demonstration_loader_without_accelerator_prepare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The offline loader remains owned by its official DistributedSampler."""
    accelerator = _Accelerator([])
    trainer = object.__new__(SFTTrainer)
    trainer.config = object()
    trainer.accelerator = accelerator
    preprocess_func = lambda row: row
    pipeline_io_contract = object()
    trainer.adapter = SimpleNamespace(
        preprocess_func=preprocess_func,
        pipeline_io_contract=pipeline_io_contract,
    )
    expected_loader = object()
    received: dict[str, Any] = {}

    def build_loader(*args: Any, **kwargs: Any) -> Any:
        received["args"] = args
        received["kwargs"] = kwargs
        return expected_loader

    monkeypatch.setattr(
        "flow_factory.trainers.offline.sft.build_offline_train_dataloader",
        build_loader,
    )

    loader, loaders_by_source = trainer._build_train_dataloader()

    assert loader is expected_loader
    assert loaders_by_source == {}
    assert received == {
        "args": (trainer.config, accelerator, preprocess_func),
        "kwargs": {
            "supervision_type": "demonstration",
            "pipeline_io_contract": pipeline_io_contract,
        },
    }
    assert accelerator.prepare_calls == 0


def test_sft_averages_time_terms_inside_one_microstep_and_steps_on_gas_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T loss terms do not multiply accumulation or optimizer cadence."""
    accelerator = _Accelerator([False, True])
    adapter = _Adapter()
    trainer, optimizer_windows = _trainer(accelerator, adapter)
    batch = _batch()
    monkeypatch.setattr(
        "flow_factory.trainers.offline.sft.sample_offline_timesteps",
        lambda *args, **kwargs: torch.tensor([[1.0, 1.0], [3.0, 3.0]]),
    )

    trainer.optimize_batch(batch)
    assert trainer.step == 0
    trainer.optimize_batch(batch)

    assert accelerator.accumulate_calls == 2
    assert len(accelerator.backward_losses) == 2
    torch.testing.assert_close(torch.stack(accelerator.backward_losses), torch.tensor([5.0, 5.0]))
    assert trainer.step == 1
    assert len(optimizer_windows) == 1
    torch.testing.assert_close(
        torch.stack(optimizer_windows[0]["loss"]),
        torch.tensor([5.0, 5.0]),
    )
    assert len(optimizer_windows[0]["flow_matching_loss"]) == 2
    assert all(value.ndim == 0 for value in optimizer_windows[0]["flow_matching_loss"])
    assert not trainer._loss_info

    # The same decoded batch is encoded again instead of retaining target latents.
    assert adapter.encode_calls == 2
    assert adapter.train_calls == 2
    assert adapter.forward_calls == 4
    assert all(
        model_batch["prompt_embeds"].shape == (2, 3) for model_batch in adapter.forward_batches
    )
    assert all(model_batch["output_ids"].shape == (2, 1) for model_batch in adapter.forward_batches)
    assert all("output_ids" not in condition for condition in adapter.encode_conditions)


def test_sft_forward_failure_does_not_backward_step_or_mutate_loss_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial T loop cannot be mistaken for a completed microstep."""
    accelerator = _Accelerator([False, True])
    adapter = _Adapter(fail_forward_call=4)
    trainer, optimizer_windows = _trainer(accelerator, adapter)
    monkeypatch.setattr(
        "flow_factory.trainers.offline.sft.sample_offline_timesteps",
        lambda *args, **kwargs: torch.tensor([[1.0, 1.0], [3.0, 3.0]]),
    )

    trainer.optimize_batch(_batch())
    retained_loss = trainer._loss_info["loss"][0].clone()
    with pytest.raises(RuntimeError, match="injected SFT forward failure"):
        trainer.optimize_batch(_batch())

    assert accelerator.accumulate_calls == 2
    assert len(accelerator.backward_losses) == 1
    assert optimizer_windows == []
    assert trainer.step == 0
    assert len(trainer._loss_info["loss"]) == 1
    torch.testing.assert_close(trainer._loss_info["loss"][0], retained_loss)
    assert len(trainer._loss_info["flow_matching_loss"]) == 1


def test_sft_rejects_non_demonstration_batches_before_model_side_effects() -> None:
    """Algorithm branch selection follows typed offline supervision, not field guessing."""
    accelerator = _Accelerator([])
    adapter = _Adapter()
    trainer, _ = _trainer(accelerator, adapter)

    with pytest.raises(TypeError, match="expected OfflineBatch"):
        trainer.optimize_batch(object())

    invalid = _batch()
    invalid = OfflineBatch(
        condition=invalid.condition,
        condition_ids=invalid.condition_ids,
        record_ids=invalid.record_ids,
        sources=invalid.sources,
        source_ids=invalid.source_ids,
        model_inputs=invalid.model_inputs,
        supervision_type="preference",
        output=PreferenceOutputBatch(
            chosen_media=invalid.output.target_media,
            rejected_media=invalid.output.target_media,
        ),
        metadata_json=invalid.metadata_json,
    )
    with pytest.raises(ValueError, match="supervision_type='demonstration'"):
        trainer.optimize_batch(invalid)

    assert adapter.train_calls == 0
    assert adapter.encode_calls == 0
    assert accelerator.accumulate_calls == 0
