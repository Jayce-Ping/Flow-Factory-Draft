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

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
import torch

from flow_factory.contracts import MediaType
from flow_factory.contracts.execution import (
    OFFLINE_EXECUTION_CONTRACT,
    AcquisitionMode,
    FeedbackMode,
)
from flow_factory.data_utils.offline_dataset import (
    DecodedMedia,
    DemonstrationOutputBatch,
    OfflineBatch,
    PreferenceOutputBatch,
)
from flow_factory.models.output_state import (
    EncodedOutputState,
    GeometrySignature,
    MediaGeometrySignature,
)
from flow_factory.samples import ComponentTimes, LatentState, MultiModalStepOutput, NoisedState
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.offline import offline_dpo as offline_dpo_module
from flow_factory.trainers.offline.offline_dpo import OfflineDPOTrainer


class _TrainingArgs(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _FakeAccelerator:
    def __init__(self, sync_schedule: list[bool]) -> None:
        self.device = torch.device("cpu")
        self.sync_gradients = False
        self.sync_schedule = sync_schedule
        self.accumulate_roots: list[Any] = []
        self.backward_losses: list[torch.Tensor] = []

    @contextmanager
    def accumulate(self, root: Any) -> Iterator[None]:
        index = len(self.accumulate_roots)
        self.accumulate_roots.append(root)
        self.sync_gradients = self.sync_schedule[index]
        yield

    def backward(self, loss: torch.Tensor) -> None:
        self.backward_losses.append(loss.detach().clone())
        loss.backward()

    def prepare(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("offline train dataloader must not use Accelerator.prepare")


class _FakeAdapter:
    trajectory_component_order = ("latent",)
    preprocess_func = object()

    def __init__(self, autocast_active: Any) -> None:
        self.policy_weight = torch.nn.Parameter(torch.tensor(0.7))
        self.autocast_active = autocast_active
        self.train_calls = 0
        self.encode_calls: list[str] = []
        self.time_calls: list[tuple[float, torch.Tensor]] = []
        self.drawn_noise: list[LatentState] = []
        self.reused_noise: list[LatentState] = []
        self.forward_events: list[tuple[float, bool, bool, bool]] = []
        self.ref_scope_enters = 0
        self.ref_scope_exits = 0
        self._ref_active = False

    def train(self, mode: bool = True) -> None:
        assert mode is True
        self.train_calls += 1

    def encode_output_state(
        self,
        media_batch: tuple[tuple[DecodedMedia, ...], ...],
        condition: dict[str, Any],
        generator: torch.Generator | None = None,
    ) -> EncodedOutputState:
        del condition, generator
        arm = str(media_batch[0][0].payload)
        self.encode_calls.append(arm)
        arm_value = 0.0 if arm == "chosen" else 1.0
        batch_size = len(media_batch)
        clean = torch.full((batch_size, 2), arm_value, dtype=torch.float32)
        signature = GeometrySignature(
            media=(
                MediaGeometrySignature(
                    type=MediaType.IMAGE,
                    height=16,
                    width=16,
                ),
            )
        )
        return EncodedOutputState(
            clean_state=LatentState({"latent": clean}),
            forward_context={
                "arm_token": torch.full((batch_size, 1), arm_value, dtype=torch.float32)
            },
            decode_context={"height": 16, "width": 16},
            geometry_signatures=tuple(signature for _ in range(batch_size)),
        )

    def build_training_component_times(
        self,
        primary_timesteps: torch.Tensor,
        *,
        batch: dict[str, Any],
    ) -> ComponentTimes:
        arm = float(batch["arm_token"][0].item())
        self.time_calls.append((arm, primary_timesteps.detach().clone()))
        sigma = primary_timesteps.float() / 1000.0
        return ComponentTimes(
            timestep={"latent": primary_timesteps},
            next_timestep={"latent": torch.zeros_like(primary_timesteps)},
            sigma={"latent": sigma},
            next_sigma={"latent": torch.zeros_like(sigma)},
        )

    def add_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        generator: torch.Generator | None = None,
    ) -> NoisedState:
        del generator
        noise = LatentState(
            {
                "latent": torch.full_like(
                    clean_state.components["latent"],
                    float(len(self.drawn_noise) + 1),
                )
            }
        )
        self.drawn_noise.append(noise)
        return self._apply_noise(clean_state, times, noise)

    def apply_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        noise: LatentState,
    ) -> NoisedState:
        assert noise is self.drawn_noise[-1]
        self.reused_noise.append(noise)
        return self._apply_noise(clean_state, times, noise)

    @staticmethod
    def _apply_noise(
        clean_state: LatentState,
        times: ComponentTimes,
        noise: LatentState,
    ) -> NoisedState:
        clean = clean_state.components["latent"]
        noise_tensor = noise.components["latent"]
        sigma = times.sigma["latent"].reshape(clean.shape[0], 1)
        noised = clean * (1.0 - sigma) + noise_tensor * sigma
        return NoisedState(
            state=LatentState({"latent": noised}),
            target_velocity=LatentState({"latent": noise_tensor - clean}),
            noise=noise,
        )

    def forward_state(
        self,
        *,
        batch: dict[str, Any],
        state: LatentState,
        times: ComponentTimes,
        **kwargs: Any,
    ) -> MultiModalStepOutput:
        del kwargs
        arm = batch["arm_token"]
        timestep = times.timestep["latent"].float().reshape(arm.shape[0], 1) / 1000.0
        grad_enabled = torch.is_grad_enabled()
        autocast_enabled = bool(self.autocast_active())
        self.forward_events.append(
            (float(arm[0].item()), self._ref_active, grad_enabled, autocast_enabled)
        )
        assert autocast_enabled
        if self._ref_active:
            assert not grad_enabled
            velocity = 0.2 * timestep - 0.3 * arm
        else:
            assert grad_enabled
            velocity = self.policy_weight * (1.0 + timestep) + 0.4 * arm
        velocity = velocity.expand_as(state.components["latent"])
        return MultiModalStepOutput(velocity=LatentState({"latent": velocity}))

    @contextmanager
    def use_ref_parameters(self) -> Iterator[None]:
        assert not self._ref_active
        assert not torch.is_grad_enabled()
        self.ref_scope_enters += 1
        self._ref_active = True
        try:
            yield
        finally:
            self._ref_active = False
            self.ref_scope_exits += 1

    @staticmethod
    def reduce_latent_values(
        values: dict[str, torch.Tensor],
        *,
        state: LatentState,
    ) -> torch.Tensor:
        del state
        return values["latent"].flatten(1).mean(dim=1)


def _media_batch(arm: str, batch_size: int = 2) -> tuple[tuple[DecodedMedia, ...], ...]:
    return tuple(
        (
            DecodedMedia(
                type="image",
                path=f"{arm}-{index}.png",
                payload=arm,
            ),
        )
        for index in range(batch_size)
    )


def _preference_batch(batch_size: int = 2) -> OfflineBatch:
    return OfflineBatch(
        condition={"prompt_embeds": torch.ones(batch_size, 2)},
        condition_ids=tuple(f"condition-{index}" for index in range(batch_size)),
        record_ids=tuple(f"record-{index}" for index in range(batch_size)),
        sources=tuple("source" for _ in range(batch_size)),
        source_ids=torch.zeros(batch_size, dtype=torch.long),
        model_inputs=(),
        supervision_type="preference",
        output=PreferenceOutputBatch(
            chosen_media=_media_batch("chosen", batch_size),
            rejected_media=_media_batch("rejected", batch_size),
        ),
        metadata_json=tuple("{}" for _ in range(batch_size)),
    )


def _trainer(sync_schedule: list[bool]) -> tuple[OfflineDPOTrainer, _FakeAdapter]:
    trainer = object.__new__(OfflineDPOTrainer)
    trainer.accelerator = _FakeAccelerator(sync_schedule)
    trainer.training_args = _TrainingArgs(
        weighting_scheme="uniform",
        num_train_timesteps=3,
        timestep_range=(0.1, 0.9),
        time_shift=1.0,
        beta=2.0,
    )
    trainer.model_bundle = object()
    autocast_depth = 0

    @contextmanager
    def autocast() -> Iterator[None]:
        nonlocal autocast_depth
        autocast_depth += 1
        try:
            yield
        finally:
            autocast_depth -= 1

    trainer.autocast = autocast
    adapter = _FakeAdapter(lambda: autocast_depth > 0)
    trainer.adapter = adapter
    return trainer, adapter


def test_offline_dpo_declares_dataset_execution_without_online_dpo_stages() -> None:
    assert OfflineDPOTrainer.__bases__ == (BaseTrainer,)
    assert OfflineDPOTrainer.paradigm == "decoupled"
    assert OfflineDPOTrainer.execution_contract is OFFLINE_EXECUTION_CONTRACT
    assert OfflineDPOTrainer.execution_contract.acquisition is AcquisitionMode.DATASET
    assert OfflineDPOTrainer.execution_contract.feedback is FeedbackMode.NONE
    assert "sample" not in OfflineDPOTrainer.__dict__
    assert "prepare_feedback" not in OfflineDPOTrainer.__dict__
    assert "optimize" not in OfflineDPOTrainer.__dict__


def test_offline_dpo_builds_preference_loader_without_accelerator_prepare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = object.__new__(OfflineDPOTrainer)
    trainer.config = object()
    trainer.accelerator = _FakeAccelerator([])
    trainer.adapter = SimpleNamespace(
        preprocess_func=object(),
        pipeline_io_contract=object(),
    )
    sentinel = object()
    received: dict[str, Any] = {}

    def fake_builder(**kwargs: Any) -> Any:
        received.update(kwargs)
        return sentinel

    monkeypatch.setattr(offline_dpo_module, "build_offline_train_dataloader", fake_builder)

    dataloader, source_loaders = trainer._build_train_dataloader()

    assert dataloader is sentinel
    assert source_loaders == {}
    assert received == {
        "config": trainer.config,
        "accelerator": trainer.accelerator,
        "preprocess_func": trainer.adapter.preprocess_func,
        "supervision_type": "preference",
        "pipeline_io_contract": trainer.adapter.pipeline_io_contract,
    }


def test_offline_dpo_rejects_non_preference_batch_boundaries() -> None:
    trainer = object.__new__(OfflineDPOTrainer)

    with pytest.raises(TypeError, match="exact OfflineBatch"):
        trainer.optimize_batch(object())

    batch = _preference_batch()
    demonstration = OfflineBatch(
        condition=batch.condition,
        condition_ids=batch.condition_ids,
        record_ids=batch.record_ids,
        sources=batch.sources,
        source_ids=batch.source_ids,
        model_inputs=batch.model_inputs,
        supervision_type="demonstration",
        output=DemonstrationOutputBatch(target_media=_media_batch("chosen")),
        metadata_json=batch.metadata_json,
    )
    with pytest.raises(ValueError, match="supervision_type='preference'"):
        trainer.optimize_batch(demonstration)


def test_offline_dpo_pairs_noise_reference_scopes_and_microbatch_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, adapter = _trainer([False, True])
    objective_values: list[torch.Tensor] = []
    original_objective = offline_dpo_module.dpo_objective

    def recording_objective(**kwargs: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics = original_objective(**kwargs)
        objective_values.append(loss.detach().clone())
        return loss, metrics

    monkeypatch.setattr(offline_dpo_module, "dpo_objective", recording_objective)
    optimizer_steps: list[dict[str, list[torch.Tensor]]] = []

    def apply_optimizer_step(
        loss_info: dict[str, list[torch.Tensor]],
    ) -> dict[str, list[torch.Tensor]]:
        optimizer_steps.append({name: list(values) for name, values in loss_info.items()})
        return defaultdict(list)

    trainer._apply_optimizer_step = apply_optimizer_step
    batch = _preference_batch()

    trainer.optimize_batch(batch)

    assert optimizer_steps == []
    assert len(trainer._offline_dpo_loss_info["loss"]) == 1
    assert adapter.encode_calls == ["chosen", "rejected"]

    trainer.optimize_batch(batch)

    assert adapter.train_calls == 2
    assert adapter.encode_calls == ["chosen", "rejected", "chosen", "rejected"]
    assert len(trainer.accelerator.accumulate_roots) == 2
    assert trainer.accelerator.accumulate_roots == [trainer.model_bundle, trainer.model_bundle]
    assert len(trainer.accelerator.backward_losses) == 2
    assert len(optimizer_steps) == 1
    assert all(len(values) == 2 for values in optimizer_steps[0].values())

    timesteps_per_batch = trainer.training_args.num_train_timesteps
    assert len(objective_values) == 2 * timesteps_per_batch
    for batch_index, backward_loss in enumerate(trainer.accelerator.backward_losses):
        start = batch_index * timesteps_per_batch
        expected = torch.stack(objective_values[start : start + timesteps_per_batch]).mean()
        torch.testing.assert_close(backward_loss, expected)

    assert len(adapter.drawn_noise) == 2 * timesteps_per_batch
    assert len(adapter.reused_noise) == len(adapter.drawn_noise)
    assert all(reused is drawn for reused, drawn in zip(adapter.reused_noise, adapter.drawn_noise))
    assert len(adapter.time_calls) == 4 * timesteps_per_batch
    for index in range(0, len(adapter.time_calls), 2):
        chosen_arm, chosen_times = adapter.time_calls[index]
        rejected_arm, rejected_times = adapter.time_calls[index + 1]
        assert chosen_arm == 0.0
        assert rejected_arm == 1.0
        torch.testing.assert_close(chosen_times, rejected_times)

    expected_reference_forwards = 4 * timesteps_per_batch
    expected_reference_scopes = 2 * timesteps_per_batch
    assert adapter.ref_scope_enters == expected_reference_scopes
    assert adapter.ref_scope_exits == expected_reference_scopes
    reference_events = [event for event in adapter.forward_events if event[1]]
    policy_events = [event for event in adapter.forward_events if not event[1]]
    assert len(reference_events) == expected_reference_forwards
    assert len(policy_events) == expected_reference_forwards
    assert all(not grad_enabled and autocast for _, _, grad_enabled, autocast in reference_events)
    assert all(grad_enabled and autocast for _, _, grad_enabled, autocast in policy_events)


def test_offline_dpo_reencodes_both_arms_on_every_batch() -> None:
    trainer, adapter = _trainer([True, True])
    trainer._apply_optimizer_step = lambda loss_info: defaultdict(list)
    batch = _preference_batch(batch_size=1)

    trainer.optimize_batch(batch)
    trainer.optimize_batch(batch)

    assert adapter.encode_calls == ["chosen", "rejected", "chosen", "rejected"]
