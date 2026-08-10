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

from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

import pytest
import torch

from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import (
    BaseSample,
    ComponentTrajectory,
    IndexedTrajectoryTensor,
    LatentState,
    StructuredTrajectory,
)
from flow_factory.scheduler import SchedulerGroup, SDESchedulerOutput


class AdapterFake(BaseAdapter):
    """Minimal concrete adapter exercising the replay hooks."""

    def load_pipeline(self) -> Any:
        """Return an unused pipeline fake."""
        raise NotImplementedError

    def decode_latents(self, latents: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return latents unchanged."""
        return latents

    def inference(self, **kwargs: Any) -> List[BaseSample]:
        """Return no samples."""
        return []

    def forward(self, **kwargs: Any) -> SDESchedulerOutput:
        """Return a deterministic scheduler output honouring ``return_kwargs``."""
        self.forward_kwargs = kwargs
        requested = set(kwargs.get("return_kwargs") or ())
        latents = kwargs["latents"]
        # Schedulers broadcast per-sample statistics to (B, 1, ..., 1).
        broadcast_shape = (latents.shape[0],) + (1,) * (latents.ndim - 1)
        return SDESchedulerOutput(
            next_latents=latents + 1 if "next_latents" in requested else None,
            next_latents_mean=(latents + 2 if "next_latents_mean" in requested else None),
            std_dev_t=torch.full(broadcast_shape, 0.25) if "std_dev_t" in requested else None,
            dt=torch.full(broadcast_shape, -0.5) if "dt" in requested else None,
            log_prob=torch.tensor([0.75, 0.5]) if kwargs.get("compute_log_prob") else None,
            velocity=latents + 3 if "velocity" in requested else None,
        )


class StructuredAdapterFake(AdapterFake):
    """Adapter fake declaring the structured video/audio component contract."""

    trajectory_component_order = ("video", "audio")


class SchedulerFake:
    """Small scheduler-like object recording lifecycle dispatch."""

    def __init__(self, train_timesteps: Optional[torch.Tensor] = None) -> None:
        self.train_timesteps = (
            torch.tensor([0, 2, 4]) if train_timesteps is None else train_timesteps
        )
        self.seeds: List[int] = []
        self.modes: List[Any] = []
        self.noise_level = 0.7

    def step(self) -> None:
        """Provide scheduler compatibility."""

    def set_seed(self, seed: int) -> None:
        """Record one seed dispatch."""
        self.seeds.append(seed)

    def train(self, mode: bool = True) -> None:
        """Record one train-mode dispatch."""
        self.modes.append(("train", mode))

    def eval(self) -> None:
        """Record one eval-mode dispatch."""
        self.modes.append(("eval", None))


def _adapter() -> AdapterFake:
    adapter = object.__new__(AdapterFake)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake())
    adapter.scheduler_group = adapter.build_scheduler_group()
    return adapter


def _structured_adapter() -> StructuredAdapterFake:
    adapter = object.__new__(StructuredAdapterFake)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake())
    return adapter


def _legacy_callback_batch() -> Any:
    return BaseSample.stack(
        [
            BaseSample(
                timesteps=torch.tensor([1000.0, 500.0, 0.0]),
                all_latents=torch.tensor([[1.0], [2.0], [3.0]]),
                latent_index_map=torch.tensor([0, 1, 2]),
                log_probs=torch.tensor([0.1, 0.2]),
                log_prob_index_map=torch.tensor([0, 1]),
                extra_kwargs={
                    "velocity": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    "callback_index_map": torch.tensor([-1, 0]),
                },
            ),
            BaseSample(
                timesteps=torch.tensor([1000.0, 500.0, 0.0]),
                all_latents=torch.tensor([[10.0], [20.0], [30.0]]),
                latent_index_map=torch.tensor([0, 1, 2]),
                log_probs=torch.tensor([0.3, 0.4]),
                log_prob_index_map=torch.tensor([0, 1]),
                extra_kwargs={
                    "velocity": torch.tensor([[11.0, 12.0], [13.0, 14.0]]),
                    "callback_index_map": torch.tensor([-1, 0]),
                },
            ),
        ]
    )


def _structured_callback_batch() -> Any:
    samples = []
    for offset in (0.0, 100.0):
        samples.append(
            BaseSample(
                trajectory=StructuredTrajectory(
                    components={
                        "video": ComponentTrajectory(
                            states=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) + offset,
                            timesteps=torch.tensor([1000.0, 500.0, 0.0]),
                            state_index_map=torch.tensor([0, 1, 2]),
                        ),
                        "audio": ComponentTrajectory(
                            states=torch.tensor([[7.0], [8.0], [9.0]]) + offset,
                            timesteps=torch.tensor([800.0, 300.0, 0.0]),
                            state_index_map=torch.tensor([0, 1, 2]),
                        ),
                    },
                    callbacks={
                        "velocity": {
                            "video": IndexedTrajectoryTensor(
                                values=torch.tensor([[1.0, 2.0]]) + offset,
                                index_map=torch.tensor([-1, 0]),
                            ),
                            "audio": IndexedTrajectoryTensor(
                                values=torch.tensor([[3.0]]) + offset,
                                index_map=torch.tensor([-1, 0]),
                            ),
                        }
                    },
                )
            )
        )
    return BaseSample.stack(samples)


def test_legacy_replay_callback_uses_shared_callback_index_map() -> None:
    callback = _adapter().get_replay_callback(_legacy_callback_batch(), 1, "velocity")

    assert callback.component_names == ("latent",)
    assert torch.equal(
        callback.components["latent"],
        torch.tensor([[1.0, 2.0], [11.0, 12.0]]),
    )


def test_legacy_replay_callback_rejects_uncollected_position() -> None:
    with pytest.raises(
        ValueError,
        match=r"callback 'velocity'.*callback_index_map.*rollout position 0.*sentinel -1",
    ):
        _adapter().get_replay_callback(_legacy_callback_batch(), 0, "velocity")


def test_legacy_replay_callback_requires_stored_field() -> None:
    with pytest.raises(
        ValueError,
        match=r"next_latents_mean.*get_replay_callback.*step_index=1",
    ):
        _adapter().get_replay_callback(_legacy_callback_batch(), 1, "next_latents_mean")


def test_structured_replay_callback_reads_each_component_index_map() -> None:
    adapter = _structured_adapter()
    batch = _structured_callback_batch()

    callback = adapter.get_replay_callback(batch, 1, "velocity")

    assert callback.component_names == ("video", "audio")
    assert torch.equal(
        callback.components["video"],
        torch.tensor([[1.0, 2.0], [101.0, 102.0]]),
    )
    assert torch.equal(callback.components["audio"], torch.tensor([[3.0], [103.0]]))


def test_structured_replay_callback_reports_component_for_uncollected_position() -> None:
    with pytest.raises(
        ValueError,
        match=r"callback 'velocity' component 'video'.*rollout position 0.*sentinel -1",
    ):
        _structured_adapter().get_replay_callback(_structured_callback_batch(), 0, "velocity")


def test_structured_replay_callback_requires_declared_component_order() -> None:
    with pytest.raises(
        ValueError,
        match=r"trajectory_component_order.*\('latent',\).*\('video', 'audio'\)",
    ):
        _adapter().get_replay_callback(_structured_callback_batch(), 1, "velocity")


def test_structured_replay_callback_requires_named_field() -> None:
    with pytest.raises(
        ValueError,
        match=r"callback field 'next_latents_mean'.*\('velocity',\).*step_index=1",
    ):
        _structured_adapter().get_replay_callback(
            _structured_callback_batch(), 1, "next_latents_mean"
        )


def test_default_component_reducer_means_each_component_over_non_batch_elements() -> None:
    torch.manual_seed(11)
    video, audio = torch.randn(2, 3, 4), torch.randn(2, 5)

    reduced = _structured_adapter().reduce_component_latent_values({"video": video, "audio": audio})

    assert tuple(reduced) == ("video", "audio")
    assert torch.equal(reduced["video"], video.flatten(1).mean(dim=1))
    assert torch.equal(reduced["audio"], audio.flatten(1).mean(dim=1))


def test_component_reducer_requires_declared_component_order() -> None:
    adapter = _structured_adapter()

    with pytest.raises(ValueError, match=r"\('video', 'audio'\).*\('audio', 'video'\)"):
        adapter.reduce_component_latent_values(
            {"audio": torch.zeros(2, 5), "video": torch.zeros(2, 3, 4)}
        )


def test_component_reducer_rejects_unbatched_component_values() -> None:
    adapter = _adapter()

    with pytest.raises(ValueError, match=r"values\['latent'\].*batched.*\(\)"):
        adapter.reduce_component_latent_values({"latent": torch.tensor(1.0)})


class ComponentReducerOverrideFake(StructuredAdapterFake):
    """Adapter whose per-component reduction override is configurable per test."""

    result: Any = None

    def _reduce_component_latent_values(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        state: Optional[LatentState] = None,
    ) -> Mapping[str, torch.Tensor]:
        """Return the preconfigured override result."""
        return self.result


def _override_adapter(result: Any) -> ComponentReducerOverrideFake:
    adapter = object.__new__(ComponentReducerOverrideFake)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake())
    adapter.scheduler_group = adapter.build_scheduler_group()
    adapter.result = result
    return adapter


def test_component_reducer_passes_state_context_to_the_override() -> None:
    """A masked adapter averages only the active positions the state marks."""
    values = {"video": torch.tensor([[1.0, 3.0], [2.0, 4.0]]), "audio": torch.tensor([[10.0]])}

    captured: Dict[str, Any] = {}

    class CapturingFake(ComponentReducerOverrideFake):
        def _reduce_component_latent_values(
            self,
            values: Mapping[str, torch.Tensor],
            *,
            state: Optional[LatentState] = None,
        ) -> Mapping[str, torch.Tensor]:
            captured["state"] = state
            return {name: value.reshape(1, -1)[:, :1].mean(dim=1) for name, value in values.items()}

    adapter = object.__new__(CapturingFake)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake())
    adapter.scheduler_group = adapter.build_scheduler_group()
    state = LatentState({"video": torch.zeros(1, 2), "audio": torch.zeros(1, 1)})

    reduced = adapter.reduce_component_latent_values(
        {"video": torch.tensor([[1.0, 3.0]]), "audio": torch.tensor([[10.0]])}, state=state
    )

    assert captured["state"] is state
    assert torch.equal(reduced["video"], torch.tensor([1.0]))
    assert values["audio"].shape == (1, 1)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            {"audio": torch.zeros(2), "video": torch.zeros(2)},
            r"reduce_component_latent_values.*ComponentReducerOverrideFake.*component order "
            r"\('video', 'audio'\).*\('audio', 'video'\)",
        ),
        (
            {"video": torch.zeros(2, 1), "audio": torch.zeros(2)},
            r"reduce_component_latent_values.*\['video'\].*\(2,\).*\(2, 1\)",
        ),
        (
            {"video": torch.zeros(2), "audio": torch.zeros(3)},
            r"reduce_component_latent_values.*\['audio'\].*\(2,\).*\(3,\)",
        ),
        (
            {"video": torch.zeros(2), "audio": torch.zeros(2, dtype=torch.float64)},
            r"reduce_component_latent_values.*\['audio'\].*float32.*float64",
        ),
        (
            torch.zeros(2),
            r"reduce_component_latent_values.*ComponentReducerOverrideFake.*Mapping.*Tensor",
        ),
    ],
    ids=["order", "rank", "batch", "dtype", "type"],
)
def test_component_reducer_validates_an_override_result(result: Any, message: str) -> None:
    adapter = _override_adapter(result)

    with pytest.raises((TypeError, ValueError), match=message):
        adapter.reduce_component_latent_values(
            {"video": torch.ones(2, 3), "audio": torch.ones(2, 5)}
        )


def test_component_reducer_rejects_an_override_result_that_drops_the_input_batch() -> None:
    adapter = _override_adapter({"video": torch.zeros(3), "audio": torch.zeros(3)})

    with pytest.raises(
        ValueError, match=r"reduce_component_latent_values.*\['video'\].*\(2,\).*\(3,\)"
    ):
        adapter.reduce_component_latent_values(
            {"video": torch.ones(2, 3), "audio": torch.ones(2, 5)}
        )


def test_default_state_active_numel_counts_non_batch_elements() -> None:
    state = LatentState({"video": torch.zeros(2, 3, 4), "audio": torch.zeros(2, 5)})

    assert _structured_adapter().get_state_active_numel(state) == {"video": 12, "audio": 5}


def test_state_active_numel_requires_declared_component_order() -> None:
    state = LatentState({"audio": torch.zeros(2, 5), "video": torch.zeros(2, 3, 4)})

    with pytest.raises(
        ValueError,
        match=r"trajectory_component_order.*\('video', 'audio'\).*\('audio', 'video'\)",
    ):
        _structured_adapter().get_state_active_numel(state)


def test_train_step_indices_return_primary_scheduler_positions() -> None:
    adapter = _adapter()

    indices = adapter.get_train_step_indices()

    assert torch.equal(indices, torch.tensor([0, 2, 4]))


def test_train_step_indices_accept_aligned_group_with_different_timestep_values() -> None:
    adapter = _structured_adapter()
    video = SchedulerFake(torch.tensor([1, 3]))
    audio = SchedulerFake(torch.tensor([1, 3]))
    adapter.pipeline.scheduler = video
    adapter.scheduler_group = SchedulerGroup({"video": video, "audio": audio}, primary_name="video")

    assert torch.equal(adapter.get_train_step_indices(), torch.tensor([1, 3]))


def test_train_step_indices_reject_misaligned_group_members() -> None:
    adapter = _structured_adapter()
    video = SchedulerFake(torch.tensor([1, 3]))
    audio = SchedulerFake(torch.tensor([1, 2]))
    adapter.pipeline.scheduler = video
    adapter.scheduler_group = SchedulerGroup({"video": video, "audio": audio}, primary_name="video")

    with pytest.raises(
        ValueError,
        match=r"component 'audio'.*\[1, 2\].*primary 'video'.*\[1, 3\]",
    ):
        adapter.get_train_step_indices()


def test_legacy_forward_state_wraps_component_statistics() -> None:
    adapter = _adapter()
    batch = _legacy_callback_batch()
    replay = adapter.get_replay_step(batch, 0)

    output = adapter.forward_state(
        batch=batch,
        state=replay.state,
        times=replay.times,
        next_state=replay.next_state,
        compute_log_prob=True,
        return_fields=("log_prob", "next_latents_mean", "std_dev_t", "dt"),
    )

    assert torch.equal(output.std_dev_t["latent"], torch.full((2, 1), 0.25))
    assert torch.equal(output.dt["latent"], torch.full((2, 1), -0.5))
    assert torch.equal(output.component_log_probs["latent"], output.log_prob)
    assert output.velocity is None


def test_legacy_forward_state_omits_absent_component_statistics() -> None:
    adapter = _adapter()
    batch = _legacy_callback_batch()
    replay = adapter.get_replay_step(batch, 0)

    output = adapter.forward_state(
        batch=batch,
        state=replay.state,
        times=replay.times,
        next_state=replay.next_state,
        compute_log_prob=False,
        return_fields=("velocity",),
    )

    assert output.std_dev_t is None
    assert output.dt is None
    assert output.log_prob is None
    assert output.component_log_probs is None


def _legacy_terminal_step_batch() -> Any:
    # Adapters store one timestep per denoising step (T), while latents/log-probs
    # keep T + 1 rollout positions, so the final transition has no stored t_next.
    return BaseSample.stack(
        [
            BaseSample(
                timesteps=torch.tensor([1000.0, 500.0]),
                all_latents=torch.tensor([[1.0], [2.0], [3.0]]),
                latent_index_map=torch.tensor([0, 1, 2]),
            ),
            BaseSample(
                timesteps=torch.tensor([1000.0, 500.0]),
                all_latents=torch.tensor([[10.0], [20.0], [30.0]]),
                latent_index_map=torch.tensor([0, 1, 2]),
            ),
        ]
    )


def test_legacy_replay_uses_zero_next_timestep_at_the_terminal_step() -> None:
    replay = _adapter().get_replay_step(_legacy_terminal_step_batch(), 1)

    assert torch.equal(replay.times.timestep["latent"], torch.tensor([500.0, 500.0]))
    assert torch.equal(replay.times.next_timestep["latent"], torch.tensor(0))
    assert torch.equal(replay.next_state.components["latent"], torch.tensor([[3.0], [30.0]]))


def test_legacy_replay_step_exposes_joint_log_prob_as_component_log_prob() -> None:
    replay = _adapter().get_replay_step(_legacy_callback_batch(), 1)

    assert tuple(replay.component_log_probs) == ("latent",)
    assert replay.component_log_probs["latent"] is replay.log_prob


def test_set_trajectory_seed_dispatches_once_for_legacy_scheduler() -> None:
    adapter = _adapter()

    adapter.set_trajectory_seed(11)
    adapter.scheduler_group.train(mode=True)
    adapter.scheduler_group.eval()

    assert adapter.scheduler.seeds == [11]
    assert adapter.scheduler.modes == [("train", True), ("eval", None)]


class _InitAdapterFake(AdapterFake):
    """Adapter fake driving ``BaseAdapter.__init__`` scheduler-group invariants."""

    def __init__(self, group_factory: Any, **kwargs: Any) -> None:
        self._group_factory = group_factory
        super().__init__(**kwargs)

    def build_component_runtime(self) -> Any:
        """Return a minimal runtime exposing an eager pipeline."""
        return SimpleNamespace(
            pipeline=SimpleNamespace(scheduler=SchedulerFake()),
            declared_component_names=("scheduler",),
        )

    def load_scheduler(self) -> Any:
        """Return the canonical scheduler unchanged."""
        return self.pipeline.scheduler

    def build_scheduler_group(self) -> SchedulerGroup:
        """Return the group under test."""
        return self._group_factory(self)


def _init_kwargs() -> Any:
    return {
        "config": SimpleNamespace(
            model_args=SimpleNamespace(),
            training_args=SimpleNamespace(),
            eval_args=SimpleNamespace(),
        ),
        "accelerator": SimpleNamespace(),
    }


def test_base_adapter_init_rejects_scheduler_group_name_mismatch() -> None:
    def factory(adapter: Any) -> SchedulerGroup:
        return SchedulerGroup({"video": adapter.scheduler}, primary_name="video")

    with pytest.raises(
        ValueError,
        match=r"trajectory_component_order.*\('latent',\).*\('video',\)",
    ):
        _InitAdapterFake(factory, **_init_kwargs())


def test_base_adapter_init_rejects_non_canonical_primary_scheduler() -> None:
    def factory(adapter: Any) -> SchedulerGroup:
        return SchedulerGroup({"latent": SchedulerFake()}, primary_name="latent")

    with pytest.raises(
        ValueError,
        match=r"SchedulerGroup.primary.*canonical pipeline scheduler.*'latent'",
    ):
        _InitAdapterFake(factory, **_init_kwargs())
