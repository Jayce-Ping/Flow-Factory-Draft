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
from typing import Any, Dict, List

import pytest
import torch

from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import (
    BaseSample,
    ComponentTimes,
    ComponentTrajectory,
    LatentState,
    StructuredTrajectory,
)
from flow_factory.scheduler import SDESchedulerOutput


class AdapterFake(BaseAdapter):
    """Minimal concrete adapter for legacy bridge tests."""

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
        """Record and return a deterministic scheduler output."""
        self.forward_kwargs = kwargs
        latents = kwargs["latents"]
        # Schedulers broadcast per-sample statistics to (B, 1, ..., 1).
        broadcast_shape = (latents.shape[0],) + (1,) * (latents.ndim - 1)
        return SDESchedulerOutput(
            next_latents=latents + 1,
            next_latents_mean=latents + 2,
            std_dev_t=torch.full(broadcast_shape, 0.25),
            dt=torch.full(broadcast_shape, -0.5),
            log_prob=torch.full((latents.shape[0],), 0.75),
            velocity=latents + 3,
        )


class StructuredAdapterFake(AdapterFake):
    """Adapter fake declaring the structured video/audio component contract."""

    trajectory_component_order = ("video", "audio")


class SchedulerFake:
    """Small scheduler-like object for adapter group tests."""

    def step(self) -> None:
        """Provide scheduler compatibility."""


def _adapter() -> AdapterFake:
    adapter = object.__new__(AdapterFake)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake())
    return adapter


def _structured_adapter() -> StructuredAdapterFake:
    adapter = object.__new__(StructuredAdapterFake)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake())
    return adapter


def _legacy_batch() -> Any:
    return BaseSample.stack(
        [
            BaseSample(
                timesteps=torch.tensor([1000.0, 500.0, 0.0]),
                all_latents=torch.tensor([[1.0], [2.0], [3.0]]),
                latent_index_map=torch.tensor([0, 1, 2]),
                log_probs=torch.tensor([0.1, 0.2]),
                log_prob_index_map=torch.tensor([0, 1]),
                prompt_embeds=torch.tensor([4.0]),
            ),
            BaseSample(
                timesteps=torch.tensor([1000.0, 500.0, 0.0]),
                all_latents=torch.tensor([[10.0], [20.0], [30.0]]),
                latent_index_map=torch.tensor([0, 1, 2]),
                log_probs=torch.tensor([0.3, 0.4]),
                log_prob_index_map=torch.tensor([0, 1]),
                prompt_embeds=torch.tensor([5.0]),
            ),
        ]
    )


def _structured_batch() -> Any:
    samples = []
    for offset in (0.0, 100.0):
        samples.append(
            BaseSample(
                trajectory=StructuredTrajectory(
                    components={
                        "video": ComponentTrajectory(
                            states=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) + offset,
                            timesteps=torch.tensor([1000.0, 500.0, 0.0]),
                            sigmas=torch.tensor([1.0, 0.5, 0.0]),
                            state_index_map=torch.tensor([0, 1, 2]),
                        ),
                        "audio": ComponentTrajectory(
                            states=torch.tensor([[7.0], [8.0], [9.0]]) + offset,
                            timesteps=torch.tensor([800.0, 300.0, 0.0]),
                            sigmas=torch.tensor([0.8, 0.3, 0.0]),
                            state_index_map=torch.tensor([0, 1, 2]),
                        ),
                    },
                    log_probs=torch.tensor([0.1, 0.2]) + offset,
                    log_prob_index_map=torch.tensor([0, 1]),
                )
            )
        )
    return BaseSample.stack(samples)


def _structured_terminal_only_batch() -> Any:
    samples = []
    for offset in (0.0, 100.0):
        samples.append(
            BaseSample(
                trajectory=StructuredTrajectory(
                    components={
                        "video": ComponentTrajectory(
                            states=torch.tensor([[5.0, 6.0]]) + offset,
                            timesteps=torch.tensor([1000.0, 500.0, 0.0]),
                            state_index_map=torch.tensor([-1, -1, 0]),
                        ),
                        "audio": ComponentTrajectory(
                            states=torch.tensor([[9.0]]) + offset,
                            timesteps=torch.tensor([800.0, 300.0, 0.0]),
                            state_index_map=torch.tensor([-1, -1, 0]),
                        ),
                    }
                )
            )
        )
    return BaseSample.stack(samples)


def test_default_adapter_scheduler_group_reuses_canonical_scheduler() -> None:
    adapter = _adapter()

    group = adapter.build_scheduler_group()

    assert group.names == ("latent",)
    assert group.primary is adapter.scheduler


def test_legacy_terminal_and_replay_hooks_preserve_existing_indexing() -> None:
    adapter = _adapter()
    batch = _legacy_batch()

    terminal = adapter.get_terminal_state(batch)
    replay = adapter.get_replay_step(batch, 1)

    assert torch.equal(terminal.components["latent"], batch["all_latents"][:, 2])
    assert torch.equal(replay.state.components["latent"], batch["all_latents"][:, 1])
    assert torch.equal(replay.next_state.components["latent"], batch["all_latents"][:, 2])
    assert torch.equal(replay.times.timestep["latent"], batch["timesteps"][:, 1])
    assert torch.equal(replay.times.next_timestep["latent"], batch["timesteps"][:, 2])
    assert torch.equal(replay.log_prob, batch["log_probs"][:, 1])


def test_structured_replay_keeps_component_shapes_and_independent_times() -> None:
    replay = _structured_adapter().get_replay_step(_structured_batch(), 0)

    assert replay.state.components["video"].shape == (2, 2)
    assert replay.state.components["audio"].shape == (2, 1)
    assert replay.times.timestep["video"].tolist() == [1000.0, 1000.0]
    assert replay.times.timestep["audio"].tolist() == [800.0, 800.0]
    assert replay.times.sigma["video"].tolist() == [1.0, 1.0]
    assert replay.times.sigma["audio"].tolist() == pytest.approx([0.8, 0.8])


def test_legacy_terminal_needs_only_latents_and_index_map() -> None:
    batch = BaseSample.stack(
        [
            BaseSample(
                all_latents=torch.tensor([[1.0], [2.0]]),
                latent_index_map=torch.tensor([0, 1]),
            ),
            BaseSample(
                all_latents=torch.tensor([[10.0], [20.0]]),
                latent_index_map=torch.tensor([0, 1]),
            ),
        ]
    )

    terminal = _adapter().get_terminal_state(batch)

    assert torch.equal(terminal.components["latent"], torch.tensor([[2.0], [20.0]]))


def test_legacy_replay_allows_missing_log_probability_trajectory() -> None:
    batch = _legacy_batch()
    batch["log_probs"] = None
    batch["log_prob_index_map"] = None

    replay = _adapter().get_replay_step(batch, 0)

    assert replay.log_prob is None


def test_legacy_terminal_rejects_uncollected_index_sentinel() -> None:
    batch = _legacy_batch()
    batch["latent_index_map"] = torch.tensor([0, 1, -1])

    with pytest.raises(
        ValueError,
        match=r"latent_index_map.*rollout position 2.*sentinel -1.*\[0, 1, -1\]",
    ):
        _adapter().get_terminal_state(batch)


def test_legacy_terminal_only_map_returns_compact_state_zero() -> None:
    batch = BaseSample.stack(
        [
            BaseSample(
                all_latents=torch.tensor([[3.0]]),
                latent_index_map=torch.tensor([-1, -1, 0]),
            ),
            BaseSample(
                all_latents=torch.tensor([[30.0]]),
                latent_index_map=torch.tensor([-1, -1, 0]),
            ),
        ]
    )

    terminal = _adapter().get_terminal_state(batch)

    assert torch.equal(terminal.components["latent"], torch.tensor([[3.0], [30.0]]))


@pytest.mark.parametrize(
    ("map_name", "map_value", "message"),
    [
        (
            "latent_index_map",
            torch.tensor([-1, 1, 2]),
            r"latent_index_map.*rollout position 0.*sentinel -1.*\[-1, 1, 2\]",
        ),
        (
            "latent_index_map",
            torch.tensor([0, -1, 2]),
            r"latent_index_map.*rollout position 1.*sentinel -1.*\[0, -1, 2\]",
        ),
        (
            "log_prob_index_map",
            torch.tensor([-1, 1]),
            r"log_prob_index_map.*rollout position 0.*sentinel -1.*\[-1, 1\]",
        ),
    ],
)
def test_legacy_replay_rejects_uncollected_index_sentinels(
    map_name: str, map_value: torch.Tensor, message: str
) -> None:
    batch = _legacy_batch()
    batch[map_name] = map_value

    with pytest.raises(ValueError, match=message):
        _adapter().get_replay_step(batch, 0)


@pytest.mark.parametrize("operation", ["terminal", "replay"])
def test_structured_hooks_require_declared_component_order(operation: str) -> None:
    adapter = _adapter()
    batch = _structured_batch()

    with pytest.raises(
        ValueError,
        match=r"trajectory_component_order.*\('latent',\).*\('video', 'audio'\)",
    ):
        if operation == "terminal":
            adapter.get_terminal_state(batch)
        else:
            adapter.get_replay_step(batch, 0)


@pytest.mark.parametrize("operation", ["terminal", "replay"])
def test_structured_hooks_reject_unbatched_trajectory(operation: str) -> None:
    batch = _legacy_batch()
    batch["trajectory"] = _structured_batch().samples[0].trajectory
    adapter = _structured_adapter()

    with pytest.raises(
        ValueError,
        match=r"batched StructuredTrajectory.*component 'video'.*states.*timesteps",
    ):
        if operation == "terminal":
            adapter.get_terminal_state(batch)
        else:
            adapter.get_replay_step(batch, 0)


def test_structured_terminal_only_sparse_map_returns_compact_state_zero() -> None:
    terminal = _structured_adapter().get_terminal_state(_structured_terminal_only_batch())

    assert torch.equal(
        terminal.components["video"],
        torch.tensor([[5.0, 6.0], [105.0, 106.0]]),
    )
    assert torch.equal(terminal.components["audio"], torch.tensor([[9.0], [109.0]]))


def test_structured_replay_rejects_requested_uncollected_state_position() -> None:
    with pytest.raises(
        ValueError,
        match=r"component 'video' state_index_map.*rollout position 0.*sentinel -1",
    ):
        _structured_adapter().get_replay_step(_structured_terminal_only_batch(), 0)


def test_structured_replay_rejects_requested_uncollected_log_prob_position() -> None:
    batch = _structured_batch()
    batch.trajectory.log_prob_index_map = torch.tensor([-1, 0])

    with pytest.raises(
        ValueError,
        match=r"structured log_prob_index_map.*rollout position 0.*sentinel -1",
    ):
        _structured_adapter().get_replay_step(batch, 0)


def test_legacy_forward_state_preserves_arguments_and_wraps_output() -> None:
    adapter = _adapter()
    batch = _legacy_batch()
    replay = adapter.get_replay_step(batch, 0)

    output = adapter.forward_state(
        batch=batch,
        state=replay.state,
        times=replay.times,
        next_state=replay.next_state,
        compute_log_prob=True,
        return_fields=("log_prob", "velocity"),
        noise_level=0.7,
        guidance_scale=3.0,
    )

    assert adapter.forward_kwargs["t"] is replay.times.timestep["latent"]
    assert adapter.forward_kwargs["t_next"] is replay.times.next_timestep["latent"]
    assert adapter.forward_kwargs["latents"] is replay.state.components["latent"]
    assert adapter.forward_kwargs["next_latents"] is replay.next_state.components["latent"]
    assert adapter.forward_kwargs["return_kwargs"] == ("log_prob", "velocity")
    assert adapter.forward_kwargs["prompt_embeds"] is batch["prompt_embeds"]
    assert adapter.forward_kwargs["guidance_scale"] == 3.0
    assert "trajectory" not in adapter.forward_kwargs
    assert "all_latents" not in adapter.forward_kwargs
    assert torch.equal(
        output.next_state.components["latent"], replay.state.components["latent"] + 1
    )
    assert torch.equal(output.log_prob, torch.tensor([0.75, 0.75]))


def test_forward_state_removes_batch_level_state_owned_keys() -> None:
    adapter = _adapter()
    batch = _legacy_batch()
    replay = adapter.get_replay_step(batch, 0)
    batch["noise_level"] = 99.0
    batch["next_latents"] = torch.full_like(replay.next_state.components["latent"], -99.0)

    adapter.forward_state(
        batch=batch,
        state=replay.state,
        times=replay.times,
        next_state=replay.next_state,
        noise_level=0.7,
    )

    assert adapter.forward_kwargs["noise_level"] == 0.7
    assert adapter.forward_kwargs["next_latents"] is replay.next_state.components["latent"]


@pytest.mark.parametrize("collision", ["t", "t_next", "latents", "next_latents", "return_kwargs"])
def test_forward_state_rejects_state_owned_kwarg_collisions(collision: str) -> None:
    adapter = _adapter()
    batch = _legacy_batch()
    replay = adapter.get_replay_step(batch, 0)

    with pytest.raises(ValueError, match=rf"state-owned.*{collision}"):
        adapter.forward_state(
            batch=batch,
            state=replay.state,
            times=replay.times,
            next_state=replay.next_state,
            **{collision: object()},
        )


def test_default_forward_process_noise_is_single_component_and_deterministic() -> None:
    clean = LatentState({"latent": torch.tensor([[1.0, 2.0], [3.0, 4.0]])})
    times = ComponentTimes(
        timestep={"latent": torch.tensor([500.0, 500.0])},
        next_timestep={"latent": torch.tensor([0.0, 0.0])},
        sigma={"latent": torch.tensor([0.25, 0.5])},
        next_sigma={"latent": torch.tensor([0.0, 0.0])},
    )
    generator = torch.Generator().manual_seed(123)
    expected_generator = torch.Generator().manual_seed(123)
    expected_noise = torch.randn(clean.components["latent"].shape, generator=expected_generator)

    noised = _adapter().add_forward_process_noise(clean, times, generator=generator)

    sigma = torch.tensor([[0.25], [0.5]])
    assert torch.equal(noised.noise.components["latent"], expected_noise)
    assert torch.equal(
        noised.state.components["latent"],
        (1 - sigma) * clean.components["latent"] + sigma * expected_noise,
    )
    assert torch.equal(
        noised.target_velocity.components["latent"],
        expected_noise - clean.components["latent"],
    )


def test_default_heterogeneous_noise_and_axes_require_adapter_override() -> None:
    adapter = _adapter()
    state = LatentState({"video": torch.zeros(1, 2), "audio": torch.zeros(1, 1)})
    times = ComponentTimes(
        timestep={"video": torch.ones(1), "audio": torch.ones(1)},
        next_timestep={"video": torch.zeros(1), "audio": torch.zeros(1)},
        sigma={"video": torch.ones(1), "audio": torch.ones(1)},
    )

    with pytest.raises(ValueError, match=r"exactly.*latent.*video.*audio"):
        adapter.add_forward_process_noise(state, times)
    with pytest.raises(ValueError, match=r"component.*latent.*video"):
        adapter.resolve_component_latent_axes("video", torch.zeros(1, 2, 3))


def test_reduce_latent_values_uses_global_element_weighting() -> None:
    adapter = _structured_adapter()
    values: Dict[str, torch.Tensor] = {
        "video": torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
        "audio": torch.tensor([[10.0], [20.0]]),
    }

    assert torch.equal(adapter.reduce_latent_values(values), torch.tensor([14.0 / 3.0, 26.0 / 3.0]))
    assert torch.equal(
        adapter.reduce_latent_values(
            {"video": torch.tensor([2.0, 3.0]), "audio": torch.tensor([10.0, 20.0])},
            active_numel={"video": 2, "audio": 1},
        ),
        torch.tensor([14.0 / 3.0, 26.0 / 3.0]),
    )

    with pytest.raises(ValueError, match=r"active_numel.*audio.*positive.*0"):
        adapter.reduce_latent_values(
            {"video": torch.tensor([1.0]), "audio": torch.tensor([1.0])},
            active_numel={"video": 1, "audio": 0},
        )


def test_reduce_latent_values_preserves_single_component_scalar_scale() -> None:
    values = torch.tensor([2.0, 3.0])

    reduced = _adapter().reduce_latent_values(
        {"latent": values},
        active_numel={"latent": 17},
    )

    assert reduced is values


def test_reduce_latent_values_treats_empty_active_numel_as_no_override() -> None:
    values = torch.tensor([2.0, 3.0])

    reduced = _adapter().reduce_latent_values({"latent": values}, active_numel={})

    assert reduced is values


def test_reduce_latent_values_supports_partial_active_numel_override() -> None:
    reduced = _structured_adapter().reduce_latent_values(
        {
            "video": torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
            "audio": torch.tensor([10.0, 20.0]),
        },
        active_numel={"audio": 3},
    )

    assert torch.equal(reduced, torch.tensor([34.0 / 5.0, 66.0 / 5.0]))


@pytest.mark.parametrize(
    ("values", "active_numel", "message"),
    [
        (
            {"audio": torch.ones(2), "video": torch.ones(2)},
            None,
            r"trajectory_component_order.*video.*audio.*audio.*video",
        ),
        (
            {"video": torch.ones(2), "audio": torch.ones(2, dtype=torch.float64)},
            None,
            r"compatible dtype.*video.*float32.*audio.*float64",
        ),
        (
            {"video": torch.ones(2), "audio": torch.ones(2)},
            {"missing": 1},
            r"active_numel.*unknown.*missing.*video.*audio",
        ),
    ],
)
def test_reduce_latent_values_validates_component_contract(
    values: Dict[str, torch.Tensor],
    active_numel: Dict[str, int] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _structured_adapter().reduce_latent_values(values, active_numel=active_numel)


def test_scheduler_setter_refreshes_group_primary() -> None:
    adapter = _adapter()
    adapter.scheduler_group = adapter.build_scheduler_group()
    replacement = SchedulerFake()

    adapter.scheduler = replacement

    assert adapter.scheduler_group.primary is replacement


def test_scheduler_setter_before_group_initialization_updates_pipeline_only() -> None:
    adapter = _adapter()
    replacement = SchedulerFake()

    adapter.scheduler = replacement

    assert adapter.pipeline.scheduler is replacement
    assert not hasattr(adapter, "scheduler_group")


@pytest.mark.parametrize("operation", ["terminal", "replay"])
def test_structured_hooks_reject_plain_dict_batch_informatively(operation: str) -> None:
    batch = dict(_structured_batch())
    adapter = _structured_adapter()

    with pytest.raises(
        TypeError,
        match=r"expected StackedSampleBatch.*received dict",
    ):
        if operation == "terminal":
            adapter.get_terminal_state(batch)
        else:
            adapter.get_replay_step(batch, 0)
