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
from typing import Any, List

import pytest
import torch
from diffusers.utils.torch_utils import randn_tensor

from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import BaseSample, ComponentTimes, LatentState
from flow_factory.scheduler import SchedulerGroup, SDESchedulerOutput
from flow_factory.utils.base import to_broadcast_tensor
from flow_factory.utils.noise_schedule import flow_match_sigma


class SchedulerFake:
    """Small scheduler-like object recording seed dispatch."""

    def __init__(self) -> None:
        self.seeds: List[int] = []

    def step(self) -> None:
        """Provide scheduler compatibility."""

    def set_seed(self, seed: int) -> None:
        """Record one seed dispatch."""
        self.seeds.append(seed)


class AdapterFake(BaseAdapter):
    """Minimal concrete adapter exercising the forward-process hooks."""

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
        """Return a deterministic scheduler output."""
        latents = kwargs["latents"]
        return SDESchedulerOutput(velocity=latents + 3)


class StructuredAdapterFake(AdapterFake):
    """Adapter fake declaring the structured video/audio component contract."""

    trajectory_component_order = ("video", "audio")


class DataWardAdapterFake(StructuredAdapterFake):
    """Structured adapter whose velocity points from noise toward clean data."""

    flow_velocity_direction = "data"


class OrderedDrawAdapterFake(StructuredAdapterFake):
    """Heterogeneous adapter overriding only the ordered noise draw."""

    def add_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        *,
        generator: Any = None,
    ) -> Any:
        """Draw one noise tensor per component, in declared component order."""
        noise = {}
        for name in self.trajectory_component_order:
            component = clean_state.components[name]
            noise[name] = randn_tensor(
                component.shape,
                generator=generator,
                device=component.device,
                dtype=component.dtype,
            )
        return self.apply_forward_process_noise(clean_state, times, LatentState(noise))


def _adapter() -> AdapterFake:
    adapter = object.__new__(AdapterFake)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake())
    adapter.scheduler_group = adapter.build_scheduler_group()
    return adapter


def _structured_adapter(cls: type = StructuredAdapterFake) -> StructuredAdapterFake:
    adapter = object.__new__(cls)
    video = SchedulerFake()
    adapter.pipeline = SimpleNamespace(scheduler=video)
    adapter.scheduler_group = SchedulerGroup(
        {"video": video, "audio": SchedulerFake()}, primary_name="video"
    )
    return adapter


def _heterogeneous_times() -> ComponentTimes:
    return ComponentTimes(
        timestep={"video": torch.tensor([750.0, 250.0]), "audio": torch.tensor([400.0, 900.0])},
        next_timestep={"video": torch.zeros(2), "audio": torch.zeros(2)},
        sigma={"video": torch.tensor([0.75, 0.25]), "audio": torch.tensor([0.4, 0.9])},
        next_sigma={"video": torch.zeros(2), "audio": torch.zeros(2)},
    )


def _latent_times(primary: torch.Tensor) -> ComponentTimes:
    sigma = flow_match_sigma(primary)
    return ComponentTimes(
        timestep={"latent": primary},
        next_timestep={"latent": torch.zeros_like(primary)},
        sigma={"latent": sigma},
        next_sigma={"latent": torch.zeros_like(sigma)},
    )


def _terminal_only_legacy_batch() -> Any:
    """Legacy rollout that stored only the terminal latent (``trajectory_indices=[-1]``)."""
    samples = []
    for offset in (0.0, 100.0):
        samples.append(
            BaseSample(
                timesteps=torch.tensor([1000.0, 500.0]),
                all_latents=torch.tensor([[7.0, 8.0]]) + offset,
                latent_index_map=torch.tensor([-1, -1, 0]),
            )
        )
    return BaseSample.stack(samples)


def test_default_component_times_map_the_primary_coordinate() -> None:
    adapter = _adapter()
    primary = torch.tensor([1000.0, 250.0])

    times = adapter.build_training_component_times(primary)

    assert tuple(times.timestep) == ("latent",)
    assert torch.equal(times.timestep["latent"], primary)
    assert torch.equal(times.next_timestep["latent"], torch.zeros(2))
    assert torch.equal(times.sigma["latent"], torch.tensor([1.0, 0.25]))
    assert torch.equal(times.next_sigma["latent"], torch.zeros(2))


def test_default_component_times_consume_no_randomness() -> None:
    adapter = _adapter()
    state_before = torch.get_rng_state()

    adapter.build_training_component_times(torch.tensor([1000.0, 250.0]))

    assert torch.equal(torch.get_rng_state(), state_before)


def test_default_component_times_reject_an_unbatched_coordinate() -> None:
    adapter = _adapter()

    with pytest.raises(ValueError, match=r"primary_timesteps.*\(B,\).*\(2, 3\)"):
        adapter.build_training_component_times(torch.zeros(2, 3))


def test_default_component_times_reject_a_non_tensor_coordinate() -> None:
    adapter = _adapter()

    with pytest.raises(TypeError, match=r"torch.Tensor primary_timesteps.*received list"):
        adapter.build_training_component_times([1000.0, 250.0])


def test_default_component_times_require_the_single_latent_component_order() -> None:
    adapter = _structured_adapter()

    with pytest.raises(
        ValueError,
        match=r"\('latent',\).*build_training_component_times.*\('video', 'audio'\)",
    ):
        adapter.build_training_component_times(torch.tensor([1000.0, 250.0]))


def test_default_noising_matches_the_legacy_interpolation_bit_for_bit() -> None:
    adapter = _adapter()
    clean = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    primary = torch.tensor([700.0, 300.0])
    times = _latent_times(primary)

    torch.manual_seed(11)
    legacy_noise = randn_tensor(clean.shape, device=clean.device, dtype=clean.dtype)
    legacy_sigma = to_broadcast_tensor(flow_match_sigma(primary), clean)
    legacy_state = (1 - legacy_sigma) * clean + legacy_sigma * legacy_noise

    torch.manual_seed(11)
    noised = adapter.add_forward_process_noise(LatentState({"latent": clean}), times)

    assert torch.equal(noised.noise.components["latent"], legacy_noise)
    assert torch.equal(noised.state.components["latent"], legacy_state)
    assert torch.equal(noised.target_velocity.components["latent"], legacy_noise - clean)


def test_velocity_projection_recovers_clean_state_for_both_flow_directions() -> None:
    clean = LatentState(
        {
            "video": torch.tensor([[1.0, 2.0]]),
            "audio": torch.tensor([[3.0, 4.0, 5.0]]),
        }
    )
    noise = LatentState(
        {
            "video": torch.tensor([[5.0, 6.0]]),
            "audio": torch.tensor([[9.0, 10.0, 11.0]]),
        }
    )
    times = ComponentTimes(
        timestep={"video": torch.tensor([750.0]), "audio": torch.tensor([250.0])},
        next_timestep={"video": torch.zeros(1), "audio": torch.zeros(1)},
        sigma={"video": torch.tensor([0.75]), "audio": torch.tensor([0.25])},
        next_sigma={"video": torch.zeros(1), "audio": torch.zeros(1)},
    )
    noised = {
        name: (1 - to_broadcast_tensor(times.sigma[name], value)) * value
        + to_broadcast_tensor(times.sigma[name], value) * noise.components[name]
        for name, value in clean.components.items()
    }

    noise_ward = _structured_adapter()
    noise_ward_velocity = LatentState(
        {name: noise.components[name] - clean.components[name] for name in clean.component_names}
    )
    data_ward = _structured_adapter(DataWardAdapterFake)
    data_ward_velocity = LatentState(
        {name: clean.components[name] - noise.components[name] for name in clean.component_names}
    )

    standard_prediction = noise_ward.project_velocity_to_clean_state(
        LatentState(noised), times, noise_ward_velocity
    )
    data_ward_prediction = data_ward.project_velocity_to_clean_state(
        LatentState(noised), times, data_ward_velocity
    )

    for name in clean.component_names:
        assert torch.equal(standard_prediction.components[name], clean.components[name])
        assert torch.equal(data_ward_prediction.components[name], clean.components[name])


@pytest.mark.parametrize(
    ("state_dtype", "velocity_dtype", "expected_dtype"),
    [
        (torch.float16, torch.float32, torch.float32),
        (torch.float32, torch.bfloat16, torch.float32),
        (torch.float16, torch.bfloat16, torch.float32),
    ],
)
def test_velocity_projection_promotes_mixed_storage_and_compute_dtypes(
    state_dtype: torch.dtype,
    velocity_dtype: torch.dtype,
    expected_dtype: torch.dtype,
) -> None:
    adapter = _structured_adapter()
    state = LatentState(
        {
            "video": torch.tensor([[0.5, 1.0]], dtype=state_dtype),
            "audio": torch.tensor([[1.5, 2.0]], dtype=state_dtype),
        }
    )
    velocity = LatentState(
        {
            "video": torch.tensor([[0.25, 0.5]], dtype=velocity_dtype),
            "audio": torch.tensor([[0.75, 1.0]], dtype=velocity_dtype),
        }
    )
    times = ComponentTimes(
        timestep={"video": torch.tensor([500.0]), "audio": torch.tensor([250.0])},
        next_timestep={"video": torch.zeros(1), "audio": torch.zeros(1)},
        sigma={"video": torch.tensor([0.5]), "audio": torch.tensor([0.25])},
        next_sigma={"video": torch.zeros(1), "audio": torch.zeros(1)},
    )

    projected = adapter.project_velocity_to_clean_state(state, times, velocity)

    for name in state.component_names:
        promoted_state = state.components[name].to(expected_dtype)
        promoted_velocity = velocity.components[name].to(expected_dtype)
        expected = (
            promoted_state
            - to_broadcast_tensor(times.sigma[name], promoted_state) * promoted_velocity
        )
        assert projected.components[name].dtype == expected_dtype
        assert torch.equal(projected.components[name], expected)


def test_velocity_projection_rejects_an_unknown_direction() -> None:
    adapter = _structured_adapter(DataWardAdapterFake)
    adapter.flow_velocity_direction = "sideways"
    state = LatentState({"video": torch.zeros(1, 2), "audio": torch.zeros(1, 3)})
    velocity = LatentState({"video": torch.zeros(1, 2), "audio": torch.zeros(1, 3)})
    times = ComponentTimes(
        timestep={"video": torch.zeros(1), "audio": torch.zeros(1)},
        next_timestep={"video": torch.zeros(1), "audio": torch.zeros(1)},
        sigma={"video": torch.zeros(1), "audio": torch.zeros(1)},
        next_sigma={"video": torch.zeros(1), "audio": torch.zeros(1)},
    )

    with pytest.raises(ValueError, match=r"flow_velocity_direction.*'noise'.*'data'.*'sideways'"):
        adapter.project_velocity_to_clean_state(state, times, velocity)


def test_default_noising_consumes_exactly_one_random_draw() -> None:
    adapter = _adapter()
    clean = torch.zeros(2, 3, 4)
    times = _latent_times(torch.tensor([700.0, 300.0]))

    torch.manual_seed(12)
    adapter.add_forward_process_noise(LatentState({"latent": clean}), times)
    after_hook = torch.randn(5)

    torch.manual_seed(12)
    randn_tensor(clean.shape, device=clean.device, dtype=clean.dtype)
    after_legacy = torch.randn(5)

    assert torch.equal(after_hook, after_legacy)


def test_default_noising_uses_an_explicit_generator() -> None:
    adapter = _adapter()
    clean = torch.zeros(2, 3, 4)
    times = _latent_times(torch.tensor([700.0, 300.0]))

    noised = adapter.add_forward_process_noise(
        LatentState({"latent": clean}),
        times,
        generator=torch.Generator().manual_seed(13),
    )
    expected = randn_tensor(
        clean.shape,
        generator=torch.Generator().manual_seed(13),
        device=clean.device,
        dtype=clean.dtype,
    )

    assert torch.equal(noised.noise.components["latent"], expected)


def test_default_noising_keeps_the_component_dtype() -> None:
    """``to_broadcast_tensor`` casts sigma to the latent dtype; promotion would diverge."""
    adapter = _adapter()
    clean = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    primary = torch.tensor([700.0, 300.0])
    times = _latent_times(primary)

    torch.manual_seed(14)
    noised = adapter.add_forward_process_noise(LatentState({"latent": clean}), times)

    torch.manual_seed(14)
    legacy_noise = randn_tensor(clean.shape, device=clean.device, dtype=clean.dtype)
    legacy_sigma = to_broadcast_tensor(flow_match_sigma(primary), clean)
    assert noised.state.components["latent"].dtype is torch.bfloat16
    assert torch.equal(
        noised.state.components["latent"],
        (1 - legacy_sigma) * clean + legacy_sigma * legacy_noise,
    )


def test_default_noising_requires_the_single_latent_component_order() -> None:
    adapter = _structured_adapter()
    times = _latent_times(torch.tensor([700.0, 300.0]))

    with pytest.raises(
        ValueError,
        match=r"\('latent',\).*add_forward_process_noise.*\('video', 'audio'\)",
    ):
        adapter.add_forward_process_noise(LatentState({"latent": torch.zeros(2, 4)}), times)


def test_default_noising_requires_component_sigmas() -> None:
    adapter = _adapter()
    primary = torch.tensor([700.0, 300.0])
    times = ComponentTimes(
        timestep={"latent": primary}, next_timestep={"latent": torch.zeros_like(primary)}
    )

    with pytest.raises(ValueError, match=r"sigma.*\('latent',\).*received None"):
        adapter.add_forward_process_noise(LatentState({"latent": torch.zeros(2, 4)}), times)


def test_explicit_noise_application_supports_heterogeneous_components() -> None:
    adapter = _structured_adapter()
    video = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    audio = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    video_noise = torch.full_like(video, 0.5)
    audio_noise = torch.full_like(audio, -1.5)
    times = _heterogeneous_times()
    video_sigma = times.sigma["video"]
    audio_sigma = times.sigma["audio"]

    noised = adapter.apply_forward_process_noise(
        LatentState({"video": video, "audio": audio}),
        times,
        LatentState({"video": video_noise, "audio": audio_noise}),
    )

    for name, clean, noise, sigma in (
        ("video", video, video_noise, video_sigma),
        ("audio", audio, audio_noise, audio_sigma),
    ):
        sigma_broadcast = to_broadcast_tensor(sigma, clean)
        assert torch.equal(
            noised.state.components[name],
            (1 - sigma_broadcast) * clean + sigma_broadcast * noise,
        )
        assert torch.equal(noised.target_velocity.components[name], noise - clean)


def test_explicit_noise_application_consumes_no_randomness() -> None:
    adapter = _adapter()
    clean = torch.zeros(2, 3)
    times = _latent_times(torch.tensor([700.0, 300.0]))
    state_before = torch.get_rng_state()

    adapter.apply_forward_process_noise(
        LatentState({"latent": clean}), times, LatentState({"latent": torch.ones(2, 3)})
    )

    assert torch.equal(torch.get_rng_state(), state_before)


def test_explicit_noise_application_reuses_the_given_noise_state() -> None:
    """DPO builds the rejected arm from the chosen arm's noise; it must not be copied."""
    adapter = _adapter()
    noise = LatentState({"latent": torch.ones(2, 3)})
    times = _latent_times(torch.tensor([700.0, 300.0]))

    noised = adapter.apply_forward_process_noise(
        LatentState({"latent": torch.zeros(2, 3)}), times, noise
    )

    assert noised.noise is noise


def test_explicit_noise_application_rejects_a_foreign_component_order() -> None:
    adapter = _structured_adapter()
    times = ComponentTimes(
        timestep={"audio": torch.zeros(2), "video": torch.zeros(2)},
        next_timestep={"audio": torch.zeros(2), "video": torch.zeros(2)},
        sigma={"audio": torch.zeros(2), "video": torch.zeros(2)},
        next_sigma={"audio": torch.zeros(2), "video": torch.zeros(2)},
    )

    with pytest.raises(
        ValueError,
        match=r"apply_forward_process_noise.*clean_state.*\('video', 'audio'\).*"
        r"\('audio', 'video'\)",
    ):
        adapter.apply_forward_process_noise(
            LatentState({"audio": torch.zeros(2, 5), "video": torch.zeros(2, 3)}),
            times,
            LatentState({"audio": torch.zeros(2, 5), "video": torch.zeros(2, 3)}),
        )


def test_explicit_noise_application_rejects_a_noise_shape_mismatch() -> None:
    adapter = _adapter()
    times = _latent_times(torch.tensor([700.0, 300.0]))

    with pytest.raises(ValueError, match=r"noise component 'latent'.*\(2, 3\).*received.*\(2, 4\)"):
        adapter.apply_forward_process_noise(
            LatentState({"latent": torch.zeros(2, 3)}),
            times,
            LatentState({"latent": torch.zeros(2, 4)}),
        )


def test_explicit_noise_application_rejects_a_noise_dtype_mismatch() -> None:
    adapter = _adapter()
    times = _latent_times(torch.tensor([700.0, 300.0]))

    with pytest.raises(ValueError, match=r"noise component 'latent'.*torch.float32.*torch.float64"):
        adapter.apply_forward_process_noise(
            LatentState({"latent": torch.zeros(2, 3)}),
            times,
            LatentState({"latent": torch.zeros(2, 3, dtype=torch.float64)}),
        )


def test_explicit_noise_application_rejects_a_mismatched_component_batch_size() -> None:
    adapter = _structured_adapter()
    times = ComponentTimes(
        timestep={"video": torch.zeros(2), "audio": torch.zeros(3)},
        next_timestep={"video": torch.zeros(2), "audio": torch.zeros(3)},
        sigma={"video": torch.zeros(2), "audio": torch.zeros(3)},
        next_sigma={"video": torch.zeros(2), "audio": torch.zeros(3)},
    )

    with pytest.raises(ValueError, match=r"clean_state component 'audio'.*batch size 2.*\(3, 5\)"):
        adapter.apply_forward_process_noise(
            LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(3, 5)}),
            times,
            LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(3, 5)}),
        )


def test_explicit_noise_application_rejects_an_unbatched_primary_component() -> None:
    adapter = _adapter()
    times = _latent_times(torch.tensor([700.0, 300.0]))

    with pytest.raises(
        ValueError,
        match=r"clean_state component 'latent'.*leading batch dimension.*received shape \(\)",
    ):
        adapter.apply_forward_process_noise(
            LatentState({"latent": torch.tensor(1.0)}),
            times,
            LatentState({"latent": torch.tensor(2.0)}),
        )


def test_explicit_noise_application_rejects_a_sigma_without_one_value_per_sample() -> None:
    adapter = _adapter()
    primary = torch.tensor([700.0, 300.0])
    times = ComponentTimes(
        timestep={"latent": primary},
        next_timestep={"latent": torch.zeros_like(primary)},
        sigma={"latent": torch.zeros(2, 3)},
        next_sigma={"latent": torch.zeros(2, 3)},
    )

    with pytest.raises(
        ValueError, match=r"sigma for component 'latent'.*one value per sample.*\(2, 3\)"
    ):
        adapter.apply_forward_process_noise(
            LatentState({"latent": torch.zeros(2, 3)}),
            times,
            LatentState({"latent": torch.zeros(2, 3)}),
        )


def test_an_override_draws_per_component_and_reuses_the_application_hook() -> None:
    """Heterogeneous adapters replace only the ordered draw, in component order."""
    adapter = _structured_adapter(OrderedDrawAdapterFake)
    clean = {"video": torch.zeros(2, 3, 4), "audio": torch.zeros(2, 5)}
    times = _heterogeneous_times()

    torch.manual_seed(21)
    noised = adapter.add_forward_process_noise(LatentState(clean), times)

    torch.manual_seed(21)
    expected_video = randn_tensor((2, 3, 4))
    expected_audio = randn_tensor((2, 5))
    assert torch.equal(noised.noise.components["video"], expected_video)
    assert torch.equal(noised.noise.components["audio"], expected_audio)
    audio_sigma = to_broadcast_tensor(times.sigma["audio"], clean["audio"])
    assert torch.equal(noised.state.components["audio"], audio_sigma * expected_audio)


def test_terminal_state_reads_a_terminal_only_sparse_index_map() -> None:
    """Decoupled rollouts store one latent, so ``all_latents[:, -1]`` is not the contract."""
    adapter = _adapter()
    batch = _terminal_only_legacy_batch()

    terminal = adapter.get_terminal_state(batch)

    assert terminal.component_names == ("latent",)
    assert torch.equal(terminal.components["latent"], batch["all_latents"][:, 0])


def test_trajectory_seed_dispatch_reaches_every_component_scheduler_once() -> None:
    adapter = _adapter()

    adapter.set_trajectory_seed(7)

    assert adapter.pipeline.scheduler.seeds == [7]
