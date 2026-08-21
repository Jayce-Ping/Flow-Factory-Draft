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
from typing import Any, Dict, List, Optional

import pytest
import torch
from diffusers.utils.torch_utils import randn_tensor

from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import (
    BaseSample,
    ComponentTimes,
    ComponentTrajectory,
    IndexedTrajectoryTensor,
    LatentState,
    StructuredTrajectory,
)
from flow_factory.scheduler import SchedulerGroup, SDESchedulerOutput

VIDEO_MASK = torch.tensor([[False], [True], [True]])
AUDIO_MASK = torch.ones(2, 1, dtype=torch.bool)


class SchedulerFake:
    """Small scheduler-like object for adapter group construction."""

    def step(self) -> None:
        """Provide scheduler compatibility."""


class StructuredAdapterFake(BaseAdapter):
    """Adapter fake declaring the ordered video/audio component contract."""

    trajectory_component_order = ("video", "audio")

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
        return SDESchedulerOutput(velocity=kwargs["latents"])

    def add_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Any:
        """Draw one full-shape noise tensor per component, in component order."""
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


def _adapter() -> StructuredAdapterFake:
    adapter = object.__new__(StructuredAdapterFake)
    video = SchedulerFake()
    adapter.pipeline = SimpleNamespace(scheduler=video)
    adapter.scheduler_group = SchedulerGroup(
        {"video": video, "audio": SchedulerFake()}, primary_name="video"
    )
    return adapter


def _sample(offset: float, *, masked: bool, audio_masked: Optional[bool] = None) -> BaseSample:
    audio_masked = masked if audio_masked is None else audio_masked
    return BaseSample(
        trajectory=StructuredTrajectory(
            components={
                "video": ComponentTrajectory(
                    states=torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) + offset,
                    timesteps=torch.tensor([1000.0, 0.0]),
                    state_index_map=torch.tensor([0, 1]),
                    active_mask=VIDEO_MASK if masked else None,
                ),
                "audio": ComponentTrajectory(
                    states=torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) + offset,
                    timesteps=torch.tensor([1000.0, 0.0]),
                    state_index_map=torch.tensor([0, 1]),
                    active_mask=AUDIO_MASK if audio_masked else None,
                ),
            },
            callbacks={
                "velocity": {
                    "video": IndexedTrajectoryTensor(
                        values=torch.ones(1, 3, 2) * offset,
                        index_map=torch.tensor([0, -1]),
                    ),
                    "audio": IndexedTrajectoryTensor(
                        values=torch.ones(1, 2, 2) * offset,
                        index_map=torch.tensor([0, -1]),
                    ),
                }
            },
        )
    )


def _batch(*, masked: bool = True, audio_masked: Optional[bool] = None) -> Any:
    return BaseSample.stack(
        [
            _sample(0.0, masked=masked, audio_masked=audio_masked),
            _sample(100.0, masked=masked, audio_masked=audio_masked),
        ]
    )


def _masked_state() -> LatentState:
    return LatentState(
        {
            "video": torch.tensor([[[1.0, 3.0], [2.0, 4.0], [6.0, 8.0]]] * 2),
            "audio": torch.tensor([[[10.0, 20.0], [30.0, 40.0]]] * 2),
        },
        active_masks={
            "video": VIDEO_MASK.unsqueeze(0).expand(2, 3, 1),
            "audio": AUDIO_MASK.unsqueeze(0).expand(2, 2, 1),
        },
    )


def _times() -> ComponentTimes:
    return ComponentTimes(
        timestep={"video": torch.tensor([500.0, 500.0]), "audio": torch.tensor([500.0, 500.0])},
        next_timestep={"video": torch.zeros(2), "audio": torch.zeros(2)},
        sigma={"video": torch.tensor([0.5, 0.25]), "audio": torch.tensor([0.5, 0.25])},
        next_sigma={"video": torch.zeros(2), "audio": torch.zeros(2)},
    )


def test_structured_terminal_state_attaches_component_masks() -> None:
    terminal = _adapter().get_terminal_state(_batch())

    assert tuple(terminal.active_masks) == ("video", "audio")
    assert torch.equal(terminal.active_masks["video"], VIDEO_MASK.unsqueeze(0).expand(2, 3, 1))
    assert torch.equal(terminal.active_masks["audio"], AUDIO_MASK.unsqueeze(0).expand(2, 2, 1))


def test_structured_replay_attaches_masks_to_both_states() -> None:
    replay = _adapter().get_replay_step(_batch(), 0)

    assert tuple(replay.state.active_masks) == ("video", "audio")
    assert tuple(replay.next_state.active_masks) == ("video", "audio")


def test_structured_replay_callback_attaches_component_masks() -> None:
    callback = _adapter().get_replay_callback(_batch(), 0, "velocity")

    assert tuple(callback.active_masks) == ("video", "audio")


def test_structured_hooks_reject_partial_component_mask_presence() -> None:
    with pytest.raises(
        ValueError,
        match=r"active_mask.*all components.*\('video', 'audio'\).*missing.*\('audio',\)",
    ):
        _adapter().get_terminal_state(_batch(masked=True, audio_masked=False))


def test_maskless_structured_states_stay_maskless() -> None:
    terminal = _adapter().get_terminal_state(_batch(masked=False))

    assert terminal.active_masks is None


def test_masked_noising_keeps_inactive_elements_clean_with_zero_target() -> None:
    adapter = _adapter()
    clean = _masked_state()
    times = _times()

    torch.manual_seed(31)
    noised = adapter.add_forward_process_noise(clean, times)

    video = noised.state.components["video"]
    target = noised.target_velocity.components["video"]
    assert torch.equal(video[:, 0], clean.components["video"][:, 0])
    assert torch.equal(target[:, 0], torch.zeros_like(target[:, 0]))
    assert not torch.equal(video[:, 1], clean.components["video"][:, 1])
    assert tuple(noised.state.active_masks) == ("video", "audio")
    assert tuple(noised.target_velocity.active_masks) == ("video", "audio")


def test_masked_noising_consumes_the_same_randomness_as_the_unmasked_draw() -> None:
    adapter = _adapter()
    clean = _masked_state()
    times = _times()

    torch.manual_seed(32)
    adapter.add_forward_process_noise(clean, times)
    after_masked = torch.randn(4)

    torch.manual_seed(32)
    adapter.add_forward_process_noise(LatentState(dict(clean.components)), times)
    after_unmasked = torch.randn(4)

    assert torch.equal(after_masked, after_unmasked)


def test_active_numel_counts_broadcast_active_elements() -> None:
    active = _adapter().get_state_active_numel(_masked_state())

    assert active == {"video": 4, "audio": 4}


def test_active_numel_without_masks_uses_the_full_component_numel() -> None:
    state = LatentState({"video": torch.zeros(2, 3, 2), "audio": torch.zeros(2, 2, 2)})

    assert _adapter().get_state_active_numel(state) == {"video": 6, "audio": 4}


def test_active_numel_rejects_a_varying_batch_active_count() -> None:
    state = LatentState(
        {"video": torch.zeros(2, 3, 2), "audio": torch.zeros(2, 2, 2)},
        active_masks={
            "video": torch.tensor([[[True], [True], [False]], [[True], [False], [False]]]),
            "audio": torch.ones(2, 2, 1, dtype=torch.bool),
        },
    )

    with pytest.raises(ValueError, match=r"component 'video'.*constant.*per sample.*\[4, 2\]"):
        _adapter().get_state_active_numel(state)


def test_active_numel_rejects_a_fully_inactive_component() -> None:
    state = LatentState(
        {"video": torch.zeros(2, 3, 2), "audio": torch.zeros(2, 2, 2)},
        active_masks={
            "video": torch.zeros(2, 3, 1, dtype=torch.bool),
            "audio": torch.ones(2, 2, 1, dtype=torch.bool),
        },
    )

    with pytest.raises(ValueError, match=r"component 'video'.*positive.*received 0"):
        _adapter().get_state_active_numel(state)


def test_masked_component_reducer_averages_only_active_elements() -> None:
    adapter = _adapter()
    state = _masked_state()
    values = {
        "video": torch.tensor([[[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]]] * 2),
        "audio": torch.tensor([[[1.0, 3.0], [5.0, 7.0]]] * 2),
    }

    reduced = adapter.reduce_component_latent_values(values, state=state)

    assert torch.equal(reduced["video"], torch.tensor([3.0, 3.0]))
    assert torch.equal(reduced["audio"], torch.tensor([4.0, 4.0]))


def test_masked_global_reducer_uses_active_sums_and_counts() -> None:
    adapter = _adapter()
    state = _masked_state()
    values = {
        "video": torch.tensor([[[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]]] * 2),
        "audio": torch.tensor([[[1.0, 3.0], [5.0, 7.0]]] * 2),
    }

    reduced = adapter.reduce_latent_values(values, state=state)

    assert torch.equal(reduced, torch.tensor([28.0 / 8.0, 28.0 / 8.0]))


def test_masked_global_reducer_keeps_explicit_active_numel_for_pre_reduced_values() -> None:
    adapter = _adapter()
    state = _masked_state()

    reduced = adapter.reduce_latent_values(
        {"video": torch.tensor([2.0, 3.0]), "audio": torch.tensor([10.0, 20.0])},
        active_numel={"video": 2, "audio": 1},
        state=state,
    )

    assert torch.equal(reduced, torch.tensor([14.0 / 3.0, 26.0 / 3.0]))


def test_maskless_reducers_stay_bit_identical() -> None:
    adapter = _adapter()
    values: Dict[str, torch.Tensor] = {
        "video": torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
        "audio": torch.tensor([[10.0], [20.0]]),
    }
    state = LatentState({"video": values["video"].clone(), "audio": values["audio"].clone()})

    assert torch.equal(
        adapter.reduce_latent_values(values, state=state),
        torch.tensor([14.0 / 3.0, 26.0 / 3.0]),
    )
    assert torch.equal(
        adapter.reduce_component_latent_values(values, state=state)["video"],
        torch.tensor([2.0, 3.0]),
    )


def test_masked_reducers_reject_a_fully_inactive_component() -> None:
    adapter = _adapter()
    state = LatentState(
        {"video": torch.zeros(2, 3, 2), "audio": torch.zeros(2, 2, 2)},
        active_masks={
            "video": torch.zeros(2, 3, 1, dtype=torch.bool),
            "audio": torch.ones(2, 2, 1, dtype=torch.bool),
        },
    )

    with pytest.raises(ValueError, match=r"component 'video'.*positive.*active"):
        adapter.reduce_component_latent_values(
            {"video": torch.zeros(2, 3, 2), "audio": torch.zeros(2, 2, 2)},
            state=state,
        )
