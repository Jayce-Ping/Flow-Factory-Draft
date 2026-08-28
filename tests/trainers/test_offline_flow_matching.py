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

from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import torch

from flow_factory.contracts import MediaType
from flow_factory.models.output_state import (
    EncodedOutputState,
    GeometrySignature,
    MediaGeometrySignature,
)
from flow_factory.samples import ComponentTimes, LatentState, NoisedState
from flow_factory.trainers.common.flow_matching import (
    build_noised_output_state,
    flow_matching_per_sample_loss,
    sample_offline_timesteps,
    validate_preference_component_times,
    validate_preference_output_states,
)


def _signature(height: int = 16, width: int = 16) -> GeometrySignature:
    return GeometrySignature(
        media=(MediaGeometrySignature(type=MediaType.IMAGE, height=height, width=width),)
    )


def _encoded(
    values: torch.Tensor,
    *,
    signature: GeometrySignature | None = None,
    forward_context: dict[str, Any] | None = None,
    mask: torch.Tensor | None = None,
) -> EncodedOutputState:
    return EncodedOutputState(
        clean_state=LatentState(
            {"latent": values},
            active_masks=None if mask is None else {"latent": mask},
        ),
        forward_context={} if forward_context is None else forward_context,
        decode_context={},
        geometry_signatures=tuple((signature or _signature()) for _ in range(values.shape[0])),
    )


@pytest.mark.parametrize("scheme", ["logit_normal", "uniform"])
def test_offline_timestep_sampling_materializes_independent_batch_coordinates(
    scheme: str,
) -> None:
    args = SimpleNamespace(
        weighting_scheme=scheme,
        num_train_timesteps=3,
        timestep_range=(0.0, 0.99),
        time_shift=1.0,
        logit_mean=0.0,
        logit_std=1.0,
    )

    timesteps = sample_offline_timesteps(
        args,
        batch_size=4,
        device="cpu",
        generator=torch.Generator().manual_seed(7),
    )

    assert timesteps.shape == (3, 4)
    assert timesteps.is_contiguous()
    assert any(not torch.equal(row, row[:1].expand_as(row)) for row in timesteps)


def test_build_noised_output_state_reuses_explicit_noise_without_another_draw() -> None:
    times = ComponentTimes(
        timestep={"latent": torch.tensor([500.0, 250.0])},
        next_timestep={"latent": torch.zeros(2)},
        sigma={"latent": torch.tensor([0.5, 0.25])},
        next_sigma={"latent": torch.zeros(2)},
    )
    clean = LatentState({"latent": torch.zeros(2, 2)})
    shared_noise = LatentState({"latent": torch.ones(2, 2)})
    events: list[str] = []

    class Adapter:
        def build_training_component_times(self, primary: torch.Tensor, *, batch: Any):
            events.append(f"times:{batch['arm']}")
            return times

        def add_forward_process_noise(self, *args: Any, **kwargs: Any):
            raise AssertionError("explicit shared noise must not draw again")

        def apply_forward_process_noise(self, state: Any, component_times: Any, noise: Any):
            events.append("apply")
            return NoisedState(state=state, target_velocity=noise, noise=noise)

    returned_times, noised = build_noised_output_state(
        Adapter(),
        clean,
        torch.tensor([500.0, 250.0]),
        batch=MappingProxyType({"arm": "rejected"}),
        noise=shared_noise,
    )

    assert returned_times is times
    assert noised.noise is shared_noise
    assert events == ["times:rejected", "apply"]

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_noised_output_state(
            Adapter(),
            clean,
            torch.tensor([500.0, 250.0]),
            batch={"arm": "chosen"},
            generator=torch.Generator(),
            noise=shared_noise,
        )


def test_preference_arms_share_coordinates_and_noise_but_keep_their_own_batches() -> None:
    seen_batches: list[str] = []

    class Adapter:
        def build_training_component_times(self, primary: torch.Tensor, *, batch: Any):
            seen_batches.append(batch["arm"])
            sigma = primary.to(torch.float64).div(1000).to(primary.dtype)
            return ComponentTimes(
                timestep={"latent": primary},
                next_timestep={"latent": torch.zeros_like(primary)},
                sigma={"latent": sigma},
                next_sigma={"latent": torch.zeros_like(sigma)},
            )

        def add_forward_process_noise(
            self,
            clean_state: LatentState,
            times: ComponentTimes,
            *,
            generator: torch.Generator | None,
        ) -> NoisedState:
            clean = clean_state.components["latent"]
            noise = LatentState(
                {
                    "latent": torch.randn(
                        clean.shape,
                        generator=generator,
                        device=clean.device,
                        dtype=clean.dtype,
                    )
                }
            )
            return self.apply_forward_process_noise(clean_state, times, noise)

        def apply_forward_process_noise(
            self,
            clean_state: LatentState,
            times: ComponentTimes,
            noise: LatentState,
        ) -> NoisedState:
            clean = clean_state.components["latent"]
            sigma = times.sigma["latent"].unsqueeze(1).to(clean)
            noised = (1 - sigma) * clean + sigma * noise.components["latent"]
            target = noise.components["latent"] - clean
            return NoisedState(
                state=LatentState({"latent": noised}),
                target_velocity=LatentState({"latent": target}),
                noise=noise,
            )

    adapter = Adapter()
    primary = torch.tensor([725.0, 125.0])
    chosen_times, chosen = build_noised_output_state(
        adapter,
        LatentState({"latent": torch.zeros(2, 3)}),
        primary,
        batch={"arm": "chosen"},
        generator=torch.Generator().manual_seed(19),
    )
    rejected_times, rejected = build_noised_output_state(
        adapter,
        LatentState({"latent": torch.ones(2, 3)}),
        primary,
        batch={"arm": "rejected"},
        noise=chosen.noise,
    )

    assert seen_batches == ["chosen", "rejected"]
    assert torch.equal(chosen_times.timestep["latent"], rejected_times.timestep["latent"])
    assert torch.equal(chosen_times.sigma["latent"], rejected_times.sigma["latent"])
    assert rejected.noise is chosen.noise


def test_preference_component_times_reject_context_dependent_schedule_drift() -> None:
    chosen = ComponentTimes(
        timestep={"latent": torch.tensor([500.0, 250.0])},
        next_timestep={"latent": torch.zeros(2)},
        sigma={"latent": torch.tensor([0.5, 0.25])},
        next_sigma={"latent": torch.zeros(2)},
    )
    rejected = ComponentTimes(
        timestep={"latent": chosen.timestep["latent"].clone()},
        next_timestep={"latent": chosen.next_timestep["latent"].clone()},
        sigma={"latent": torch.tensor([0.5, 0.2])},
        next_sigma={"latent": chosen.next_sigma["latent"].clone()},
    )

    with pytest.raises(ValueError, match="component times values mismatch"):
        validate_preference_component_times(chosen, rejected)

    rejected.sigma = {"latent": chosen.sigma["latent"].clone()}
    validate_preference_component_times(chosen, rejected)


def test_flow_matching_loss_computes_fp32_errors_before_adapter_reduction() -> None:
    predicted = LatentState(
        {
            "video": torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float16),
            "audio": torch.tensor([[5.0], [7.0]], dtype=torch.float16),
        }
    )
    target = LatentState(
        {
            "video": torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float16),
            "audio": torch.tensor([[2.0], [3.0]], dtype=torch.float16),
        }
    )
    state = LatentState(
        {
            "video": torch.zeros(2, 2, dtype=torch.float16),
            "audio": torch.zeros(2, 1, dtype=torch.float16),
        }
    )
    received: dict[str, Any] = {}

    class Adapter:
        def reduce_latent_values(self, values: Any, *, state: Any):
            received["values"] = values
            received["state"] = state
            total = torch.cat([value.flatten(1) for value in values.values()], dim=1)
            return total.mean(dim=1)

    noised = NoisedState(state=state, target_velocity=target, noise=target)
    loss = flow_matching_per_sample_loss(Adapter(), predicted, noised)

    torch.testing.assert_close(loss, torch.tensor([(1.0 + 4.0 + 9.0) / 3, 7.0]))
    assert all(value.dtype is torch.float32 for value in received["values"].values())
    assert received["state"] is state


def test_preference_state_validation_accepts_content_specific_context_values() -> None:
    chosen = _encoded(
        torch.zeros(2, 4),
        forward_context={"context": torch.zeros(2, 3), "label": "chosen"},
    )
    rejected = _encoded(
        torch.ones(2, 4),
        forward_context={"context": torch.ones(2, 3), "label": "rejected"},
    )

    validate_preference_output_states(chosen, rejected)


@pytest.mark.parametrize("mismatch", ["shape", "geometry", "mask", "context"])
def test_preference_state_validation_rejects_incompatible_forward_processes(
    mismatch: str,
) -> None:
    chosen_mask = torch.ones(2, 1, dtype=torch.bool)
    rejected_mask = chosen_mask.clone()
    chosen = _encoded(
        torch.zeros(2, 4),
        mask=chosen_mask if mismatch == "mask" else None,
        forward_context={"ids": torch.zeros(4, 3)},
    )
    rejected = _encoded(
        torch.zeros(2, 5) if mismatch == "shape" else torch.ones(2, 4),
        signature=_signature(32, 16) if mismatch == "geometry" else None,
        mask=rejected_mask.logical_not() if mismatch == "mask" else None,
        forward_context={
            "ids": torch.zeros(5 if mismatch == "context" else 4, 3),
        },
    )

    with pytest.raises((TypeError, ValueError), match="preference arm"):
        validate_preference_output_states(chosen, rejected)
