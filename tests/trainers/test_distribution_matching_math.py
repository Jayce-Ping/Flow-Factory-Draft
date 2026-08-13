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

from typing import Any, List

import torch

from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import BaseSample, LatentState
from flow_factory.scheduler import SDESchedulerOutput
from flow_factory.trainers.distillation.distribution_matching import (
    add_flow_noise,
    dmd_generator_loss,
    flow_matching_loss,
    flow_velocity_target,
    tdm_fake_loss,
    tdm_generator_loss,
    velocity_to_x0,
)


class AdapterFake(BaseAdapter):
    """Minimal concrete adapter for objective reduction tests."""

    trajectory_component_order = ("latent",)

    def load_pipeline(self) -> Any:
        raise NotImplementedError

    def decode_latents(self, latents: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return latents

    def inference(self, **kwargs: Any) -> List[BaseSample]:
        return []

    def forward(self, **kwargs: Any) -> SDESchedulerOutput:
        raise NotImplementedError


def _adapter() -> AdapterFake:
    return object.__new__(AdapterFake)


def _state(tensor: torch.Tensor) -> LatentState:
    return LatentState({"latent": tensor})


def test_flow_x0_round_trip() -> None:
    torch.manual_seed(0)
    x0 = _state(torch.randn(2, 4, 3, 3))
    noise = _state(torch.randn(2, 4, 3, 3))
    for value in (0.1, 0.5, 0.9):
        sigma = {"latent": torch.full((2,), value)}
        x_t = add_flow_noise(x0, noise, sigma)
        velocity = flow_velocity_target(x0, noise)
        recovered = velocity_to_x0(x_t, velocity, sigma)
        torch.testing.assert_close(recovered.components["latent"], x0.components["latent"])


def test_dmd_generator_loss_gradient_matches_normalized_direction() -> None:
    adapter = _adapter()
    x0_gen = _state(torch.randn(2, 4, 3, 3, requires_grad=True))
    x0_real = _state(torch.randn(2, 4, 3, 3))
    x0_fake = _state(torch.randn(2, 4, 3, 3))
    loss = dmd_generator_loss(adapter, x0_gen, x0_real, x0_fake)
    loss.backward()

    gen = x0_gen.components["latent"].detach()
    real = x0_real.components["latent"]
    fake = x0_fake.components["latent"]
    per_sample = (gen - real).abs().flatten(1).mean(dim=1).clamp_min(1e-6)
    direction = (fake - real) / per_sample.reshape(-1, 1, 1, 1)
    expected = direction / gen.numel()
    torch.testing.assert_close(x0_gen.components["latent"].grad, expected, atol=1e-5, rtol=1e-5)


def test_flow_matching_loss_is_velocity_mse() -> None:
    adapter = _adapter()
    v_pred = _state(torch.tensor([[[1.0, 3.0]]]))
    v_target = _state(torch.tensor([[[0.0, 1.0]]]))
    loss = flow_matching_loss(adapter, v_pred, v_target, state=v_pred)
    torch.testing.assert_close(loss, torch.tensor(2.5))


def test_tdm_fake_loss_uses_per_sample_snr_weights() -> None:
    adapter = _adapter()
    x0_fake = _state(torch.ones(2, 1, 1, 1, requires_grad=True))
    x0_target = _state(torch.zeros(2, 1, 1, 1))
    low = tdm_fake_loss(
        adapter,
        x0_fake,
        x0_target,
        sigma=torch.tensor([0.9, 0.9]),
        importance=torch.ones(2),
        snr_gamma=5.0,
    )
    high = tdm_fake_loss(
        adapter,
        x0_fake,
        x0_target,
        sigma=torch.tensor([0.1, 0.1]),
        importance=torch.ones(2),
        snr_gamma=5.0,
    )
    mixed = tdm_fake_loss(
        adapter,
        x0_fake,
        x0_target,
        sigma=torch.tensor([0.9, 0.1]),
        importance=torch.ones(2),
        snr_gamma=5.0,
    )
    assert mixed.item() != low.item()
    assert mixed.item() != high.item()
    assert low.item() < high.item()


def test_tdm_fake_loss_is_finite_with_unit_importance() -> None:
    adapter = _adapter()
    x0_fake = _state(torch.randn(2, 4, 3, 3, requires_grad=True))
    x0_target = _state(torch.randn(2, 4, 3, 3))
    loss = tdm_fake_loss(
        adapter,
        x0_fake,
        x0_target,
        sigma=torch.tensor([0.4, 0.6]),
        importance=torch.ones(2),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert x0_fake.components["latent"].grad is not None


def test_tdm_generator_huber_correction_has_real_minus_fake_sign() -> None:
    adapter = _adapter()
    x0_student = _state(torch.zeros(2, 1, 2, 2, requires_grad=True))
    x0_real = _state(torch.ones(2, 1, 2, 2))
    x0_fake = _state(torch.full((2, 1, 2, 2), 3.0))
    loss = tdm_generator_loss(
        adapter,
        x0_student,
        x0_real,
        x0_fake,
        use_huber=True,
        huber_c=1e-3,
    )
    loss.backward()
    grad = x0_student.components["latent"].grad
    assert grad is not None
    assert torch.all(grad > 0)
