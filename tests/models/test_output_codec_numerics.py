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

"""Tests for role-neutral VAE encoding primitives used by conditions and outputs."""

from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch

from flow_factory.models.configured_image_output import (
    encode_shift_scale_vae_image,
    retrieve_vae_latents,
)
from flow_factory.models.flux._output import (
    encode_flux1_vae_image,
    encode_flux2_output_images,
    encode_flux2_vae_image,
)
from flow_factory.models.qwen_image._output import encode_qwen_vae_image


class _Posterior:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value
        self.mode_calls = 0
        self.sample_generators: list[Optional[torch.Generator]] = []

    def mode(self) -> torch.Tensor:
        self.mode_calls += 1
        return self.value

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        self.sample_generators.append(generator)
        return self.value + 7.0


class _ShiftScaleVAE:
    def __init__(self) -> None:
        self.config = SimpleNamespace(shift_factor=1.25, scaling_factor=2.5)
        self.posteriors: list[_Posterior] = []

    def encode(self, values: torch.Tensor) -> Any:
        posterior = _Posterior(values + 3.0)
        self.posteriors.append(posterior)
        return SimpleNamespace(latent_dist=posterior)


def test_retrieve_vae_latents_selects_argmax_or_sample_and_forwards_generator() -> None:
    """Posterior selection stays explicit and sampling receives the caller's generator."""
    posterior = _Posterior(torch.ones(1, 2))
    encoder_output = SimpleNamespace(latent_dist=posterior)
    generator = torch.Generator().manual_seed(23)

    argmax = retrieve_vae_latents(
        encoder_output,
        sample_mode="argmax",
        source="condition",
    )
    sampled = retrieve_vae_latents(
        encoder_output,
        sample_mode="sample",
        generator=generator,
        source="target",
    )

    assert torch.equal(argmax, posterior.value)
    assert torch.equal(sampled, posterior.value + 7.0)
    assert posterior.mode_calls == 1
    assert posterior.sample_generators == [generator]


@pytest.mark.parametrize(
    ("sample_mode", "error_type", "message"),
    [
        (None, TypeError, "sample_mode to be str"),
        ("mode", ValueError, "sample_mode in"),
    ],
)
def test_retrieve_vae_latents_requires_an_explicit_supported_selection(
    sample_mode: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Callers cannot inherit an implicit global posterior policy."""
    with pytest.raises(error_type, match=message):
        retrieve_vae_latents(
            SimpleNamespace(latent_dist=_Posterior(torch.ones(1))),
            sample_mode=sample_mode,  # type: ignore[arg-type]
            source="test",
        )


def test_shift_scale_primitive_shares_normalization_but_not_posterior_policy() -> None:
    """Condition and output roles share math while selecting their official posterior path."""
    pixels = torch.randn(2, 3, 8, 8)
    adapter = SimpleNamespace(vae=_ShiftScaleVAE())
    generator = torch.Generator().manual_seed(29)

    condition_latents = encode_shift_scale_vae_image(
        adapter,
        pixels,
        sample_mode="argmax",
        source="condition",
    )
    output_latents = encode_shift_scale_vae_image(
        adapter,
        pixels,
        sample_mode="sample",
        generator=generator,
        source="output",
    )

    assert torch.equal(condition_latents, (pixels + 3.0 - 1.25) * 2.5)
    assert torch.equal(
        output_latents,
        (adapter.vae.posteriors[1].value + 7.0 - 1.25) * 2.5,
    )
    assert adapter.vae.posteriors[0].mode_calls == 1
    assert adapter.vae.posteriors[1].sample_generators == [generator]


def test_flux1_wrapper_delegates_to_the_role_neutral_shift_scale_primitive() -> None:
    """FLUX.1 condition and target orchestration cannot drift numerically."""
    pixels = torch.randn(1, 3, 8, 8)
    adapter = SimpleNamespace(vae=_ShiftScaleVAE())

    assert torch.equal(
        encode_flux1_vae_image(adapter, pixels, sample_mode="argmax"),
        (pixels + 3.0 - 1.25) * 2.5,
    )


def test_flux2_primitive_applies_posterior_argmax_patchify_and_batch_norm() -> None:
    """FLUX.2 keeps the official deterministic transform in one shared helper."""
    pixels = torch.randn(2, 2, 4, 4)
    posterior = _Posterior(pixels + 1.0)
    vae = SimpleNamespace(
        encode=lambda values: SimpleNamespace(latent_dist=posterior),
        bn=SimpleNamespace(
            running_mean=torch.tensor([0.25, -0.5]),
            running_var=torch.tensor([0.75, 1.25]),
        ),
        config=SimpleNamespace(batch_norm_eps=1e-5),
    )
    adapter = SimpleNamespace(
        vae=vae,
        pipeline=SimpleNamespace(_patchify_latents=lambda values: values),
    )

    actual = encode_flux2_vae_image(adapter, pixels, sample_mode="argmax")

    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + 1e-5)
    assert torch.allclose(actual, (pixels + 1.0 - mean) / std)
    assert posterior.mode_calls == 1


def test_flux2_output_orchestration_samples_with_the_passed_generator() -> None:
    """The target-only wrapper fixes selection to sample without replacing its RNG."""
    pixels = torch.randn(1, 2, 4, 4)
    posterior = _Posterior(pixels)
    vae = SimpleNamespace(
        encode=lambda values: SimpleNamespace(latent_dist=posterior),
        bn=SimpleNamespace(running_mean=torch.zeros(2), running_var=torch.ones(2)),
        config=SimpleNamespace(batch_norm_eps=0.0),
    )
    pipeline = SimpleNamespace(
        _patchify_latents=lambda values: values,
        _prepare_latent_ids=lambda values: torch.zeros(4, 3),
        _pack_latents=lambda values: values.flatten(2).transpose(1, 2),
    )
    adapter = SimpleNamespace(vae=vae, pipeline=pipeline, device=torch.device("cpu"))
    generator = torch.Generator().manual_seed(31)

    encoded = encode_flux2_output_images(adapter, pixels, generator)

    assert encoded.latents.shape == (1, 16, 2)
    assert posterior.sample_generators == [generator]
    assert posterior.mode_calls == 0


def test_qwen_primitive_normalizes_five_dimensional_latents_per_channel() -> None:
    """Qwen condition and target roles share one BCFHW channel transform."""
    pixels = torch.randn(2, 2, 1, 4, 4)
    posterior = _Posterior(pixels + 2.0)
    vae = SimpleNamespace(
        encode=lambda values: SimpleNamespace(latent_dist=posterior),
        config=SimpleNamespace(latents_mean=[1.0, -2.0], latents_std=[2.0, 4.0]),
    )
    adapter = SimpleNamespace(vae=vae)

    actual = encode_qwen_vae_image(adapter, pixels, sample_mode="argmax")

    mean = torch.tensor([1.0, -2.0]).view(1, 2, 1, 1, 1)
    std = torch.tensor([2.0, 4.0]).view(1, 2, 1, 1, 1)
    assert torch.equal(actual, (pixels + 2.0 - mean) / std)
    assert posterior.mode_calls == 1


@pytest.mark.parametrize(
    ("shift", "scale", "message"),
    [
        (None, 1.0, "shift_factor"),
        (0.0, 0.0, "scaling_factor > 0"),
    ],
)
def test_shift_scale_primitive_rejects_ambiguous_vae_normalization(
    shift: object,
    scale: object,
    message: str,
) -> None:
    """A missing or invalid VAE normalization fails at the shared boundary."""
    vae = _ShiftScaleVAE()
    vae.config = SimpleNamespace(shift_factor=shift, scaling_factor=scale)

    with pytest.raises((TypeError, ValueError), match=message):
        encode_shift_scale_vae_image(
            SimpleNamespace(vae=vae),
            torch.zeros(1, 1, 2, 2),
            sample_mode="argmax",
        )
