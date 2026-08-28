from types import MethodType
from typing import Any

import pytest
import torch

from flow_factory.models.bagel.bagel import BagelAdapter, BagelSample
from flow_factory.samples import BaseSample
from flow_factory.trainers.distillation.distillation_runtime import (
    validate_media_free_rollout,
    without_media_decoding,
)


def _adapter(decoder: Any) -> BagelAdapter:
    adapter = object.__new__(BagelAdapter)
    adapter.decode_latents = MethodType(decoder, adapter)
    return adapter


def _result(batch_size: int = 2) -> dict[str, Any]:
    initial = torch.zeros(batch_size, 4, 8)
    terminal = torch.ones(batch_size, 4, 8)
    return {
        "final_latents": terminal,
        "all_latents": [initial, terminal],
        "all_log_probs": None,
        "timesteps": torch.tensor([1000.0]),
        "latent_index_map": torch.tensor([0, 1]),
        "log_prob_index_map": None,
        "callback_results": {},
        "callback_index_map": None,
    }


def test_bagel_assembles_samples_from_one_batched_decode() -> None:
    calls: list[tuple[tuple[int, ...], tuple[int, int] | None]] = []

    def decode(
        _self: BagelAdapter,
        latents: torch.Tensor,
        image_shape: tuple[int, int] | None = None,
    ) -> list[torch.Tensor]:
        calls.append((tuple(latents.shape), image_shape))
        return [torch.full((3, 8, 8), index) for index in range(latents.shape[0])]

    adapter = _adapter(decode)
    samples = adapter._assemble_samples(
        _result(),
        prompts=["first", "second"],
        condition_images_list=None,
        height=64,
        width=64,
    )

    assert calls == [((2, 4, 8), (64, 64))]
    assert all(isinstance(sample, BagelSample) for sample in samples)
    assert torch.equal(samples[0].image, torch.zeros(3, 8, 8))
    assert torch.equal(samples[1].image, torch.ones(3, 8, 8))


def test_bagel_media_free_samples_keep_replay_trajectory() -> None:
    def decode(
        _self: BagelAdapter,
        latents: torch.Tensor,
        image_shape: tuple[int, int] | None = None,
    ) -> list[torch.Tensor]:
        del latents, image_shape
        raise AssertionError("TDM must not invoke the real Bagel decoder")

    adapter = _adapter(decode)
    validate_media_free_rollout(adapter, algorithm_name="TDM")

    with without_media_decoding(adapter, algorithm_name="TDM"):
        samples = adapter._assemble_samples(
            _result(),
            prompts=["first", "second"],
            condition_images_list=None,
            height=64,
            width=64,
        )

    assert [sample.image for sample in samples] == [None, None]
    batch = BaseSample.stack(samples)
    replay = adapter.get_replay_step(batch, 0)
    assert replay.state.components["latent"].shape == (2, 4, 8)
    assert replay.next_state.components["latent"].shape == (2, 4, 8)

    with pytest.raises(AssertionError, match="real Bagel decoder"):
        adapter.decode_latents(torch.zeros(2, 4, 8), image_shape=(64, 64))


def test_bagel_maps_reference_guidance_to_text_cfg() -> None:
    adapter = object.__new__(BagelAdapter)

    assert adapter.reference_guidance_kwargs(4.0) == {"cfg_text_scale": 4.0}
