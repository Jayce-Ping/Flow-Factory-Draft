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

"""Tests for MiniMax H3 offline audiovisual target encoding."""

from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
import torch

from flow_factory.contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaBinding,
    InputMediaOrder,
    MediaType,
    NegativePromptPolicy,
    RateRequirement,
    validate_pipeline_model_input,
)
from flow_factory.data_utils.offline_dataset import DecodedMedia
from flow_factory.data_utils.schema import MediaAsset, NormalizedModelInput
from flow_factory.models.minimax_h3._output import (
    MiniMaxH3AVOutputCodec,
    prepare_h3_target_audio,
    prepare_h3_target_video,
    resolve_h3_output_geometry,
    validate_h3_encoded_output_geometry,
)
from flow_factory.models.minimax_h3.adapters import (
    MiniMaxH3FL2VAAdapter,
    MiniMaxH3Ref2VAAdapter,
    MiniMaxH3T2VAAdapter,
)
from flow_factory.samples import ComponentTimes, LatentState
from flow_factory.scheduler import MiniMaxH3SDEScheduler
from flow_factory.trainers.common.flow_matching import (
    build_noised_output_state,
    flow_matching_per_sample_loss,
    validate_preference_component_times,
    validate_preference_output_states,
)


class _Posterior:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values
        self.sample_generator: Optional[torch.Generator] = None
        self.sample_calls = 0
        self.mode_calls = 0

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        self.sample_generator = generator
        self.sample_calls += 1
        return self.values

    def mode(self) -> torch.Tensor:
        self.mode_calls += 1
        return self.values


class _VideoVAE:
    def __init__(self, posterior: _Posterior) -> None:
        self.posterior = posterior
        self.config = SimpleNamespace(
            latents_mean=[0.0] * 24,
            latents_std=[1.0] * 24,
        )
        self.encoded_pixels: Optional[torch.Tensor] = None

    def encode(self, pixels: torch.Tensor) -> Any:
        self.encoded_pixels = pixels
        return SimpleNamespace(latent_dist=self.posterior)


class _AudioVAE:
    hop_length = 800

    def __init__(self, posterior: _Posterior) -> None:
        self.posterior = posterior
        self.config = SimpleNamespace(
            sampling_rate=32000,
            latents_mean=[0.0] * 32,
            latents_std=[1.0] * 32,
        )
        self.encoded_waveform: Optional[torch.Tensor] = None

    def encode(self, waveform: torch.Tensor) -> Any:
        self.encoded_waveform = waveform
        return SimpleNamespace(latent_dist=self.posterior)


class _TinyH3Transformer(torch.nn.Module):
    """Gradient-bearing stand-in that preserves H3's two-output contract."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.keyframe_noise_aug = 0.999

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states * self.weight, audio_hidden_states * self.weight


class _Adapter:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.video_posterior = _Posterior(torch.full((1, 24, 7, 2, 2), 2.0005))
        self.audio_posterior = _Posterior(torch.full((2, 32, 37), 3.0))
        self.vae = _VideoVAE(self.video_posterior)
        self.audio_vae = _AudioVAE(self.audio_posterior)
        self.transformer = _TinyH3Transformer()
        self.pipeline = SimpleNamespace(
            fps=24,
            pixel_mean=(0.485, 0.456, 0.406),
            pixel_std=(0.229, 0.224, 0.225),
            vae_spatial_compression_ratio=16,
            vae_frames_per_chunk=17,
            vae_latents_per_chunk=5,
            vae_latent_channels=24,
            patch_size=(1, 2, 2),
            audio_channels=2,
            audio_latent_channels=32,
            audio_sampling_rate=32000,
            min_duration=0.5,
            max_duration=15.0,
        )
        self.training_args = SimpleNamespace(
            height=32,
            width=32,
            num_frames=22,
            frame_rate=24.0,
        )

    def get_component(self, name: str) -> Any:
        return getattr(self, name)


def _base_adapter(
    *,
    latent_storage_dtype: str,
    adapter_type: type = MiniMaxH3T2VAAdapter,
) -> Any:
    components = _Adapter()
    components.transformer_ref = components.transformer
    adapter = object.__new__(adapter_type)
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.training_args = SimpleNamespace(
        height=32,
        width=32,
        num_frames=22,
        frame_rate=24.0,
        latent_storage_dtype=latent_storage_dtype,
    )
    adapter.pipeline = components.pipeline
    adapter.component_runtime = SimpleNamespace(
        get_component=lambda name: components.get_component(name)
    )
    adapter.scheduler = MiniMaxH3SDEScheduler(
        shift=12.0,
        dynamics_type="Flow-SDE",
        sde_steps=[0, 1],
        num_sde_steps=2,
    )
    adapter.audio_scheduler = MiniMaxH3SDEScheduler(
        shift=3.0,
        dynamics_type="Flow-SDE",
        sde_steps=[0, 1],
        num_sde_steps=2,
    )
    adapter.scheduler.set_timesteps(2, device="cpu")
    adapter.audio_scheduler.set_timesteps(2, device="cpu")
    adapter._effective_pipeline_io_contract = adapter.pipeline_io_contract
    adapter._condition_state_preparer = adapter.build_condition_state_preparer()
    adapter._output_state_codec = adapter.build_output_state_codec()
    return adapter


def _condition(
    *,
    video_condition_rows: int = 0,
    audio_condition_rows: int = 0,
) -> dict[str, Any]:
    text_rows = 2
    audio_rows = 74 + audio_condition_rows
    video_rows = 7 + video_condition_rows
    sequence_length = text_rows + audio_rows + video_rows
    return {
        "height": [32],
        "width": [32],
        "num_frames": [22],
        "num_latent_frames": [7],
        "latent_height": [2],
        "latent_width": [2],
        "num_audio_latents": [37],
        "position_ids": torch.zeros(1, sequence_length, 3, dtype=torch.float64),
        "token_tags": torch.zeros(1, sequence_length, dtype=torch.long),
        "text_indices": torch.arange(text_rows).unsqueeze(0),
        "audio_indices": torch.arange(text_rows, text_rows + audio_rows).unsqueeze(0),
        "video_indices": torch.arange(text_rows + audio_rows, sequence_length).unsqueeze(0),
        "num_condition_video_rows": [video_condition_rows],
        "num_condition_audio_rows": [audio_condition_rows],
    }


def _media(*, audio_samples: int = 37 * 800) -> tuple[tuple[DecodedMedia, ...], ...]:
    return (
        (
            DecodedMedia(
                type="video",
                path="target.mp4",
                payload=np.zeros((22, 32, 32, 3), dtype=np.uint8),
                fps=24.0,
            ),
            DecodedMedia(
                type="audio",
                path="target.wav",
                payload=torch.linspace(-1.0, 1.0, audio_samples).unsqueeze(0),
                sample_rate=32000,
            ),
        ),
    )


def test_t2va_declares_exact_configured_single_sample_av_contract() -> None:
    contract = MiniMaxH3T2VAAdapter.pipeline_io_contract

    assert contract is not None
    assert contract.input_media.rules == ()
    assert contract.negative_prompt is NegativePromptPolicy.UNSUPPORTED
    assert contract.geometry_source is GeometrySource.CONFIGURED
    assert contract.batch_capability is BatchCapability.SINGLE_SAMPLE
    assert tuple(item.type for item in contract.output_media.items) == (
        MediaType.VIDEO,
        MediaType.AUDIO,
    )
    assert contract.output_media.items[0].fps is RateRequirement.REQUIRED
    assert contract.output_media.items[1].sample_rate is RateRequirement.REQUIRED
    MiniMaxH3T2VAAdapter.validate_offline_output_capability()


def test_fl2va_declares_ordered_first_last_input_and_complete_offline_codec() -> None:
    contract = MiniMaxH3FL2VAAdapter.pipeline_io_contract

    assert contract is not None
    assert contract.input_media.binding is InputMediaBinding.GROUPED_BY_TYPE
    assert contract.input_media.order is InputMediaOrder.WITHIN_TYPE
    assert len(contract.input_media.rules) == 1
    rule = contract.input_media.rules[0]
    assert rule.format.type is MediaType.IMAGE
    assert (rule.min_count, rule.max_count) == (1, 2)
    assert rule.slots == ("first_frame", "last_frame")
    assert rule.required_slots == ()
    MiniMaxH3FL2VAAdapter.validate_offline_output_capability()


def test_ref2va_declares_global_ordered_multimodal_input_and_complete_codec() -> None:
    contract = MiniMaxH3Ref2VAAdapter.pipeline_io_contract

    assert contract is not None
    assert contract.input_media.binding is InputMediaBinding.ORDERED_REFERENCES
    assert contract.input_media.order is InputMediaOrder.GLOBAL
    rules = contract.input_media.rules
    assert tuple(rule.format.type for rule in rules) == (
        MediaType.IMAGE,
        MediaType.VIDEO,
        MediaType.AUDIO,
    )
    assert tuple((rule.min_count, rule.max_count) for rule in rules) == (
        (0, 9),
        (0, 3),
        (0, 3),
    )
    assert rules[1].format.fps is RateRequirement.OPTIONAL
    assert rules[2].format.sample_rate is RateRequirement.OPTIONAL
    assert contract.input_media.min_total_count == 1
    assert contract.input_media.max_total_count == 12
    assert contract.input_media.required_any_types == (MediaType.IMAGE, MediaType.VIDEO)
    MiniMaxH3Ref2VAAdapter.validate_offline_output_capability()


def test_ref2va_contract_fails_before_decode_for_empty_audio_only_and_overall_limit() -> None:
    contract = MiniMaxH3Ref2VAAdapter.pipeline_io_contract

    def model_input(*media: MediaAsset) -> NormalizedModelInput:
        return NormalizedModelInput(
            prompt="describe",
            negative_prompt=None,
            media=media,
        )

    with pytest.raises(ValueError, match="at least 1 input media item"):
        validate_pipeline_model_input(model_input(), contract)
    with pytest.raises(ValueError, match="whose type is in.*image.*video"):
        validate_pipeline_model_input(
            model_input(MediaAsset(type="audio", path="voice.wav")),
            contract,
        )
    with pytest.raises(ValueError, match="at most 12 input media item"):
        validate_pipeline_model_input(
            model_input(
                *(
                    [MediaAsset(type="image", path=f"image-{index}.png") for index in range(9)]
                    + [MediaAsset(type="video", path=f"video-{index}.mp4") for index in range(3)]
                    + [MediaAsset(type="audio", path="voice.wav")]
                )
            ),
            contract,
        )


def test_codec_uses_deterministic_av_modes_without_condition_fp16_rounding() -> None:
    adapter = _Adapter()
    generator = torch.Generator().manual_seed(17)

    encoded = MiniMaxH3AVOutputCodec(adapter).encode_output_state(
        _media(audio_samples=100),
        _condition(),
        generator,
    )

    assert adapter.video_posterior.sample_calls == 0
    assert adapter.video_posterior.sample_generator is None
    assert adapter.video_posterior.mode_calls == 1
    assert adapter.audio_posterior.mode_calls == 1
    assert adapter.audio_posterior.sample_calls == 0
    assert adapter.vae.encoded_pixels is not None
    assert adapter.vae.encoded_pixels.shape == (1, 3, 22, 32, 32)
    assert adapter.audio_vae.encoded_waveform is not None
    assert adapter.audio_vae.encoded_waveform.shape == (2, 1, 37 * 800)
    assert encoded.clean_state.component_names == ("video", "audio")
    assert encoded.clean_state.components["video"].shape == (1, 7, 96)
    assert encoded.clean_state.components["audio"].shape == (1, 74, 32)
    mode_value = adapter.video_posterior.values.flatten()[0]
    condition_rounded_value = mode_value.to(torch.float16).to(torch.float32)
    assert mode_value != condition_rounded_value
    assert encoded.clean_state.components["video"].flatten()[0] == mode_value
    assert encoded.forward_context == {}
    assert encoded.decode_context["geometry"] == {
        "height": 32,
        "width": 32,
        "num_frames": 22,
        "num_latent_frames": 7,
        "latent_height": 2,
        "latent_width": 2,
        "num_audio_latents": 37,
    }
    signature = encoded.geometry_signatures[0]
    assert signature.media[0].fps == 24.0
    assert signature.media[1].samples == 37 * 800
    assert signature.media[1].sample_rate == 32000
    validate_h3_encoded_output_geometry(adapter, _media(), _condition(), encoded)


def test_geometry_hook_independently_enforces_pipeline_duration_bounds() -> None:
    adapter = _Adapter()
    adapter.pipeline.min_duration = 5.0

    with pytest.raises(ValueError, match="outside the pipeline contract"):
        resolve_h3_output_geometry(adapter, _condition())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("height", 64, "cached output canvas"),
        ("num_frames", 23, "cached output frame count"),
    ],
)
def test_geometry_hook_rejects_cache_from_another_training_geometry(
    field: str,
    value: int,
    message: str,
) -> None:
    adapter = _Adapter()
    setattr(adapter.training_args, field, value)

    with pytest.raises(ValueError, match=message):
        resolve_h3_output_geometry(adapter, _condition())


def test_geometry_hook_rejects_invalid_input_owned_flat_layout() -> None:
    adapter = _Adapter()
    condition = _condition()
    encoded = MiniMaxH3AVOutputCodec(adapter).encode_output_state(_media(), condition)

    condition["audio_indices"] = condition["audio_indices"][:, :-1]
    with pytest.raises(ValueError, match=r"audio layout expected 0 condition \+ 74 target rows"):
        validate_h3_encoded_output_geometry(adapter, _media(), condition, encoded)


@pytest.mark.parametrize(
    ("video_condition_rows", "audio_condition_rows"),
    [(1, 0), (3, 5)],
)
def test_codec_accepts_conditioned_layout_while_encoding_target_only_rows(
    video_condition_rows: int,
    audio_condition_rows: int,
) -> None:
    adapter = _Adapter()
    condition = _condition(
        video_condition_rows=video_condition_rows,
        audio_condition_rows=audio_condition_rows,
    )

    encoded = MiniMaxH3AVOutputCodec(adapter).encode_output_state(_media(), condition)

    assert encoded.clean_state.components["video"].shape == (1, 7, 96)
    assert encoded.clean_state.components["audio"].shape == (1, 74, 32)
    validate_h3_encoded_output_geometry(adapter, _media(), condition, encoded)


def test_base_lifecycle_casts_clean_state_without_output_context_dtype_drift() -> None:
    adapter = _base_adapter(latent_storage_dtype="fp16")

    encoded = adapter.encode_output_state(_media(), _condition())

    assert encoded.clean_state.components["video"].dtype is torch.float16
    assert encoded.clean_state.components["audio"].dtype is torch.float16
    assert encoded.forward_context == {}


def test_offline_flow_objective_sums_modality_means_without_changing_joint_reducer() -> None:
    adapter = object.__new__(MiniMaxH3T2VAAdapter)
    values = {
        "video": torch.full((2, 7, 96), 2.0),
        "audio": torch.full((2, 74, 32), 5.0),
    }

    offline = adapter.reduce_flow_matching_objective_values(values)
    joint = adapter.reduce_latent_values(values)

    assert torch.equal(offline, torch.full((2,), 7.0))
    assert not torch.equal(joint, offline)


@pytest.mark.parametrize(
    "adapter_type",
    [MiniMaxH3T2VAAdapter, MiniMaxH3FL2VAAdapter, MiniMaxH3Ref2VAAdapter],
)
def test_adapter_decode_routes_both_components_with_cached_geometry(
    monkeypatch: pytest.MonkeyPatch,
    adapter_type: type,
) -> None:
    encoded = MiniMaxH3AVOutputCodec(_Adapter()).encode_output_state(_media(), _condition())
    adapter = object.__new__(adapter_type)
    observed: dict[str, Any] = {}

    def decode(self: Any, state: LatentState, **kwargs: Any) -> str:
        observed["state"] = state
        observed.update(kwargs)
        return "decoded-av"

    monkeypatch.setattr(adapter_type, "decode_latents", decode)

    assert adapter.decode_output_state(encoded, output_type="np") == "decoded-av"
    assert observed["state"] is encoded.clean_state
    assert observed["geometry"] == encoded.decode_context["geometry"]
    assert observed["output_type"] == "np"


def test_flat_offline_condition_binds_layout_and_state_native_empty_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = object.__new__(MiniMaxH3T2VAAdapter)
    state = LatentState(
        {
            "video": torch.ones(1, 7, 96, dtype=torch.float16),
            "audio": torch.ones(1, 74, 32, dtype=torch.float16),
        }
    )
    times = ComponentTimes(
        timestep={"video": torch.tensor([500.0]), "audio": torch.tensor([500.0])},
        next_timestep={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
    )
    observed: dict[str, Any] = {}

    def record_forward(self: Any, **kwargs: Any) -> str:
        observed.update(kwargs)
        return "bound"

    monkeypatch.setattr(MiniMaxH3T2VAAdapter, "forward", record_forward)
    result = adapter._forward_state(
        batch=SimpleNamespace(),
        state=state,
        times=times,
        next_state=None,
        compute_log_prob=False,
        return_fields=("velocity",),
        noise_level=0.0,
        forward_kwargs={**_condition(), "prompt_embeds": torch.zeros(1, 2, 4)},
    )

    assert result == "bound"
    assert torch.equal(observed["layout"]["video_indices"], _condition()["video_indices"][0])
    for component in ("video", "audio"):
        prefix = observed["condition_prefixes"][component]
        clean = state.components[component]
        assert prefix.shape == (1, 0, clean.shape[-1])
        assert prefix.dtype is clean.dtype
        assert prefix.device == clean.device


def test_h3_sft_path_runs_codec_cast_noising_and_velocity_only_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _base_adapter(latent_storage_dtype="fp32")
    condition = {**_condition(), "prompt_embeds": torch.zeros(1, 2, 4)}
    encoded = adapter.encode_output_state(_media(), condition)
    times, noised = build_noised_output_state(
        adapter,
        encoded.clean_state,
        torch.tensor([500.0]),
        batch=condition,
        generator=torch.Generator().manual_seed(31),
    )

    def unexpected_scheduler_step(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("offline velocity-only forward must not step schedulers")

    monkeypatch.setattr(adapter.scheduler, "step", unexpected_scheduler_step)
    monkeypatch.setattr(adapter.audio_scheduler, "step", unexpected_scheduler_step)

    output = adapter._forward_state(
        batch=SimpleNamespace(),
        state=noised.state,
        times=times,
        next_state=None,
        compute_log_prob=False,
        return_fields=("velocity",),
        noise_level=0.0,
        forward_kwargs=condition,
    )
    loss = flow_matching_per_sample_loss(adapter, output.velocity, noised).mean()
    loss.backward()

    transformer = adapter.get_component("transformer")
    assert output.velocity.component_names == ("video", "audio")
    assert output.velocity.components["video"].shape == (1, 7, 96)
    assert output.velocity.components["audio"].shape == (1, 74, 32)
    assert transformer.weight.grad is not None


def test_h3_offline_dpo_arms_reuse_structured_noise_through_real_forward() -> None:
    adapter = _base_adapter(latent_storage_dtype="fp32")
    condition = {**_condition(), "prompt_embeds": torch.zeros(1, 2, 4)}
    chosen = adapter.encode_output_state(_media(), condition)
    rejected = adapter.encode_output_state(_media(), condition)
    validate_preference_output_states(chosen, rejected)

    primary_timesteps = torch.tensor([650.0])
    chosen_times, chosen_noised = build_noised_output_state(
        adapter,
        chosen.clean_state,
        primary_timesteps,
        batch=condition,
        generator=torch.Generator().manual_seed(41),
    )
    rejected_times, rejected_noised = build_noised_output_state(
        adapter,
        rejected.clean_state,
        primary_timesteps,
        batch=condition,
        noise=chosen_noised.noise,
    )
    validate_preference_component_times(chosen_times, rejected_times)

    outputs = [
        adapter._forward_state(
            batch=SimpleNamespace(),
            state=noised.state,
            times=times,
            next_state=None,
            compute_log_prob=False,
            return_fields=("velocity",),
            noise_level=0.0,
            forward_kwargs=condition,
        )
        for times, noised in (
            (chosen_times, chosen_noised),
            (rejected_times, rejected_noised),
        )
    ]

    assert rejected_noised.noise is chosen_noised.noise
    for component in ("video", "audio"):
        assert torch.equal(
            rejected_noised.noise.components[component],
            chosen_noised.noise.components[component],
        )
        assert outputs[0].velocity.components[component].shape == (
            chosen_noised.state.components[component].shape
        )
        assert outputs[1].velocity.components[component].shape == (
            rejected_noised.state.components[component].shape
        )


@pytest.mark.parametrize(
    ("adapter_type", "audio_condition_rows"),
    [(MiniMaxH3FL2VAAdapter, 0), (MiniMaxH3Ref2VAAdapter, 2)],
)
def test_conditioned_h3_prepares_one_prefix_for_codec_and_velocity_forward(
    monkeypatch: pytest.MonkeyPatch,
    adapter_type: type,
    audio_condition_rows: int,
) -> None:
    adapter = _base_adapter(latent_storage_dtype="fp32", adapter_type=adapter_type)
    condition = {
        **_condition(
            video_condition_rows=1,
            audio_condition_rows=audio_condition_rows,
        ),
        "prompt_embeds": torch.zeros(1, 2, 4),
        "condition_latents": [[torch.ones(1, 24, 1, 2, 2)]],
        "audio_condition_latents": [
            [torch.ones(audio_condition_rows, 32)] if audio_condition_rows else []
        ],
    }
    prefixes = {
        "video": torch.full((1, 1, 96), 3.0),
        "audio": torch.full((1, audio_condition_rows, 32), 4.0),
    }
    calls = 0

    def prepare(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        return prefixes

    monkeypatch.setattr(
        "flow_factory.models.minimax_h3._condition.prepare_h3_condition_prefixes",
        prepare,
    )

    prepared = adapter.prepare_condition_state(condition)
    chosen = adapter.encode_output_state(_media(), prepared)
    rejected = adapter.encode_output_state(_media(), prepared)
    model_batch = dict(prepared.model_forward_condition())
    times, noised = build_noised_output_state(
        adapter,
        chosen.clean_state,
        torch.tensor([500.0]),
        batch=model_batch,
        generator=torch.Generator().manual_seed(31),
    )
    output = adapter._forward_state(
        batch=SimpleNamespace(),
        state=noised.state,
        times=times,
        next_state=None,
        compute_log_prob=False,
        return_fields=("velocity",),
        noise_level=0.0,
        forward_kwargs=model_batch,
    )

    assert calls == 1
    validate_preference_output_states(chosen, rejected)
    assert model_batch["condition_prefixes"] is prefixes
    assert chosen.clean_state.components["video"].shape == (1, 7, 96)
    assert output.velocity.components["video"].shape == (1, 7, 96)
    assert output.velocity.components["audio"].shape == (1, 74, 32)


def test_conditioned_offline_replay_still_requires_prefix_binder() -> None:
    adapter = object.__new__(MiniMaxH3FL2VAAdapter)
    state = LatentState(
        {
            "video": torch.ones(1, 7, 96),
            "audio": torch.ones(1, 74, 32),
        }
    )
    times = ComponentTimes(
        timestep={"video": torch.tensor([500.0]), "audio": torch.tensor([500.0])},
        next_timestep={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
    )

    with pytest.raises(ValueError, match="conditioned-prefix binder"):
        adapter._forward_state(
            batch=SimpleNamespace(),
            state=state,
            times=times,
            next_state=None,
            compute_log_prob=False,
            return_fields=("velocity",),
            noise_level=0.0,
            forward_kwargs={**_condition(), "prompt_embeds": torch.zeros(1, 2, 4)},
        )


def test_audio_alignment_trims_or_zero_pads_on_the_exact_latent_grid() -> None:
    short = prepare_h3_target_audio(
        torch.tensor([[1.0, 2.0, 3.0]]),
        source_sample_rate=32000,
        target_sample_rate=32000,
        target_samples=5,
        target_duration_seconds=5 / 32000,
    )
    assert short.shape == (2, 5)
    assert torch.equal(short[0], torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0]))
    assert torch.equal(short[0], short[1])

    long = prepare_h3_target_audio(
        torch.arange(14, dtype=torch.float32).reshape(2, 7),
        source_sample_rate=32000,
        target_sample_rate=32000,
        target_samples=5,
        target_duration_seconds=5 / 32000,
    )
    assert torch.equal(long, torch.tensor([[0, 1, 2, 3, 4], [7, 8, 9, 10, 11]]).float())


def test_audio_alignment_truncates_on_source_clock_before_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def convert(
        waveform: torch.Tensor,
        from_rate: int,
        to_rate: int,
        to_channels: int,
    ) -> torch.Tensor:
        observed["waveform"] = waveform.clone()
        observed["rates"] = (from_rate, to_rate, to_channels)
        return waveform.repeat_interleave(2, dim=-1).expand(2, -1).contiguous()

    monkeypatch.setattr("flow_factory.models.minimax_h3._output.convert_audio", convert)
    result = prepare_h3_target_audio(
        torch.arange(8, dtype=torch.float32).unsqueeze(0),
        source_sample_rate=10,
        target_sample_rate=20,
        target_samples=10,
        target_duration_seconds=0.5,
    )

    assert torch.equal(observed["waveform"], torch.arange(5, dtype=torch.float32).unsqueeze(0))
    assert observed["rates"] == (10, 20, 2)
    assert result.shape == (2, 10)


def test_target_video_fails_when_source_cannot_cover_configured_timeline() -> None:
    with pytest.raises(ValueError, match="too short"):
        prepare_h3_target_video(
            np.zeros((4, 8, 8, 3), dtype=np.uint8),
            source_fps=24.0,
            target_frames=5,
            target_fps=24.0,
            height=8,
            width=8,
        )


def test_target_video_matches_official_h3_fps_filter_for_30_to_24() -> None:
    frames = np.zeros((10, 1, 1, 3), dtype=np.uint8)
    frames[:, 0, 0, :] = np.arange(10, dtype=np.uint8)[:, None]

    pixels = prepare_h3_target_video(
        frames,
        source_fps=30.0,
        target_frames=8,
        target_fps=24.0,
        height=1,
        width=1,
    )

    values = pixels[0, 0, :, 0, 0].mul(255).round().to(torch.uint8)
    assert torch.equal(values, torch.tensor([0, 1, 3, 4, 5, 6, 8, 9], dtype=torch.uint8))
