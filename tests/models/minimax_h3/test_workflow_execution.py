# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch

from flow_factory.models.minimax_h3.adapters import (
    MiniMaxH3FL2VAAdapter,
    MiniMaxH3Ref2VAAdapter,
    MiniMaxH3T2VAAdapter,
)
from flow_factory.samples import (
    ComponentTimes,
    LatentState,
    MiniMaxH3T2VASample,
    StackedSampleBatch,
    StructuredTrajectory,
)
from flow_factory.scheduler import MiniMaxH3SDEScheduler


def _adapter(adapter_class: type, transformer: Any = None) -> Any:
    adapter = object.__new__(adapter_class)
    adapter.pipeline = SimpleNamespace()
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.scheduler = SimpleNamespace(shift=12.0)
    adapter.audio_scheduler = SimpleNamespace(shift=3.0)
    adapter.training_args = SimpleNamespace(latent_storage_dtype=None)
    adapter.get_component = lambda name: transformer or SimpleNamespace()
    adapter.on_load_components = lambda components, device=None: None
    return adapter


@pytest.mark.parametrize(
    ("adapter_class", "kwargs", "expected"),
    [
        (
            MiniMaxH3T2VAAdapter,
            {},
            {"prompt": "describe", "height": 64, "width": 96, "num_frames": 5},
        ),
        (
            MiniMaxH3FL2VAAdapter,
            {"images": [["first", "last"]]},
            {
                "prompt": "describe",
                "image": "first",
                "last_image": "last",
                "height": 64,
                "width": 96,
                "num_frames": 5,
            },
        ),
    ],
)
def test_preprocess_uses_exact_workflow_inputs_and_b1(
    monkeypatch, adapter_class: type, kwargs: Dict[str, Any], expected: Dict[str, Any]
) -> None:
    calls: List[Any] = []
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.encode_h3_workflow_inputs",
        lambda pipeline, values, workflow: calls.append((workflow, values))
        or {"prompt_embeds": torch.zeros(1, 2, 4)},
        raising=False,
    )
    adapter = _adapter(adapter_class)

    result = adapter.preprocess_func(
        prompt=["describe"], height=64, width=96, num_frames=5, **kwargs
    )

    assert calls == [(adapter.workflow, expected)]
    assert result["prompt_embeds"].shape[0] == 1
    with pytest.raises(ValueError, match=r"workflow=.*B=1.*received"):
        adapter.preprocess_func(prompt=["one", "two"], height=64, width=96, num_frames=5)


def test_preprocess_adds_outer_batch_to_arrow_cache_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.encode_h3_workflow_inputs",
        lambda *args, **kwargs: {
            "prompt_embeds": torch.zeros(1, 2, 4),
            "text_token_tags": torch.tensor([1, 1]),
            "height": 64,
            "keyframe_anchors": (),
        },
    )
    adapter = _adapter(MiniMaxH3T2VAAdapter)

    result = adapter.preprocess_func(
        prompt=["describe"],
        height=64,
        width=96,
        num_frames=124,
    )

    assert result["prompt_embeds"].shape == (1, 2, 4)
    assert len(result["text_token_tags"]) == 1
    torch.testing.assert_close(result["text_token_tags"][0], torch.tensor([1, 1]))
    assert result["height"] == [64]
    assert result["keyframe_anchors"] == [[]]


def test_ref_preprocess_builds_ordered_pinned_objects_without_returning_them(monkeypatch) -> None:
    constructed: List[Any] = []

    def reference_type(kind: str):
        return lambda **kwargs: constructed.append((kind, kwargs)) or SimpleNamespace(kind=kind)

    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.require_minimax_h3_support",
        lambda: SimpleNamespace(
            ImageReference=reference_type("image"),
            VideoReference=reference_type("video"),
            AudioReference=reference_type("audio"),
        ),
    )
    encoded_inputs: Dict[str, Any] = {}
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.encode_h3_workflow_inputs",
        lambda pipeline, values, workflow: encoded_inputs.update(values)
        or {"prompt_embeds": torch.zeros(1, 2, 4)},
        raising=False,
    )
    adapter = _adapter(MiniMaxH3Ref2VAAdapter)
    references = [
        {"kind": "image", "path": "i.png", "media": "image"},
        {
            "kind": "video",
            "path": "v.mp4",
            "frames": "frames",
            "fps": 24.0,
            "audio": torch.zeros(2, 8),
            "sample_rate": 32000,
        },
        {"kind": "audio", "path": "a.wav", "media": torch.ones(1, 8), "sample_rate": 16000},
    ]

    result = adapter.preprocess_func(
        prompt=["describe"],
        references=[references],
        reference_manifest=["manifest"],
        height=64,
        width=96,
        num_frames=5,
    )

    assert [kind for kind, _ in constructed] == ["image", "video", "audio"]
    assert constructed[1][1]["frames"] == "frames"
    assert "video" not in constructed[1][1]
    assert [ref.kind for ref in encoded_inputs["references"]] == ["image", "video", "audio"]
    assert result["reference_manifest"] == ["manifest"]
    assert "references" not in result
    assert all(not isinstance(value, SimpleNamespace) for value in result.values())


def _state(value: float = 0.0) -> LatentState:
    return LatentState(
        {
            "video": torch.full((1, 2, 96), value, dtype=torch.float32),
            "audio": torch.full((1, 3, 32), value, dtype=torch.float32),
        }
    )


def _times(value: float = 1.0, next_value: float = 0.0) -> ComponentTimes:
    return ComponentTimes(
        timestep={name: torch.tensor([value * 1000]) for name in ("video", "audio")},
        next_timestep={name: torch.tensor([next_value * 1000]) for name in ("video", "audio")},
        sigma={name: torch.tensor([value]) for name in ("video", "audio")},
        next_sigma={name: torch.tensor([next_value]) for name in ("video", "audio")},
    )


@pytest.mark.parametrize(
    "adapter_class",
    [MiniMaxH3T2VAAdapter, MiniMaxH3FL2VAAdapter, MiniMaxH3Ref2VAAdapter],
)
def test_training_times_map_primary_video_coordinate_to_audio_shift(
    adapter_class: type,
) -> None:
    adapter = _adapter(adapter_class)

    times = adapter.build_training_component_times(torch.tensor([500.0]))

    assert tuple(times.timestep) == ("video", "audio")
    assert times.timestep["video"].item() == pytest.approx(500.0)
    assert times.timestep["audio"].item() != pytest.approx(500.0)
    assert times.next_timestep["video"].item() == 0
    assert times.next_timestep["audio"].item() == 0


@pytest.mark.parametrize("primary_dtype", [torch.float32, torch.float64])
def test_training_times_reuse_exact_independent_scheduler_grid_coordinates(
    primary_dtype: torch.dtype,
) -> None:
    adapter = _adapter(MiniMaxH3T2VAAdapter)
    adapter.scheduler = MiniMaxH3SDEScheduler(shift=12.0)
    adapter.audio_scheduler = MiniMaxH3SDEScheduler(shift=3.0)
    adapter.scheduler.set_timesteps(4)
    adapter.audio_scheduler.set_timesteps(4)
    primary_timesteps = (adapter.scheduler.sigmas * 1000).to(primary_dtype)

    times = adapter.build_training_component_times(primary_timesteps)

    for component, scheduler in (
        ("video", adapter.scheduler),
        ("audio", adapter.audio_scheduler),
    ):
        assert torch.equal(times.sigma[component], scheduler.sigmas.to(primary_dtype))
        assert torch.equal(
            times.timestep[component],
            (scheduler.sigmas * 1000).to(primary_dtype),
        )


def test_training_times_do_not_snap_nearby_float64_interior_coordinates() -> None:
    adapter = _adapter(MiniMaxH3T2VAAdapter)
    adapter.scheduler = MiniMaxH3SDEScheduler(shift=12.0)
    adapter.audio_scheduler = MiniMaxH3SDEScheduler(shift=3.0)
    adapter.scheduler.set_timesteps(4)
    adapter.audio_scheduler.set_timesteps(4)
    grid_timestep = (adapter.scheduler.sigmas * 1000)[1:2].to(torch.float64)
    primary_timesteps = grid_timestep + 1e-6
    assert torch.equal(primary_timesteps.to(torch.float32), grid_timestep.to(torch.float32))

    times = adapter.build_training_component_times(primary_timesteps)

    assert not torch.equal(
        times.sigma["video"],
        adapter.scheduler.sigmas[1:2].to(torch.float64),
    )
    assert not torch.equal(
        times.timestep["audio"],
        (adapter.audio_scheduler.sigmas * 1000)[1:2].to(torch.float64),
    )


def test_inference_collects_structured_target_only_trajectory(monkeypatch) -> None:
    calls: List[Any] = []
    prepared_values: List[Dict[str, Any]] = []
    prefixes = {
        "video": torch.ones(1, 1, 96),
        "audio": torch.ones(1, 1, 32),
    }
    schedules = {
        "video": (torch.tensor([1000.0, 500.0, 0.0]), torch.tensor([1.0, 0.5, 0.0])),
        "audio": (torch.tensor([1000.0, 300.0, 0.0]), torch.tensor([1.0, 0.3, 0.0])),
    }

    def prepare(pipeline, values, **kwargs):
        del pipeline, kwargs
        prepared_values.append(values)
        return _state(), prefixes

    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.prepare_h3_rollout_state",
        prepare,
        raising=False,
    )
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.build_h3_schedule_plan",
        lambda *args, **kwargs: SimpleNamespace(schedules=schedules),
        raising=False,
    )

    def forward(*args, **kwargs):
        calls.append((args[1], kwargs))
        value = float(len(calls))
        return SimpleNamespace(
            next_state=_state(value),
            next_state_mean=_state(value + 20),
            log_prob=torch.tensor([value]),
            component_log_probs={
                "video": torch.tensor([value]),
                "audio": torch.tensor([value]),
            },
            velocity=_state(value + 10),
        )

    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.forward_h3_state", forward, raising=False
    )
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.decode_h3_targets",
        lambda *args, **kwargs: (torch.zeros(1, 2, 3, 4, 4), torch.zeros(1, 2, 16), 32000),
        raising=False,
    )
    adapter = _adapter(MiniMaxH3T2VAAdapter, transformer=torch.nn.Linear(1, 1))
    adapter_forward = adapter.forward
    adapter_forward_calls: List[Dict[str, Any]] = []
    adapter_decode = adapter.decode_latents
    adapter_decode_calls: List[Dict[str, Any]] = []
    adapter.forward = lambda **kwargs: adapter_forward_calls.append(kwargs) or adapter_forward(
        **kwargs
    )
    adapter.decode_latents = lambda latents, **kwargs: adapter_decode_calls.append(
        {"latents": latents, **kwargs}
    ) or adapter_decode(latents, **kwargs)

    samples = adapter.inference(
        prompt=["describe"],
        prompt_embeds=torch.zeros(1, 2, 4),
        layout={
            "video_indices": torch.arange(3),
            "audio_indices": torch.arange(3, 7),
            "text_indices": torch.arange(7, 9),
            "num_condition_video_rows": 1,
            "num_condition_audio_rows": 1,
        },
        geometry={
            "num_latent_frames": 1,
            "latent_height": 2,
            "latent_width": 2,
            "num_audio_latents": 3,
        },
        num_inference_steps=2,
        trajectory_indices=[0, 2],
        compute_log_prob=True,
        extra_call_back_kwargs=["velocity", "next_latents", "next_latents_mean"],
    )

    assert len(calls) == 2
    assert prepared_values[0]["num_condition_video_rows"] == 1
    assert prepared_values[0]["num_condition_audio_rows"] == 1
    assert prepared_values[0]["num_latent_frames"] == 1
    assert len(adapter_forward_calls) == 2
    assert len(adapter_decode_calls) == 1
    assert adapter_decode_calls[0]["geometry"]["num_latent_frames"] == 1
    sample = samples[0]
    assert isinstance(sample.trajectory, StructuredTrajectory)
    assert sample.trajectory.components["video"].state_index_map.tolist() == [0, -1, 1]
    assert sample.trajectory.log_prob_index_map.tolist() == [0, -1]
    assert sample.timesteps is sample.all_latents is sample.latent_index_map is None
    assert sample.log_probs is sample.log_prob_index_map is None
    assert sample.audio.shape == (2, 16)
    assert sample.audio_sample_rate == 32000
    assert prefixes["video"].shape[1] == 1
    assert sample.trajectory.components["video"].states.shape[-1] == 96
    assert sample.trajectory.components["audio"].states.shape[-1] == 32
    assert sample.trajectory.callbacks["velocity"]["video"].index_map.tolist() == [0, -1]
    assert sample.trajectory.callbacks["velocity"]["audio"].values.shape[-1] == 32
    assert torch.equal(
        sample.trajectory.callbacks["next_latents"]["video"].values[0],
        _state(1).components["video"][0],
    )
    assert torch.equal(
        sample.trajectory.callbacks["next_latents_mean"]["audio"].values[0],
        _state(21).components["audio"][0],
    )

    final_only = adapter.inference(
        prompt=["describe"],
        prompt_embeds=torch.zeros(1, 2, 4),
        layout={
            "video_indices": torch.arange(3),
            "audio_indices": torch.arange(3, 7),
            "text_indices": torch.arange(7, 9),
            "num_condition_video_rows": 1,
            "num_condition_audio_rows": 1,
        },
        geometry={
            "num_latent_frames": 1,
            "latent_height": 2,
            "latent_width": 2,
            "num_audio_latents": 3,
        },
        num_inference_steps=2,
        trajectory_indices=[2],
        compute_log_prob=True,
    )[0]
    assert final_only.trajectory is not None
    assert final_only.trajectory.log_probs is None
    assert final_only.trajectory.log_prob_index_map is None


def test_forward_state_uses_prepared_component_and_forward_parity(monkeypatch) -> None:
    weight = torch.nn.Parameter(torch.tensor(2.0))

    class Transformer:
        pass

    prepared = Transformer()
    prepared.weight = weight
    calls: List[Any] = []

    def forward(transformer, state, *args, **kwargs):
        calls.append((transformer, kwargs))
        scaled = LatentState(
            {name: values * transformer.weight for name, values in state.components.items()}
        )
        return SimpleNamespace(
            next_state=scaled,
            log_prob=torch.ones(1),
            component_log_probs={"video": torch.ones(1), "audio": torch.ones(1)},
            velocity=scaled,
        )

    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.forward_h3_state", forward, raising=False
    )
    adapter = _adapter(MiniMaxH3T2VAAdapter, transformer=prepared)
    batch = StackedSampleBatch(
        [
            MiniMaxH3T2VASample(
                prompt="describe",
                prompt_embeds=torch.zeros(2, 4),
                extra_kwargs={
                    "condition_prefixes": {
                        "video": torch.zeros(1, 96),
                        "audio": torch.zeros(1, 32),
                    },
                    "layout": {},
                    "advantage": torch.tensor(9),
                },
            )
        ]
    )
    state = _state(1)
    times = _times()

    bridged = adapter.forward_state(
        batch=batch,
        state=state,
        times=times,
        compute_log_prob=True,
        trainer_metadata="drop",
    )
    direct = adapter.forward(
        state=state,
        times=times,
        compute_log_prob=True,
        prompt_embeds=batch["prompt_embeds"],
        condition_prefixes=batch["condition_prefixes"],
        layout=batch["layout"],
    )
    bridged.next_state.components["video"].sum().backward()

    assert torch.equal(
        bridged.next_state.components["video"], direct.next_state.components["video"]
    )
    assert calls[0][0] is prepared
    assert weight.grad is not None
    assert "advantage" not in calls[0][1]
    assert "trainer_metadata" not in calls[0][1]


def test_forward_rejects_replay_batch_argument() -> None:
    adapter = _adapter(MiniMaxH3T2VAAdapter)

    with pytest.raises(
        ValueError,
        match=r"workflow='t2va' forward received unsupported arguments=\('batch',\)",
    ):
        adapter.forward(batch=SimpleNamespace(), state=_state(1), times=_times())
