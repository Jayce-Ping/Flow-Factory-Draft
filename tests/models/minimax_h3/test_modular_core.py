# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flow_factory.models.minimax_h3._common import MINIMAX_H3_COMPONENT_ORDER
from flow_factory.samples import ComponentTimes, LatentState
from flow_factory.scheduler import SDESchedulerOutput


class FakePipelineState:
    def __init__(self, values):
        self.values = values


class RecordingBlock:
    def __init__(self, name, output=None, error=None):
        self.name = name
        self.output = output or {}
        self.error = error

    def __call__(self, pipeline, state):
        pipeline.calls.append(self.name)
        if self.error is not None:
            raise self.error
        state.values.update(self.output)
        return pipeline, state


@dataclasses.dataclass(frozen=True)
class FakeSymbols:
    PipelineState: type = FakePipelineState


def test_block_executor_propagates_shared_state_and_selects_outputs(monkeypatch):
    from flow_factory.models.minimax_h3 import blocks

    monkeypatch.setattr(blocks, "require_minimax_h3_support", lambda: FakeSymbols())
    pipeline = SimpleNamespace(calls=[])
    first = RecordingBlock("first", {"tensor": torch.tensor([1.0])})
    second = RecordingBlock("second", {"result": "ok"})

    selected = blocks.run_h3_blocks(
        pipeline, [first, second], {"input": 3}, requested_outputs=("tensor", "result")
    )

    assert pipeline.calls == ["first", "second"]
    assert selected["tensor"] is first.output["tensor"]
    assert selected["result"] == "ok"


def test_block_executor_reports_missing_output_and_preserves_exception(monkeypatch):
    from flow_factory.models.minimax_h3 import blocks

    monkeypatch.setattr(blocks, "require_minimax_h3_support", lambda: FakeSymbols())
    with pytest.raises(ValueError, match="workflow='t2va'.*field='missing'"):
        blocks.run_h3_blocks(
            SimpleNamespace(calls=[]),
            [RecordingBlock("first")],
            {},
            requested_outputs=("missing",),
            workflow="t2va",
        )
    cause = RuntimeError("upstream")
    with pytest.raises(RuntimeError, match="upstream") as raised:
        blocks.run_h3_blocks(
            SimpleNamespace(calls=[]),
            [RecordingBlock("first", error=cause)],
            {},
            requested_outputs=(),
        )
    assert raised.value is cause


def test_row_timestep_oracle_and_transition_schedules():
    from flow_factory.models.minimax_h3.layout import build_h3_schedule_plan, build_row_timesteps
    from flow_factory.scheduler import MiniMaxH3SDEScheduler

    video_scheduler = MiniMaxH3SDEScheduler(shift=12.0, dynamics_type="ODE")
    audio_scheduler = MiniMaxH3SDEScheduler(shift=3.0, dynamics_type="ODE")
    layout = {
        "video_indices": torch.tensor([2, 5, 6]),
        "audio_indices": torch.tensor([3, 4]),
        "text_indices": torch.tensor([0, 1]),
        "num_condition_video_rows": 1,
        "num_condition_audio_rows": 1,
    }
    plan = build_h3_schedule_plan(video_scheduler, audio_scheduler, 2, layout, torch.device("cpu"))

    assert len(plan.row_timestep_plan) == 2
    assert len(plan.schedules["video"][1]) == 3
    assert plan.schedules["video"][1][-1].item() == 0
    unique, inverse = build_row_timesteps(layout, 0.2, 0.4, 0.999)
    oracle = torch.tensor([0.2, 0.2, 0.999, 1.0, 0.4, 0.2, 0.2])
    expected_unique, expected_inverse = torch.unique(oracle, sorted=True, return_inverse=True)
    torch.testing.assert_close(unique, expected_unique)
    torch.testing.assert_close(inverse, expected_inverse)


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.calls = []

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        attention_kwargs=None,
        token_tags=None,
        position_ids=None,
        video_indices=None,
        audio_indices=None,
        text_indices=None,
    ):
        self.calls.append(
            {
                "hidden_states": hidden_states,
                "audio_hidden_states": audio_hidden_states,
                "token_tags": token_tags,
            }
        )
        return hidden_states * self.weight, audio_hidden_states * self.weight


def _state(video_rows=2, audio_rows=3):
    return LatentState(
        {
            "video": torch.ones(1, video_rows, 96),
            "audio": torch.ones(1, audio_rows, 32),
        }
    )


def test_transformer_uses_supplied_component_keeps_grad_and_slices_prefixes():
    from flow_factory.models.minimax_h3.denoise import run_h3_joint_transformer

    transformer = FakeTransformer()
    prefixes = {
        "video": torch.full((1, 1, 96), 3.0),
        "audio": torch.full((1, 2, 32), 4.0),
    }
    layout = {
        "video_indices": torch.tensor([2, 5, 6]),
        "audio_indices": torch.tensor([3, 4, 7, 8, 9]),
        "text_indices": torch.tensor([0, 1]),
        "num_condition_video_rows": 1,
        "num_condition_audio_rows": 2,
        "token_tags": torch.arange(10),
        "position_ids": torch.zeros(10, 3),
    }
    times = ComponentTimes(
        timestep={"video": torch.tensor([800.0]), "audio": torch.tensor([600.0])},
        next_timestep={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
        sigma={"video": torch.tensor([0.8]), "audio": torch.tensor([0.6])},
        next_sigma={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
    )
    velocity = run_h3_joint_transformer(
        transformer,
        _state(),
        prefixes,
        torch.ones(1, 2, 8),
        times,
        layout,
    )

    assert len(transformer.calls) == 1
    assert transformer.calls[0]["hidden_states"].shape == (1, 3, 96)
    assert velocity.components["video"].shape == (1, 2, 96)
    assert velocity.components["audio"].shape == (1, 3, 32)
    velocity.components["video"].sum().backward()
    assert transformer.weight.grad is not None


def test_transformer_rejects_batch_greater_than_one():
    from flow_factory.models.minimax_h3.denoise import run_h3_joint_transformer

    bad = LatentState(
        {
            "video": torch.ones(2, 2, 96),
            "audio": torch.ones(2, 3, 32),
        }
    )
    with pytest.raises(ValueError, match="workflow='t2va'.*B=1.*received B=2"):
        run_h3_joint_transformer(
            FakeTransformer(),
            bad,
            {"video": torch.empty(2, 0, 96), "audio": torch.empty(2, 0, 32)},
            torch.ones(2, 1, 8),
            None,
            {},
            workflow="t2va",
        )


class FakeScheduler:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def step(self, velocity, timestep, state, **kwargs):
        self.calls.append(
            (self.name, kwargs["sigma"], kwargs["sigma_next"], kwargs["next_latents"])
        )
        return SDESchedulerOutput(
            next_latents=state - velocity,
            next_latents_mean=state - velocity,
            std_dev_t=torch.ones(1),
            dt=torch.ones(1),
            log_prob=torch.ones(1),
            velocity=velocity,
        )


def test_component_step_is_video_then_audio_and_target_only():
    from flow_factory.models.minimax_h3.denoise import step_h3_components

    calls = []
    times = ComponentTimes(
        timestep={"video": torch.tensor([800.0]), "audio": torch.tensor([600.0])},
        next_timestep={"video": torch.tensor([400.0]), "audio": torch.tensor([300.0])},
        sigma={"video": torch.tensor([0.8]), "audio": torch.tensor([0.6])},
        next_sigma={"video": torch.tensor([0.4]), "audio": torch.tensor([0.3])},
    )
    output = step_h3_components(
        _state(),
        _state(),
        times,
        FakeScheduler("video", calls),
        FakeScheduler("audio", calls),
        next_state=_state(),
        compute_log_prob=True,
    )
    assert [call[0] for call in calls] == list(MINIMAX_H3_COMPONENT_ORDER)
    assert output.next_state.components["video"].shape == (1, 2, 96)
    assert output.log_prob.shape == (1,)


def test_dependency_probe_error_names_pin(monkeypatch):
    from flow_factory.models.minimax_h3 import dependency

    monkeypatch.setattr(dependency, "_SYMBOLS", None)
    monkeypatch.setattr(dependency, "_IMPORT_ERROR", ImportError("missing"))
    with pytest.raises(ImportError, match="f53d552036a0d1bd5570782a39cd40cfabf112bc"):
        dependency.require_minimax_h3_support()


def test_pyproject_pins_exact_h3_diffusers_revision():
    text = Path("pyproject.toml").read_text()
    requirement = (
        "diffusers @ git+https://github.com/huggingface/diffusers.git@"
        "f53d552036a0d1bd5570782a39cd40cfabf112bc"
    )
    assert text.count(requirement) == 1
    assert '"diffusers>=0.37.0"' not in text
