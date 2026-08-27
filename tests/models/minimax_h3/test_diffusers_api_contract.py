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

import inspect
from importlib import metadata

import numpy as np
import pytest
import torch
from packaging.version import Version
from PIL import Image

from flow_factory.models.minimax_h3 import dependency
from flow_factory.models.runtime import ModularPipelineRuntime

EXECUTED_NO_WEIGHT_BLOCKS = frozenset(
    {
        "ResizeStep",
        "RefSetupStep",
        "NoKeyframeAnchorsStep",
        "PrepareLayoutStep",
        "RefPrepareLayoutStep",
        "PrepareConditionLatentsStep",
        "PrepareLatentsStep",
        "FL2VAPrepareLatentsStep",
        "Ref2VAPrepareLatentsStep",
        "SetTimestepsStep",
        "AfterDenoiseStep",
    }
)
SIGNATURE_ONLY_BLOCK_REASONS = {
    "TextEncoderStep": "requires checkpoint-backed text encoder, tokenizer, and processor",
    "FL2VATextEncoderStep": "requires checkpoint-backed text encoder, tokenizer, and processor",
    "Ref2VATextEncoderStep": "requires checkpoint-backed text encoder, tokenizer, and processor",
    "KeyframeEncoderStep": "requires the checkpoint-backed video VAE",
    "ReferenceEncoderStep": "requires checkpoint-backed video and audio VAEs",
    "VideoDecodeStep": "requires the checkpoint-backed video VAE",
    "AudioDecodeStep": "requires the checkpoint-backed audio VAE",
}


@pytest.fixture(scope="module")
def h3_symbols():
    if dependency._SYMBOLS is None:
        pytest.skip(
            "installed diffusers does not expose MiniMax H3 modular symbols; "
            "H3-specific execution tests do not run; the environment-independent "
            "pyproject minimum-version contract is covered separately"
        )
    return dependency.require_minimax_h3_support()


def test_h3_capable_environment_uses_supported_diffusers_release(
    h3_symbols,
) -> None:
    """Verify the installed release after it passes the H3 feature gate."""
    del h3_symbols
    installed = metadata.version("diffusers")
    required = dependency.MINIMAX_H3_DIFFUSERS_MIN_VERSION
    assert Version(installed) >= Version(required), (
        f"MiniMax H3 requires diffusers>={required}, received {installed}"
    )


def test_real_h3_symbols_preserve_workflows_and_callable_surfaces(h3_symbols) -> None:
    state_values = {"probe": object()}
    state = h3_symbols.PipelineState(values=state_values)
    assert state.values == state_values
    assert h3_symbols.MiniMaxH3Blocks._workflow_map == {
        "t2va": {"prompt": True},
        "fl2va": (
            {"prompt": True, "image": True},
            {"prompt": True, "last_image": True},
        ),
        "ref2va": {"prompt": True, "references": True},
    }

    block_names = dependency._BLOCK_FIELDS
    for block_name in block_names:
        block = getattr(h3_symbols, block_name)()
        signature = inspect.signature(block)
        parameters = tuple(signature.parameters.values())
        assert len(parameters) >= 2
        assert parameters[0].kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        assert parameters[1].kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )

    row_timesteps = h3_symbols.SetTimestepsStep.build_row_timesteps(
        torch.tensor([2]),
        torch.tensor([1]),
        0,
        0,
        1,
        0.2,
        0.4,
        0.999,
        1.0,
    )
    assert len(row_timesteps) == 2
    assert all(isinstance(value, torch.Tensor) for value in row_timesteps)

    assert tuple(inspect.signature(h3_symbols.ImageReference).parameters) == ("image",)
    assert tuple(inspect.signature(h3_symbols.VideoReference).parameters) == (
        "frames",
        "fps",
        "audio",
        "sample_rate",
    )
    assert tuple(inspect.signature(h3_symbols.AudioReference).parameters) == (
        "audio",
        "sample_rate",
    )


def test_real_h3_blocks_execute_no_weight_pipeline_state_transitions(h3_symbols) -> None:
    executed = set()

    t2va_pipeline = h3_symbols.MiniMaxH3ModularPipeline.from_config({"workflow": "t2va"})
    scheduler_class = t2va_pipeline.get_component_spec("scheduler").type_hint
    t2va_pipeline.register_components(
        scheduler=scheduler_class(shift=12.0),
        audio_scheduler=scheduler_class(shift=3.0),
    )
    assert t2va_pipeline.scheduler.config.shift == 12.0
    assert t2va_pipeline.audio_scheduler.config.shift == 3.0
    t2va_state = h3_symbols.PipelineState(
        values={
            "text_token_tags": torch.tensor([1, 1]),
            "height": 32,
            "width": 32,
            "num_frames": 124,
        }
    )
    returned_pipeline, t2va_state = h3_symbols.NoKeyframeAnchorsStep()(
        t2va_pipeline, t2va_state
    )
    assert returned_pipeline is t2va_pipeline
    assert t2va_state.values["keyframe_anchors"] == ()
    executed.add("NoKeyframeAnchorsStep")

    _, t2va_state = h3_symbols.PrepareLayoutStep()(t2va_pipeline, t2va_state)
    assert t2va_state.values["position_ids"].shape[1] == 3
    assert t2va_state.values["num_latent_frames"] == 37
    executed.add("PrepareLayoutStep")

    t2va_state.values.update(
        {
            "generator": torch.Generator().manual_seed(0),
            "latents": torch.zeros(1, 24, 37, 2, 2),
            "audio_latents": torch.zeros(2, 32, 207),
        }
    )
    _, t2va_state = h3_symbols.PrepareLatentsStep()(t2va_pipeline, t2va_state)
    assert t2va_state.values["latents"].shape == (37, 96)
    assert t2va_state.values["audio_latents"].shape == (414, 32)
    executed.add("PrepareLatentsStep")

    condition_state = h3_symbols.PipelineState(
        values={
            "generator": torch.Generator().manual_seed(0),
            "num_condition_video_rows": 1,
            "condition_latents": [torch.zeros(1, 24, 1, 2, 2)],
        }
    )
    _, condition_state = h3_symbols.PrepareConditionLatentsStep()(
        t2va_pipeline, condition_state
    )
    assert condition_state.values["condition_rows"].shape == (1, 96)
    executed.add("PrepareConditionLatentsStep")

    t2va_state.values["num_inference_steps"] = 3
    _, t2va_state = h3_symbols.SetTimestepsStep()(t2va_pipeline, t2va_state)
    assert len(t2va_state.values["timesteps"]) == 2
    assert len(t2va_state.values["audio_timesteps"]) == 2
    assert len(t2va_state.values["row_timestep_plan"]) == 2
    executed.add("SetTimestepsStep")

    _, t2va_state = h3_symbols.AfterDenoiseStep()(t2va_pipeline, t2va_state)
    assert t2va_state.values["latents"].shape == (1, 24, 37, 2, 2)
    assert t2va_state.values["audio_latents"].shape == (2, 32, 207)
    executed.add("AfterDenoiseStep")

    fl2va_pipeline = h3_symbols.MiniMaxH3ModularPipeline.from_config({"workflow": "fl2va"})
    fl2va_pipeline.load_components(names=["image_processor"])
    resize_state = h3_symbols.PipelineState(
        values={
            "image": Image.new("RGB", (16, 16)),
            "last_image": None,
            "height": 32,
            "width": 32,
        }
    )
    _, resize_state = h3_symbols.ResizeStep()(fl2va_pipeline, resize_state)
    assert resize_state.values["keyframe_anchors"] == ("first",)
    assert resize_state.values["keyframes"][0].size == (32, 32)
    executed.add("ResizeStep")

    fl2va_state = h3_symbols.PipelineState(
        values={
            "condition_rows": torch.ones(1, 96),
            "latents": torch.zeros(2, 96),
        }
    )
    _, fl2va_state = h3_symbols.FL2VAPrepareLatentsStep()(fl2va_pipeline, fl2va_state)
    assert fl2va_state.values["latents"].shape == (3, 96)
    assert torch.equal(fl2va_state.values["latents"][0], torch.ones(96))
    executed.add("FL2VAPrepareLatentsStep")

    ref2va_pipeline = h3_symbols.MiniMaxH3ModularPipeline.from_config({"workflow": "ref2va"})
    reference = h3_symbols.VideoReference(
        frames=np.zeros((1, 32, 32, 3), dtype=np.uint8),
        fps=24.0,
        audio=None,
        sample_rate=None,
    )
    setup_state = h3_symbols.PipelineState(
        values={
            "references": [reference],
            "height": 32,
            "width": 32,
            "num_frames": 124,
        }
    )
    _, setup_state = h3_symbols.RefSetupStep()(ref2va_pipeline, setup_state)
    normalized_references = setup_state.values["normalized_references"]
    assert len(normalized_references) == 1
    assert normalized_references[0].kind == "video"
    executed.add("RefSetupStep")

    ref_layout_state = h3_symbols.PipelineState(
        values={
            "text_token_tags": torch.tensor([1, 1]),
            "normalized_references": normalized_references,
            "condition_latents": [torch.zeros(1, 24, 1, 2, 2)],
            "audio_condition_latents": [],
            "height": 32,
            "width": 32,
            "num_frames": 124,
        }
    )
    _, ref_layout_state = h3_symbols.RefPrepareLayoutStep()(ref2va_pipeline, ref_layout_state)
    assert ref_layout_state.values["position_ids"].shape[1] == 3
    assert ref_layout_state.values["num_condition_video_rows"] == 1
    assert ref_layout_state.values["num_condition_audio_rows"] == 0
    executed.add("RefPrepareLayoutStep")

    ref_latents_state = h3_symbols.PipelineState(
        values={
            "condition_rows": torch.ones(1, 96),
            "latents": torch.zeros(2, 96),
            "audio_condition_latents": [torch.ones(1, 32)],
            "num_condition_audio_rows": 1,
            "audio_latents": torch.zeros(2, 32),
        }
    )
    _, ref_latents_state = h3_symbols.Ref2VAPrepareLatentsStep()(
        ref2va_pipeline, ref_latents_state
    )
    assert ref_latents_state.values["latents"].shape == (3, 96)
    assert ref_latents_state.values["audio_latents"].shape == (3, 32)
    executed.add("Ref2VAPrepareLatentsStep")

    assert executed == EXECUTED_NO_WEIGHT_BLOCKS
    assert EXECUTED_NO_WEIGHT_BLOCKS | set(SIGNATURE_ONLY_BLOCK_REASONS) == set(
        dependency._BLOCK_FIELDS
    )


@pytest.mark.parametrize(
    ("workflow", "target_component", "absent_target"),
    [
        ("t2va", "transformer", "transformer_ref"),
        ("fl2va", "transformer", "transformer_ref"),
        ("ref2va", "transformer_ref", "transformer"),
    ],
)
def test_real_h3_no_weight_workflows_expose_component_specs(
    h3_symbols,
    workflow: str,
    target_component: str,
    absent_target: str,
) -> None:
    pipeline_class = h3_symbols.MiniMaxH3ModularPipeline
    pipeline = pipeline_class.from_config({"workflow": workflow})

    assert isinstance(pipeline.pretrained_component_names, list)
    assert isinstance(pipeline.config_component_names, list)
    assert isinstance(pipeline.components, dict)
    assert callable(pipeline.get_component_spec)
    assert callable(pipeline.load_components)

    declared_names = set([*pipeline.pretrained_component_names, *pipeline.config_component_names])
    assert target_component in declared_names
    assert absent_target not in declared_names
    assert {
        "scheduler",
        "text_encoder",
        "tokenizer",
        "processor",
        "vae",
        "audio_vae",
    } <= declared_names

    component_spec = pipeline.get_component_spec(target_component)
    assert pipeline.get_component_spec(target_component) == component_spec
    assert pipeline.get_component_spec(target_component) is not component_spec

    runtime = ModularPipelineRuntime(pipeline)
    assert set(runtime.declared_component_names) == declared_names
