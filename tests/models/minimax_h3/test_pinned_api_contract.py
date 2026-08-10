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
import json
from importlib import metadata

import pytest
import torch

from flow_factory.models.minimax_h3 import dependency
from flow_factory.models.runtime import ModularPipelineRuntime


def _read_diffusers_direct_url() -> tuple[str, dict]:
    distribution = metadata.distribution("diffusers")
    version = distribution.version
    direct_url_text = distribution.read_text("direct_url.json")
    assert direct_url_text is not None, (
        "installed diffusers must expose direct_url.json for exact revision verification; "
        f"version={version!r}, direct_url.json={direct_url_text!r}"
    )
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as error:
        raise AssertionError(
            "installed diffusers direct_url.json must contain valid JSON; "
            f"version={version!r}, direct_url.json={direct_url_text!r}"
        ) from error
    assert isinstance(direct_url, dict), (
        "installed diffusers direct_url.json must contain an object; "
        f"version={version!r}, direct_url.json={direct_url!r}"
    )
    return version, direct_url


@pytest.fixture(scope="module")
def pinned_symbols():
    if dependency._SYMBOLS is None:
        pytest.skip(
            "installed diffusers does not expose MiniMax H3 modular symbols; "
            "the dependency failure contract is covered separately"
        )
    return dependency.require_minimax_h3_support()


def test_installed_diffusers_revision_matches_required_commit(pinned_symbols) -> None:
    version, direct_url = _read_diffusers_direct_url()
    vcs_info = direct_url.get("vcs_info")
    diagnostic = f"version={version!r}, direct_url.json={direct_url!r}"

    assert isinstance(vcs_info, dict), diagnostic
    assert vcs_info.get("vcs") == "git", diagnostic
    assert vcs_info.get("requested_revision") == dependency.MINIMAX_H3_DIFFUSERS_COMMIT, diagnostic
    assert vcs_info.get("commit_id") == dependency.MINIMAX_H3_DIFFUSERS_COMMIT, diagnostic


def test_real_pinned_symbols_preserve_workflows_and_callable_surfaces(pinned_symbols) -> None:
    state_values = {"probe": object()}
    state = pinned_symbols.PipelineState(values=state_values)
    assert state.values == state_values
    assert pinned_symbols.MiniMaxH3Blocks._workflow_map == {
        "t2va": {"prompt": True},
        "fl2va": (
            {"prompt": True, "image": True},
            {"prompt": True, "last_image": True},
        ),
        "ref2va": {"prompt": True, "references": True},
    }

    block_names = dependency._BLOCK_FIELDS
    for block_name in block_names:
        block = getattr(pinned_symbols, block_name)()
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

    row_timesteps = pinned_symbols.SetTimestepsStep.build_row_timesteps(
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

    assert tuple(inspect.signature(pinned_symbols.ImageReference).parameters) == ("image",)
    assert tuple(inspect.signature(pinned_symbols.VideoReference).parameters) == (
        "frames",
        "fps",
        "audio",
        "sample_rate",
    )
    assert tuple(inspect.signature(pinned_symbols.AudioReference).parameters) == (
        "audio",
        "sample_rate",
    )


@pytest.mark.parametrize(
    ("workflow", "target_component", "absent_target"),
    [
        ("t2va", "transformer", "transformer_ref"),
        ("fl2va", "transformer", "transformer_ref"),
        ("ref2va", "transformer_ref", "transformer"),
    ],
)
def test_real_pinned_no_weight_workflows_expose_component_specs(
    pinned_symbols,
    workflow: str,
    target_component: str,
    absent_target: str,
) -> None:
    pipeline_class = pinned_symbols.MiniMaxH3ModularPipeline
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
