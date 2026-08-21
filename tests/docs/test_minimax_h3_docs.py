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

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_LINKS = (
    "examples/grpo/lora/minimax_h3_t2va/debug.yaml",
    "examples/grpo/lora/minimax_h3_fl2va/default.yaml",
    "examples/grpo/lora/minimax_h3_ref2va/default.yaml",
)


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_readme_documents_h3_links_pin_and_limits() -> None:
    text = _text("README.md")
    assert (
        "The configurations under `examples/` have been verified to yield measurable "
        "performance gains."
    ) not in text
    assert "Validation status varies by example" in text
    assert "hardware and reward-trend evidence" in text
    assert "MiniMax H3 T2VA has real-weight LoRA validation" in text
    assert "FL2VA and Ref2VA remain" in text

    for model_type, link in zip(
        ("minimax-h3-t2va", "minimax-h3-fl2va", "minimax-h3-ref2va"),
        EXAMPLE_LINKS,
    ):
        assert model_type in text
        assert link in text
        assert (ROOT / link).is_file()

    for required in (
        "4e0466f3e5260f0d78b5e2b68ffbf27d819cc6db",
        "pip install -e .",
        "PyAV >=18.0.0",
        "B=1",
        "no CFG",
        "shift 12",
        "shift 3",
        "data-ward velocity",
        "N transitions",
        "N + 1 states",
        "61 GB",
        "completed long-run reward trend is not claimed",
        "pip install 'diffusers @ git+https://github.com/huggingface/diffusers.git@",
        "[Datasets](guidance/datasets.md)",
    ):
        assert required in text


def test_examples_readme_links_h3_and_separates_validation_levels() -> None:
    text = _text("examples/README.md")
    for root_link in EXAMPLE_LINKS:
        relative_link = f"../{root_link}"
        assert relative_link in text
        assert (ROOT / root_link).is_file()
    assert "Schema/API validated only" in text
    assert "hardware" in text
    assert "reward" in text
    assert "61 GB" in text
    assert "ImageBind" in text
    assert "facebookresearch/ImageBind.git" in text
    assert "NonCommercial" in text


def test_new_model_guide_documents_component_runtime_boundaries() -> None:
    text = _text("guidance/new_model.md")
    for required in (
        "ClassicPipelineRuntime",
        "ModularPipelineRuntime",
        "PseudoPipelineRuntime",
        "build_component_runtime()",
        "canonical lookup",
        "prepared/replacement override",
        "declared specs",
        "materialized modules",
        "materialize_components(None)",
        "adapter.pipeline",
        "adapter.scheduler",
        "scheduler_group",
        "trajectory_component_order",
        "ModelBundle",
        "RoutedComponentProxy",
    ):
        assert required in text


def test_workflow_guide_documents_structured_trajectory_contract_and_algorithms() -> None:
    text = _text("guidance/workflow.md")
    for required in (
        "StructuredTrajectory",
        "trajectory is None",
        "structured trajectories only",
        "`-1` means",
        "represented by `None`",
        "T + 1",
        "log-probability and callback maps have length T",
        '("video", "audio")',
        "GRPO",
        "GRPO-Guard",
        "DPPO",
        "DiffusionNFT",
        "AWM",
        "DPO",
        "DGPO",
        "CRD",
        "DiffusionOPD",
        "neutral guidance `1.0`",
        "framework-interface compatibility",
        "numerical parity",
    ):
        assert required in text
