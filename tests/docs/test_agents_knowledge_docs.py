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

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE = ROOT / ".agents/knowledge"
TOPICS = (
    "component_runtime.md",
    "structured_trajectory.md",
    "minimax_h3.md",
)
SKILLS = (
    "ff-develop",
    "ff-debug",
    "ff-review",
    "ff-new-model",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _heading_anchors(path: Path) -> set[str]:
    anchors = set()
    for heading in re.findall(r"(?m)^#{1,6}\s+(.+)$", _text(path)):
        normalized = re.sub(r"[^\w\s-]", "", heading.lower())
        anchors.add(re.sub(r"[\s-]+", "-", normalized).strip("-"))
    return anchors


def test_new_topics_are_leaf_docs_with_bidirectional_routing() -> None:
    routing = _text(KNOWLEDGE / "README.md")
    for topic in TOPICS:
        path = KNOWLEDGE / "topics" / topic
        assert path.is_file()
        assert f"](topics/{topic})" in routing

        text = _text(path)
        assert "## Cross-refs" in text
        assert "constraints.md" in text
        assert "architecture.md" in text
        assert not re.search(r"[\u4e00-\u9fff]", text)


def test_leaf_docs_cover_runtime_trajectory_and_h3_contracts() -> None:
    runtime = _text(KNOWLEDGE / "topics/component_runtime.md")
    for required in (
        "ClassicPipelineRuntime",
        "ModularPipelineRuntime",
        "PseudoPipelineRuntime",
        "canonical",
        "override",
        "declared",
        "materialized",
        "materialize_components(None)",
        "SchedulerGroup",
        "ModelBundle",
        "Failure modes",
        "Review checklist",
    ):
        assert required in runtime

    trajectory = _text(KNOWLEDGE / "topics/structured_trajectory.md")
    for required in (
        "StructuredTrajectory",
        "component_order",
        "T + 1",
        "`-1`",
        "`None`",
        "callbacks",
        "active masks",
        "get_terminal_state",
        "get_replay_step",
        "get_replay_callback",
        "GRPO",
        "DPPO",
        "DiffusionNFT",
        "AWM",
        "DPO",
        "DGPO",
        "CRD",
        "DiffusionOPD",
        "bit-parity",
        "degrees of freedom",
        "homogeneous",
        "guidance `1.0`",
    ):
        assert required in trajectory

    h3 = _text(KNOWLEDGE / "topics/minimax_h3.md")
    for required in (
        "f53d552036a0d1bd5570782a39cd40cfabf112bc",
        "minimax-h3-t2va",
        "minimax-h3-fl2va",
        "minimax-h3-ref2va",
        "transformer_ref",
        "B=1",
        "no CFG",
        "first",
        "last",
        "ordered",
        "shift 12",
        "shift 3",
        "data-ward velocity",
        "N transitions",
        "PyAV",
        "61 GB",
        "Upgrade checklist",
    ):
        assert required in h3


def test_relevant_skills_link_down_to_all_new_topics() -> None:
    for skill in SKILLS:
        text = _text(ROOT / ".agents/skills" / skill / "SKILL.md")
        for topic in TOPICS:
            assert f"](../../knowledge/topics/{topic})" in text


def test_constraints_and_architecture_remain_concise_indexes() -> None:
    constraints = _text(KNOWLEDGE / "constraints.md")
    assert (
        "Quick index: **#1-5** Registry | **#6-10** Training Pipeline | "
        "**#11-14** Base Classes | **#15-17** Config | **#18-20** Distributed | "
        "**#21-27** Code Quality | **#28-29** Agent Workflow"
    ) in constraints
    assert "topics/component_runtime.md" in constraints
    assert "topics/structured_trajectory.md" in constraints
    assert "topics/minimax_h3.md" in constraints
    assert "state maps have length `T + 1`" in constraints
    assert "transition maps have length `T`" in constraints
    assert "`-1` means uncollected" in constraints
    assert "whole absent category is `None`" in constraints

    architecture = _text(KNOWLEDGE / "architecture.md")
    for topic in TOPICS:
        assert f"](topics/{topic})" in architecture
    assert "#### Component runtime enumeration boundaries" not in architecture
    assert "#### Structured trajectory bridge ownership boundaries" not in architecture


def test_adapter_gotchas_are_append_only() -> None:
    text = _text(KNOWLEDGE / "topics/adapter_conventions.md")
    gotchas = text.split("## Numbered Gotchas (append-only)", 1)[1].split("## Cross-refs", 1)[0]
    numbers = [int(value) for value in re.findall(r"(?m)^(\d+)\. ", gotchas)]
    assert numbers == list(range(1, 16))
    assert "12. **LTX2 rollouts publish structured trajectories only**" in gotchas
    assert "13. **Runtime and trajectory ownership stay explicit**" in gotchas


def test_new_markdown_links_resolve() -> None:
    paths = [KNOWLEDGE / "README.md"]
    paths.extend(KNOWLEDGE / "topics" / topic for topic in TOPICS)
    paths.extend(ROOT / ".agents/skills" / skill / "SKILL.md" for skill in SKILLS)

    for source in paths:
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", _text(source)):
            if "://" in target or target.startswith("#"):
                continue
            clean_target = target.split("#", 1)[0]
            resolved_target = (source.parent / clean_target).resolve()
            assert (
                resolved_target.exists()
            ), f"broken local link in {source.relative_to(ROOT)}: {target}"
            if "#" in target:
                anchor = target.split("#", 1)[1]
                assert anchor in _heading_anchors(
                    resolved_target
                ), f"broken anchor in {source.relative_to(ROOT)}: {target}"
