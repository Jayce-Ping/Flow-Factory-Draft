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
GUIDE = ROOT / "guidance/datasets.md"
README = ROOT / "README.md"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_dataset_guide_covers_shared_formats_and_h3_workflows() -> None:
    text = _text(GUIDE)

    for required in (
        "train.txt",
        "train.jsonl",
        "image_dir",
        "video_dir",
        "audio_dir",
        "minimax-h3-t2va",
        "minimax-h3-fl2va",
        "minimax-h3-ref2va",
        '"prompt"',
        '"images"',
        '"references"',
        "first frame",
        "last frame",
        "image",
        "video",
        "audio",
        "audio_path",
        "sample_rate",
        "reference_manifest",
        "condition_prefixes",
        "transformer_ref",
        "log-probability",
        "B=1",
        "guidance_scale: 1.0",
    ):
        assert required in text


def test_dataset_guide_states_reference_order_and_training_boundaries() -> None:
    text = " ".join(_text(GUIDE).split())

    for required in (
        "Array order is semantically significant",
        "At least one image or video reference is required",
        "Reference media is not stored in the Arrow cache",
        "Reference condition rows are not trajectory states",
        "Reference condition rows do not contribute policy log-probability degrees of freedom",
        "Gradients flow through `transformer_ref`",
    ):
        assert required in text


def test_dataset_guide_local_links_resolve() -> None:
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", _text(GUIDE)):
        if "://" in target or target.startswith("#"):
            continue
        resolved = (GUIDE.parent / target.split("#", 1)[0]).resolve()
        assert resolved.exists(), f"broken local link in {GUIDE.relative_to(ROOT)}: {target}"


def test_readme_routes_to_dataset_guide_and_replaces_news() -> None:
    text = _text(README)
    news = text.split("# 🔥 News", 1)[1].split("# 📕 Table of Contents", 1)[0]

    assert "[Datasets](guidance/datasets.md)" in text
    assert "See the [Dataset Guide](guidance/datasets.md)" in text
    assert "**[2026-08-11]**" in news
    assert "minimax-h3-t2va" in news
    assert "minimax-h3-fl2va" in news
    assert "minimax-h3-ref2va" in news
    assert "[2026-04-25]" not in news
    assert "[2026-02-01]" not in news
