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
EXPECTED_DIFFUSERS_REFERENCE = (
    "diffusers @ git+https://github.com/huggingface/diffusers.git"
    "@4e0466f3e5260f0d78b5e2b68ffbf27d819cc6db"
)


def test_project_metadata_pins_exact_diffusers_direct_reference_in_any_environment() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    diffusers_requirements = re.findall(r'"(diffusers[^"]*)"', pyproject)

    assert diffusers_requirements == [EXPECTED_DIFFUSERS_REFERENCE], (
        "pyproject.toml must contain exactly one diffusers direct reference at the required "
        f"commit; expected={EXPECTED_DIFFUSERS_REFERENCE!r}, observed={diffusers_requirements!r}"
    )
