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

from pathlib import Path


def test_multirole_examples_remain_unpublished() -> None:
    repository_root = Path(__file__).parents[2]
    for example_name in ("dmd2", "tdm", "tdm_r1"):
        assert not (repository_root / "examples" / example_name).exists()
    algorithms = (repository_root / "guidance" / "algorithms.md").read_text()
    assert "ttur_fake_updates" in algorithms
    assert "fake first" in algorithms.lower()
