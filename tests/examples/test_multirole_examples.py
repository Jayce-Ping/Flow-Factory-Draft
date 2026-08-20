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

import yaml


def test_only_gpu_validated_multirole_example_is_published() -> None:
    repository_root = Path(__file__).parents[2]
    for example_name in ("dmd2", "tdm_r1"):
        assert not (repository_root / "examples" / example_name).exists()
    tdm_path = repository_root / "examples" / "tdm" / "lora" / "sd3_5" / "ocr.yaml"
    config = yaml.safe_load(tdm_path.read_text())
    assert config["train"]["trainer_type"] == "tdm"
    assert config["train"]["num_inference_steps"] == 4
    assert "trajectory_steps" not in config["train"]
    assert config["train"]["tdm_importance_clip"] == 20.0
    assert config["train"]["replay_rtol"] == config["train"]["replay_atol"] == 0
    assert [optimizer["name"] for optimizer in config["optimizers"]] == [
        "generator",
        "fake",
    ]
    assert config["scheduler"] == {"dynamics_type": "ODE"}
    algorithms = (repository_root / "guidance" / "algorithms.md").read_text()
    assert "ttur_fake_updates" in algorithms
    assert "fake first" in algorithms.lower()
