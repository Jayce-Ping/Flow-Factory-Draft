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

import pytest
import torch

from flow_factory.hparams import ModelArguments


def test_scalar_frozen_dtype_remains_a_backward_compatible_global_policy() -> None:
    arguments = ModelArguments(frozen_parameters_dtype="bf16")

    assert arguments.frozen_parameters_dtype is torch.bfloat16
    assert arguments.to_dict()["frozen_parameters_dtype"] == "bfloat16"


def test_component_frozen_dtype_policy_normalizes_and_injects_default_null() -> None:
    arguments = ModelArguments(
        frozen_parameters_dtype={
            "transformers": "bf16",
            "vae": "fp32",
            "audio_vae": None,
        }
    )

    assert arguments.frozen_parameters_dtype == {
        "transformers": torch.bfloat16,
        "vae": torch.float32,
        "audio_vae": None,
        "default": None,
    }
    assert arguments.to_dict()["frozen_parameters_dtype"] == {
        "transformers": "bfloat16",
        "vae": "float32",
        "audio_vae": None,
        "default": None,
    }


@pytest.mark.parametrize("invalid", (1, ["bf16"]))
def test_frozen_dtype_policy_rejects_non_dtype_non_mapping_values(invalid) -> None:
    with pytest.raises(TypeError, match=r"frozen_parameters_dtype.*dtype, mapping, or None"):
        ModelArguments(frozen_parameters_dtype=invalid)


def test_frozen_dtype_policy_rejects_invalid_selector_type() -> None:
    with pytest.raises(TypeError, match=r"selector.*non-empty str.*int"):
        ModelArguments(frozen_parameters_dtype={1: "bf16"})


def test_frozen_dtype_policy_rejects_unknown_dtype_name() -> None:
    with pytest.raises(ValueError, match=r"\['vae'\].*known dtype.*int8"):
        ModelArguments(frozen_parameters_dtype={"vae": "int8"})
