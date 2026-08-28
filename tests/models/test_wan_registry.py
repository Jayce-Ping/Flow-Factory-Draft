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

from typing import get_args, get_type_hints

import pytest

from flow_factory.hparams import ModelArguments
from flow_factory.models.registry import get_model_adapter_class, list_registered_models


def test_removed_wan_v2v_adapter_is_not_publicly_available() -> None:
    registered_models = list_registered_models()
    model_type_values = set(get_args(get_type_hints(ModelArguments)["model_type"]))

    assert "wan2_v2v" not in registered_models
    assert "wan2_v2v" not in model_type_values

    with pytest.raises(ImportError, match="wan2_v2v"):
        get_model_adapter_class("wan2_v2v")
