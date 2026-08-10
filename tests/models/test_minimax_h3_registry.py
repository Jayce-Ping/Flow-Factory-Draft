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

import flow_factory.models.minimax_h3.dependency as dependency
from flow_factory.hparams import ModelArguments
from flow_factory.models.abc import BaseAdapter
from flow_factory.models.registry import get_model_adapter_class, list_registered_models

EXPECTED_ADAPTERS = {
    "minimax-h3-t2va": "MiniMaxH3T2VAAdapter",
    "minimax-h3-fl2va": "MiniMaxH3FL2VAAdapter",
    "minimax-h3-ref2va": "MiniMaxH3Ref2VAAdapter",
}


def test_registry_lazy_resolves_all_h3_workflows_without_installed_symbols(monkeypatch) -> None:
    monkeypatch.setattr(dependency, "_SYMBOLS", None)
    monkeypatch.setattr(dependency, "_IMPORT_ERROR", ImportError("H3 intentionally unavailable"))

    registered = list_registered_models()
    for key, class_name in EXPECTED_ADAPTERS.items():
        adapter_class = get_model_adapter_class(key)
        assert registered[key].endswith(f".{class_name}")
        assert adapter_class.__name__ == class_name
        assert adapter_class.__bases__ == (BaseAdapter,)


def test_model_type_literal_contains_all_h3_registry_keys() -> None:
    model_type = get_type_hints(ModelArguments)["model_type"]
    literal_values = set(get_args(model_type))

    assert set(EXPECTED_ADAPTERS).issubset(literal_values)
    for key in EXPECTED_ADAPTERS:
        assert ModelArguments(model_type=key).model_type == key


def test_minimax_h3_package_exports_all_workflow_adapters() -> None:
    import flow_factory.models.minimax_h3 as minimax_h3

    assert {name for name in EXPECTED_ADAPTERS.values() if hasattr(minimax_h3, name)} == set(
        EXPECTED_ADAPTERS.values()
    )
