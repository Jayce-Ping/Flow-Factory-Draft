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

import ast
import importlib
import importlib.machinery
import inspect
import sys
import types
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, get_args, get_type_hints

import pytest
import torch

_CLASSIC_ADAPTERS = (
    ("flow_factory.models.stable_diffusion.sd3_5", "SD3_5Adapter"),
    ("flow_factory.models.flux.flux1", "Flux1Adapter"),
    ("flow_factory.models.flux.flux1_kontext", "Flux1KontextAdapter"),
    ("flow_factory.models.flux.flux2", "Flux2Adapter"),
    ("flow_factory.models.flux.flux2_klein", "Flux2KleinAdapter"),
    ("flow_factory.models.qwen_image.qwen_image", "QwenImageAdapter"),
    (
        "flow_factory.models.qwen_image.qwen_image_edit_plus",
        "QwenImageEditPlusAdapter",
    ),
    ("flow_factory.models.z_image.z_image", "ZImageAdapter"),
    ("flow_factory.models.wan.wan2_t2v", "Wan2_T2V_Adapter"),
    ("flow_factory.models.wan.wan2_i2v", "Wan2_I2V_Adapter"),
    ("flow_factory.models.wan.wan2_v2v", "Wan2_V2V_Adapter"),
    ("flow_factory.models.ltx2.ltx2_t2av", "LTX2_T2AV_Adapter"),
    ("flow_factory.models.ltx2.ltx2_i2av", "LTX2_I2AV_Adapter"),
)


@pytest.mark.parametrize(("module_name", "adapter_name"), _CLASSIC_ADAPTERS)
def test_classic_adapter_loaders_delegate_to_shared_dtype_loader(
    module_name: str,
    adapter_name: str,
) -> None:
    adapter_class = getattr(importlib.import_module(module_name), adapter_name)
    assert not adapter_class.supports_fsdp2_cpu_efficient_loading
    adapter = object.__new__(adapter_class)
    adapter.model_args = SimpleNamespace(model_name_or_path="model")
    calls = []
    sentinel = object()

    def fake_loader(
        self: Any,
        pipeline_class: type,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> object:
        calls.append((pipeline_class, pretrained_model_name_or_path, kwargs))
        return sentinel

    adapter._load_diffusers_pipeline = MethodType(fake_loader, adapter)

    assert adapter.load_pipeline() is sentinel
    assert len(calls) == 1
    assert calls[0][1] == "model"


def test_sensenova_loader_forwards_custom_loader_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("flow_factory.models.sensenova.sensenova")
    adapter = object.__new__(module.SenseNovaAdapter)
    adapter.model_args = SimpleNamespace(
        model_name_or_path="model",
        extra_kwargs={"trust_remote_code": True},
    )
    sentinel = object()
    calls = []

    def fake_from_pretrained(model_name_or_path: str, **kwargs: Any) -> object:
        calls.append((model_name_or_path, kwargs))
        return sentinel

    monkeypatch.setattr(
        module.SenseNovaPseudoPipeline,
        "from_pretrained",
        fake_from_pretrained,
    )

    assert adapter.load_pipeline() is sentinel
    assert calls == [
        (
            "model",
            {"low_cpu_mem_usage": False, "trust_remote_code": True},
        )
    ]


def test_sensenova_loader_applies_transformer_load_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("flow_factory.models.sensenova.sensenova")
    adapter = object.__new__(module.SenseNovaAdapter)
    adapter.model_args = SimpleNamespace(model_name_or_path="model", extra_kwargs={})
    adapter._component_load_dtype_manifest = None
    adapter._component_load_dtype_overrides = {"transformers": torch.bfloat16}
    calls = []

    def fake_from_pretrained(model_name_or_path: str, **kwargs: Any) -> object:
        calls.append((model_name_or_path, kwargs))
        return object()

    monkeypatch.setattr(
        module.SenseNovaPseudoPipeline,
        "from_pretrained",
        fake_from_pretrained,
    )

    adapter.load_pipeline()

    assert calls == [
        (
            "model",
            {"low_cpu_mem_usage": False, "dtype": torch.bfloat16},
        )
    ]


def test_bagel_loader_source_keeps_custom_pipeline_boundary() -> None:
    source_path = (
        Path(__file__).parents[2] / "src" / "flow_factory" / "models" / "bagel" / "bagel.py"
    )
    tree = ast.parse(source_path.read_text())
    load_pipeline = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "load_pipeline"
    )

    calls = [
        node
        for node in ast.walk(load_pipeline)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
    ]
    assert len(calls) == 1
    assert "BagelPseudoPipeline.from_pretrained" in ast.unparse(calls[0].func)
    assert any(keyword.arg == "component_dtypes" for keyword in calls[0].keywords)


def test_bagel_adapter_imports_with_its_optional_kernel_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flow_factory.utils.imports as import_utils

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    flash_attn.flash_attn_varlen_func = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)
    monkeypatch.setattr(import_utils, "is_flash_attn_available", lambda *args: True)
    monkeypatch.setattr(import_utils, "get_flash_attn_version", lambda: "test")

    module = importlib.import_module("flow_factory.models.bagel.bagel")

    assert module.BagelAdapter.__name__ == "BagelAdapter"
    assert not module.BagelAdapter.supports_fsdp2_cpu_efficient_loading
    assert "_freeze_components" not in module.BagelAdapter.__dict__


def test_all_registered_adapters_have_an_audited_load_pipeline() -> None:
    from flow_factory.models.registry import list_registered_models

    registered_paths = set(list_registered_models().values())
    audited_paths = {
        f"{module_name}.{adapter_name}" for module_name, adapter_name in _CLASSIC_ADAPTERS
    } | {
        "flow_factory.models.bagel.bagel.BagelAdapter",
        "flow_factory.models.sensenova.sensenova.SenseNovaAdapter",
        "flow_factory.models.minimax_h3.adapters.MiniMaxH3T2VAAdapter",
        "flow_factory.models.minimax_h3.adapters.MiniMaxH3FL2VAAdapter",
        "flow_factory.models.minimax_h3.adapters.MiniMaxH3Ref2VAAdapter",
    }

    assert audited_paths == registered_paths


def test_model_type_literal_matches_registered_model_keys() -> None:
    from flow_factory.hparams.model_args import ModelArguments
    from flow_factory.models.registry import list_registered_models

    model_type = get_type_hints(ModelArguments)["model_type"]

    assert set(get_args(model_type)) == set(list_registered_models())


def test_model_specific_load_dtype_defaults_are_explicit_and_narrow() -> None:
    from flow_factory.models.minimax_h3.adapters import (
        MiniMaxH3FL2VAAdapter,
        MiniMaxH3Ref2VAAdapter,
        MiniMaxH3T2VAAdapter,
    )
    from flow_factory.models.wan.wan2_i2v import Wan2_I2V_Adapter
    from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter
    from flow_factory.models.wan.wan2_v2v import Wan2_V2V_Adapter
    from flow_factory.models.z_image.z_image import ZImageAdapter

    assert {
        adapter.component_load_dtype_defaults
        for adapter in (
            MiniMaxH3T2VAAdapter,
            MiniMaxH3FL2VAAdapter,
            MiniMaxH3Ref2VAAdapter,
        )
    } == {torch.bfloat16}
    assert all(
        adapter.supports_fsdp2_cpu_efficient_loading
        for adapter in (
            MiniMaxH3T2VAAdapter,
            MiniMaxH3FL2VAAdapter,
            MiniMaxH3Ref2VAAdapter,
        )
    )
    expected_wan_defaults = {
        "transformers": torch.bfloat16,
        "text_encoders": torch.bfloat16,
        "vae": torch.float32,
    }
    assert Wan2_T2V_Adapter.component_load_dtype_defaults == expected_wan_defaults
    assert Wan2_V2V_Adapter.component_load_dtype_defaults == expected_wan_defaults
    assert Wan2_I2V_Adapter.component_load_dtype_defaults == {
        **expected_wan_defaults,
        "image_encoder": torch.float32,
    }
    assert ZImageAdapter.component_load_dtype_defaults == {"transformer": torch.float32}


def test_priority_model_loading_topologies_remain_explicit() -> None:
    from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter
    from flow_factory.models.qwen_image.qwen_image_edit_plus import (
        QwenImageEditPlusAdapter,
    )
    from flow_factory.models.sensenova.sensenova import SenseNovaAdapter
    from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter

    inference_source = inspect.getsource(LTX2_T2AV_Adapter.inference_modules.fget)
    assert all(
        component in inference_source for component in ("vae", "audio_vae", "connectors", "vocoder")
    )
    assert Wan2_T2V_Adapter.component_load_dtype_defaults["transformers"] is torch.bfloat16
    assert Wan2_T2V_Adapter.ddp_find_unused_parameters
    assert QwenImageEditPlusAdapter.ddp_find_unused_parameters
    assert not SenseNovaAdapter.supports_fsdp2_cpu_efficient_loading
