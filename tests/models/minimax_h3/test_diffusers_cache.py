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

from contextlib import contextmanager
from typing import Any, Dict

import pytest
import torch
from diffusers.hooks import FirstBlockCacheConfig
from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerBlock,
)
from peft import LoraConfig, get_peft_model

from flow_factory.acceleration.torch_compile import CompileAccelerator
from flow_factory.models.minimax_h3._chunking import (
    install_h3_in_forward_block_checkpointing,
)
from flow_factory.models.minimax_h3._diffusers_cache import (
    H3_DIFFUSERS_CACHE_POLICIES,
    h3_diffusers_cache_context,
    prepare_h3_diffusers_cache,
)
from flow_factory.models.minimax_h3.adapters import (
    MiniMaxH3FL2VAAdapter,
    MiniMaxH3Ref2VAAdapter,
    MiniMaxH3T2VAAdapter,
)


class FSDPMiniMaxH3TransformerBlock(MiniMaxH3TransformerBlock):
    """Represent the runtime subclass installed by FSDP2 composable APIs."""


class IncompatibleMiniMaxH3TransformerBlock(MiniMaxH3TransformerBlock):
    """Provide an isolated registry key for incompatible metadata testing."""


def _tiny_h3_transformer() -> MiniMaxH3Transformer3DModel:
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=1,
        attention_head_dim=12,
        hidden_size=12,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=24,
        in_channels=4,
        audio_in_channels=4,
        patch_size=(1, 1, 1),
        text_dim=12,
        freq_dim=4,
        time_embed_hidden_dim=12,
        time_embed_dim=12,
        rope_freq_dim=2,
    ).eval()


def _tiny_h3_inputs() -> Dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.randn(1, 1, 4),
        "audio_hidden_states": torch.randn(1, 1, 4),
        "encoder_hidden_states": torch.randn(1, 1, 12),
        "timestep": torch.tensor([0.5]),
        "timestep_indices": torch.zeros(3, dtype=torch.long),
        "token_tags": torch.tensor([0, 1, 2], dtype=torch.long),
        "position_ids": torch.zeros(3, 3),
        "video_indices": torch.tensor([0]),
        "audio_indices": torch.tensor([2]),
        "text_indices": torch.tensor([1]),
    }


@pytest.mark.parametrize("runtime_mode", ["eager", "peft", "fsdp2_checkpointed"])
def test_real_h3_first_block_cache_skips_tail_and_recomputes_after_reset(
    runtime_mode: str,
) -> None:
    torch.manual_seed(7)
    base_transformer = _tiny_h3_transformer()
    transformer = base_transformer
    if runtime_mode == "peft":
        transformer = get_peft_model(
            base_transformer,
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                init_lora_weights="gaussian",
            ),
        )
    elif runtime_mode == "fsdp2_checkpointed":
        for block in base_transformer.transformer_blocks:
            block.__class__ = FSDPMiniMaxH3TransformerBlock
        assert install_h3_in_forward_block_checkpointing(base_transformer) == 3

    prepare_h3_diffusers_cache("first_block", transformer)
    runtime_block_class = type(base_transformer.transformer_blocks[0])
    metadata = TransformerBlockRegistry.get(runtime_block_class)
    assert metadata.return_hidden_states_index == 0
    assert metadata.return_encoder_hidden_states_index is None
    prepare_h3_diffusers_cache("first_block", transformer)
    assert TransformerBlockRegistry.get(runtime_block_class) is metadata

    tail_attention_calls = []
    handle = base_transformer.transformer_blocks[-1].attn.register_forward_pre_hook(
        lambda *_args: tail_attention_calls.append(None)
    )
    transformer.enable_cache(FirstBlockCacheConfig(threshold=0.05))
    inputs = _tiny_h3_inputs()

    def run() -> Any:
        with torch.no_grad(), transformer.cache_context("minimax_h3_t2va"):
            return transformer(**inputs, return_dict=False)

    try:
        first = run()
        second = run()
        assert len(tail_attention_calls) == 1
        torch.testing.assert_close(second, first)

        transformer._reset_stateful_cache()
        third = run()
        assert len(tail_attention_calls) == 2
        torch.testing.assert_close(third, first)
    finally:
        if transformer.is_cache_enabled:
            transformer.disable_cache()
        handle.remove()


@pytest.mark.parametrize("marker_target", ["forward", "_compiled_call_impl"])
def test_h3_cache_rejects_grad_consistent_torch_compile_before_registry_mutation(
    marker_target: str,
) -> None:
    transformer = _tiny_h3_transformer()
    if marker_target == "forward":
        CompileAccelerator._wrap_forward_grad_consistent(None, transformer)
    else:

        def compiled_call(*args: Any, **kwargs: Any) -> Any:
            return transformer.forward(*args, **kwargs)

        setattr(compiled_call, "_ff_grad_consistent", True)
        transformer._compiled_call_impl = compiled_call

    TransformerBlockRegistry._register()
    registry_before = dict(TransformerBlockRegistry._registry)
    with pytest.raises(ValueError, match="cannot be combined with torch_compile"):
        prepare_h3_diffusers_cache("first_block", transformer)

    assert TransformerBlockRegistry._registry == registry_before


def test_h3_cache_rejects_short_main_block_stack_before_diffusers_hook_failure() -> None:
    transformer = _tiny_h3_transformer()
    transformer.transformer_blocks = torch.nn.ModuleList([transformer.transformer_blocks[0]])

    with pytest.raises(TypeError, match=r"at least two blocks.*length=1"):
        prepare_h3_diffusers_cache("first_block", transformer)


def test_h3_cache_rejects_incompatible_existing_registry_metadata() -> None:
    transformer = _tiny_h3_transformer()
    for block in transformer.transformer_blocks:
        block.__class__ = IncompatibleMiniMaxH3TransformerBlock
    TransformerBlockRegistry.register(
        IncompatibleMiniMaxH3TransformerBlock,
        TransformerBlockMetadata(
            return_hidden_states_index=1,
            return_encoder_hidden_states_index=0,
        ),
    )

    with pytest.raises(RuntimeError, match="incompatible registered metadata"):
        prepare_h3_diffusers_cache("first_block", transformer)


def test_h3_cache_context_resets_state_after_forward_exception() -> None:
    events = []

    class CacheOwner:
        is_cache_enabled = True

        @contextmanager
        def cache_context(self, name: str):
            events.append(("enter", name))
            yield

        def _reset_stateful_cache(self) -> None:
            events.append(("reset",))

    with pytest.raises(RuntimeError, match="forward failed"):
        with h3_diffusers_cache_context(CacheOwner(), workflow="ref2va"):
            raise RuntimeError("forward failed")

    assert events == [("enter", "minimax_h3_ref2va"), ("reset",)]


def test_disabled_h3_cache_context_is_a_noop() -> None:
    class CacheOwner:
        is_cache_enabled = False

        def cache_context(self, name: str):
            raise AssertionError(f"disabled cache unexpectedly entered context {name}")

    with h3_diffusers_cache_context(CacheOwner(), workflow="t2va"):
        pass


@pytest.mark.parametrize(
    ("adapter_class", "component_name"),
    [
        (MiniMaxH3T2VAAdapter, "transformer"),
        (MiniMaxH3FL2VAAdapter, "transformer"),
        (MiniMaxH3Ref2VAAdapter, "transformer_ref"),
    ],
)
def test_all_h3_workflows_declare_first_block_cache_only(
    adapter_class: type,
    component_name: str,
) -> None:
    assert adapter_class.supports_diffusers_cache
    assert adapter_class.supported_diffusers_cache_policies is H3_DIFFUSERS_CACHE_POLICIES
    assert adapter_class.supported_diffusers_cache_policies == frozenset({"first_block"})
    assert adapter_class.transformer_component_name == component_name
