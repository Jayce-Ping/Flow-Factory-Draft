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

"""Bridge MiniMax H3 to diffusers feature-cache hooks."""

from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch

try:
    from diffusers.hooks._helpers import (
        TransformerBlockMetadata,
        TransformerBlockRegistry,
    )
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3TransformerBlock,
    )
    from diffusers.utils.torch_utils import unwrap_module

    _CACHE_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as import_error:
    TransformerBlockMetadata = None
    TransformerBlockRegistry = None
    MiniMaxH3TransformerBlock = None
    unwrap_module = None
    _CACHE_IMPORT_ERROR = import_error

H3_DIFFUSERS_CACHE_POLICIES = frozenset({"first_block"})


def prepare_h3_diffusers_cache(policy: str, transformer: Any) -> None:
    """Install the diffusers metadata required by an H3 cache policy.

    Args:
        policy: User-facing diffusers cache policy identifier.
        transformer: Prepared H3 transformer route that will receive the policy.

    Raises:
        RuntimeError: If the pinned diffusers cache compatibility surface is unavailable.
        TypeError: If the prepared route does not resolve to the expected H3 block stack.
        ValueError: If the policy is unsupported or combined with ``torch_compile``.
    """
    if policy not in H3_DIFFUSERS_CACHE_POLICIES:
        raise ValueError(
            "MiniMax H3 diffusers cache expected policy in "
            f"{sorted(H3_DIFFUSERS_CACHE_POLICIES)}, received {policy!r}"
        )
    if _CACHE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "MiniMax H3 FirstBlockCache requires diffusers>=0.40.0 with "
            "diffusers.hooks._helpers.TransformerBlockRegistry and "
            "MiniMaxH3TransformerBlock available"
        ) from _CACHE_IMPORT_ERROR

    base_transformer = getattr(transformer, "inner", transformer)
    get_base_model = getattr(base_transformer, "get_base_model", None)
    if callable(get_base_model):
        base_transformer = get_base_model()
    if getattr(getattr(base_transformer, "forward", None), "_ff_grad_consistent", False) or getattr(
        getattr(base_transformer, "_compiled_call_impl", None),
        "_ff_grad_consistent",
        False,
    ):
        raise ValueError(
            "MiniMax H3 FirstBlockCache cannot be combined with torch_compile because "
            "diffusers 0.40.0 retains cross-step autograd graphs in its cache; remove "
            "either acceleration entry"
        )
    transformer_blocks = getattr(base_transformer, "transformer_blocks", None)
    if not isinstance(transformer_blocks, torch.nn.ModuleList) or len(transformer_blocks) < 2:
        block_count = (
            len(transformer_blocks) if isinstance(transformer_blocks, torch.nn.ModuleList) else None
        )
        raise TypeError(
            "MiniMax H3 FirstBlockCache expected a transformer_blocks ModuleList with at "
            f"least two blocks, received {type(transformer_blocks).__name__} with "
            f"length={block_count}"
        )
    unwrapped_blocks = tuple(unwrap_module(block) for block in transformer_blocks)
    unexpected_blocks = tuple(
        type(block).__name__
        for block in unwrapped_blocks
        if not isinstance(block, MiniMaxH3TransformerBlock)
    )
    if unexpected_blocks:
        raise TypeError(
            "MiniMax H3 FirstBlockCache expected MiniMaxH3TransformerBlock entries, "
            f"received unexpected block types={unexpected_blocks}"
        )

    block_classes = tuple(dict.fromkeys(type(block) for block in unwrapped_blocks))
    missing_classes = []
    for block_class in block_classes:
        try:
            metadata = TransformerBlockRegistry.get(block_class)
        except ValueError:
            missing_classes.append(block_class)
            continue
        if (
            metadata.return_hidden_states_index != 0
            or metadata.return_encoder_hidden_states_index is not None
            or metadata.hidden_states_argument_name != "hidden_states"
        ):
            raise RuntimeError(
                "MiniMax H3 FirstBlockCache found incompatible registered metadata for "
                f"block class {block_class.__name__}: "
                f"return_hidden_states_index={metadata.return_hidden_states_index!r}, "
                "return_encoder_hidden_states_index="
                f"{metadata.return_encoder_hidden_states_index!r}, "
                f"hidden_states_argument_name={metadata.hidden_states_argument_name!r}"
            )
    for block_class in missing_classes:
        TransformerBlockRegistry.register(
            model_class=block_class,
            metadata=TransformerBlockMetadata(
                return_hidden_states_index=0,
                return_encoder_hidden_states_index=None,
            ),
        )


def reset_h3_diffusers_cache(transformer: Any, *, workflow: str) -> None:
    """Reset stateful cache hooks before one independent H3 generation.

    Args:
        transformer: Prepared H3 transformer route that owns diffusers cache hooks.
        workflow: H3 workflow identifier used in diagnostics.

    Raises:
        TypeError: If an enabled cache does not expose the required reset method.
    """
    if not getattr(transformer, "is_cache_enabled", False):
        return
    reset = getattr(transformer, "_reset_stateful_cache", None)
    if not callable(reset):
        raise TypeError(
            f"MiniMax H3 workflow={workflow!r} enabled diffusers cache expected callable "
            f"_reset_stateful_cache(), received {type(reset).__name__}"
        )
    reset()


@contextmanager
def h3_diffusers_cache_context(transformer: Any, *, workflow: str) -> Iterator[None]:
    """Select one H3 cache state while an enabled transformer runs.

    Args:
        transformer: Prepared H3 transformer route that owns diffusers cache hooks.
        workflow: H3 workflow identifier used for the cache state and diagnostics.

    Yields:
        ``None`` while the prepared transformer forward executes.

    Raises:
        TypeError: If an enabled cache does not expose ``cache_context``.
    """
    if not getattr(transformer, "is_cache_enabled", False):
        yield
        return
    cache_context = getattr(transformer, "cache_context", None)
    if not callable(cache_context):
        raise TypeError(
            f"MiniMax H3 workflow={workflow!r} enabled diffusers cache expected callable "
            f"cache_context(), received {type(cache_context).__name__}"
        )
    try:
        with cache_context(f"minimax_h3_{workflow}"):
            yield
    except BaseException:
        reset_h3_diffusers_cache(transformer, workflow=workflow)
        raise
