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

"""Bound peak memory for token-local MiniMax H3 operations."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

H3_MAX_FEED_FORWARD_TOKENS = 1024
H3_MAX_ATTENTION_NORM_TOKENS = 1024


class _ChunkedFeedForward(nn.Module):
    """Run one existing feed-forward network over remainder-safe token chunks.

    The upstream Diffusers H3 blocks expose their feed-forward layers as a sole
    ``net`` ModuleList. Registering that same ModuleList directly keeps parameter
    identities and ``ff.net.*`` state-dict/LoRA paths unchanged.
    """

    def __init__(self, net: nn.ModuleList, *, max_tokens: int) -> None:
        super().__init__()
        self.net = net
        self.max_tokens = _positive_int(max_tokens, "max_tokens")

    def _forward_chunk(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the token-local network without requiring an even split."""
        if hidden_states.ndim < 3:
            raise ValueError(
                "MiniMax H3 feed-forward chunking expected [batch, tokens, hidden] "
                f"input, received shape={tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[1] <= self.max_tokens:
            return self._forward_chunk(hidden_states)
        chunks = hidden_states.split(self.max_tokens, dim=1)
        if torch.is_grad_enabled():
            return torch.cat(
                [
                    checkpoint(
                        self._forward_chunk,
                        chunk,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                    for chunk in chunks
                ],
                dim=1,
            )
        return torch.cat(
            [self._forward_chunk(chunk) for chunk in chunks],
            dim=1,
        )


class _ChunkedRMSNorm(nn.RMSNorm):
    """Bound row-wise RMSNorm temporaries while preserving its parameter path."""

    def __init__(self, norm: nn.RMSNorm, *, max_tokens: int) -> None:
        nn.Module.__init__(self)
        self.normalized_shape = norm.normalized_shape
        self.eps = norm.eps
        self.elementwise_affine = norm.elementwise_affine
        self.register_parameter("weight", norm.weight)
        self.max_tokens = _positive_int(max_tokens, "max_tokens")

    def _forward_chunk(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return nn.functional.rms_norm(
            hidden_states,
            self.normalized_shape,
            self.weight,
            self.eps,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize independent token rows without materializing a full FP32 copy."""
        if hidden_states.ndim < 3:
            raise ValueError(
                "MiniMax H3 RMSNorm chunking expected [batch, tokens, ..., hidden] "
                f"input, received shape={tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[1] <= self.max_tokens:
            return self._forward_chunk(hidden_states)
        return torch.cat(
            [self._forward_chunk(chunk) for chunk in hidden_states.split(self.max_tokens, dim=1)],
            dim=1,
        )


def install_h3_feed_forward_chunking(
    transformer: nn.Module,
    *,
    max_tokens: int = H3_MAX_FEED_FORWARD_TOKENS,
) -> int:
    """Install bounded feed-forward execution on both H3 repeated stacks.

    Installation happens immediately after pretrained component materialization,
    before Flow-Factory resume loading, LoRA injection, gradient checkpointing, or
    distributed wrapping. The mutation is idempotent and preserves every parameter
    object.

    Returns:
        Number of feed-forward layers configured across both stacks.
    """
    max_tokens = _positive_int(max_tokens, "max_tokens")
    blocks = tuple(_h3_repeated_blocks(transformer))
    configured = 0
    for name, block in blocks:
        feed_forward = getattr(block, "ff", None)
        if isinstance(feed_forward, _ChunkedFeedForward):
            if feed_forward.max_tokens != max_tokens:
                raise ValueError(
                    f"MiniMax H3 {name}.ff already uses max_tokens="
                    f"{feed_forward.max_tokens}, received conflicting {max_tokens}"
                )
            configured += 1
            continue
        net = getattr(feed_forward, "net", None)
        if not isinstance(net, nn.ModuleList):
            raise TypeError(
                f"MiniMax H3 {name}.ff expected a sole net ModuleList, received "
                f"{type(feed_forward).__name__} with net={type(net).__name__}"
            )
        if tuple(feed_forward._modules) != ("net",):
            raise TypeError(
                f"MiniMax H3 {name}.ff expected only the net child module, received "
                f"children={tuple(feed_forward._modules)}"
            )
        if not (
            len(net) == 3
            and type(net[0]).__name__ == "SwiGLU"
            and isinstance(net[1], nn.Dropout)
            and net[1].p == 0.0
            and isinstance(net[2], nn.Linear)
        ):
            raise TypeError(
                f"MiniMax H3 {name}.ff expected SwiGLU, Dropout(0), Linear; "
                f"received={[type(module).__name__ for module in net]}"
            )
        if tuple(feed_forward.named_parameters(recurse=False)) or tuple(
            feed_forward.named_buffers(recurse=False)
        ):
            raise TypeError(f"MiniMax H3 {name}.ff expected no direct parameters or buffers")
        _reject_execution_hooks(feed_forward, f"{name}.ff")
        replacement = _ChunkedFeedForward(net, max_tokens=max_tokens)
        replacement.train(feed_forward.training)
        block.ff = replacement
        configured += 1
    return configured


def install_h3_attention_norm_chunking(
    transformer: nn.Module,
    *,
    max_tokens: int = H3_MAX_ATTENTION_NORM_TOKENS,
) -> int:
    """Install bounded Q/K head normalization on both H3 repeated stacks."""
    max_tokens = _positive_int(max_tokens, "max_tokens")
    configured = 0
    for block_name, block in _h3_repeated_blocks(transformer):
        attention = getattr(block, "attn", None)
        if not isinstance(attention, nn.Module):
            raise TypeError(
                f"MiniMax H3 {block_name}.attn expected nn.Module, received "
                f"{type(attention).__name__}"
            )
        for norm_name in ("norm_q", "norm_k"):
            path = f"{block_name}.attn.{norm_name}"
            norm = getattr(attention, norm_name, None)
            if isinstance(norm, _ChunkedRMSNorm):
                if norm.max_tokens != max_tokens:
                    raise ValueError(
                        f"MiniMax H3 {path} already uses max_tokens={norm.max_tokens}, "
                        f"received conflicting {max_tokens}"
                    )
                configured += 1
                continue
            if not isinstance(norm, nn.RMSNorm):
                raise TypeError(
                    f"MiniMax H3 {path} expected nn.RMSNorm, received " f"{type(norm).__name__}"
                )
            if tuple(norm.named_children()) or tuple(norm.named_buffers(recurse=False)):
                raise TypeError(f"MiniMax H3 {path} expected no child modules or direct buffers")
            expected_parameters = ("weight",) if norm.elementwise_affine else ()
            if (
                tuple(name for name, _ in norm.named_parameters(recurse=False))
                != expected_parameters
            ):
                raise TypeError(
                    f"MiniMax H3 {path} expected direct parameters={expected_parameters}"
                )
            _reject_execution_hooks(norm, path)
            replacement = _ChunkedRMSNorm(norm, max_tokens=max_tokens)
            replacement.train(norm.training)
            setattr(attention, norm_name, replacement)
            configured += 1
    return configured


def _h3_repeated_blocks(transformer: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    token_refiner = getattr(transformer, "token_refiner", None)
    stacks = (
        ("token_refiner.refiner_blocks", getattr(token_refiner, "refiner_blocks", None)),
        ("transformer_blocks", getattr(transformer, "transformer_blocks", None)),
    )
    for stack_name, stack in stacks:
        if not isinstance(stack, nn.ModuleList) or not stack:
            raise TypeError(
                f"MiniMax H3 expected non-empty {stack_name} ModuleList, received "
                f"{type(stack).__name__}"
            )
        for index, block in enumerate(stack):
            yield f"{stack_name}.{index}", block


def _reject_execution_hooks(module: nn.Module, path: str) -> None:
    hook_fields = (
        "_forward_pre_hooks",
        "_forward_hooks",
        "_backward_pre_hooks",
        "_backward_hooks",
    )
    active_hooks = tuple(field for field in hook_fields if getattr(module, field, None))
    if active_hooks or getattr(module, "_hf_hook", None) is not None:
        raise TypeError(
            f"MiniMax H3 {path} must be configured before execution hooks; "
            f"received hooks={active_hooks}, hf_hook="
            f"{type(getattr(module, '_hf_hook', None)).__name__}"
        )


def _positive_int(value: int, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"MiniMax H3 {field} expected a positive int, received {value!r}")
    return value
