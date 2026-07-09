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

# src/flow_factory/models/model_bundle.py
"""Single-root module bundle for distributed training.

`ModelBundle` wraps several trainable / frozen-shardable modules (e.g. one or
more transformers, plus a value critic) as ONE ``nn.Module`` so that
``accelerator.prepare`` wraps a single root. This is required by DeepSpeed (one
engine per ``prepare``) and FSDP2 (one root) — they cannot wrap multiple models.
It also lets a frozen member (e.g. Wan2.2's inactive transformer) be FSDP-sharded
for memory while only the ``requires_grad`` subset is trained.

`RoutedComponentProxy` is the transparent proxy installed as an adapter component
after ``prepare``: calling it routes the forward through the (wrapped) bundle
root — which is what drives DDP's gradient reducer / FSDP's param all-gather /
the DeepSpeed engine — while attribute access (``.config``, ``.dtype``,
``.parameters()``, ``cache_context``, ``disable_adapter`` ...) delegates to the
inner module so existing adapter code is unchanged.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ModelBundle(nn.Module):
    """Bundle of named submodules dispatched through a single ``forward``.

    The single ``nn.Module`` passed to ``accelerator.prepare`` so the whole
    distributed wrapper (DDP / FSDP / DeepSpeed) has exactly one root. Members
    may be trainable or frozen-but-sharded; only the ``requires_grad`` subset is
    optimized (the trainer keeps using ``adapter.get_trainable_parameters()``).

    Args:
        members: Mapping from component name (e.g. ``"transformer"``,
            ``"value_critic"``) to its ``nn.Module``. Must be non-empty.
    """

    def __init__(self, members: Dict[str, nn.Module]):
        super().__init__()
        if not members:
            raise ValueError(
                "ModelBundle requires at least one member module, got an empty mapping."
            )
        for name, module in members.items():
            if not isinstance(module, nn.Module):
                raise TypeError(
                    f"ModelBundle member '{name}' must be an nn.Module, "
                    f"got {type(module).__name__}."
                )
        self.members = nn.ModuleDict(members)

    @property
    def _no_split_modules(self):
        """Aggregate the members' ``_no_split_modules`` so accelerate's FSDP
        ``TRANSFORMER_BASED_WRAP`` can discover the transformer block class(es).

        accelerate's ``set_auto_wrap_policy`` reads ``getattr(root, "_no_split_modules")``
        off the single root passed to ``prepare`` (this bundle). Without this the
        list is empty, ``transformer_layer_cls`` becomes an empty set, and the whole
        bundle is wrapped as ONE FSDP unit -> one monolithic flat param (the full
        unsharded model materialized on every rank) -> OOM at init for large models
        (e.g. Wan2.2 A14B: ~53GB flat param). We surface the block-class *names* that
        the underlying model classes already declare (e.g. diffusers
        ``WanTransformer3DModel._no_split_modules == ['WanTransformerBlock']``);
        accelerate then resolves each name to its class via ``get_module_class_from_name``
        over this root's submodule tree and wraps per-block. Returns ``None`` when no
        member declares any (accelerate keeps its prior fallback unchanged).
        """
        names: list[str] = []
        # Walk members' submodule tree (NOT self, to avoid recursing on this property);
        # collect every ``_no_split_modules`` a nested module declares.
        for module in self.members.modules():
            nsm = getattr(module, "_no_split_modules", None)
            if nsm:
                names.extend(nsm)
        deduped = list(dict.fromkeys(names))
        return deduped or None

    def forward(self, component_name: str, /, *args, **kwargs):
        """Dispatch the call to ``members[component_name]``.

        ``component_name`` is positional-only so it never collides with a
        member's own keyword arguments.

        Args:
            component_name: Name of the member to invoke.
            *args: Positional args forwarded to the member.
            **kwargs: Keyword args forwarded to the member.

        Returns:
            The member's forward output.

        Raises:
            KeyError: If ``component_name`` is not a member of this bundle.
        """
        if component_name not in self.members:
            raise KeyError(
                f"ModelBundle has no member '{component_name}'. "
                f"Available members: {list(self.members.keys())}."
            )
        return self.members[component_name](*args, **kwargs)


class RoutedComponentProxy:
    """Callable proxy that routes a component's forward through a `ModelBundle`.

    Installed as an adapter component (in ``adapter._components``) after
    ``prepare``. Calling it invokes the prepared bundle root (driving grad-sync /
    param-gather), while every attribute access falls through to the inner
    module, so adapter code that does ``self.transformer(...)`` /
    ``self.transformer.config`` keeps working unchanged.

    Not an ``nn.Module`` on purpose: it must not register the inner module as a
    submodule (that would double-count parameters under the bundle root). Use
    ``BaseAdapter._unwrap`` to recover the inner module.
    """

    def __init__(self, bundle: nn.Module, name: str, inner: nn.Module):
        # Bypass __setattr__/__getattr__ for the proxy's own fields.
        object.__setattr__(self, "_bundle", bundle)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "inner", inner)

    def __call__(self, *args, **kwargs):
        bundle = object.__getattribute__(self, "_bundle")
        name = object.__getattribute__(self, "_name")
        return bundle(name, *args, **kwargs)

    def __getattr__(self, attr: str):
        # __getattr__ only fires when normal lookup misses, so the proxy's own
        # fields (_bundle/_name/inner) never reach here.
        return getattr(object.__getattribute__(self, "inner"), attr)

    def __repr__(self) -> str:
        name = object.__getattribute__(self, "_name")
        inner = object.__getattribute__(self, "inner")
        return f"RoutedComponentProxy(name={name!r}, inner={type(inner).__name__})"
