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

# src/flow_factory/acceleration/diffusers_cache.py
"""Lossy rollout-only feature caching via diffusers' native ``CacheMixin``.

Zero extra dependency: diffusers transformers already inherit ``enable_cache`` /
``disable_cache`` (they are ``CacheMixin`` models — the same machinery behind the
``cache_context(...)`` calls Flow-Factory adapters already use). This accelerator
enables a cache policy for the duration of one rollout epoch and tears it down on
exit so the Stage-6 training forward stays exact.

Only valid in the rollout slot of a decoupled / distillation trainer — the
paradigm validator enforces this (``constraints.md`` #7).
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, List

from diffusers.hooks import (
    FasterCacheConfig,
    FirstBlockCacheConfig,
    MagCacheConfig,
    PyramidAttentionBroadcastConfig,
    TaylorSeerCacheConfig,
)

from .abc import BaseAccelerator
from ..utils.logger_utils import setup_logger

if TYPE_CHECKING:
    import torch

    from ..models.abc import BaseAdapter

logger = setup_logger(__name__)

# Map the user-facing ``policy`` string to its diffusers config class.
_POLICY_CONFIGS = {
    "first_block": FirstBlockCacheConfig,
    "faster": FasterCacheConfig,
    "pyramid": PyramidAttentionBroadcastConfig,
    "taylorseer": TaylorSeerCacheConfig,
    "magcache": MagCacheConfig,
}


class DiffusersCacheAccelerator(BaseAccelerator):
    """Enable a diffusers cache policy on every transformer during rollout.

    Lossy and rollout-scoped. The default policy is ``first_block`` (FirstBlockCache,
    aka FBCache) which is robust across models and needs only a single ``threshold``.

    Parameters (from the entry's ``params``):
        policy: One of ``first_block`` / ``faster`` / ``pyramid`` / ``taylorseer`` /
            ``magcache``. Defaults to ``first_block``.
        <other>: All remaining params are forwarded verbatim to the selected
            diffusers config class (e.g. ``threshold`` for FirstBlockCache).

    Note:
        Caching reuses block outputs across denoising steps, so the per-step
        ``cache_context`` the adapter opens around its transformer call must persist
        cache state across the loop. Adapters that already wrap their forward in
        ``transformer.cache_context(...)`` (Qwen-Image, Wan2, LTX2, FLUX.2-Klein)
        are cache-ready; verify the reward distribution before/after enabling.
    """

    safety = "lossy"
    stage = "rollout"

    def _build_config(self):
        params = dict(self.params)
        policy = params.pop("policy", "first_block")
        if policy not in _POLICY_CONFIGS:
            raise ValueError(
                f"DiffusersCacheAccelerator: unknown policy={policy!r}; "
                f"expected one of {sorted(_POLICY_CONFIGS)}."
            )
        config_cls = _POLICY_CONFIGS[policy]
        try:
            return config_cls(**params)
        except TypeError as e:
            raise ValueError(
                f"DiffusersCacheAccelerator: invalid parameters {sorted(params)} for policy "
                f"'{policy}' ({config_cls.__name__}): {e}"
            ) from e

    @contextmanager
    def rollout_context(self, adapter: "BaseAdapter") -> Iterator[None]:
        transformer_names = adapter.transformer_names
        if not transformer_names:
            raise ValueError("DiffusersCacheAccelerator: adapter exposes no transformer to cache.")

        enabled: List["torch.nn.Module"] = []
        try:
            for name in transformer_names:
                transformer = adapter.get_component(name)
                if not hasattr(transformer, "enable_cache"):
                    raise ValueError(
                        f"DiffusersCacheAccelerator: component '{name}' is not a diffusers "
                        "CacheMixin (no `enable_cache`); use a different accelerator."
                    )
                # Defensive: clear any stale cache left enabled by a prior epoch.
                if getattr(transformer, "is_cache_enabled", False):
                    transformer.disable_cache()
                transformer.enable_cache(self._build_config())
                enabled.append(transformer)
                # diffusers' HookRegistry caches its child-registry list the first time
                # a `cache_context` sets a context. If a `cache_context` ran while the
                # cache was DISABLED (e.g. during eval, where the adapter still opens
                # `transformer.cache_context(...)`), that cache was populated EMPTY --
                # before the per-block cache hooks above existed -- so a later
                # `_set_context` never reaches the freshly added block hooks and the
                # block forward raises "No context is set". Invalidate the stale cache on
                # the unwrapped module (the exact object the adapter's `cache_context`
                # targets) so the next context build rediscovers the new hooks. Safe/no-op
                # when no such registry exists yet (the common no-eval path).
                unwrapped = adapter.get_component_unwrapped(name)
                cache_hook = getattr(unwrapped, "_diffusers_hook", None)
                if cache_hook is not None:
                    cache_hook._child_registries_cache = None
                logger.info("DiffusersCacheAccelerator: cache enabled for '%s'.", name)
            yield
        finally:
            for transformer in enabled:
                transformer.disable_cache()
