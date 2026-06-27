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

# src/flow_factory/acceleration/cache_dit.py
"""Lossy rollout-only feature caching via the optional ``cache-dit`` library.

`cache-dit <https://github.com/vipshop/cache-dit>`_ offers richer cache policies
(DBCache, TaylorSeer, ...) than diffusers' built-ins. It is an optional dependency
(``pip install flow-factory[acceleration]``); when absent, this accelerator raises
a clear install hint instead of failing at import (``constraints.md`` #22a).

Only valid in the rollout slot of a decoupled / distillation trainer — the
paradigm validator enforces this (``constraints.md`` #7).
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

from .abc import BaseAccelerator
from ..utils.logger_utils import setup_logger

try:
    import cache_dit
except ImportError:
    cache_dit = None

if TYPE_CHECKING:
    from ..models.abc import BaseAdapter

logger = setup_logger(__name__)


class CacheDitAccelerator(BaseAccelerator):
    """Enable cache-dit caching on the adapter pipeline during rollout.

    Lossy and rollout-scoped. All params are forwarded verbatim to
    ``cache_dit.enable_cache`` (see the cache-dit docs for policy options such as
    ``cache_type``/``Fn_compute_blocks``/``rdt``), and cleared on context exit so
    the Stage-6 training forward stays exact.

    cache-dit operates on ``adapter.pipeline``, whose ``transformer`` attribute is
    the unwrapped inner module; rollout forwards still route through it via the
    prepared bundle, so the cache hooks fire as expected.
    """

    safety = "lossy"
    stage = "rollout"

    def __init__(self, **params) -> None:
        super().__init__(**params)
        if cache_dit is None:
            raise ImportError(
                "CacheDitAccelerator requires the optional 'cache-dit' package. "
                "Install it with `pip install flow-factory[acceleration]` (or "
                "`pip install cache-dit`)."
            )

    @contextmanager
    def rollout_context(self, adapter: "BaseAdapter") -> Iterator[None]:
        pipeline = adapter.pipeline
        cache_dit.enable_cache(pipeline, **self.params)
        logger.info("CacheDitAccelerator: cache-dit enabled on the rollout pipeline.")
        try:
            yield
        finally:
            cache_dit.disable_cache(pipeline)
