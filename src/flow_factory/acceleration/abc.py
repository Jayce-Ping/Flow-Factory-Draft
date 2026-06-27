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

# src/flow_factory/acceleration/abc.py
"""Abstract base class for the model-agnostic acceleration plugin layer.

An accelerator is a pluggable speedup applied to a model adapter's transformer(s)
without touching trainer or model math. Each accelerator declares two invariants
that the validator (:mod:`flow_factory.acceleration.validator`) enforces against
the trainer's RL paradigm:

* ``safety`` — ``"lossless"`` (numerically ~identical; safe for any algorithm and
  any stage because the same module backs both rollout ``inference()`` and the
  training ``forward()``) or ``"lossy"`` (changes ``noise_pred``; only safe during
  rollout for **decoupled / distillation** algorithms — see ``constraints.md`` #7
  and #20a).
* ``stage`` — ``"both"`` (a one-time mutation applied via :meth:`setup`, e.g.
  ``torch.compile``) or ``"rollout"`` (a per-epoch context applied via
  :meth:`rollout_context`, e.g. feature caching).

Subclasses implement only what they need: ``setup`` defaults to a no-op,
``rollout_context`` defaults to yielding without modification.
"""

from __future__ import annotations

from abc import ABC
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Literal

if TYPE_CHECKING:
    from ..models.abc import BaseAdapter


class BaseAccelerator(ABC):
    """Base class for all acceleration plugins.

    Subclasses MUST define the ``safety`` and ``stage`` class attributes. They
    are validated at construction time by ``__init_subclass__`` so a misdeclared
    accelerator fails fast at import rather than mid-training.

    Args:
        **params: Accelerator-specific parameters forwarded verbatim from the
            ``acceleration`` config block (e.g. ``mode`` for the compile
            accelerator). Stored on ``self.params``.
    """

    safety: ClassVar[Literal["lossless", "lossy"]]
    stage: ClassVar[Literal["rollout", "both"]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Skip intermediate ABCs that intentionally leave the markers unset.
        if getattr(cls, "__abstractmethods__", None):
            return
        for attr in ("safety", "stage"):
            if not hasattr(cls, attr):
                raise TypeError(
                    f"Accelerator '{cls.__name__}' must define the class attribute "
                    f"'{attr}'. Declare it on the class body (e.g. `safety = 'lossless'`)."
                )
        if cls.safety not in ("lossless", "lossy"):
            raise TypeError(
                f"Accelerator '{cls.__name__}' has invalid safety={cls.safety!r}; "
                "expected 'lossless' or 'lossy'."
            )
        if cls.stage not in ("rollout", "both"):
            raise TypeError(
                f"Accelerator '{cls.__name__}' has invalid stage={cls.stage!r}; "
                "expected 'rollout' or 'both'."
            )

    def __init__(self, **params: Any) -> None:
        self.params = params

    def setup(self, adapter: "BaseAdapter") -> None:
        """Apply a one-time mutation to the adapter's prepared transformer(s).

        Called once from ``BaseTrainer._initialization`` after
        ``accelerator.prepare()`` and after the routing proxies are installed, so
        attribute access (``compile``, ``set_attention_backend``, ...) reaches the
        inner module while forwards still route through the prepared bundle root.

        Args:
            adapter: The model adapter whose transformer(s) to accelerate.
        """
        return None

    @contextmanager
    def rollout_context(self, adapter: "BaseAdapter") -> Iterator[None]:
        """Wrap one epoch of rollout (Stage 3) with stage-scoped acceleration.

        The default implementation is a no-op. Stateful accelerators (feature
        caching) enable their state on enter and tear it down on exit so nothing
        leaks into the Stage-6 training forward.

        Args:
            adapter: The model adapter being sampled from.

        Yields:
            ``None``; the rollout loop runs inside the ``with`` block.
        """
        yield
