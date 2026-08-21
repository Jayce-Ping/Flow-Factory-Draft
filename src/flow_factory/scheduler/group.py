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

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, Dict, Iterable, Tuple, Union


class SchedulerGroup(Mapping[str, Any]):
    """Expose an immutable ordered mapping of component schedulers.

    Args:
        schedulers: Ordered scheduler mapping or ordered name/scheduler pairs.
        primary_name: Name used by the legacy single-scheduler compatibility API.
    """

    def __init__(
        self,
        schedulers: Union[Mapping[str, Any], Iterable[Tuple[str, Any]]],
        *,
        primary_name: str,
    ) -> None:
        items = list(schedulers.items()) if isinstance(schedulers, Mapping) else list(schedulers)
        if not items:
            raise ValueError("expected a non-empty SchedulerGroup, received no schedulers")

        scheduler_map: Dict[str, Any] = {}
        for name, scheduler in items:
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "expected each SchedulerGroup component name to be a non-empty string, "
                    f"received {name!r}"
                )
            if name in scheduler_map:
                raise ValueError(
                    f"expected unique SchedulerGroup component names, received duplicate {name!r}"
                )
            scheduler_map[name] = scheduler

        if not isinstance(primary_name, str) or not primary_name:
            raise ValueError(
                "expected primary_name to be a non-empty string, " f"received {primary_name!r}"
            )
        if primary_name not in scheduler_map:
            raise ValueError(
                f"primary_name received unknown {primary_name!r}; expected one of "
                f"{tuple(scheduler_map)}"
            )
        for name, scheduler in scheduler_map.items():
            step = getattr(scheduler, "step", None)
            if not callable(step):
                raise TypeError(
                    f"expected scheduler-like object with callable step for component {name!r}, "
                    f"received {type(scheduler).__name__} without step"
                )
        self._schedulers = MappingProxyType(scheduler_map)
        self._names = tuple(scheduler_map)
        self._primary_name = primary_name

    def __getitem__(self, name: str) -> Any:
        return self._schedulers[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    @property
    def names(self) -> Tuple[str, ...]:
        """Return immutable component names in dispatch order."""
        return self._names

    @property
    def primary_name(self) -> str:
        """Return the primary scheduler component name."""
        return self._primary_name

    @property
    def primary(self) -> Any:
        """Return the primary scheduler."""
        return self._schedulers[self._primary_name]

    def _dispatch(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        for name in self._names:
            scheduler = self._schedulers[name]
            method = getattr(scheduler, method_name, None)
            if not callable(method):
                raise TypeError(
                    f"expected scheduler component {name!r} to provide callable "
                    f"{method_name}, received {type(scheduler).__name__}"
                )
            method(*args, **kwargs)

    def eval(self) -> None:
        """Switch every scheduler to evaluation mode in declared order."""
        self._dispatch("eval")

    def train(self, mode: bool = True) -> None:
        """Switch every scheduler to training mode in declared order.

        Args:
            mode: Training mode passed unchanged to each scheduler.
        """
        self._dispatch("train", mode=mode)

    def rollout(self, *args: Any, **kwargs: Any) -> None:
        """Switch every scheduler to rollout mode in declared order.

        Args:
            *args: Positional arguments passed unchanged to each scheduler.
            **kwargs: Keyword arguments passed unchanged to each scheduler.
        """
        self._dispatch("rollout", *args, **kwargs)

    def set_seed(self, seed: int) -> None:
        """Set every scheduler seed in declared order.

        Args:
            seed: Seed passed unchanged to each scheduler.
        """
        if not isinstance(seed, int):
            raise TypeError(
                f"expected int seed for SchedulerGroup.set_seed, received "
                f"{type(seed).__name__}: {seed!r}"
            )
        self._dispatch("set_seed", seed)
