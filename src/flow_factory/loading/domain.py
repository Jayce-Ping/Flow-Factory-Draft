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

"""Immutable physical-root ownership for model components."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class ComponentRole(str, Enum):
    TARGET = "target"
    AUXILIARY = "auxiliary"
    REWARD = "reward"
    HOST = "host"


def _label(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"expected str for ComponentDescriptor.{field}, "
            f"received {type(value).__name__}: {value!r}"
        )
    if not value:
        raise ValueError(f"expected non-empty ComponentDescriptor.{field}")
    return value


@dataclass(frozen=True)
class ComponentDescriptor:
    """A logical component routed to a physical ownership root."""

    name: str
    root: str
    role: ComponentRole
    path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _label(self.name, "name"))
        object.__setattr__(self, "root", _label(self.root, "root"))
        if not isinstance(self.role, ComponentRole):
            raise TypeError(
                "expected ComponentRole for ComponentDescriptor.role, "
                f"received {type(self.role).__name__}: {self.role!r}"
            )
        path = tuple(self.path.split(".")) if isinstance(self.path, str) else tuple(self.path)
        if any(not isinstance(segment, str) or not segment for segment in path):
            raise ValueError(f"expected non-empty string path segments, received {path!r}")
        object.__setattr__(self, "path", path)

    @property
    def physical_path(self) -> str:
        return ".".join((self.root, *self.path))


@dataclass(frozen=True)
class LoadRequest:
    """One exactly-once ownership request for a physical root."""

    root: str
    role: ComponentRole
    descriptors: Mapping[str, ComponentDescriptor]

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptors", MappingProxyType(dict(self.descriptors)))

    @property
    def logical_names(self) -> tuple[str, ...]:
        return tuple(self.descriptors)

    @property
    def routes(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {name: descriptor.path for name, descriptor in self.descriptors.items()}
        )


@dataclass(frozen=True)
class LoadPlan:
    """Exactly-once physical roots and their logical routes."""

    requests: Mapping[str, LoadRequest]
    descriptors: Mapping[str, ComponentDescriptor]

    def __post_init__(self) -> None:
        object.__setattr__(self, "requests", MappingProxyType(dict(self.requests)))
        object.__setattr__(self, "descriptors", MappingProxyType(dict(self.descriptors)))

    def __iter__(self) -> Iterator[LoadRequest]:
        return iter(self.requests.values())

    def __len__(self) -> int:
        return len(self.requests)

    @property
    def roots(self) -> tuple[str, ...]:
        return tuple(self.requests)

    @property
    def routes(self) -> Mapping[str, str]:
        return MappingProxyType(
            {name: descriptor.physical_path for name, descriptor in self.descriptors.items()}
        )

    def request_for_root(self, root: str) -> LoadRequest:
        try:
            return self.requests[root]
        except KeyError as error:
            raise KeyError(
                f"unknown physical root={root!r}; available={list(self.requests)}"
            ) from error

    def request_for_component(self, name: str) -> LoadRequest:
        try:
            descriptor = self.descriptors[name]
        except KeyError as error:
            raise KeyError(
                f"unknown logical component={name!r}; available={list(self.descriptors)}"
            ) from error
        return self.requests[descriptor.root]


def _root_role(root: str, descriptors: Mapping[str, ComponentDescriptor]) -> ComponentRole:
    roles = {descriptor.role for descriptor in descriptors.values()}
    if len(roles) == 1:
        return next(iter(roles))
    if roles == {ComponentRole.TARGET, ComponentRole.AUXILIARY}:
        return ComponentRole.TARGET
    raise ValueError(
        f"physical root={root!r} has incompatible roles=" f"{sorted(role.value for role in roles)}"
    )


class LoadPlanner:
    """Deduplicate logical component declarations by physical root."""

    def build(self, descriptors: Iterable[ComponentDescriptor]) -> LoadPlan:
        logical: dict[str, ComponentDescriptor] = {}
        for descriptor in descriptors:
            if not isinstance(descriptor, ComponentDescriptor):
                raise TypeError(
                    "expected ComponentDescriptor, "
                    f"received {type(descriptor).__name__}: {descriptor!r}"
                )
            previous = logical.get(descriptor.name)
            if previous is not None and previous != descriptor:
                raise ValueError(
                    f"logical component={descriptor.name!r} has conflicting descriptors"
                )
            logical[descriptor.name] = descriptor

        grouped: dict[str, dict[str, ComponentDescriptor]] = {}
        for descriptor in logical.values():
            grouped.setdefault(descriptor.root, {})[descriptor.name] = descriptor

        requests = {
            root: LoadRequest(
                root=root,
                role=_root_role(root, root_descriptors),
                descriptors=root_descriptors,
            )
            for root, root_descriptors in grouped.items()
        }
        return LoadPlan(requests=requests, descriptors=logical)
