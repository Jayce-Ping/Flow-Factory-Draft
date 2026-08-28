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

"""Immutable domain objects for planning component materialization."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, unique
from types import MappingProxyType


@unique
class ComponentRole(str, Enum):
    """Ownership role of one physical component root."""

    TARGET = "target"
    AUXILIARY = "auxiliary"
    REWARD = "reward"
    HOST = "host"


@unique
class ComponentStage(str, Enum):
    """Trainer stages in which a component must be available."""

    PREPROCESS = "preprocess"
    OPTIMIZE = "optimize"
    ROLLOUT = "rollout"
    REWARD = "reward"
    EVALUATE = "evaluate"


@unique
class MaterializationMode(str, Enum):
    """Amount of component state requested from a component source."""

    FULL = "full"
    META = "meta"
    CONFIG_ONLY = "config_only"


def _validated_label(value: object, *, field_name: str, owner: str) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"expected str for {owner}.{field_name}, " f"received {type(value).__name__}: {value!r}"
        )
    if not value or value != value.strip():
        raise ValueError(
            f"expected non-empty, trimmed str for {owner}.{field_name}, received {value!r}"
        )
    return value


def _normalized_path(path: object, *, owner: str) -> tuple[str, ...]:
    if isinstance(path, str):
        raw_segments: tuple[object, ...] = () if not path else tuple(path.split("."))
    elif isinstance(path, Iterable):
        raw_segments = tuple(path)
    else:
        raise TypeError(
            f"expected dotted str or iterable[str] for {owner}.path, "
            f"received {type(path).__name__}: {path!r}"
        )

    segments: list[str] = []
    for index, segment in enumerate(raw_segments):
        segments.append(
            _validated_label(
                segment,
                field_name=f"path[{index}]",
                owner=owner,
            )
        )
    return tuple(segments)


def _normalized_stages(stages: object, *, owner: str) -> frozenset[ComponentStage]:
    if isinstance(stages, (str, bytes)) or not isinstance(stages, Iterable):
        raise TypeError(
            f"expected iterable[ComponentStage] for {owner}.stages, "
            f"received {type(stages).__name__}: {stages!r}"
        )

    normalized: set[ComponentStage] = set()
    for stage in stages:
        if not isinstance(stage, ComponentStage):
            raise TypeError(
                f"expected ComponentStage entries for {owner}.stages, "
                f"received {type(stage).__name__}: {stage!r}"
            )
        normalized.add(stage)
    return frozenset(normalized)


@dataclass(frozen=True)
class ComponentDescriptor:
    """A logical component name routed to a physical ``root + path``."""

    name: str
    root: str
    role: ComponentRole
    stages: frozenset[ComponentStage]
    path: tuple[str, ...] = ()
    mode: MaterializationMode = MaterializationMode.FULL

    def __post_init__(self) -> None:
        owner = type(self).__name__
        object.__setattr__(
            self,
            "name",
            _validated_label(self.name, field_name="name", owner=owner),
        )
        object.__setattr__(
            self,
            "root",
            _validated_label(self.root, field_name="root", owner=owner),
        )
        if not isinstance(self.role, ComponentRole):
            raise TypeError(
                f"expected ComponentRole for {owner}.role, "
                f"received {type(self.role).__name__}: {self.role!r}"
            )
        if not isinstance(self.mode, MaterializationMode):
            raise TypeError(
                f"expected MaterializationMode for {owner}.mode, "
                f"received {type(self.mode).__name__}: {self.mode!r}"
            )
        object.__setattr__(self, "path", _normalized_path(self.path, owner=owner))
        object.__setattr__(
            self,
            "stages",
            _normalized_stages(self.stages, owner=owner),
        )

    @property
    def logical_name(self) -> str:
        """Return the public component name."""

        return self.name

    @property
    def physical_path(self) -> str:
        """Return the dot-separated route starting at the physical root."""

        return ".".join((self.root, *self.path))

    @property
    def materialization_mode(self) -> MaterializationMode:
        """Return the requested materialization mode."""

        return self.mode


def _frozen_descriptors(
    descriptors: object,
    *,
    owner: str,
) -> Mapping[str, ComponentDescriptor]:
    if not isinstance(descriptors, Mapping):
        raise TypeError(
            f"expected Mapping[str, ComponentDescriptor] for {owner}.descriptors, "
            f"received {type(descriptors).__name__}: {descriptors!r}"
        )

    copied: dict[str, ComponentDescriptor] = {}
    for name, descriptor in descriptors.items():
        _validated_label(name, field_name="descriptors key", owner=owner)
        if not isinstance(descriptor, ComponentDescriptor):
            raise TypeError(
                f"expected ComponentDescriptor for {owner}.descriptors[{name!r}], "
                f"received {type(descriptor).__name__}: {descriptor!r}"
            )
        if name != descriptor.name:
            raise ValueError(
                f"expected descriptor mapping key to equal logical name for {owner}; "
                f"received key={name!r}, descriptor.name={descriptor.name!r}"
            )
        copied[name] = descriptor
    return MappingProxyType(copied)


def _resolved_root_role(
    root: str,
    descriptors: Sequence[ComponentDescriptor],
) -> ComponentRole:
    roles = frozenset(descriptor.role for descriptor in descriptors)
    if len(roles) == 1:
        return next(iter(roles))
    if roles == frozenset({ComponentRole.TARGET, ComponentRole.AUXILIARY}):
        return ComponentRole.TARGET

    logical_names = sorted(descriptor.name for descriptor in descriptors)
    role_names = sorted(role.value for role in roles)
    raise ValueError(
        "expected one role per physical root, except TARGET may promote an AUXILIARY "
        f"root; received root={root!r}, roles={role_names}, logical_names={logical_names}"
    )


@dataclass(frozen=True)
class LoadRequest:
    """One exactly-once materialization request for a physical root."""

    root: str
    role: ComponentRole
    stages: frozenset[ComponentStage]
    mode: MaterializationMode
    descriptors: Mapping[str, ComponentDescriptor] = field(repr=False)

    def __post_init__(self) -> None:
        owner = type(self).__name__
        root = _validated_label(self.root, field_name="root", owner=owner)
        if not isinstance(self.role, ComponentRole):
            raise TypeError(
                f"expected ComponentRole for {owner}.role, "
                f"received {type(self.role).__name__}: {self.role!r}"
            )
        if not isinstance(self.mode, MaterializationMode):
            raise TypeError(
                f"expected MaterializationMode for {owner}.mode, "
                f"received {type(self.mode).__name__}: {self.mode!r}"
            )
        stages = _normalized_stages(self.stages, owner=owner)
        descriptors = _frozen_descriptors(self.descriptors, owner=owner)
        if not descriptors:
            raise ValueError(
                f"expected at least one descriptor for physical root {root!r}, received none"
            )

        for descriptor in descriptors.values():
            if descriptor.root != root:
                raise ValueError(
                    f"expected all descriptors for physical root {root!r}; "
                    f"received logical_name={descriptor.name!r}, root={descriptor.root!r}"
                )

        descriptor_values = tuple(descriptors.values())
        expected_role = _resolved_root_role(root, descriptor_values)
        if self.role is not expected_role:
            raise ValueError(
                f"expected role={expected_role.value!r} for physical root {root!r}, "
                f"received role={self.role.value!r}"
            )

        modes = frozenset(descriptor.mode for descriptor in descriptor_values)
        if modes != frozenset({self.mode}):
            raise ValueError(
                f"expected mode={self.mode.value!r} for every descriptor of root {root!r}; "
                f"received modes={sorted(mode.value for mode in modes)}"
            )

        expected_stages = frozenset(
            stage for descriptor in descriptor_values for stage in descriptor.stages
        )
        if stages != expected_stages:
            raise ValueError(
                f"expected merged stages={sorted(stage.value for stage in expected_stages)} "
                f"for physical root {root!r}; "
                f"received stages={sorted(stage.value for stage in stages)}"
            )

        object.__setattr__(self, "root", root)
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "descriptors", descriptors)

    @property
    def logical_names(self) -> tuple[str, ...]:
        """Return logical names served by this physical load."""

        return tuple(self.descriptors)

    @property
    def routes(self) -> Mapping[str, tuple[str, ...]]:
        """Return immutable logical-name to root-relative-path routes."""

        return MappingProxyType(
            {name: descriptor.path for name, descriptor in self.descriptors.items()}
        )

    @property
    def materialization_mode(self) -> MaterializationMode:
        """Return the requested materialization mode."""

        return self.mode


def _frozen_requests(requests: object, *, owner: str) -> Mapping[str, LoadRequest]:
    if not isinstance(requests, Mapping):
        raise TypeError(
            f"expected Mapping[str, LoadRequest] for {owner}.requests, "
            f"received {type(requests).__name__}: {requests!r}"
        )

    copied: dict[str, LoadRequest] = {}
    for root, request in requests.items():
        _validated_label(root, field_name="requests key", owner=owner)
        if not isinstance(request, LoadRequest):
            raise TypeError(
                f"expected LoadRequest for {owner}.requests[{root!r}], "
                f"received {type(request).__name__}: {request!r}"
            )
        if root != request.root:
            raise ValueError(
                f"expected request mapping key to equal physical root for {owner}; "
                f"received key={root!r}, request.root={request.root!r}"
            )
        copied[root] = request
    return MappingProxyType(copied)


@dataclass(frozen=True)
class LoadPlan:
    """Immutable physical-root requests and their logical component routes."""

    requests: Mapping[str, LoadRequest]
    descriptors: Mapping[str, ComponentDescriptor] = field(repr=False)

    def __post_init__(self) -> None:
        owner = type(self).__name__
        requests = _frozen_requests(self.requests, owner=owner)
        descriptors = _frozen_descriptors(self.descriptors, owner=owner)

        planned_descriptors: dict[str, ComponentDescriptor] = {}
        for request in requests.values():
            for name, descriptor in request.descriptors.items():
                if name in planned_descriptors:
                    previous = planned_descriptors[name]
                    raise ValueError(
                        f"expected logical component {name!r} in one physical request; "
                        f"received roots={previous.root!r} and {descriptor.root!r}"
                    )
                planned_descriptors[name] = descriptor

        if planned_descriptors != dict(descriptors):
            missing = sorted(set(descriptors) - set(planned_descriptors))
            unexpected = sorted(set(planned_descriptors) - set(descriptors))
            mismatched = sorted(
                name
                for name in set(descriptors) & set(planned_descriptors)
                if descriptors[name] != planned_descriptors[name]
            )
            raise ValueError(
                "expected LoadPlan.descriptors to exactly match request descriptors; "
                f"received missing={missing}, unexpected={unexpected}, "
                f"mismatched={mismatched}"
            )

        object.__setattr__(self, "requests", requests)
        object.__setattr__(self, "descriptors", descriptors)

    def __iter__(self) -> Iterator[LoadRequest]:
        """Iterate once over each physical-root request."""

        return iter(self.requests.values())

    def __len__(self) -> int:
        """Return the number of physical roots to materialize."""

        return len(self.requests)

    @property
    def roots(self) -> tuple[str, ...]:
        """Return physical roots in deterministic planning order."""

        return tuple(self.requests)

    @property
    def routes(self) -> Mapping[str, str]:
        """Return immutable logical-name to full physical-path routes."""

        return MappingProxyType(
            {name: descriptor.physical_path for name, descriptor in self.descriptors.items()}
        )

    def request_for_root(self, root: str) -> LoadRequest:
        """Resolve a declared physical root to its exactly-once request."""

        root = _validated_label(root, field_name="root", owner=type(self).__name__)
        if root not in self.requests:
            raise KeyError(
                f"expected a declared physical root, received {root!r}; "
                f"available={list(self.requests)}"
            )
        return self.requests[root]

    def request_for_component(self, name: str) -> LoadRequest:
        """Resolve a logical component or alias to its physical request."""

        name = _validated_label(name, field_name="name", owner=type(self).__name__)
        if name not in self.descriptors:
            raise KeyError(
                f"expected a declared logical component, received {name!r}; "
                f"available={list(self.descriptors)}"
            )
        return self.requests[self.descriptors[name].root]

    def requests_for_stage(self, stage: ComponentStage) -> tuple[LoadRequest, ...]:
        """Return physical requests needed by one trainer stage."""

        if not isinstance(stage, ComponentStage):
            raise TypeError(
                f"expected ComponentStage for LoadPlan.requests_for_stage, "
                f"received {type(stage).__name__}: {stage!r}"
            )
        return tuple(request for request in self.requests.values() if stage in request.stages)


class LoadPlanner:
    """Compile logical component declarations into immutable root requests."""

    __slots__ = ()

    def build(self, descriptors: Iterable[ComponentDescriptor]) -> LoadPlan:
        """Validate declarations, merge stages, and deduplicate physical roots."""

        if isinstance(descriptors, (str, bytes, Mapping)) or not isinstance(descriptors, Iterable):
            raise TypeError(
                "expected iterable[ComponentDescriptor] for LoadPlanner.build, "
                f"received {type(descriptors).__name__}: {descriptors!r}"
            )

        logical_descriptors: dict[str, ComponentDescriptor] = {}
        for index, descriptor in enumerate(descriptors):
            if not isinstance(descriptor, ComponentDescriptor):
                raise TypeError(
                    "expected ComponentDescriptor entries for LoadPlanner.build, "
                    f"received index={index}, type={type(descriptor).__name__}, "
                    f"value={descriptor!r}"
                )

            previous = logical_descriptors.get(descriptor.name)
            if previous is None:
                logical_descriptors[descriptor.name] = descriptor
                continue

            conflicts = {
                field_name: (getattr(previous, field_name), getattr(descriptor, field_name))
                for field_name in ("root", "path", "role", "mode")
                if getattr(previous, field_name) != getattr(descriptor, field_name)
            }
            if conflicts:
                raise ValueError(
                    f"expected one route, role, and mode for logical component "
                    f"{descriptor.name!r}; received conflicts={conflicts!r}"
                )
            logical_descriptors[descriptor.name] = replace(
                previous,
                stages=previous.stages | descriptor.stages,
            )

        grouped: dict[str, dict[str, ComponentDescriptor]] = {}
        for descriptor in logical_descriptors.values():
            grouped.setdefault(descriptor.root, {})[descriptor.name] = descriptor

        requests: dict[str, LoadRequest] = {}
        for root, root_descriptors in grouped.items():
            descriptor_values = tuple(root_descriptors.values())
            modes = frozenset(descriptor.mode for descriptor in descriptor_values)
            if len(modes) != 1:
                raise ValueError(
                    f"expected one materialization mode for physical root {root!r}; "
                    f"received modes={sorted(mode.value for mode in modes)}, "
                    f"logical_names={sorted(root_descriptors)}"
                )
            mode = next(iter(modes))
            role = _resolved_root_role(root, descriptor_values)
            stages = frozenset(
                stage for descriptor in descriptor_values for stage in descriptor.stages
            )
            requests[root] = LoadRequest(
                root=root,
                role=role,
                stages=stages,
                mode=mode,
                descriptors=root_descriptors,
            )

        return LoadPlan(
            requests=requests,
            descriptors=logical_descriptors,
        )

    def plan(self, descriptors: Iterable[ComponentDescriptor]) -> LoadPlan:
        """Alias for :meth:`build` for call sites that prefer planner terminology."""

        return self.build(descriptors)
