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

"""Immutable model-role declarations and parameter ownership."""

import copy
from collections.abc import Iterable, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterator, List, Literal, Mapping, Optional, Tuple, cast

import torch
from peft import PeftModel

RoleName = Literal["generator", "fake", "surrogate", "reference"]
RoleStorageMode = Literal["lora", "full", "snapshot"]

_ROLE_ORDER: Tuple[RoleName, ...] = ("generator", "fake", "surrogate", "reference")
_STORAGE_MODES: Tuple[RoleStorageMode, ...] = ("lora", "full", "snapshot")


@dataclass(frozen=True)
class ModelRoleSpec:
    """Declare one immutable model role."""

    name: RoleName
    trainable: bool
    storage_mode: RoleStorageMode
    component_routes: Mapping[str, str]
    adapter_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate and detach declaration values."""
        if self.name not in _ROLE_ORDER:
            raise ValueError(
                f"expected one of {_ROLE_ORDER} for model role name, received {self.name!r}"
            )
        if not isinstance(self.trainable, bool):
            raise TypeError(
                "expected bool for model role trainable state "
                f"of role {self.name!r}, received {type(self.trainable).__name__}: "
                f"{self.trainable!r}"
            )
        if self.storage_mode not in _STORAGE_MODES:
            raise ValueError(
                f"expected storage_mode for role {self.name!r} to be one of "
                f"{_STORAGE_MODES}, received {self.storage_mode!r}"
            )
        if not isinstance(self.component_routes, Mapping):
            raise TypeError(
                f"expected component_routes for role {self.name!r} to be a mapping, "
                f"received {type(self.component_routes).__name__}: {self.component_routes!r}"
            )

        detached_routes: Dict[str, str] = {}
        for component_name, route_name in self.component_routes.items():
            if not isinstance(component_name, str):
                raise TypeError(
                    f"expected string component name for role {self.name!r}, received "
                    f"{type(component_name).__name__}: {component_name!r}"
                )
            if not component_name:
                raise ValueError(
                    f"expected a non-empty string component name for role {self.name!r}, "
                    f"received {component_name!r}"
                )
            if not isinstance(route_name, str):
                raise TypeError(
                    f"expected string route for role {self.name!r} component "
                    f"{component_name!r}, received {type(route_name).__name__}: "
                    f"{route_name!r}"
                )
            if not route_name:
                raise ValueError(
                    f"expected a non-empty string route for role {self.name!r} component "
                    f"{component_name!r}, received {route_name!r}"
                )
            detached_routes[component_name] = route_name
        if self.adapter_name is not None and not isinstance(self.adapter_name, str):
            raise TypeError(
                f"expected adapter_name for role {self.name!r} to be None or a string, "
                f"received {type(self.adapter_name).__name__}: {self.adapter_name!r}"
            )
        if self.adapter_name == "":
            raise ValueError(
                f"expected adapter_name for role {self.name!r} to be None or a non-empty "
                f"string, received {self.adapter_name!r}"
            )
        object.__setattr__(self, "component_routes", MappingProxyType(detached_routes))


@dataclass(frozen=True)
class RoleParameter:
    """Record one parameter owned by a model role."""

    role_name: RoleName
    component_name: str
    parameter_name: str
    parameter: torch.nn.Parameter


class ModelRoleRegistry:
    """Store immutable role declarations and identity-based parameter ownership.

    Args:
        adapter: Model adapter exposing canonical ``trainable_component_names``.
    """

    def __init__(self, adapter: "BaseAdapter") -> None:
        component_names = getattr(adapter, "trainable_component_names", None)
        if component_names is None:
            raise TypeError(
                "expected adapter to expose trainable_component_names, "
                f"received {type(adapter).__name__}"
            )
        if isinstance(component_names, (str, bytes)) or not isinstance(component_names, Iterable):
            raise TypeError(
                "expected adapter trainable_component_names to be a non-string iterable, "
                f"received {type(component_names).__name__}: {component_names!r} "
                f"from adapter type {type(adapter).__name__}"
            )
        canonical_components = tuple(component_names)
        if any(not isinstance(name, str) or not name for name in canonical_components):
            raise ValueError(
                "expected adapter trainable_component_names to contain non-empty strings, "
                f"received {canonical_components!r}"
            )
        if len(set(canonical_components)) != len(canonical_components):
            raise ValueError(
                "expected unique adapter trainable_component_names, "
                f"received {canonical_components!r}"
            )

        self._adapter = adapter
        self._canonical_components = canonical_components
        self._specs: Dict[RoleName, ModelRoleSpec] = {}
        self._parameter_records: Dict[RoleName, List[RoleParameter]] = {}
        self._parameter_owners: Dict[int, RoleParameter] = {}
        self._route_owners: Dict[str, Tuple[RoleName, str]] = {}
        self._bundle_members: Dict[str, torch.nn.Module] = {}
        self._parameter_emas: Dict[str, Dict[str, Any]] = {}
        self._active_role: Optional[RoleName] = None
        self._active_context: Optional[ExitStack] = None
        self._is_frozen = False

    def declare(self, spec: ModelRoleSpec) -> None:
        """Declare a model role.

        Args:
            spec: Immutable role declaration.
        """
        attempted_role = getattr(spec, "name", "<unknown>")
        self._require_mutable(attempted_role, "declare role")
        if not isinstance(spec, ModelRoleSpec):
            raise TypeError(
                "expected ModelRoleSpec for role declaration, "
                f"received {type(spec).__name__}: {spec!r}"
            )
        if spec.name in self._specs:
            raise ValueError(f"model role is already declared: {spec.name!r}")
        if not self._specs and spec.name != "generator":
            raise ValueError(
                "expected the first declared model role to be 'generator', "
                f"received {spec.name!r}"
            )
        if self._specs:
            previous_role = self.role_names[-1]
            if _ROLE_ORDER.index(spec.name) <= _ROLE_ORDER.index(previous_role):
                raise ValueError(
                    f"expected model role {spec.name!r} after {previous_role!r} to follow "
                    f"canonical order {_ROLE_ORDER}, received declaration order "
                    f"{self.role_names + (spec.name,)!r}"
                )
        if spec.name == "reference" and spec.trainable:
            raise ValueError(
                "expected reference role to be non-trainable, "
                f"received trainable={spec.trainable!r}"
            )
        if spec.name != "reference" and not spec.trainable:
            raise ValueError(
                f"expected role {spec.name!r} to be trainable, "
                f"received trainable={spec.trainable!r}"
            )

        declared_components = tuple(spec.component_routes)
        unknown_components = tuple(
            name for name in declared_components if name not in self._canonical_components
        )
        if unknown_components:
            raise ValueError(
                f"expected role {spec.name!r} component routes to use canonical trainable "
                f"components {self._canonical_components!r}, received unknown components "
                f"{unknown_components!r}"
            )
        if spec.name == "generator":
            expected_routes = {name: name for name in self._canonical_components}
            received_routes = dict(spec.component_routes)
            if received_routes != expected_routes:
                raise ValueError(
                    "expected generator to map every canonical trainable component to its "
                    f"canonical route {expected_routes!r}, received {received_routes!r}"
                )

        pending_routes: Dict[str, str] = {}
        for component_name, route_name in spec.component_routes.items():
            if route_name in pending_routes:
                raise ValueError(
                    f"route collision for route {route_name!r} in role {spec.name!r}: "
                    f"components {pending_routes[route_name]!r} and {component_name!r}"
                )
            if route_name in self._route_owners:
                owner_role, owner_component = self._route_owners[route_name]
                if owner_component != component_name:
                    raise ValueError(
                        f"route collision for route {route_name!r}: role {owner_role!r} "
                        f"component {owner_component!r} already owns it; role {spec.name!r} "
                        f"component {component_name!r} attempted to reuse it"
                    )
                shared_storage_is_valid = (
                    spec.storage_mode == "lora" and spec.adapter_name is not None
                ) or (
                    spec.name == "reference"
                    and not spec.trainable
                    and spec.storage_mode == "snapshot"
                )
                if not shared_storage_is_valid:
                    raise ValueError(
                        f"shared route {route_name!r} for canonical component "
                        f"{component_name!r} is invalid for role {spec.name!r} with "
                        f"storage_mode={spec.storage_mode!r}; route is already owned by role "
                        f"{owner_role!r}. Shared routes require a named LoRA adapter or the "
                        "frozen reference snapshot context."
                    )
            pending_routes[route_name] = component_name

        self._specs[spec.name] = spec
        self._parameter_records[spec.name] = []
        for route_name, component_name in pending_routes.items():
            self._route_owners.setdefault(route_name, (spec.name, component_name))

    def register_parameter(
        self,
        role_name: RoleName,
        component_name: str,
        parameter_name: str,
        parameter: torch.nn.Parameter,
    ) -> None:
        """Register one optimizer parameter with its owning role.

        Args:
            role_name: Declared trainable role.
            component_name: Canonical component containing the parameter.
            parameter_name: Name relative to the component.
            parameter: Parameter object whose identity is being registered.
        """
        self._require_mutable(role_name, "register parameter ownership")
        spec = self._get_declared_spec(role_name)
        if not spec.trainable:
            raise ValueError(
                f"role {role_name!r} is non-trainable and cannot own optimizer parameters"
            )
        if component_name not in spec.component_routes:
            raise KeyError(
                f"role {role_name!r} has no route for component {component_name!r}; "
                f"declared components are {tuple(spec.component_routes)!r}"
            )
        if not isinstance(parameter_name, str) or not parameter_name:
            raise ValueError(
                f"expected a non-empty parameter_name for role {role_name!r} component "
                f"{component_name!r}, received {parameter_name!r}"
            )
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError(
                f"expected torch.nn.Parameter for role {role_name!r} component "
                f"{component_name!r} parameter {parameter_name!r}, received "
                f"{type(parameter).__name__}: {parameter!r}"
            )

        existing_owner = self._parameter_owners.get(id(parameter))
        if existing_owner is not None:
            raise ValueError(
                f"parameter identity is already owned by role {existing_owner.role_name!r} "
                f"component {existing_owner.component_name!r} parameter "
                f"{existing_owner.parameter_name!r}; role {role_name!r} component "
                f"{component_name!r} parameter {parameter_name!r} attempted to register it"
            )
        record = RoleParameter(
            role_name=role_name,
            component_name=component_name,
            parameter_name=parameter_name,
            parameter=parameter,
        )
        self._parameter_records[role_name].append(record)
        self._parameter_owners[id(parameter)] = record

    def materialize(self, required_trainable_roles: Sequence[RoleName]) -> None:
        """Materialize and register all requested trainable model roles.

        Args:
            required_trainable_roles: Trainable roles to materialize, including generator.
        """
        required_roles = self._validate_required_trainable_roles(required_trainable_roles)
        self._require_mutable(required_roles, "materialize model roles")
        declared_trainable_roles = tuple(
            role_name for role_name, spec in self._specs.items() if spec.trainable
        )
        if required_roles != declared_trainable_roles:
            raise ValueError(
                "expected required_trainable_roles to exactly match declared trainable roles "
                f"{declared_trainable_roles!r}, received {required_roles!r}"
            )
        if "reference" not in self._specs:
            raise ValueError(
                "expected a declared frozen reference role before materialization, "
                f"received roles {self.role_names!r}"
            )

        self._bundle_members = self._canonical_bundle_members()
        generator_spec = self._specs["generator"]
        if generator_spec.storage_mode == "lora":
            self._materialize_lora_roles(required_roles)
        elif generator_spec.storage_mode == "full":
            self._materialize_full_roles(required_roles)
        else:
            raise ValueError(
                "expected generator storage_mode to be 'lora' or 'full', "
                f"received {generator_spec.storage_mode!r}"
            )
        self.activate("generator")
        self.freeze()

    @staticmethod
    def _validate_required_trainable_roles(
        required_trainable_roles: object,
    ) -> Tuple[RoleName, ...]:
        """Validate and detach a requested trainable-role sequence."""
        if isinstance(required_trainable_roles, (str, bytes)) or not isinstance(
            required_trainable_roles, Sequence
        ):
            raise TypeError(
                "expected required_trainable_roles to be a non-string sequence, "
                f"received {type(required_trainable_roles).__name__}: "
                f"{required_trainable_roles!r}"
            )
        for role_index, role_name in enumerate(required_trainable_roles):
            if not isinstance(role_name, str):
                raise TypeError(
                    "expected string role name in required_trainable_roles at "
                    f"index {role_index}, received {type(role_name).__name__}: {role_name!r}"
                )

        required_roles = tuple(required_trainable_roles)
        if not required_roles or required_roles[0] != "generator":
            raise ValueError(
                "expected required_trainable_roles to start with 'generator', "
                f"received {required_roles!r}"
            )
        if len(set(required_roles)) != len(required_roles):
            raise ValueError(
                "expected unique required_trainable_roles, "
                f"received duplicates in {required_roles!r}"
            )
        invalid_roles = tuple(
            role_name
            for role_name in required_roles
            if role_name not in ("generator", "fake", "surrogate")
        )
        if invalid_roles:
            raise ValueError(
                "expected required_trainable_roles to contain only "
                f"('generator', 'fake', 'surrogate'), received {invalid_roles!r} "
                f"in {required_roles!r}"
            )
        canonical_order = tuple(
            role_name
            for role_name in ("generator", "fake", "surrogate")
            if role_name in required_roles
        )
        if required_roles != canonical_order:
            raise ValueError(
                f"expected required_trainable_roles in canonical order {canonical_order!r}, "
                f"received {required_roles!r}"
            )
        return cast(Tuple[RoleName, ...], required_roles)

    def bundle_members(self) -> Dict[str, torch.nn.Module]:
        """Return detached bundle-route to module mappings.

        Returns:
            Materialized modules keyed by stable bundle route.
        """
        return dict(self._bundle_members)

    def activate(self, role_name: RoleName) -> None:
        """Activate a materialized role.

        Args:
            role_name: Declared role to activate.
        """
        spec = self._get_declared_spec(role_name)
        if not self._bundle_members:
            raise RuntimeError(
                f"cannot activate role {role_name!r}: model roles are not materialized"
            )

        previous_context = self._active_context
        if previous_context is not None:
            previous_context.close()
        self._active_context = None

        next_context = ExitStack()
        generator_storage_mode = self._specs["generator"].storage_mode
        if generator_storage_mode == "lora":
            if role_name == "reference":
                for component_name in self._canonical_components:
                    component = self._get_peft_component(component_name)
                    next_context.enter_context(component.disable_adapter())
            else:
                if spec.adapter_name is None:
                    raise ValueError(
                        f"expected named LoRA adapter for role {role_name!r}, "
                        f"received adapter_name={spec.adapter_name!r}"
                    )
                for component_name in self._canonical_components:
                    self._get_peft_component(component_name).set_adapter(spec.adapter_name)
        elif role_name == "reference":
            next_context.enter_context(self._adapter.use_ref_parameters())

        self._restore_trainable_role_parameters()
        self._active_context = next_context
        self._active_role = role_name

    @property
    def active_role(self) -> RoleName:
        """Return the currently active materialized role."""
        if self._active_role is None:
            raise RuntimeError("model roles are not materialized and no role is active")
        return self._active_role

    @contextmanager
    def use(self, role_name: RoleName) -> Iterator[None]:
        """Temporarily activate a role and restore the exact previous role.

        Args:
            role_name: Declared role to use inside the context.

        Yields:
            Control while the requested role is active.
        """
        previous_role = self.active_role
        self.activate(role_name)
        try:
            yield
        finally:
            self.activate(previous_role)

    @contextmanager
    def use_generator_for_export(self) -> Iterator[None]:
        """Activate only the canonical generator route for an export."""
        with self.use("generator"):
            yield

    def freeze(self) -> None:
        """Validate all declarations and prevent further mutation."""
        if self._is_frozen:
            return
        if not self._specs:
            raise ValueError(
                "expected a declared generator role before freezing, received no roles"
            )
        for role_name, spec in self._specs.items():
            if spec.trainable and not self._parameter_records[role_name]:
                raise ValueError(
                    f"trainable role {role_name!r} must own at least one parameter before freeze"
                )
        self._is_frozen = True

    @property
    def is_frozen(self) -> bool:
        """Return whether the registry rejects further mutation."""
        return self._is_frozen

    @property
    def role_names(self) -> Tuple[RoleName, ...]:
        """Return declared role names in declaration order."""
        return tuple(self._specs)

    def get_spec(self, role_name: RoleName) -> ModelRoleSpec:
        """Return an immutable role declaration.

        Args:
            role_name: Declared role name.

        Returns:
            Immutable declaration for the requested role.
        """
        return self._get_declared_spec(role_name)

    def parameters(self, role_name: RoleName) -> Tuple[torch.nn.Parameter, ...]:
        """Return parameters owned by a role.

        Args:
            role_name: Declared role name.

        Returns:
            Parameters in registration order.
        """
        records = self.parameter_records(role_name)
        return tuple(record.parameter for record in records)

    def parameter_records(self, role_name: RoleName) -> Tuple[RoleParameter, ...]:
        """Return immutable parameter ownership records.

        Args:
            role_name: Declared role name.

        Returns:
            Ownership records in registration order.
        """
        self._get_declared_spec(role_name)
        return tuple(self._parameter_records[role_name])

    def create_parameter_ema(self, role_name: RoleName, snapshot_name: str) -> None:
        """Create a detached named parameter snapshot for one trainable role.

        Args:
            role_name: Trainable role whose parameters initialize the snapshot.
            snapshot_name: Unique non-empty snapshot identifier.
        """
        spec = self._get_declared_spec(role_name)
        if not spec.trainable:
            raise ValueError(
                f"expected trainable role for parameter EMA, received role {role_name!r}"
            )
        if not isinstance(snapshot_name, str) or not snapshot_name:
            raise ValueError(
                "expected non-empty string snapshot_name for parameter EMA, "
                f"received {snapshot_name!r}"
            )
        if snapshot_name in self._parameter_emas:
            raise ValueError(f"parameter EMA snapshot already exists: {snapshot_name!r}")
        records = self.parameter_records(role_name)
        self._parameter_emas[snapshot_name] = {
            "role_name": role_name,
            "parameters": {
                f"{record.component_name}.{record.parameter_name}": record.parameter.detach().clone()
                for record in records
            },
            "update_count": 0,
        }

    @contextmanager
    def use_parameter_ema(self, snapshot_name: str) -> Iterator[None]:
        """Temporarily swap one parameter EMA into its existing model role.

        Args:
            snapshot_name: Existing snapshot identifier.

        Yields:
            Control while the snapshot tensors are installed.
        """
        snapshot = self._get_parameter_ema(snapshot_name)
        role_name = cast(RoleName, snapshot["role_name"])
        records = self.parameter_records(role_name)
        live_parameters = [record.parameter.detach().clone() for record in records]
        with self.use(role_name):
            try:
                for record in records:
                    key = f"{record.component_name}.{record.parameter_name}"
                    record.parameter.data.copy_(snapshot["parameters"][key])
                yield
            finally:
                for record, live in zip(records, live_parameters):
                    record.parameter.data.copy_(live)

    def update_parameter_ema(self, snapshot_name: str, decay: float) -> None:
        """Update one parameter EMA from its live role parameters.

        Args:
            snapshot_name: Existing snapshot identifier.
            decay: Finite EMA decay in ``[0, 1]``.
        """
        if isinstance(decay, bool) or not isinstance(decay, (int, float)):
            raise TypeError(
                "expected numeric decay for parameter EMA, "
                f"received {type(decay).__name__}: {decay!r}"
            )
        decay_value = float(decay)
        if not 0 <= decay_value <= 1:
            raise ValueError(f"expected parameter EMA decay in [0, 1], received {decay_value!r}")
        snapshot = self._get_parameter_ema(snapshot_name)
        role_name = cast(RoleName, snapshot["role_name"])
        for record in self.parameter_records(role_name):
            key = f"{record.component_name}.{record.parameter_name}"
            snapshot["parameters"][key].lerp_(record.parameter.detach(), 1 - decay_value)
        snapshot["update_count"] += 1

    def parameter_ema_tensors(self, snapshot_name: str) -> Tuple[torch.Tensor, ...]:
        """Return one snapshot in the owning role's parameter registration order."""
        snapshot = self._get_parameter_ema(snapshot_name)
        role_name = cast(RoleName, snapshot["role_name"])
        return tuple(
            snapshot["parameters"][f"{record.component_name}.{record.parameter_name}"]
            .detach()
            .clone()
            for record in self.parameter_records(role_name)
        )

    def parameter_ema_state_dict(self) -> Dict[str, Any]:
        """Return exact checkpoint state for parameter EMA snapshots."""
        snapshots = {
            snapshot_name: {
                "role_name": snapshot["role_name"],
                "parameters": {
                    name: tensor.detach().clone() for name, tensor in snapshot["parameters"].items()
                },
            }
            for snapshot_name, snapshot in self._parameter_emas.items()
        }
        return {
            "version": 1,
            "snapshots": snapshots,
            "update_counts": {
                snapshot_name: snapshot["update_count"]
                for snapshot_name, snapshot in self._parameter_emas.items()
            },
        }

    def load_parameter_ema_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore exact checkpoint state for declared parameter EMA snapshots.

        Args:
            state: State produced by :meth:`parameter_ema_state_dict`.
        """
        if not isinstance(state, Mapping):
            raise TypeError(
                "expected parameter EMA state as a mapping, "
                f"received {type(state).__name__}: {state!r}"
            )
        if set(state) != {"version", "snapshots", "update_counts"} or state["version"] != 1:
            raise ValueError(
                "expected parameter EMA state keys ('snapshots', 'update_counts', 'version') "
                f"and version 1, received keys={tuple(sorted(state))!r}, "
                f"version={state.get('version')!r}"
            )
        snapshots = state["snapshots"]
        update_counts = state["update_counts"]
        if not isinstance(snapshots, Mapping) or not isinstance(update_counts, Mapping):
            raise TypeError(
                "expected parameter EMA snapshots and update_counts as mappings, received "
                f"{type(snapshots).__name__} and {type(update_counts).__name__}"
            )
        expected_names = tuple(self._parameter_emas)
        if tuple(snapshots) != expected_names or tuple(update_counts) != expected_names:
            raise ValueError(
                f"expected parameter EMA snapshots {expected_names!r}, received "
                f"snapshots={tuple(snapshots)!r}, update_counts={tuple(update_counts)!r}"
            )
        for snapshot_name in expected_names:
            target = self._parameter_emas[snapshot_name]
            source = snapshots[snapshot_name]
            if not isinstance(source, Mapping) or source.get("role_name") != target["role_name"]:
                raise ValueError(
                    f"expected parameter EMA role {target['role_name']!r} for "
                    f"snapshot {snapshot_name!r}, received {source!r}"
                )
            source_parameters = source.get("parameters")
            if not isinstance(source_parameters, Mapping) or tuple(source_parameters) != tuple(
                target["parameters"]
            ):
                received = (
                    tuple(source_parameters)
                    if isinstance(source_parameters, Mapping)
                    else source_parameters
                )
                raise ValueError(
                    f"expected parameter names {tuple(target['parameters'])!r} for "
                    f"snapshot {snapshot_name!r}, received {received!r}"
                )
            for name, target_tensor in target["parameters"].items():
                source_tensor = source_parameters[name]
                if (
                    not isinstance(source_tensor, torch.Tensor)
                    or source_tensor.shape != target_tensor.shape
                    or source_tensor.dtype != target_tensor.dtype
                ):
                    raise ValueError(
                        f"expected tensor shape={tuple(target_tensor.shape)}, "
                        f"dtype={target_tensor.dtype} for parameter EMA "
                        f"{snapshot_name!r}/{name!r}, received {source_tensor!r}"
                    )
                target_tensor.copy_(source_tensor)
            update_count = update_counts[snapshot_name]
            if (
                not isinstance(update_count, int)
                or isinstance(update_count, bool)
                or update_count < 0
            ):
                raise ValueError(
                    f"expected non-negative int update count for parameter EMA "
                    f"{snapshot_name!r}, received {update_count!r}"
                )
            target["update_count"] = update_count

    def _get_parameter_ema(self, snapshot_name: str) -> Dict[str, Any]:
        """Return one parameter EMA snapshot or fail with available names."""
        if snapshot_name not in self._parameter_emas:
            raise KeyError(
                f"parameter EMA snapshot {snapshot_name!r} is not declared; "
                f"available snapshots are {tuple(self._parameter_emas)!r}"
            )
        return self._parameter_emas[snapshot_name]

    def resolve_route(self, role_name: RoleName, component_name: str) -> str:
        """Resolve a canonical component to its role-specific bundle route.

        Args:
            role_name: Declared role name.
            component_name: Canonical component name.

        Returns:
            Bundle route declared for the role and component.
        """
        spec = self._get_declared_spec(role_name)
        if component_name not in spec.component_routes:
            raise KeyError(
                f"role {role_name!r} has no route for component {component_name!r}; "
                f"declared components are {tuple(spec.component_routes)!r}"
            )
        return spec.component_routes[component_name]

    def metadata(self) -> Dict[str, Any]:
        """Return detached serializable role metadata.

        Returns:
            Role declarations and parameter names without parameter objects.
        """
        roles = []
        for role_name, spec in self._specs.items():
            records = self._parameter_records[role_name]
            roles.append(
                {
                    "name": spec.name,
                    "trainable": spec.trainable,
                    "storage_mode": spec.storage_mode,
                    "component_routes": dict(spec.component_routes),
                    "adapter_name": spec.adapter_name,
                    "parameters": [
                        {
                            "component_name": record.component_name,
                            "parameter_name": record.parameter_name,
                        }
                        for record in records
                    ],
                }
            )
        return {"is_frozen": self._is_frozen, "roles": roles}

    def training_state_dict(self) -> Dict[str, Any]:
        """Return versioned role metadata for training-state compatibility."""
        roles = []
        for role_name, spec in self._specs.items():
            roles.append(
                {
                    "name": role_name,
                    "trainable": spec.trainable,
                    "storage_mode": spec.storage_mode,
                    "component_routes": dict(spec.component_routes),
                    "adapter_name": spec.adapter_name,
                    "parameters": [
                        {
                            "component_name": record.component_name,
                            "parameter_name": record.parameter_name,
                            "shape": list(record.parameter.shape),
                        }
                        for record in self._parameter_records[role_name]
                    ],
                }
            )
        return {"version": 1, "roles": roles}

    def load_training_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate checkpoint role metadata without mutating model parameters.

        Args:
            state: Versioned role metadata produced by :meth:`training_state_dict`.
        """
        if not isinstance(state, Mapping):
            raise TypeError(
                "expected multi-role metadata state as a mapping, "
                f"received {type(state).__name__}: {state!r}"
            )
        expected_keys = {"version", "roles"}
        received_keys = set(state)
        if received_keys != expected_keys:
            raise ValueError(
                "multi-role metadata state keys mismatch: expected "
                f"{tuple(sorted(expected_keys))!r}, "
                f"received {tuple(sorted(received_keys))!r}"
            )
        received_version = state.get("version")
        if (
            not isinstance(received_version, int)
            or isinstance(received_version, bool)
            or received_version != 1
        ):
            raise ValueError(
                "multi-role metadata version mismatch: expected 1, "
                f"received {received_version!r}"
            )
        received_roles = state.get("roles")
        if not isinstance(received_roles, list):
            raise TypeError(
                "expected multi-role metadata roles as a list, "
                f"received {type(received_roles).__name__}: {received_roles!r}"
            )

        expected = self.training_state_dict()
        expected_roles = expected["roles"]
        expected_role_order = tuple(role["name"] for role in expected_roles)
        received_role_order = tuple(
            role.get("name") if isinstance(role, Mapping) else role for role in received_roles
        )
        if received_role_order != expected_role_order:
            raise ValueError(
                "multi-role metadata role order mismatch: expected "
                f"{expected_role_order!r}, received {received_role_order!r}"
            )
        for expected_role, received_role in zip(expected_roles, received_roles):
            role_name = expected_role["name"]
            if not isinstance(received_role, Mapping):
                raise TypeError(
                    f"expected multi-role metadata for role {role_name!r} as a mapping, "
                    f"received {type(received_role).__name__}: {received_role!r}"
                )
            expected_role_keys = set(expected_role)
            received_role_keys = set(received_role)
            if received_role_keys != expected_role_keys:
                raise ValueError(
                    f"multi-role metadata keys mismatch for role {role_name!r}: "
                    f"expected {tuple(sorted(expected_role_keys))!r}, "
                    f"received {tuple(sorted(received_role_keys))!r}"
                )
            for field_name in (
                "trainable",
                "storage_mode",
                "component_routes",
                "adapter_name",
                "parameters",
            ):
                expected_value = expected_role[field_name]
                received_value = received_role.get(field_name)
                if received_value != expected_value:
                    raise ValueError(
                        f"multi-role metadata {field_name} mismatch for role "
                        f"{role_name!r}: expected {expected_value!r}, "
                        f"received {received_value!r}"
                    )

    def _canonical_bundle_members(self) -> Dict[str, torch.nn.Module]:
        """Collect canonical bundle members without forcing unrelated components."""
        target_module_map = getattr(self._adapter, "target_module_map", None)
        component_names = (
            tuple(target_module_map)
            if isinstance(target_module_map, Mapping)
            else self._canonical_components
        )
        members: Dict[str, torch.nn.Module] = {}
        for component_name in component_names:
            component = self._adapter.get_component(component_name)
            if not isinstance(component, torch.nn.Module):
                raise TypeError(
                    f"expected torch.nn.Module for canonical component {component_name!r}, "
                    f"received {type(component).__name__}: {component!r}"
                )
            members[component_name] = component
        return members

    def _materialize_lora_roles(self, required_roles: Tuple[RoleName, ...]) -> None:
        """Add named PEFT adapters and register exact newly created parameters."""
        generator_spec = self._specs["generator"]
        if generator_spec.adapter_name != "default":
            raise ValueError(
                "expected generator LoRA adapter_name to be 'default', "
                f"received {generator_spec.adapter_name!r}"
            )
        for component_name in self._canonical_components:
            component = self._get_peft_component(component_name)
            if "default" not in component.peft_config:
                raise ValueError(
                    f"expected canonical PEFT component {component_name!r} to contain "
                    f"the existing 'default' adapter, received {tuple(component.peft_config)!r}"
                )
            component.set_adapter("default")
            for parameter_name, parameter in component.named_parameters():
                if parameter.requires_grad:
                    self.register_parameter("generator", component_name, parameter_name, parameter)

            for role_name in required_roles[1:]:
                spec = self._specs[role_name]
                if spec.adapter_name != role_name:
                    raise ValueError(
                        f"expected LoRA adapter_name for role {role_name!r} to equal the role "
                        f"name, received {spec.adapter_name!r}"
                    )
                parameter_ids_before = {id(parameter) for parameter in component.parameters()}
                component.add_adapter(
                    spec.adapter_name,
                    copy.deepcopy(component.peft_config["default"]),
                )
                new_parameters = [
                    (parameter_name, parameter)
                    for parameter_name, parameter in component.named_parameters()
                    if id(parameter) not in parameter_ids_before
                ]
                if not new_parameters:
                    raise RuntimeError(
                        f"expected add_adapter({spec.adapter_name!r}) to create parameters "
                        f"for component {component_name!r}, received none"
                    )
                for parameter_name, parameter in new_parameters:
                    self.register_parameter(role_name, component_name, parameter_name, parameter)

    def _materialize_full_roles(self, required_roles: Tuple[RoleName, ...]) -> None:
        """Deep-copy full trainable roles and preserve target-module freezing."""
        for component_name in self._canonical_components:
            generator = self._bundle_members[component_name]
            self._apply_full_freezing(component_name, generator)
            self._register_trainable_parameters("generator", component_name, generator)
            for role_name in required_roles[1:]:
                replica = copy.deepcopy(generator)
                self._apply_full_freezing(component_name, replica)
                route_name = self.resolve_route(role_name, component_name)
                if route_name == component_name:
                    raise ValueError(
                        f"expected distinct full-model route for role {role_name!r} component "
                        f"{component_name!r}, received shared route {route_name!r}"
                    )
                self._bundle_members[route_name] = replica
                self._register_trainable_parameters(role_name, component_name, replica)

    def _apply_full_freezing(self, component_name: str, component: torch.nn.Module) -> None:
        """Apply the adapter's configured full-model target-module map."""
        target_module_map = getattr(self._adapter, "target_module_map", None)
        if not isinstance(target_module_map, Mapping) or component_name not in target_module_map:
            raise ValueError(
                f"expected target_module_map entry for component {component_name!r}, "
                f"received {target_module_map!r}"
            )
        target_modules = target_module_map[component_name]
        component.requires_grad_(False)
        if target_modules == "all":
            component.requires_grad_(True)
            return
        if not target_modules:
            return
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        for parameter_name, parameter in component.named_parameters():
            if any(target_module in parameter_name for target_module in target_modules):
                parameter.requires_grad = True

    def _register_trainable_parameters(
        self,
        role_name: RoleName,
        component_name: str,
        component: torch.nn.Module,
    ) -> None:
        """Register every trainable parameter of one full-model role component."""
        for parameter_name, parameter in component.named_parameters():
            if parameter.requires_grad:
                self.register_parameter(role_name, component_name, parameter_name, parameter)

    def _restore_trainable_role_parameters(self) -> None:
        """Keep every identity-owned trainable-role parameter trainable."""
        for role_name, spec in self._specs.items():
            if spec.trainable:
                for record in self._parameter_records[role_name]:
                    record.parameter.requires_grad = True

    def _get_peft_component(self, component_name: str) -> PeftModel:
        """Return one canonical PEFT component with detailed validation."""
        component = self._bundle_members[component_name]
        if not isinstance(component, PeftModel):
            raise TypeError(
                f"expected PeftModel for LoRA component {component_name!r}, "
                f"received {type(component).__name__}: {component!r}"
            )
        return component

    def _get_declared_spec(self, role_name: RoleName) -> ModelRoleSpec:
        """Return a declaration or fail with available role context."""
        if role_name not in self._specs:
            raise KeyError(
                f"model role {role_name!r} is not declared; available roles are {self.role_names!r}"
            )
        return self._specs[role_name]

    def _require_mutable(self, role_name: object, operation: str) -> None:
        """Reject an attempted role mutation after freeze."""
        if self._is_frozen:
            raise RuntimeError(
                f"cannot {operation} for role {role_name!r}: model role registry is frozen"
            )
