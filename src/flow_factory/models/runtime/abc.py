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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Union

import torch


class ComponentRuntime(ABC):
    """Manage canonical and distributed-prepared model components.

    Args:
        pipeline: Backend pipeline or explicit component container.
    """

    def __init__(self, pipeline: Any) -> None:
        if pipeline is None:
            raise ValueError("ComponentRuntime requires a pipeline/container instance, got None.")
        self.pipeline = pipeline
        self.prepared_components: Dict[str, Any] = {}

    @property
    @abstractmethod
    def canonical_components(self) -> Mapping[str, Any]:
        """Return canonical component specifications keyed by component name."""
        pass

    @property
    def component_names(self) -> List[str]:
        """Return canonical module/spec names in deterministic order."""
        return sorted(self.canonical_components)

    @property
    def text_encoder_names(self) -> List[str]:
        """Return canonical text encoder component names."""
        return [name for name in self.component_names if "text_encoder" in name]

    @property
    def transformer_names(self) -> List[str]:
        """Return canonical transformer component names."""
        return [name for name in self.component_names if "transformer" in name]

    def set_prepared_component(self, name: str, module: Any) -> None:
        """Install an accelerator-prepared component override.

        Args:
            name: Canonical component name to override.
            module: Prepared module or routed proxy.
        """
        if not name:
            raise ValueError("Prepared component name must be non-empty.")
        if module is None:
            raise ValueError(
                f"Prepared component '{name}' must be a module or routed proxy, got None."
            )
        self.prepared_components[name] = module

    def is_prepared(self, name: str) -> bool:
        """Return whether a component has an accelerator-prepared override.

        Args:
            name: Canonical component name.

        Returns:
            True when a prepared override is installed.
        """
        return name in self.prepared_components

    def get_component(self, name: str) -> Any:
        """Return a component with prepared-over-canonical precedence.

        Args:
            name: Canonical component name.

        Returns:
            Prepared override when present, otherwise the canonical component.
        """
        if self.is_prepared(name):
            return self.prepared_components[name]
        return self.get_canonical_component(name)

    def get_canonical_component(self, name: str) -> Any:
        """Return the canonical backend component, materializing it if needed.

        Args:
            name: Canonical component name.

        Returns:
            Canonical backend component.
        """
        self.materialize_components([name])
        component = self._get_materialized_component(name)
        if component is None:
            raise RuntimeError(
                f"Canonical component '{name}' remained unavailable after materialization; "
                f"expected={[name]}, received={self._materialized_component_names()}."
            )
        return component

    def resolve_component_names(
        self,
        components: Optional[Union[str, List[str]]] = None,
    ) -> List[str]:
        """Resolve concrete and group component names.

        Args:
            components: Component name, names, or ``None`` for all canonical names.
                The groups ``text_encoders`` and ``transformers`` are preserved.

        Returns:
            Deduplicated concrete component names in request order.
        """
        if components is None:
            return self.component_names
        requested = [components] if isinstance(components, str) else components
        resolved: List[str] = []
        for name in requested:
            if name == "text_encoders":
                resolved.extend(self.text_encoder_names)
            elif name == "transformers":
                resolved.extend(self.transformer_names)
            else:
                resolved.append(name)
        return list(dict.fromkeys(resolved))

    def materialize_components(
        self,
        components: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Materialize requested non-prepared canonical components.

        Args:
            components: Component name, names, groups, or ``None`` for all.
        """
        names = [
            name for name in self.resolve_component_names(components) if not self.is_prepared(name)
        ]
        if not names:
            return
        unknown = [name for name in names if name not in self.canonical_components]
        if unknown:
            raise ValueError(
                "Cannot materialize unknown components; "
                f"expected={unknown}, received={self.component_names}."
            )
        self._materialize_components(names)

    def load_components(
        self,
        components: Optional[Union[str, List[str]]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Materialize and move non-prepared components to a stage device.

        Args:
            components: Component name, names, groups, or ``None`` for all.
            device: Target device.
        """
        names = [
            name for name in self.resolve_component_names(components) if not self.is_prepared(name)
        ]
        self.materialize_components(names)
        for name in names:
            component = self._get_materialized_component(name)
            if component is None:
                raise RuntimeError(
                    f"Cannot load component '{name}' because it is not materialized; "
                    f"expected={[name]}, received={self._materialized_component_names()}."
                )
            if hasattr(component, "to"):
                component.to(device)

    def unload_components(
        self,
        components: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Move materialized non-prepared components to CPU.

        Args:
            components: Component name, names, groups, or ``None`` for all.
        """
        names = [
            name for name in self.resolve_component_names(components) if not self.is_prepared(name)
        ]
        for name in names:
            component = self._get_materialized_component(name)
            if component is not None and hasattr(component, "to"):
                component.to("cpu")

    @abstractmethod
    def _get_materialized_component(self, name: str) -> Any:
        """Return a canonical component without causing materialization."""
        pass

    @abstractmethod
    def _materialize_components(self, names: List[str]) -> None:
        """Materialize canonical components for the backend."""
        pass

    def _materialized_component_names(self) -> List[str]:
        """Return names that currently resolve to materialized components."""
        return [
            name
            for name in self.component_names
            if self._get_materialized_component(name) is not None
        ]
