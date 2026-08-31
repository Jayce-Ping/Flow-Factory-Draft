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

from typing import Any, Dict, List, Mapping

import torch.nn as nn

from .abc import ComponentRuntime


class ClassicPipelineRuntime(ComponentRuntime):
    """Manage an eagerly loaded DiffusionPipeline-like backend."""

    @property
    def canonical_components(self) -> Mapping[str, Any]:
        """Return components declared by the eager pipeline.

        Raises:
            TypeError: If ``pipeline.components`` is not a mapping.
        """
        declared = getattr(self.pipeline, "components", {})
        if not isinstance(declared, Mapping):
            raise TypeError(
                f"Classic pipeline components must be a mapping, got {type(declared).__name__}."
            )
        return declared

    @property
    def declared_components(self) -> Mapping[str, Any]:
        """Return pipeline declarations plus role attributes used by groups."""
        components: Dict[str, Any] = dict(self.canonical_components)
        for name, component in vars(self.pipeline).items():
            if (
                not name.startswith("_")
                and isinstance(component, nn.Module)
                and ("text_encoder" in name or "transformer" in name)
            ):
                components.setdefault(name, component)
        return components

    def _get_materialized_component(self, name: str) -> Any:
        return getattr(self.pipeline, name, None)

    def physical_route(self, name: str) -> tuple[str, tuple[str, ...]]:
        """Collapse logical aliases that reference one canonical module object.

        Args:
            name: Declared logical component name to resolve.

        Returns:
            Canonical root name and an empty nested path. A declared null component retains its
            own logical root rather than aliasing through the ``None`` singleton.
        """
        self._validate_declared_names([name])
        component = self.declared_components[name]
        if component is None:
            return name, ()
        for canonical_name, canonical_component in self.canonical_components.items():
            if component is canonical_component:
                return canonical_name, ()
        return name, ()

    def _materialize_components(self, names: List[str]) -> None:
        missing = [
            name
            for name in names
            if self._get_materialized_component(name) is None
            and not self._allows_none_component(name)
        ]
        if missing:
            raise RuntimeError(
                "Eager pipeline is missing required canonical components; "
                f"expected={missing}, received={self._materialized_component_names()}."
            )

    def _allows_none_component(self, name: str) -> bool:
        return name in self.canonical_components and self.canonical_components[name] is None
