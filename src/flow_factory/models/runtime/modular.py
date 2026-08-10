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

from typing import Any, List, Mapping

from .abc import ComponentRuntime


class ModularPipelineRuntime(ComponentRuntime):
    """Manage a lazy modular pipeline without importing model-specific classes."""

    @property
    def canonical_components(self) -> Mapping[str, Any]:
        """Return modular component specifications exposed by the pipeline."""
        components = getattr(self.pipeline, "components", None)
        if not isinstance(components, Mapping):
            raise TypeError(
                "Modular pipeline components must be a mapping of lazy specifications, "
                f"got {type(components).__name__}."
            )
        return components

    def _get_materialized_component(self, name: str) -> Any:
        return getattr(self.pipeline, name, None)

    def _materialize_components(self, names: List[str]) -> None:
        unloaded = [name for name in names if self._get_materialized_component(name) is None]
        if unloaded:
            load_components = getattr(self.pipeline, "load_components", None)
            if not callable(load_components):
                raise RuntimeError(
                    "Modular pipeline cannot materialize required components because "
                    f"`load_components(names=...)` is unavailable; expected={unloaded}, "
                    f"received={self._materialized_component_names()}."
                )
            load_components(names=unloaded)

        missing = [name for name in names if self._get_materialized_component(name) is None]
        if missing:
            raise RuntimeError(
                "Modular component materialization failed; "
                f"expected={missing}, received_specs={self.component_names}, "
                f"materialized={self._materialized_component_names()}."
            )
