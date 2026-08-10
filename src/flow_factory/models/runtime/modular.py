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
        """Return specs declared through pinned ModularPipeline's public API.

        Raises:
            TypeError: If the public spec API has an incompatible shape.
        """
        pretrained_component_names = getattr(self.pipeline, "pretrained_component_names", None)
        config_component_names = getattr(self.pipeline, "config_component_names", None)
        if not isinstance(pretrained_component_names, list) or not all(
            isinstance(name, str) and name for name in pretrained_component_names
        ):
            raise TypeError(
                "Modular pipeline pretrained_component_names must be a list of non-empty strings, "
                f"got {type(pretrained_component_names).__name__}: "
                f"{pretrained_component_names!r}."
            )
        if not isinstance(config_component_names, list) or not all(
            isinstance(name, str) and name for name in config_component_names
        ):
            raise TypeError(
                "Modular pipeline config_component_names must be a list of non-empty strings, "
                f"got {type(config_component_names).__name__}: {config_component_names!r}."
            )
        get_component_spec = getattr(self.pipeline, "get_component_spec", None)
        if not callable(get_component_spec):
            raise TypeError(
                "Modular pipeline must provide callable get_component_spec(name), "
                f"got {get_component_spec!r}."
            )

        declared_names = list(dict.fromkeys([*pretrained_component_names, *config_component_names]))
        return {name: get_component_spec(name) for name in declared_names}

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
                f"expected={missing}, received_specs={self.declared_component_names}, "
                f"materialized={self._materialized_component_names()}."
            )
