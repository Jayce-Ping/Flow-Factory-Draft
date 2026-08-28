"""Compile adapter declarations and coordinate backend-owned loading."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Iterable, List, Optional, Union

from accelerate import Accelerator

from .backend import BackendLoadRuntime, build_backend_load_runtime
from .domain import (
    ComponentDescriptor,
    ComponentRole,
    ComponentStage,
    LoadPlan,
    LoadPlanner,
    MaterializationMode,
)

_HOST_MARKERS = ("tokenizer", "processor", "scheduler")


def _expanded_names(adapter: Any, declarations: Iterable[str]) -> set[str]:
    names: set[str] = set()
    for declaration in declarations:
        names.update(adapter._resolve_component_names(declaration))
    return names


def build_adapter_load_plan(adapter: Any) -> LoadPlan:
    """Build one physical-root plan from an adapter's public lifecycle declarations."""
    runtime = adapter.component_runtime
    target_names = set(adapter.model_args.target_components)
    preprocess_names = _expanded_names(adapter, adapter.preprocessing_modules)
    inference_names = _expanded_names(adapter, adapter.inference_modules)

    descriptors = []
    for name in runtime.declared_component_names:
        root, path = runtime.physical_route(name)
        if name in target_names:
            role = ComponentRole.TARGET
        elif any(marker in name for marker in _HOST_MARKERS):
            role = ComponentRole.HOST
        else:
            role = ComponentRole.AUXILIARY

        stages = set()
        if name in target_names:
            stages.update((ComponentStage.OPTIMIZE, ComponentStage.ROLLOUT))
        if name in preprocess_names:
            stages.add(ComponentStage.PREPROCESS)
        if name in inference_names:
            stages.update((ComponentStage.ROLLOUT, ComponentStage.EVALUATE))

        descriptors.append(
            ComponentDescriptor(
                name=name,
                root=root,
                path=path,
                role=role,
                stages=stages,
                mode=(
                    MaterializationMode.CONFIG_ONLY
                    if role is ComponentRole.HOST
                    else MaterializationMode.FULL
                ),
            )
        )
    return LoadPlanner().build(descriptors)


class ModelLoadCoordinator:
    """Single trainer-facing facade for component and backend loading."""

    def __init__(self, adapter: Any, accelerator: Accelerator) -> None:
        self.adapter = adapter
        self.accelerator = accelerator
        self.plan = build_adapter_load_plan(adapter)
        self.backend: BackendLoadRuntime = build_backend_load_runtime(
            accelerator,
            self.plan,
            adapter,
        )

    def load_scope(self, role: ComponentRole) -> AbstractContextManager[None]:
        return self.backend.load_scope(role)

    def bootstrap_targets(self) -> None:
        self.backend.bootstrap_targets()

    def components_loaded(
        self,
        components: Optional[Union[str, List[str]]],
    ) -> None:
        self.backend.components_loaded(components)

    def load_components(
        self,
        components: Optional[Union[str, List[str]]],
        *,
        device: Any,
    ) -> None:
        """Materialize replicated stage components inside the backend load scope."""
        with self.load_scope(ComponentRole.AUXILIARY):
            self.adapter.on_load_components(components=components, device=device)
        self.components_loaded(components)

    def prepare(self, *objects: Any) -> Any:
        return self.backend.prepare(*objects)
