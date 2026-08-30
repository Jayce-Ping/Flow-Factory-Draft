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

"""Role-aware distributed component loading and preparation."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Sequence, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import DistributedType

from ..utils.logger_utils import setup_logger
from .domain import ComponentRole, LoadPlan

logger = setup_logger(__name__)


def configure_backend_loading(accelerator: Accelerator, adapter_class: type) -> None:
    """Apply adapter capabilities before any pretrained component is loaded."""
    if accelerator.distributed_type != DistributedType.FSDP:
        return
    plugin = accelerator.state.fsdp_plugin
    if (
        getattr(plugin, "fsdp_version", 1) >= 2
        and plugin.cpu_ram_efficient_loading
        and not getattr(adapter_class, "supports_fsdp2_cpu_efficient_loading", False)
    ):
        plugin.cpu_ram_efficient_loading = False
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "false"
        logger.info(
            "Disabled FSDP2 CPU-efficient loading for adapter=%s; "
            "its eager/mixed component source requires replicated construction.",
            adapter_class.__name__,
        )


class BackendLoadRuntime:
    """Backend-owned loading policy for one immutable component plan."""

    def __init__(self, accelerator: Accelerator, plan: LoadPlan, adapter: Any) -> None:
        self.accelerator = accelerator
        self.plan = plan
        self.adapter = adapter
        self._verified_roots: set[str] = set()

    @contextmanager
    def load_scope(self, role: ComponentRole) -> Iterator[None]:
        """Enter the third-party loading scope for one component role."""
        yield

    def bootstrap_targets(self) -> None:
        """Prepare target state that must exist before distributed wrapping."""

    def components_loaded(
        self,
        components: Optional[Union[str, List[str]]],
    ) -> None:
        """Finalize newly resident non-target components."""

    def prepare(self, *objects: Any) -> Any:
        """Prepare target roots and related objects through Accelerate."""
        return self.accelerator.prepare(*objects)

    def _logical_names(
        self,
        components: Optional[Union[str, List[str]]],
        *,
        role: ComponentRole,
    ) -> List[str]:
        names = self.adapter._resolve_component_names(components)
        selected = []
        seen_roots = set()
        for name in names:
            request = self.plan.request_for_component(name)
            if request.role is not role or request.root in seen_roots:
                continue
            seen_roots.add(request.root)
            selected.append(request.root)
        return selected


class FSDPBackendLoadRuntime(BackendLoadRuntime):
    """FSDP loading semantics isolated from trainer business logic."""

    @property
    def cpu_ram_efficient_loading(self) -> bool:
        plugin = self.accelerator.state.fsdp_plugin
        return bool(plugin.cpu_ram_efficient_loading)

    @contextmanager
    def load_scope(self, role: ComponentRole) -> Iterator[None]:
        if role is ComponentRole.TARGET or not self.cpu_ram_efficient_loading:
            yield
            return

        previous = os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING")
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "false"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("FSDP_CPU_RAM_EFFICIENT_LOADING", None)
            else:
                os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = previous

    def prepare(self, *objects: Any) -> Any:
        """Prepare roots and apply adapter-requested FSDP2 communication policy."""
        self._extend_fsdp2_wrap_policy(objects)
        prepared = super().prepare(*objects)
        plugin = self.accelerator.state.fsdp_plugin
        if (getattr(plugin, "fsdp_version", 1) or 1) < 2 or not getattr(
            self.adapter, "fsdp2_use_default_stream_unshard", False
        ):
            return prepared

        prepared_objects = prepared if isinstance(prepared, (list, tuple)) else (prepared,)
        configured = []
        for original, candidate in zip(objects, prepared_objects):
            if not isinstance(original, nn.Module):
                continue
            configure = getattr(candidate, "_set_unshard_async_op", None)
            if not callable(configure):
                raise TypeError(
                    "FSDP2 default-stream unshard requires prepared modules to expose "
                    f"_set_unshard_async_op(), received {type(candidate).__name__}"
                )
            configure(True)
            configured.append(type(candidate).__name__)
        if not configured:
            raise TypeError("FSDP2 default-stream unshard requested without a prepared module")
        logger.info("Enabled FSDP2 default-stream unshard for roots: %s", configured)
        return prepared

    def _extend_fsdp2_wrap_policy(self, objects: Sequence[Any]) -> None:
        plugin = self.accelerator.state.fsdp_plugin
        additional = tuple(getattr(self.adapter, "fsdp2_additional_wrap_module_names", ()))
        if (getattr(plugin, "fsdp_version", 1) or 1) < 2 or not additional:
            return
        configured = getattr(plugin, "transformer_cls_names_to_wrap", None)
        if configured is None:
            configured = [
                name
                for candidate in objects
                if isinstance(candidate, nn.Module)
                for name in (getattr(candidate, "_no_split_modules", None) or ())
            ]
        plugin.transformer_cls_names_to_wrap = list(dict.fromkeys([*configured, *additional]))
        logger.info(
            "Extended FSDP2 transformer wrap modules: %s",
            plugin.transformer_cls_names_to_wrap,
        )

    def bootstrap_targets(self) -> None:
        if not self.cpu_ram_efficient_loading or self.accelerator.num_processes <= 1:
            return
        synchronized = []
        for request in self.plan:
            if request.role is not ComponentRole.TARGET:
                continue
            component = self.adapter.get_component(request.root)
            self._raise_if_any_rank(
                not isinstance(component, nn.Module),
                f"target root={request.root!r} is not a module before FSDP prepare",
            )
            buffers = tuple(component.named_buffers())
            self._raise_if_any_rank(
                any(buffer.is_meta for _, buffer in buffers),
                f"target root={request.root!r} has a meta buffer before FSDP prepare",
            )
            for _, buffer in buffers:
                original_device = buffer.device
                value = buffer.detach().to(self.accelerator.device)
                dist.broadcast(value, src=0)
                buffer.data = value.to(original_device)
            synchronized.append(request.root)
        self.accelerator.wait_for_everyone()
        logger.info("FSDP target buffers synchronized: %s", synchronized)

    def components_loaded(
        self,
        components: Optional[Union[str, List[str]]],
    ) -> None:
        if self.accelerator.num_processes <= 1:
            return
        roots = [
            root
            for root in self._logical_names(components, role=ComponentRole.AUXILIARY)
            if root not in self._verified_roots
        ]
        if not roots:
            return
        self._check_replica_fingerprints(roots)
        self._verified_roots.update(roots)

    def _check_replica_fingerprints(self, roots: Sequence[str]) -> None:
        checked = []
        for root in roots:
            managed = self.adapter._should_manage_device(root)
            component = self.adapter.get_component(root)
            self._raise_if_any_rank(
                not managed or not isinstance(component, nn.Module),
                f"auxiliary root={root!r} is not a runtime-managed module",
            )
            self._check_fingerprint(root, component)
            checked.append(root)
        self.accelerator.wait_for_everyone()
        logger.info("FSDP replica sampled fingerprints matched: %s", checked)

    def _check_fingerprint(self, root: str, component: nn.Module) -> None:
        tensors = (*component.parameters(), *component.buffers())
        self._raise_if_any_rank(
            any(tensor.is_meta for tensor in tensors),
            f"replicated root={root!r} has a meta tensor after loading",
        )
        terms = []
        for tensor_index, tensor in enumerate(tensors):
            if not tensor.is_floating_point() or tensor.numel() == 0:
                continue
            flat = tensor.detach().reshape(-1)
            indices = torch.tensor(
                [0, flat.numel() // 2, flat.numel() - 1],
                device=flat.device,
                dtype=torch.long,
            )
            sampled = flat.index_select(0, indices).to(torch.float64)
            weight = float(tensor_index + 1)
            terms.append(
                torch.stack(
                    (
                        sampled.sum() * weight,
                        sampled.abs().sum() * weight,
                        sampled.square().sum() * weight,
                    )
                )
            )
        fingerprint = (
            torch.stack(terms).sum(dim=0)
            if terms
            else torch.zeros(3, device=self.accelerator.device, dtype=torch.float64)
        )
        gathered = [torch.empty_like(fingerprint) for _ in range(self.accelerator.num_processes)]
        dist.all_gather(gathered, fingerprint)
        mismatched = [
            rank
            for rank, candidate in enumerate(gathered)
            if not torch.equal(candidate, gathered[0])
        ]
        if mismatched:
            raise RuntimeError(
                f"replicated root={root!r} sampled fingerprint differs from rank 0; "
                f"mismatched_ranks={mismatched}"
            )

    def _raise_if_any_rank(self, local_failure: bool, message: str) -> None:
        failed = torch.tensor(
            int(local_failure),
            device=self.accelerator.device,
            dtype=torch.int32,
        )
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError(message)


def build_backend_load_runtime(
    accelerator: Accelerator,
    plan: LoadPlan,
    adapter: Any,
) -> BackendLoadRuntime:
    """Build the backend strategy for one adapter load plan."""
    if accelerator.distributed_type == DistributedType.FSDP:
        return FSDPBackendLoadRuntime(accelerator, plan, adapter)
    return BackendLoadRuntime(accelerator, plan, adapter)
