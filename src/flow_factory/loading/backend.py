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

    def bootstrap_targets(self) -> None:
        if not self.cpu_ram_efficient_loading or self.accelerator.num_processes <= 1:
            return
        synchronized = []
        for request in self.plan:
            if request.role is not ComponentRole.TARGET:
                continue
            component = self.adapter.get_component(request.root)
            if not isinstance(component, nn.Module):
                continue
            for buffer_name, buffer in component.named_buffers():
                if buffer.is_meta:
                    raise RuntimeError(
                        f"expected materialized target root={request.root!r} "
                        f"buffer={buffer_name!r}, received meta tensor"
                    )
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
        roots = self._logical_names(components, role=ComponentRole.AUXILIARY)
        self._verify_replicas(roots)

    def _verify_replicas(self, roots: Sequence[str]) -> None:
        verified = []
        for root in roots:
            if not self.adapter._should_manage_device(root):
                continue
            component = self.adapter.get_component(root)
            if isinstance(component, nn.Module):
                self._verify_fingerprint(root, component)
                verified.append(root)
        self.accelerator.wait_for_everyone()
        logger.info("FSDP replicated components verified: %s", verified)

    def _verify_fingerprint(self, root: str, component: nn.Module) -> None:
        terms = []
        for tensor_index, tensor in enumerate((*component.parameters(), *component.buffers())):
            if tensor.is_meta:
                raise RuntimeError(
                    f"expected materialized replicated root={root!r}, received meta tensor"
                )
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
                f"replicated root={root!r} differs from rank 0 after loading; "
                f"mismatched_ranks={mismatched}"
            )


def build_backend_load_runtime(
    accelerator: Accelerator,
    plan: LoadPlan,
    adapter: Any,
) -> BackendLoadRuntime:
    """Build the backend strategy for one adapter load plan."""
    if accelerator.distributed_type == DistributedType.FSDP:
        return FSDPBackendLoadRuntime(accelerator, plan, adapter)
    return BackendLoadRuntime(accelerator, plan, adapter)
