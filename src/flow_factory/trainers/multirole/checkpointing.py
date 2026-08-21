"""Checkpoint metadata and custom state for multi-role trainers."""

import json
import os
from collections.abc import Mapping
from typing import Any

import torch

MULTIROLE_METADATA_FILENAME = "flow_factory_multirole_metadata.json"
MULTIROLE_METADATA_VERSION = 1
MULTIROLE_STATE_KEYS = {
    "version",
    "metadata",
    "coordinator",
    "trainer_step",
    "variant_snapshots",
}


class _MultiRoleCheckpointState:
    """Delegate Accelerate custom checkpoint state to one trainer."""

    def __init__(self, trainer: Any) -> None:
        self._trainer = trainer

    def state_dict(self) -> dict[str, Any]:
        """Return multi-role counters and defensive compatibility metadata."""
        return self._trainer._multirole_state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore multi-role counters after Accelerate restores prepared state."""
        self._trainer._load_multirole_state_dict(state)

    def prepare_save(self, output_dir: str) -> None:
        """Validate a closed boundary and write metadata before Accelerate saves."""
        try:
            custom_state = self.state_dict()
        except RuntimeError as error:
            raise RuntimeError(
                "cannot checkpoint invalid multi-role training state before "
                f"model/optimizer save; validation reported: {error}"
            ) from error
        accelerator = self._trainer.accelerator
        save_on_each_node = accelerator.project_configuration.save_on_each_node
        should_write_metadata = accelerator.is_main_process or (
            save_on_each_node and accelerator.is_local_main_process
        )
        if should_write_metadata:
            os.makedirs(output_dir, exist_ok=True)
            metadata_path = os.path.join(output_dir, MULTIROLE_METADATA_FILENAME)
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(custom_state["metadata"], metadata_file, indent=2, sort_keys=True)
                metadata_file.write("\n")

    def validate_load(self, input_dir: str) -> None:
        """Validate metadata before Accelerate can mutate prepared state."""
        metadata_path = os.path.join(input_dir, MULTIROLE_METADATA_FILENAME)
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                "multi-role metadata compatibility gate expected file "
                f"{metadata_path!r}, received missing file"
            )
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        self._trainer._validate_multirole_metadata(metadata)


class MultiRoleCheckpointingMixin:
    """Own the compatibility contract for multi-role prepared-state checkpoints."""

    def _multirole_metadata(self) -> dict[str, Any]:
        """Return deterministic metadata used to validate resume compatibility."""
        role_metadata = self.adapter.component_variant_registry.training_state_dict()
        update_plan = self._role_update_plan()
        return {
            "version": MULTIROLE_METADATA_VERSION,
            "roles": role_metadata["variants"],
            "optimizer_group_roles": [
                group.get("role_name") for group in self.optimizer.param_groups
            ],
            "update_plan": [
                {
                    "role_name": phase.role_name,
                    "repeats": phase.repeats,
                }
                for phase in update_plan.phases
            ],
        }

    def _validate_multirole_metadata(self, state: Mapping[str, Any]) -> None:
        """Validate all metadata that must match before prepared-state mutation."""
        if not isinstance(state, Mapping):
            raise TypeError(
                "expected multi-role metadata as a mapping, "
                f"received {type(state).__name__}: {state!r}"
            )
        expected_keys = {"version", "roles", "optimizer_group_roles", "update_plan"}
        received_keys = set(state)
        if received_keys != expected_keys:
            raise ValueError(
                "multi-role metadata keys mismatch: expected "
                f"{tuple(sorted(expected_keys))!r}, "
                f"received {tuple(sorted(received_keys))!r}"
            )
        received_version = state.get("version")
        if (
            not isinstance(received_version, int)
            or isinstance(received_version, bool)
            or received_version != MULTIROLE_METADATA_VERSION
        ):
            raise ValueError(
                "multi-role metadata version mismatch: expected "
                f"{MULTIROLE_METADATA_VERSION}, received {received_version!r}"
            )

        self.adapter.component_variant_registry.load_training_state_dict(
            {
                "version": received_version,
                "variants": state.get("roles"),
            }
        )
        expected = self._multirole_metadata()
        for field_name in ("optimizer_group_roles", "update_plan"):
            expected_value = expected[field_name]
            received_value = state.get(field_name)
            if received_value != expected_value:
                raise ValueError(
                    f"multi-role metadata {field_name} mismatch: expected "
                    f"{expected_value!r}, received {received_value!r}"
                )

    def _primary_role(self) -> str:
        """Return the role whose optimizer step defines this run's public step."""
        return self._required_trainable_roles()[0]

    def _multirole_state_dict(self) -> dict[str, Any]:
        """Return registered custom state without duplicating prepared state."""
        coordinator_state = self.role_optimization.state_dict()
        primary_role = self._primary_role()
        primary_step = coordinator_state["role_steps"][primary_role]
        if self.step != primary_step:
            raise RuntimeError(
                "multi-role checkpoint counter mismatch: expected trainer step to equal "
                f"{primary_role!r} role step {primary_step}, received trainer_step={self.step}"
            )
        return {
            "version": 1,
            "metadata": self._multirole_metadata(),
            "coordinator": coordinator_state,
            "trainer_step": self.step,
            "variant_snapshots": self.adapter.component_variant_registry.snapshot_state_dict(),
        }

    def _load_multirole_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore custom multi-role counters after complete validation."""
        if not isinstance(state, Mapping):
            raise TypeError(
                "expected registered multi-role state as a mapping, "
                f"received {type(state).__name__}: {state!r}"
            )
        received_keys = set(state)
        if received_keys != MULTIROLE_STATE_KEYS:
            raise ValueError(
                "registered multi-role state keys mismatch: expected "
                f"{tuple(sorted(MULTIROLE_STATE_KEYS))!r}, "
                f"received {tuple(sorted(received_keys))!r}"
            )
        state_version = state["version"]
        if (
            not isinstance(state_version, int)
            or isinstance(state_version, bool)
            or state_version != 1
        ):
            raise ValueError(
                "registered multi-role state version mismatch: expected 1, "
                f"received {state_version!r}"
            )
        self._validate_multirole_metadata(state["metadata"])
        trainer_step = state["trainer_step"]
        if not isinstance(trainer_step, int) or isinstance(trainer_step, bool) or trainer_step < 0:
            raise ValueError(
                "expected non-negative int trainer_step in registered multi-role state, "
                f"received {trainer_step!r}"
            )
        coordinator_state = state["coordinator"]
        if not isinstance(coordinator_state, Mapping):
            raise TypeError(
                "expected coordinator in registered multi-role state as a mapping, "
                f"received {type(coordinator_state).__name__}: {coordinator_state!r}"
            )
        role_steps = coordinator_state.get("role_steps")
        primary_role = self._primary_role()
        if not isinstance(role_steps, Mapping) or role_steps.get(primary_role) != trainer_step:
            received_primary_step = (
                role_steps.get(primary_role) if isinstance(role_steps, Mapping) else role_steps
            )
            raise ValueError(
                "registered multi-role counter mismatch: expected trainer_step "
                f"{trainer_step} to equal {primary_role!r} role step, "
                f"received {primary_role!r} step {received_primary_step!r}"
            )
        self.role_optimization.load_state_dict(coordinator_state)
        self.adapter.component_variant_registry.load_snapshot_state_dict(state["variant_snapshots"])
        self.step = trainer_step
        self.adapter.component_variant_registry.activate(
            self.adapter.component_variant_registry.base_variant
        )

    def _register_multirole_checkpointing(self) -> None:
        """Register Accelerate metadata gates and custom state for multi-role runs."""
        if len(self._required_trainable_roles()) <= 1:
            return
        if getattr(self, "_multirole_checkpoint_registered", False):
            raise RuntimeError(
                "cannot register multi-role checkpointing twice for trainer "
                f"{type(self).__name__}"
            )

        def save_metadata_hook(
            models: list[torch.nn.Module],
            weights: list[dict[str, torch.Tensor]],
            output_dir: str,
        ) -> None:
            del models, weights
            checkpoint_state.prepare_save(output_dir)

        def load_metadata_hook(models: list[torch.nn.Module], input_dir: str) -> None:
            del models
            checkpoint_state.validate_load(input_dir)

        checkpoint_state = _MultiRoleCheckpointState(self)
        self._multirole_checkpoint_state = checkpoint_state
        self.adapter._multirole_checkpoint_state = checkpoint_state
        self.accelerator.register_save_state_pre_hook(save_metadata_hook)
        self.accelerator.register_load_state_pre_hook(load_metadata_hook)
        self.accelerator.register_for_checkpointing(checkpoint_state)
        self._multirole_checkpoint_registered = True
