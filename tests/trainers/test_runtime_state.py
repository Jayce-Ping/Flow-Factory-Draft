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

"""Tests for checkpointable trainer runtime state."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from flow_factory.trainers.common.runtime_state import (
    TRAINER_RUNTIME_METADATA_FILENAME,
    TRAINER_RUNTIME_STATE_VERSION,
    TRAINER_RUNTIME_TENSOR_PREFIX,
    TrainerRuntimeState,
)
from flow_factory.trainers.execution import TrainingProgress


class _RecordingChild:
    """Record every state restoration while exposing mutable test state."""

    def __init__(self, value: int) -> None:
        self.value = value
        self.loads: list[dict[str, Any]] = []

    def state_dict(self) -> dict[str, Any]:
        """Return the current value."""
        return {"value": self.value}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Record and restore one value."""
        payload = dict(state_dict)
        self.loads.append(payload)
        self.value = payload["value"]

    def validate_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Accept the narrow value mapping without mutation."""
        assert set(state_dict) == {"value"}


class _FailingChild(_RecordingChild):
    """Reject restoration to verify pending payload ownership."""

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Record one failed attempt without accepting the payload."""
        self.loads.append(dict(state_dict))
        raise RuntimeError("child restore failed")


class _InvalidChildState:
    """Expose the required methods but return an invalid state payload."""

    def state_dict(self) -> list[int]:
        """Return a non-mapping payload."""
        return [1]

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Accept state only to satisfy the structural interface."""
        del state_dict

    def validate_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Accept state only to satisfy the structural interface."""
        del state_dict


def _valid_state(*, child_names: tuple[str, ...] = ()) -> dict[str, Any]:
    """Return one valid serialized runtime state for mutation tests."""
    return {
        "version": TRAINER_RUNTIME_STATE_VERSION,
        "progress": {
            "optimizer_step": 7,
            "rollout_iteration": 3,
            "data_epoch": 2,
        },
        "children": {name: {"value": index + 10} for index, name in enumerate(child_names)},
    }


def test_state_dict_serializes_concrete_immutable_training_progress() -> None:
    """Progress remains a frozen value while the runtime replaces whole snapshots."""
    runtime = TrainerRuntimeState(
        TrainingProgress(optimizer_step=4, rollout_iteration=2, data_epoch=1)
    )

    assert runtime.state_dict() == {
        "version": TRAINER_RUNTIME_STATE_VERSION,
        "progress": {
            "optimizer_step": 4,
            "rollout_iteration": 2,
            "data_epoch": 1,
        },
        "children": {},
    }
    with pytest.raises(FrozenInstanceError):
        runtime.progress.optimizer_step = 5

    replacement = runtime.progress.advance_optimizer_step()
    runtime.progress = replacement
    assert runtime.progress == TrainingProgress(
        optimizer_step=5,
        rollout_iteration=2,
        data_epoch=1,
    )


def test_safe_file_round_trip_defers_named_children_until_attachment(
    tmp_path: Path,
) -> None:
    """JSON+safetensors restores progress and each late child payload exactly once."""
    checkpoint_dir = tmp_path / "state"
    source = TrainerRuntimeState(
        TrainingProgress(optimizer_step=8, rollout_iteration=5, data_epoch=3),
        child_names=("ema", "reference"),
    )
    source.attach_child("ema", _RecordingChild(21))
    source.attach_child("reference", _RecordingChild(34))
    source.prepare_save(checkpoint_dir)

    restored = TrainerRuntimeState(child_names=("ema", "reference"))
    restored.validate_load(
        checkpoint_dir,
        children={"ema": _RecordingChild(0), "reference": _RecordingChild(0)},
    )

    assert restored.progress == TrainingProgress()
    assert restored.validated_load_pending
    restored.commit_validated_load()

    assert restored.progress == TrainingProgress(
        optimizer_step=8,
        rollout_iteration=5,
        data_epoch=3,
    )
    assert restored.pending_child_names == ("ema", "reference")

    reference = _RecordingChild(0)
    ema = _RecordingChild(0)
    restored.attach_child("reference", reference)
    restored.attach_child("ema", ema)

    assert reference.value == 34
    assert reference.loads == [{"value": 34}]
    assert ema.value == 21
    assert ema.loads == [{"value": 21}]
    assert restored.pending_child_names == ()
    assert restored.state_dict() == source.state_dict()

    with pytest.raises(RuntimeError, match="already attached"):
        restored.attach_child("ema", _RecordingChild(0))
    assert ema.loads == [{"value": 21}]

    assert (checkpoint_dir / TRAINER_RUNTIME_METADATA_FILENAME).is_file()
    tensor_files = tuple(checkpoint_dir.glob(f"{TRAINER_RUNTIME_TENSOR_PREFIX}.*.safetensors"))
    assert len(tensor_files) == 1
    assert not tuple(checkpoint_dir.glob("custom_checkpoint_*.pkl"))


def test_failed_child_restore_retains_payload_for_a_successful_attachment() -> None:
    """A failed child construction cannot consume or silently discard resume state."""
    runtime = TrainerRuntimeState(child_names=("ema",))
    runtime.load_state_dict(_valid_state(child_names=("ema",)))
    failing = _FailingChild(0)

    with pytest.raises(RuntimeError, match="child restore failed"):
        runtime.attach_child("ema", failing)

    assert failing.loads == [{"value": 10}]
    assert runtime.pending_child_names == ("ema",)
    restored = _RecordingChild(0)
    runtime.attach_child("ema", restored)
    assert restored.loads == [{"value": 10}]
    assert runtime.pending_child_names == ()


def test_pending_child_payload_remains_serializable_before_construction() -> None:
    """A save between load and child construction preserves the deferred payload."""
    payload = _valid_state(child_names=("reference",))
    runtime = TrainerRuntimeState(child_names=("reference",))

    runtime.load_state_dict(payload)

    assert runtime.state_dict() == payload


def test_fresh_state_rejects_missing_or_invalid_child_state() -> None:
    """Declared children cannot silently disappear from a newly written checkpoint."""
    missing = TrainerRuntimeState(child_names=("ema",))
    with pytest.raises(RuntimeError, match="neither attached nor backed"):
        missing.state_dict()

    invalid = TrainerRuntimeState(child_names=("ema",))
    invalid.attach_child("ema", _InvalidChildState())
    with pytest.raises(TypeError, match="child 'ema' state as a mapping"):
        invalid.state_dict()


def test_child_declarations_and_interfaces_fail_fast() -> None:
    """Names and structural methods are validated before child ownership changes."""
    with pytest.raises(TypeError, match="child_names.*iterable"):
        TrainerRuntimeState(child_names="ema")
    with pytest.raises(TypeError, match="non-empty str"):
        TrainerRuntimeState(child_names=("",))
    with pytest.raises(ValueError, match="must be unique"):
        TrainerRuntimeState(child_names=("ema", "ema"))

    runtime = TrainerRuntimeState(child_names=("ema",))
    with pytest.raises(KeyError, match="was not declared"):
        runtime.attach_child("reference", _RecordingChild(0))
    with pytest.raises(TypeError, match="state_dict/load_state_dict"):
        runtime.attach_child("ema", object())


def test_load_is_single_use_and_must_precede_child_attachment() -> None:
    """The resume phase cannot replay a payload or overwrite a live child."""
    loaded = TrainerRuntimeState()
    loaded.load_state_dict(_valid_state())
    with pytest.raises(RuntimeError, match="already received"):
        loaded.load_state_dict(_valid_state())

    attached = TrainerRuntimeState(child_names=("ema",))
    child = _RecordingChild(1)
    attached.attach_child("ema", child)
    with pytest.raises(RuntimeError, match="must load before attaching children"):
        attached.load_state_dict(_valid_state(child_names=("ema",)))
    assert child.loads == []


@pytest.mark.parametrize(
    ("mutate", "error_type", "match"),
    [
        (lambda state: [], TypeError, "runtime state as a mapping"),
        (
            lambda state: {key: value for key, value in state.items() if key != "children"},
            ValueError,
            "runtime state keys mismatch",
        ),
        (
            lambda state: {**state, "unexpected": None},
            ValueError,
            "runtime state keys mismatch",
        ),
        (
            lambda state: {**state, "version": True},
            TypeError,
            "state version to be int",
        ),
        (
            lambda state: {**state, "version": TRAINER_RUNTIME_STATE_VERSION + 1},
            ValueError,
            "version mismatch",
        ),
        (
            lambda state: {**state, "progress": []},
            TypeError,
            "runtime progress as a mapping",
        ),
        (
            lambda state: {
                **state,
                "progress": {
                    key: value for key, value in state["progress"].items() if key != "data_epoch"
                },
            },
            ValueError,
            "runtime progress keys mismatch",
        ),
        (
            lambda state: {
                **state,
                "progress": {**state["progress"], "optimizer_step": False},
            },
            TypeError,
            "optimizer_step to be int",
        ),
        (
            lambda state: {
                **state,
                "progress": {**state["progress"], "data_epoch": -1},
            },
            ValueError,
            "data_epoch >= 0",
        ),
        (
            lambda state: {**state, "children": []},
            TypeError,
            "runtime children as a mapping",
        ),
        (
            lambda state: {**state, "children": {}},
            ValueError,
            "runtime children keys mismatch",
        ),
        (
            lambda state: {**state, "children": {"ema": []}},
            TypeError,
            "child 'ema' state as a mapping",
        ),
    ],
)
def test_load_rejects_malformed_schema_without_mutating_progress(mutate, error_type, match) -> None:
    """Every serialized layer is validated before restored progress becomes visible."""
    runtime = TrainerRuntimeState(
        TrainingProgress(optimizer_step=1),
        child_names=("ema",),
    )
    state = mutate(deepcopy(_valid_state(child_names=("ema",))))

    with pytest.raises(error_type, match=match):
        runtime.load_state_dict(state)

    assert runtime.progress == TrainingProgress(optimizer_step=1)
    assert runtime.pending_child_names == ()


def test_progress_assignment_requires_the_concrete_training_progress_type() -> None:
    """Mutable mappings and lookalike values cannot replace immutable progress."""
    runtime = TrainerRuntimeState()

    with pytest.raises(TypeError, match="progress to be TrainingProgress"):
        runtime.progress = {"optimizer_step": 1}  # type: ignore[assignment]
