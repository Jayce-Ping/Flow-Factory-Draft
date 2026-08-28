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

"""Tests for dependency-neutral algorithm execution semantics."""

from dataclasses import FrozenInstanceError, fields

import pytest

from flow_factory.contracts.execution import (
    OFFLINE_EXECUTION_CONTRACT,
    ONLINE_EXECUTION_CONTRACT,
    ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
    AcquisitionMode,
    ExecutionContract,
    FeedbackMode,
)


def test_execution_contract_has_only_orthogonal_algorithm_axes() -> None:
    """Loader and cycle policy are runtime derivations, not duplicate config axes."""
    assert tuple(field.name for field in fields(ExecutionContract)) == (
        "acquisition",
        "feedback",
    )


def test_predefined_contracts_distinguish_acquisition_from_feedback() -> None:
    """Reward-free distillation remains generated while offline data is dataset-owned."""
    assert ONLINE_EXECUTION_CONTRACT == ExecutionContract(
        acquisition=AcquisitionMode.GENERATION,
        feedback=FeedbackMode.RUNTIME_REWARD,
    )
    assert ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT == ExecutionContract(
        acquisition=AcquisitionMode.GENERATION,
        feedback=FeedbackMode.NONE,
    )
    assert OFFLINE_EXECUTION_CONTRACT == ExecutionContract(
        acquisition=AcquisitionMode.DATASET,
        feedback=FeedbackMode.NONE,
    )


def test_acquisition_and_feedback_are_independently_composable() -> None:
    """Future dataset algorithms may opt into runtime feedback without schema changes."""
    contract = ExecutionContract(
        acquisition=AcquisitionMode.DATASET,
        feedback=FeedbackMode.RUNTIME_REWARD,
    )

    assert contract.feedback is FeedbackMode.RUNTIME_REWARD


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"acquisition": "generation"}, "acquisition"),
        ({"feedback": "none"}, "feedback"),
    ],
)
def test_execution_contract_rejects_untyped_strings(kwargs, field_name: str) -> None:
    """Raw strings cannot silently enter an immutable algorithm contract."""
    values = {
        "acquisition": AcquisitionMode.GENERATION,
        "feedback": FeedbackMode.NONE,
    }
    values.update(kwargs)

    with pytest.raises(TypeError, match=field_name):
        ExecutionContract(**values)


def test_execution_contract_is_immutable() -> None:
    """Algorithm semantics cannot mutate after registry resolution."""
    with pytest.raises(FrozenInstanceError):
        OFFLINE_EXECUTION_CONTRACT.feedback = FeedbackMode.RUNTIME_REWARD
