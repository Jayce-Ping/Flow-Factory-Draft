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

"""Tests for algorithm-owned execution contracts in training arguments."""

from dataclasses import fields

import pytest

from flow_factory.contracts.execution import (
    ONLINE_EXECUTION_CONTRACT,
    ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
)
from flow_factory.hparams.training_args import (
    DiffusionOPDTrainingArguments,
    DMD2TrainingArguments,
    TDMR1TrainingArguments,
    TDMTrainingArguments,
    TrainingArguments,
    list_registered_training_args,
)
from flow_factory.trainers.registry import get_trainer_class, list_registered_trainers


def test_builtin_argument_and_trainer_registries_do_not_drift() -> None:
    """Every built-in algorithm pair declares equal execution semantics."""
    argument_registry = list_registered_training_args()
    trainer_registry = list_registered_trainers()

    assert set(argument_registry) == set(trainer_registry)
    for name, arguments_class in argument_registry.items():
        assert arguments_class.execution_contract == get_trainer_class(name).execution_contract


def test_reward_free_distillation_declares_feedback_independently() -> None:
    """Distillation remains generation acquisition while omitting runtime rewards."""
    for arguments_class in (
        DMD2TrainingArguments,
        TDMTrainingArguments,
        DiffusionOPDTrainingArguments,
    ):
        assert arguments_class.execution_contract is ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT

    assert TDMR1TrainingArguments.execution_contract is ONLINE_EXECUTION_CONTRACT


def test_execution_contract_is_not_a_serialized_configuration_field() -> None:
    """Users select semantics through trainer_type rather than raw contract fields."""
    arguments = TrainingArguments()

    assert "execution_contract" not in {field.name for field in fields(arguments)}
    assert "execution_contract" not in arguments.to_dict()


@pytest.mark.parametrize(
    "values",
    [
        {"execution_contract": "dataset"},
        {"extra_kwargs": {"execution_contract": "dataset"}},
    ],
)
def test_user_cannot_override_execution_contract(values: dict) -> None:
    """The selected algorithm owns its execution semantics."""
    with pytest.raises(ValueError, match="selected by trainer_type"):
        TrainingArguments.from_dict(values)
