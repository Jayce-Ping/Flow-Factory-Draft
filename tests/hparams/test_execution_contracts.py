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

"""Execution semantics declared by algorithm argument classes."""

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from flow_factory.contracts.execution import (
    ONLINE_EXECUTION_CONTRACT,
    ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
    ExecutionContract,
)
from flow_factory.hparams import Arguments, SFTTrainingArguments, get_training_args_class
from flow_factory.hparams.training_args import (
    DiffusionOPDTrainingArguments,
    DMD2TrainingArguments,
    TDMR1TrainingArguments,
    TDMTrainingArguments,
    TrainingArguments,
    list_registered_training_args,
)
from flow_factory.models.abc import BaseAdapter
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.loader import load_trainer
from flow_factory.trainers.registry import get_trainer_class, list_registered_trainers


def test_builtin_argument_and_trainer_registries_declare_matching_contracts() -> None:
    """The two existing registries cannot silently drift on execution mode."""
    trainer_names = set(list_registered_trainers())
    argument_names = set(list_registered_training_args())

    assert trainer_names == argument_names

    for trainer_name in trainer_names:
        arguments_class = get_training_args_class(trainer_name)
        trainer_class = get_trainer_class(trainer_name)

        assert arguments_class.execution_contract == trainer_class.execution_contract


def test_reward_free_distillation_arguments_declare_feedback_independently() -> None:
    """Distillation keeps rollout acquisition while omitting runtime rewards."""
    for arguments_class in (
        DMD2TrainingArguments,
        TDMTrainingArguments,
        DiffusionOPDTrainingArguments,
    ):
        assert arguments_class.execution_contract is ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT

    assert TDMR1TrainingArguments.execution_contract is ONLINE_EXECUTION_CONTRACT


def test_execution_contract_classvar_is_not_serialized_as_user_configuration() -> None:
    """Algorithm semantics stay outside the dataclass/YAML field surface."""
    arguments = DMD2TrainingArguments()

    assert "execution_contract" not in {config_field.name for config_field in fields(arguments)}
    assert "execution_contract" not in arguments.to_dict()
    assert "execution_contract" not in dict(arguments)


@pytest.mark.parametrize(
    "train_fields",
    [
        {"execution_contract": "offline"},
        {"extra_kwargs": {"execution_contract": "offline"}},
    ],
)
def test_user_cannot_override_execution_contract(train_fields: dict) -> None:
    """The selected trainer type is the only public execution-mode switch."""
    with pytest.raises(ValueError, match="selected by trainer_type"):
        Arguments.from_dict({"train": {"trainer_type": "grpo", **train_fields}})


class _ContractProbeTrainer(BaseTrainer):
    """Expose contract validation without running heavyweight trainer initialization."""


class _EquivalentContractProbeTrainer(BaseTrainer):
    """Declare equal execution semantics through an independently built value."""

    execution_contract = ExecutionContract(
        acquisition=ONLINE_EXECUTION_CONTRACT.acquisition,
        cycle_unit=ONLINE_EXECUTION_CONTRACT.cycle_unit,
        feedback=ONLINE_EXECUTION_CONTRACT.feedback,
        loader_kind=ONLINE_EXECUTION_CONTRACT.loader_kind,
    )


def test_base_trainer_rejects_argument_contract_drift() -> None:
    """A custom or stale registry pairing fails before loading runtime components."""
    trainer = object.__new__(_ContractProbeTrainer)
    trainer.training_args = DMD2TrainingArguments()

    with pytest.raises(ValueError, match="execution contract mismatch"):
        trainer._validate_execution_contract()


def test_base_trainer_accepts_value_equivalent_custom_contracts() -> None:
    """Custom integrations need semantic equality, not built-in singleton identity."""
    trainer = object.__new__(_EquivalentContractProbeTrainer)
    trainer.training_args = TrainingArguments()

    assert type(trainer).execution_contract is not ONLINE_EXECUTION_CONTRACT
    trainer._validate_execution_contract()


def test_loader_rejects_contract_drift_before_loading_model() -> None:
    """A mismatched registry pair cannot trigger model downloads or allocations."""
    config = SimpleNamespace(training_args=TrainingArguments())

    with (
        patch(
            "flow_factory.trainers.loader.get_trainer_class",
            return_value=type(
                "OfflineTrainerProbe",
                (BaseTrainer,),
                {
                    "execution_contract": ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
                },
            ),
        ),
        patch("flow_factory.trainers.loader.load_model") as load_model,
        pytest.raises(ValueError, match="execution contract mismatch"),
    ):
        load_trainer(config)

    load_model.assert_not_called()


def test_loader_surfaces_offline_adapter_blocker_before_accelerator_or_model_loading() -> None:
    """A known model-specific target blocker must not download or allocate anything."""

    class KnownUnavailableAdapter(BaseAdapter):
        output_state_codec_unavailable_reason = (
            "Target packing has no parity fixture; add and validate one first."
        )

    config = SimpleNamespace(
        training_args=SFTTrainingArguments(),
        model_args=SimpleNamespace(model_type="known_unavailable"),
    )

    with (
        patch(
            "flow_factory.trainers.loader.get_model_adapter_class",
            return_value=KnownUnavailableAdapter,
        ),
        patch("flow_factory.trainers.loader.Accelerator") as accelerator,
        patch("flow_factory.trainers.loader.load_model") as load_model,
        pytest.raises(NotImplementedError, match=r"KnownUnavailableAdapter.*parity fixture"),
    ):
        load_trainer(config)

    accelerator.assert_not_called()
    load_model.assert_not_called()
