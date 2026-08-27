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

"""Registry coverage for finite-dataset trainers."""

import pytest

from flow_factory.contracts.execution import (
    OFFLINE_EXECUTION_CONTRACT,
    ONLINE_EXECUTION_CONTRACT,
)
from flow_factory.hparams import get_training_args_class
from flow_factory.trainers.registry import get_trainer_class, list_registered_trainers


@pytest.mark.parametrize(
    ("name", "class_path", "class_name"),
    [
        (
            "sft",
            "flow_factory.trainers.offline.sft.SFTTrainer",
            "SFTTrainer",
        ),
        (
            "offline-dpo",
            "flow_factory.trainers.offline.offline_dpo.OfflineDPOTrainer",
            "OfflineDPOTrainer",
        ),
    ],
)
def test_offline_trainers_are_registered_as_lazy_import_paths(
    name: str,
    class_path: str,
    class_name: str,
) -> None:
    registered = list_registered_trainers()

    assert registered[name] == class_path
    assert get_trainer_class(name).__name__ == class_name


@pytest.mark.parametrize("name", ["sft", "offline-dpo"])
def test_offline_argument_and_trainer_contracts_match(name: str) -> None:
    trainer_class = get_trainer_class(name)
    arguments_class = get_training_args_class(name)

    assert trainer_class.execution_contract is OFFLINE_EXECUTION_CONTRACT
    assert arguments_class.execution_contract is OFFLINE_EXECUTION_CONTRACT


def test_online_dpo_registration_remains_online() -> None:
    registered = list_registered_trainers()

    assert registered["dpo"] == "flow_factory.trainers.rl.dpo.DPOTrainer"
    assert get_trainer_class("dpo").execution_contract is ONLINE_EXECUTION_CONTRACT
