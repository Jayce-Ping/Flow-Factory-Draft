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

"""Compatibility tests for dependency-neutral execution declarations."""

import subprocess
import sys

import pytest

from flow_factory.contracts.execution import (
    OFFLINE_EXECUTION_CONTRACT,
    ONLINE_EXECUTION_CONTRACT,
    ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
    AcquisitionMode,
    CycleUnit,
    ExecutionContract,
    FeedbackMode,
    LoaderKind,
)
from flow_factory.trainers import execution as trainer_execution


@pytest.mark.parametrize(
    "name,neutral_object",
    [
        ("AcquisitionMode", AcquisitionMode),
        ("CycleUnit", CycleUnit),
        ("ExecutionContract", ExecutionContract),
        ("FeedbackMode", FeedbackMode),
        ("LoaderKind", LoaderKind),
        ("OFFLINE_EXECUTION_CONTRACT", OFFLINE_EXECUTION_CONTRACT),
        ("ONLINE_EXECUTION_CONTRACT", ONLINE_EXECUTION_CONTRACT),
        (
            "ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT",
            ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
        ),
    ],
)
def test_legacy_trainer_import_reexports_neutral_objects(
    name: str,
    neutral_object: object,
) -> None:
    """Legacy imports retain class, enum, and singleton identity."""
    assert getattr(trainer_execution, name) is neutral_object


@pytest.mark.parametrize(
    "imports",
    [
        "import flow_factory.hparams; import flow_factory.trainers.execution",
        "import flow_factory.trainers.execution; import flow_factory.hparams",
    ],
)
def test_hparams_and_trainer_runtime_import_in_either_order(imports: str) -> None:
    """The neutral declaration layer must not introduce an import cycle."""
    subprocess.run(
        [sys.executable, "-c", imports],
        check=True,
        capture_output=True,
        text=True,
    )
