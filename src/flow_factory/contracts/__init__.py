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

"""Dependency-neutral framework contracts."""

from .execution import (
    OFFLINE_EXECUTION_CONTRACT,
    ONLINE_EXECUTION_CONTRACT,
    ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
    AcquisitionMode,
    CycleUnit,
    ExecutionContract,
    FeedbackMode,
    LoaderKind,
)

__all__ = [
    "AcquisitionMode",
    "CycleUnit",
    "ExecutionContract",
    "FeedbackMode",
    "LoaderKind",
    "OFFLINE_EXECUTION_CONTRACT",
    "ONLINE_EXECUTION_CONTRACT",
    "ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT",
]
