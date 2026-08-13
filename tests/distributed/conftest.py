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

import pytest

BACKEND_CHOICES = ("ddp", "fsdp2", "zero1", "zero2")
ALGORITHM_CHOICES = ("dmd2", "tdm", "tdm-r1")
STORAGE_MODE_CHOICES = ("lora", "full")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the distributed backend acceptance selector."""
    parser.addoption(
        "--backend",
        action="store",
        choices=BACKEND_CHOICES,
        help="Distributed backend contract exercised by this test run.",
    )
    parser.addoption(
        "--algorithm",
        action="store",
        choices=ALGORITHM_CHOICES,
        help="Multi-role algorithm selected for distributed acceptance.",
    )
    parser.addoption(
        "--storage-mode",
        action="store",
        choices=STORAGE_MODE_CHOICES,
        help="Parameter storage mode selected for distributed acceptance.",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Select strict distributed matrix axes from CLI or enumerate all values."""
    for fixture_name, choices in (
        ("algorithm", ALGORITHM_CHOICES),
        ("storage_mode", STORAGE_MODE_CHOICES),
    ):
        if fixture_name not in metafunc.fixturenames:
            continue
        selected = metafunc.config.getoption(
            f"--{fixture_name.replace('_', '-')}",
            default=None,
        )
        metafunc.parametrize(fixture_name, (selected,) if selected is not None else choices)


@pytest.fixture
def backend(request: pytest.FixtureRequest) -> str:
    """Return the explicitly selected distributed backend."""
    value = request.config.getoption("--backend", default=None)
    if value is None:
        pytest.skip("distributed acceptance requires an explicit --backend")
    return value
