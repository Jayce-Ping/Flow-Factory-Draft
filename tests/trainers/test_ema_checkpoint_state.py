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

"""Strict checkpoint contract tests for the real EMA wrapper."""

from copy import deepcopy
from typing import Any

import pytest
import torch

from flow_factory.ema.ema import EMA_STATE_VERSION, EMAModuleWrapper


def _wrapper() -> EMAModuleWrapper:
    """Build a deterministic two-parameter EMA wrapper."""
    parameters = [
        torch.nn.Parameter(torch.zeros(2, dtype=torch.float32)),
        torch.nn.Parameter(torch.zeros(1, 3, dtype=torch.float32)),
    ]
    return EMAModuleWrapper(
        parameters,
        decay=0.9,
        update_step_interval=2,
        device=torch.device("cpu"),
        decay_schedule="linear",
        initial_decay=0.1,
        warmup_steps=3,
    )


def _state() -> dict[str, Any]:
    """Return a valid state with non-default values."""
    wrapper = _wrapper()
    wrapper.ema_parameters[0].fill_(4.0)
    wrapper.ema_parameters[1].fill_(6.0)
    wrapper.num_updates = 7
    return wrapper.state_dict()


def test_real_ema_wrapper_round_trip_restores_tensors_and_update_count() -> None:
    """A compatible wrapper restores values without changing its schedule contract."""
    state = _state()
    restored = _wrapper()

    restored.load_state_dict(state)

    assert state["version"] == EMA_STATE_VERSION
    assert restored.num_updates == 7
    assert restored.decay == pytest.approx(0.9)
    assert restored.update_step_interval == 2
    assert restored._decay_schedule == "linear"
    assert restored._schedule_params == {
        "initial_decay": 0.1,
        "warmup_steps": 3,
    }
    torch.testing.assert_close(restored.ema_parameters[0], torch.full((2,), 4.0))
    torch.testing.assert_close(restored.ema_parameters[1], torch.full((1, 3), 6.0))
    assert restored.ema_parameters[0] is not state["ema_parameters"][0]


@pytest.mark.parametrize(
    ("mutate", "error_type", "match"),
    [
        (lambda state: state.pop("version"), ValueError, "state keys mismatch"),
        (
            lambda state: state.update(version=True),
            TypeError,
            "state version to be int",
        ),
        (
            lambda state: state.update(version=EMA_STATE_VERSION + 1),
            ValueError,
            "version mismatch",
        ),
        (
            lambda state: state.update(decay=0.8),
            ValueError,
            "decay mismatch",
        ),
        (
            lambda state: state.update(update_step_interval=3),
            ValueError,
            "update_step_interval mismatch",
        ),
        (
            lambda state: state.update(decay_schedule="constant"),
            ValueError,
            "decay_schedule mismatch",
        ),
        (
            lambda state: state.update(schedule_params={}),
            ValueError,
            "schedule_params mismatch",
        ),
        (
            lambda state: state.update(num_updates=-1),
            ValueError,
            "num_updates.*non-negative",
        ),
        (
            lambda state: state.update(ema_parameters=state["ema_parameters"][:1]),
            ValueError,
            "parameter count mismatch",
        ),
        (
            lambda state: state["ema_parameters"].__setitem__(0, torch.zeros(3)),
            ValueError,
            "parameter 0 shape mismatch",
        ),
        (
            lambda state: state["ema_parameters"].__setitem__(
                0, state["ema_parameters"][0].to(torch.float64)
            ),
            ValueError,
            "parameter 0 dtype mismatch",
        ),
    ],
)
def test_real_ema_wrapper_rejects_malformed_state_without_partial_mutation(
    mutate, error_type, match
) -> None:
    """Keys, config, count, shape, and dtype validate before tensor replacement."""
    state = deepcopy(_state())
    mutate(state)
    restored = _wrapper()
    tensors_before = [parameter.clone() for parameter in restored.ema_parameters]

    with pytest.raises(error_type, match=match):
        restored.load_state_dict(state)

    assert restored.num_updates == 0
    for actual, expected in zip(restored.ema_parameters, tensors_before, strict=True):
        torch.testing.assert_close(actual, expected)


def test_ema_state_dict_rejects_tensor_subclasses_without_collective_gather() -> None:
    """A DTensor-like tensor subclass cannot enter the replicated custom format."""

    class _ShardedTensorLike(torch.Tensor):
        """Stand in for a distributed tensor subclass."""

    wrapper = _wrapper()
    wrapper.ema_parameters[0] = wrapper.ema_parameters[0].as_subclass(_ShardedTensorLike)

    with pytest.raises(TypeError, match="replicated plain tensors.*ShardedTensorLike"):
        wrapper.state_dict()


def test_ema_state_dict_rejects_temporary_parameter_swap() -> None:
    """A state save cannot capture a policy while EMA weights are installed."""
    wrapper = _wrapper()
    live_parameters = [
        torch.nn.Parameter(torch.full_like(parameter, 9.0)) for parameter in wrapper.ema_parameters
    ]

    with wrapper.use_ema_parameters(live_parameters):
        with pytest.raises(RuntimeError, match="while EMA parameters are installed"):
            wrapper.state_dict()

    for parameter in live_parameters:
        torch.testing.assert_close(parameter, torch.full_like(parameter, 9.0))


def test_ema_state_dict_rejects_invalid_live_counter() -> None:
    """Save-time validation emits only payloads the same wrapper can restore."""
    wrapper = _wrapper()
    wrapper.num_updates = -1

    with pytest.raises(ValueError, match="num_updates.*non-negative"):
        wrapper.state_dict()
