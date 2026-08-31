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

from types import SimpleNamespace
from typing import Dict, Tuple

import pytest
import torch
import torch.nn.functional as F

from flow_factory.trainers.common.dpo_objective import dpo_objective
from flow_factory.trainers.rl import dpo as dpo_module
from flow_factory.trainers.rl.dpo import DPOTrainer


def _legacy_online_objective(
    policy_chosen_loss: torch.Tensor,
    policy_rejected_loss: torch.Tensor,
    reference_chosen_loss: torch.Tensor,
    reference_rejected_loss: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    chosen_delta = policy_chosen_loss - reference_chosen_loss
    rejected_delta = policy_rejected_loss - reference_rejected_loss
    preference_delta = chosen_delta - rejected_delta
    loss = -F.logsigmoid(-0.5 * beta * preference_delta).mean()
    with torch.no_grad():
        implicit_reward_chosen = -0.5 * beta * chosen_delta
        implicit_reward_rejected = -0.5 * beta * rejected_delta
        metrics = {
            "implicit_reward_chosen": implicit_reward_chosen,
            "implicit_reward_rejected": implicit_reward_rejected,
            "implicit_accuracy": (implicit_reward_chosen > implicit_reward_rejected).float().mean(),
        }
    return loss, metrics


def test_objective_matches_online_values_and_logging_metrics_exactly() -> None:
    inputs = (
        torch.tensor([1.0, 4.0, 2.5], dtype=torch.float64),
        torch.tensor([3.0, 2.0, 2.5], dtype=torch.float64),
        torch.tensor([2.0, 2.0, 2.5], dtype=torch.float64),
        torch.tensor([2.0, 2.0, 2.5], dtype=torch.float64),
    )

    actual_loss, actual_metrics = dpo_objective(*inputs, beta=2.0)
    expected_loss, expected_metrics = _legacy_online_objective(*inputs, beta=2.0)

    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    assert tuple(actual_metrics) == (
        "implicit_reward_chosen",
        "implicit_reward_rejected",
        "implicit_accuracy",
    )
    for name in actual_metrics:
        torch.testing.assert_close(actual_metrics[name], expected_metrics[name], rtol=0, atol=0)
    torch.testing.assert_close(actual_metrics["implicit_accuracy"], torch.tensor(1.0 / 3.0))
    assert not any(metric.requires_grad for metric in actual_metrics.values())


def test_objective_preserves_the_legacy_gradient_for_all_four_inputs() -> None:
    values = (
        torch.tensor([0.4, 1.2], dtype=torch.float64),
        torch.tensor([1.7, 0.5], dtype=torch.float64),
        torch.tensor([0.8, 0.9], dtype=torch.float64),
        torch.tensor([1.1, 0.7], dtype=torch.float64),
    )
    actual_inputs = tuple(value.clone().requires_grad_() for value in values)
    expected_inputs = tuple(value.clone().requires_grad_() for value in values)

    actual_loss, _ = dpo_objective(*actual_inputs, beta=7.0)
    expected_loss, _ = _legacy_online_objective(*expected_inputs, beta=7.0)
    actual_loss.backward()
    expected_loss.backward()

    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for actual, expected in zip(actual_inputs, expected_inputs):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


def test_extreme_preference_logits_remain_finite() -> None:
    loss, _ = dpo_objective(
        torch.tensor([1.0e6, -1.0e6]),
        torch.tensor([-1.0e6, 1.0e6]),
        torch.zeros(2),
        torch.zeros(2),
        beta=2000.0,
    )
    assert torch.isfinite(loss)


def test_online_trainer_delegates_without_changing_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = object.__new__(DPOTrainer)
    trainer.training_args = SimpleNamespace(beta=3.5)
    values = tuple(torch.tensor([float(i), float(i + 1)]) for i in range(1, 8, 2))
    sentinel_loss = torch.tensor(9.0)
    sentinel_metrics = {
        "implicit_reward_chosen": torch.tensor([10.0, 11.0]),
        "implicit_reward_rejected": torch.tensor([12.0, 13.0]),
        "implicit_accuracy": torch.tensor(0.5),
    }
    received: Dict[str, object] = {}

    def fake_objective(**kwargs: object):
        received.update(kwargs)
        return sentinel_loss, sentinel_metrics

    monkeypatch.setattr(dpo_module, "dpo_objective", fake_objective)
    loss, metrics = trainer._preference_loss(*values)

    assert received == {
        "policy_chosen_loss": values[0],
        "policy_rejected_loss": values[1],
        "reference_chosen_loss": values[2],
        "reference_rejected_loss": values[3],
        "beta": 3.5,
    }
    assert loss is sentinel_loss
    assert metrics is sentinel_metrics


@pytest.mark.parametrize(
    "invalid_loss",
    [torch.tensor(1.0), torch.empty(0), torch.ones(2, 1)],
)
def test_objective_requires_non_empty_per_sample_vectors(invalid_loss: torch.Tensor) -> None:
    with pytest.raises(ValueError, match=r"policy_chosen_loss.*shape \(B,\)"):
        dpo_objective(invalid_loss, torch.ones(2), torch.ones(2), torch.ones(2), beta=1.0)


def test_objective_rejects_misaligned_shapes_and_dtypes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        dpo_objective(torch.ones(2), torch.ones(3), torch.ones(2), torch.ones(2), beta=1.0)
    with pytest.raises(TypeError, match="same dtype"):
        dpo_objective(
            torch.ones(2, dtype=torch.float32),
            torch.ones(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float32),
            torch.ones(2, dtype=torch.float32),
            beta=1.0,
        )


@pytest.mark.parametrize("beta", [True, "1.0", float("inf"), float("nan")])
def test_objective_requires_a_finite_real_beta(beta: object) -> None:
    values = torch.ones(2)
    error_type = TypeError if isinstance(beta, (bool, str)) else ValueError
    with pytest.raises(error_type, match="beta"):
        dpo_objective(values, values, values, values, beta=beta)  # type: ignore[arg-type]
