import pytest

from flow_factory.hparams.gradient_checkpointing import (
    GradientCheckpointingSpec,
    gradient_checkpointing_enabled,
    normalize_gradient_checkpointing_policy,
    serialize_gradient_checkpointing_policy,
)
from flow_factory.hparams.training_args import TrainingArguments


@pytest.mark.parametrize("value", [True, False])
def test_bool_gradient_checkpointing_policy_is_backward_compatible(value: bool) -> None:
    arguments = TrainingArguments(enable_gradient_checkpointing=value)

    assert arguments.enable_gradient_checkpointing is value
    assert arguments.gradient_checkpointing_enabled is value
    assert arguments.to_dict()["enable_gradient_checkpointing"] is value


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ({"mode": "full"}, GradientCheckpointingSpec(mode="full")),
        ({"mode": "none"}, GradientCheckpointingSpec(mode="none")),
        (
            {"fraction": 0.25},
            GradientCheckpointingSpec(mode="fraction", fraction=0.25),
        ),
        (
            {"mode": "every_n", "every_n": 3},
            GradientCheckpointingSpec(mode="every_n", every_n=3),
        ),
        (
            {"layers": [4, 1, 4]},
            GradientCheckpointingSpec(mode="layers", layers=(4, 1)),
        ),
    ],
)
def test_selective_gradient_checkpointing_policy_normalizes(
    configured: dict,
    expected: GradientCheckpointingSpec,
) -> None:
    policy = normalize_gradient_checkpointing_policy(configured)

    assert policy == expected
    assert gradient_checkpointing_enabled(policy) is (expected.mode != "none")
    assert (
        normalize_gradient_checkpointing_policy(serialize_gradient_checkpointing_policy(policy))
        == expected
    )


@pytest.mark.parametrize(
    ("configured", "error_type", "match"),
    [
        ({"fraction": 0}, ValueError, r"fraction.*\(0, 1\]"),
        ({"fraction": True}, TypeError, r"fraction.*real number.*bool"),
        ({"every_n": 0}, ValueError, r"every_n.*positive"),
        ({"every_n": 1.5}, TypeError, r"every_n.*int.*float"),
        ({"layers": []}, ValueError, r"at least one"),
        ({"layers": [-1]}, ValueError, r"index >= 0"),
        ({"layers": ["1"]}, TypeError, r"index as int.*str"),
        ({"mode": "full", "fraction": 0.5}, ValueError, r"does not accept"),
        ({"unknown": 1}, ValueError, r"unknown"),
    ],
)
def test_invalid_gradient_checkpointing_policy_fails_fast(
    configured: dict,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        TrainingArguments(enable_gradient_checkpointing=configured)
