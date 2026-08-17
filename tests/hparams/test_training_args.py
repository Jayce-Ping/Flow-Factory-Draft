import pytest

from flow_factory.hparams import GRPOTrainingArguments


def test_max_sequence_length_is_preserved_in_training_kwargs() -> None:
    training_args = GRPOTrainingArguments(max_sequence_length=2048)
    assert training_args.max_sequence_length == 2048
    assert training_args.to_dict()["max_sequence_length"] == 2048


def test_max_sequence_length_rejects_non_integer() -> None:
    with pytest.raises(TypeError, match="expected int.*got str"):
        GRPOTrainingArguments(max_sequence_length="2048")


def test_max_sequence_length_rejects_nonpositive_integer() -> None:
    with pytest.raises(ValueError, match="max_sequence_length >= 1.*got 0"):
        GRPOTrainingArguments(max_sequence_length=0)
