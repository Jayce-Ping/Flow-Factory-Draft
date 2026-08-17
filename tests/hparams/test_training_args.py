from flow_factory.hparams import GRPOTrainingArguments
from flow_factory.utils.base import filter_kwargs


def _adapter_inference(prompt=None, max_sequence_length: int = 1024):
    """Stand-in for an adapter whose own sequence-length default is not 512."""
    return max_sequence_length


def test_unset_max_sequence_length_leaves_the_adapter_default_alone() -> None:
    """A declared field would be forwarded always and silently halve Qwen/LTX2."""
    training_args = GRPOTrainingArguments.from_dict({"seed": 7})

    forwarded = filter_kwargs(_adapter_inference, **dict(training_args))

    assert "max_sequence_length" not in forwarded


def test_configured_max_sequence_length_reaches_the_adapter() -> None:
    """Undeclared keys still travel to the adapter through extra_kwargs."""
    training_args = GRPOTrainingArguments.from_dict({"max_sequence_length": 2048})

    forwarded = filter_kwargs(_adapter_inference, **dict(training_args))

    assert forwarded["max_sequence_length"] == 2048
