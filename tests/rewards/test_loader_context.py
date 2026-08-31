from contextlib import contextmanager
from types import SimpleNamespace

from flow_factory.hparams import MultiRewardArguments, RewardArguments
from flow_factory.rewards.loader import MultiRewardLoader


def test_reward_loader_enters_backend_context_once_per_unique_model(
    monkeypatch,
) -> None:
    events = []

    @contextmanager
    def load_context():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    monkeypatch.setattr(
        "flow_factory.rewards.loader.load_reward_model",
        lambda config, accelerator: SimpleNamespace(config=config),
    )
    config = RewardArguments(name="score", reward_model="clip")

    loader = MultiRewardLoader(
        reward_args=MultiRewardArguments(reward_configs=[config]),
        accelerator=SimpleNamespace(),
        load_context=load_context,
    ).load()

    assert loader.get_unique_model_count() == 1
    assert events == ["enter", "exit"]
