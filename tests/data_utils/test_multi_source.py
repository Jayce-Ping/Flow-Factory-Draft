from flow_factory.data_utils.multi_source import (
    MultiSourceTrainDataLoader,
    WeightedSourceBatchScheduler,
)


def test_multi_source_loader_exposes_batch_size_for_deepspeed() -> None:
    loader = MultiSourceTrainDataLoader(
        {},
        WeightedSourceBatchScheduler({}, seed=42),
        batch_size=3,
    )

    assert loader.batch_size == 3
