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

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DistributedSampler

from flow_factory.data_utils.offline_dataset import (
    OFFLINE_CONDITION_ID_COLUMN,
    OfflineDataset,
    compute_offline_condition_id,
)
from flow_factory.data_utils.offline_loader import build_offline_dataloader
from flow_factory.data_utils.schema import NormalizedDatasetRecord, normalize_v2_record


def _offline_dataset(
    tmp_path: Path,
    *,
    source_name: str,
    source_id: int,
    size: int,
    supervision_type: Literal["demonstration", "preference"] = "demonstration",
) -> OfflineDataset:
    records = []
    conditions = []
    for index in range(size):
        target_path = tmp_path / source_name / f"target-{index}.png"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), color=(source_id, index, 0)).save(target_path)
        records.append(
            _record(
                tmp_path,
                prompt=f"{source_name}-{index}",
                target_path=target_path,
                supervision_type=supervision_type,
            )
        )
    for index, record in enumerate(records):
        conditions.append(
            {
                "encoded": torch.tensor([source_id, index]),
                OFFLINE_CONDITION_ID_COLUMN: compute_offline_condition_id(
                    record,
                    index=index,
                    source_name=source_name,
                ),
            }
        )
    return OfflineDataset(
        records,
        conditions,
        source_name=source_name,
        source_id=source_id,
        supervision_type=supervision_type,
    )


def _record(
    dataset_dir: Path,
    *,
    prompt: str,
    target_path: Path,
    supervision_type: Literal["demonstration", "preference"],
) -> NormalizedDatasetRecord:
    if supervision_type == "demonstration":
        supervision: Dict[str, Any] = {
            "type": "demonstration",
            "target": {"media": [{"type": "image", "path": str(target_path)}]},
        }
    else:
        supervision = {
            "type": "preference",
            "chosen": {"media": [{"type": "image", "path": str(target_path)}]},
            "rejected": {"media": [{"type": "image", "path": str(target_path)}]},
        }
    return normalize_v2_record(
        {
            "schema_version": 2,
            "input": {"prompt": prompt, "media": []},
            "supervision": supervision,
        },
        dataset_dir=dataset_dir,
    )


def _build(dataset: OfflineDataset, **overrides: Any):
    kwargs = {
        "source_weights": [1],
        "batch_size": 2,
        "num_replicas": 1,
        "rank": 0,
        "gradient_accumulation_steps": 1,
        "num_workers": 0,
        "shuffle": False,
        "seed": 17,
        "sampler_drop_last": False,
        "batch_drop_last": False,
    }
    kwargs.update(overrides)
    return build_offline_dataloader(dataset, **kwargs)


def test_single_process_still_uses_explicit_official_distributed_sampler(
    tmp_path: Path,
) -> None:
    dataset = _offline_dataset(
        tmp_path,
        source_name="single",
        source_id=0,
        size=4,
    )

    loader = _build(dataset)

    assert isinstance(loader.dataset, ConcatDataset)
    assert loader.dataset.datasets == [dataset]
    assert isinstance(loader.sampler, DistributedSampler)
    assert loader.sampler.num_replicas == 1
    assert loader.sampler.rank == 0
    assert loader.sampler.shuffle is False
    assert loader.sampler.seed == 17
    assert loader.sampler.drop_last is False
    assert loader.batch_size == 2
    assert loader.num_workers == 0
    assert loader.pin_memory is False
    assert loader.drop_last is False
    assert len(loader) == 2

    batch = next(iter(loader))
    assert batch.sources == ("single", "single")
    assert torch.equal(batch.source_ids, torch.tensor([0, 0], dtype=torch.long))
    assert torch.equal(batch.condition["encoded"], torch.tensor([[0, 0], [0, 1]]))


def test_multiple_sources_are_concatenated_and_may_share_one_batch(tmp_path: Path) -> None:
    first = _offline_dataset(
        tmp_path,
        source_name="first",
        source_id=2,
        size=2,
    )
    second = _offline_dataset(
        tmp_path,
        source_name="second",
        source_id=5,
        size=2,
    )

    loader = build_offline_dataloader(
        [first, second],
        source_weights=[1, 1.0],
        batch_size=3,
        num_replicas=1,
        rank=0,
        gradient_accumulation_steps=1,
        num_workers=0,
        shuffle=False,
        seed=9,
        sampler_drop_last=False,
        batch_drop_last=False,
    )
    batch = next(iter(loader))

    assert isinstance(loader.dataset, ConcatDataset)
    assert loader.dataset.datasets == [first, second]
    assert batch.sources == ("first", "first", "second")
    assert torch.equal(batch.source_ids, torch.tensor([2, 2, 5], dtype=torch.long))


def test_distributed_sampler_receives_rank_world_and_drop_last_explicitly(
    tmp_path: Path,
) -> None:
    dataset = _offline_dataset(
        tmp_path,
        source_name="distributed",
        source_id=0,
        size=8,
    )

    rank_zero = _build(
        dataset,
        num_replicas=2,
        rank=0,
        sampler_drop_last=True,
        batch_drop_last=True,
    )
    rank_one = _build(
        dataset,
        num_replicas=2,
        rank=1,
        sampler_drop_last=True,
        batch_drop_last=True,
    )

    assert list(rank_zero.sampler) == [0, 2, 4, 6]
    assert list(rank_one.sampler) == [1, 3, 5, 7]
    for rank, loader in enumerate((rank_zero, rank_one)):
        assert isinstance(loader.sampler, DistributedSampler)
        assert loader.sampler.num_replicas == 2
        assert loader.sampler.rank == rank
        assert loader.sampler.shuffle is False
        assert loader.sampler.seed == 17
        assert loader.sampler.drop_last is True
        assert loader.drop_last is True


def test_sampler_and_batch_drop_last_are_independent_for_uneven_geometry(
    tmp_path: Path,
) -> None:
    dataset = _offline_dataset(
        tmp_path,
        source_name="independent-tails",
        source_id=0,
        size=10,
    )

    keep_both = _build(
        dataset,
        num_replicas=3,
        rank=0,
        batch_size=2,
        sampler_drop_last=False,
        batch_drop_last=False,
    )
    drop_sampler_only = _build(
        dataset,
        num_replicas=3,
        rank=0,
        batch_size=2,
        sampler_drop_last=True,
        batch_drop_last=False,
    )
    drop_batch_only = _build(
        dataset,
        num_replicas=3,
        rank=0,
        batch_size=2,
        sampler_drop_last=False,
        batch_drop_last=True,
    )
    drop_both = _build(
        dataset,
        num_replicas=3,
        rank=0,
        batch_size=2,
        sampler_drop_last=True,
        batch_drop_last=True,
    )

    assert len(keep_both.sampler) == 4
    assert len(drop_sampler_only.sampler) == 3
    assert len(drop_batch_only.sampler) == 4
    assert len(drop_both.sampler) == 3
    assert len(keep_both) == 2
    assert len(drop_sampler_only) == 2
    assert len(drop_batch_only) == 2
    assert len(drop_both) == 1
    assert keep_both.sampler.drop_last is False and keep_both.drop_last is False
    assert drop_sampler_only.sampler.drop_last is True and drop_sampler_only.drop_last is False
    assert drop_batch_only.sampler.drop_last is False and drop_batch_only.drop_last is True
    assert drop_both.sampler.drop_last is True and drop_both.drop_last is True


def test_builder_never_calls_set_epoch(tmp_path: Path) -> None:
    dataset = _offline_dataset(
        tmp_path,
        source_name="epoch-owner",
        source_id=0,
        size=4,
    )

    with patch.object(DistributedSampler, "set_epoch", autospec=True) as set_epoch:
        _build(dataset)

    set_epoch.assert_not_called()


def test_sources_must_share_one_supervision_type(tmp_path: Path) -> None:
    demonstration = _offline_dataset(
        tmp_path,
        source_name="demonstration",
        source_id=0,
        size=2,
    )
    preference = _offline_dataset(
        tmp_path,
        source_name="preference",
        source_id=1,
        size=2,
        supervision_type="preference",
    )

    with pytest.raises(ValueError, match="must share one supervision_type"):
        build_offline_dataloader(
            [demonstration, preference],
            source_weights=[1, 1],
            batch_size=2,
            num_replicas=1,
            rank=0,
            gradient_accumulation_steps=1,
        )


def test_multiple_sources_require_unique_names_and_ids(tmp_path: Path) -> None:
    duplicate_name = [
        _offline_dataset(tmp_path, source_name="duplicate", source_id=0, size=2),
        _offline_dataset(tmp_path, source_name="duplicate", source_id=1, size=2),
    ]
    with pytest.raises(ValueError, match="source_name 'duplicate' is duplicated"):
        build_offline_dataloader(
            duplicate_name,
            source_weights=[1, 1],
            batch_size=2,
            num_replicas=1,
            rank=0,
            gradient_accumulation_steps=1,
        )

    duplicate_id = [
        _offline_dataset(tmp_path, source_name="first-id", source_id=3, size=2),
        _offline_dataset(tmp_path, source_name="second-id", source_id=3, size=2),
    ]
    with pytest.raises(ValueError, match="source_id 3 is duplicated"):
        build_offline_dataloader(
            duplicate_id,
            source_weights=[1, 1],
            batch_size=2,
            num_replicas=1,
            rank=0,
            gradient_accumulation_steps=1,
        )


@pytest.mark.parametrize(
    "weights,error_type,error_fragment",
    [
        ([1], ValueError, "one entry per offline source"),
        ([1, 2], ValueError, "must equal 1"),
        ([1, True], TypeError, "must be int or float"),
        ("11", TypeError, "numeric sequence"),
    ],
)
def test_offline_sources_require_explicit_unit_weights(
    tmp_path: Path,
    weights: Any,
    error_type: type[Exception],
    error_fragment: str,
) -> None:
    datasets = [
        _offline_dataset(tmp_path, source_name="first", source_id=0, size=2),
        _offline_dataset(tmp_path, source_name="second", source_id=1, size=2),
    ]

    with pytest.raises(error_type, match=error_fragment):
        build_offline_dataloader(
            datasets,
            source_weights=weights,
            batch_size=2,
            num_replicas=1,
            rank=0,
            gradient_accumulation_steps=1,
        )


@pytest.mark.parametrize(
    "override,error_type,error_fragment",
    [
        ({"batch_size": 0}, ValueError, "batch_size must be >= 1"),
        ({"batch_size": True}, TypeError, "batch_size must be int"),
        ({"num_replicas": 0}, ValueError, "num_replicas must be >= 1"),
        ({"rank": -1}, ValueError, "rank must be >= 0"),
        ({"num_replicas": 2, "rank": 2}, ValueError, "rank must satisfy"),
        ({"gradient_accumulation_steps": 0}, ValueError, "must be >= 1"),
        ({"num_workers": -1}, ValueError, "num_workers must be >= 0"),
        ({"seed": True}, TypeError, "seed must be int"),
        ({"shuffle": 1}, TypeError, "shuffle must be bool"),
        ({"sampler_drop_last": 0}, TypeError, "sampler_drop_last must be bool"),
        ({"batch_drop_last": 0}, TypeError, "batch_drop_last must be bool"),
        ({"pin_memory": 0}, TypeError, "pin_memory must be bool"),
    ],
)
def test_builder_rejects_invalid_runtime_geometry(
    tmp_path: Path,
    override: Dict[str, Any],
    error_type: type[Exception],
    error_fragment: str,
) -> None:
    dataset = _offline_dataset(
        tmp_path,
        source_name="validation",
        source_id=0,
        size=4,
    )

    with pytest.raises(error_type, match=error_fragment):
        _build(dataset, **override)


def test_builder_rejects_empty_or_non_offline_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one OfflineDataset"):
        build_offline_dataloader(
            [],
            source_weights=[],
            batch_size=1,
            num_replicas=1,
            rank=0,
            gradient_accumulation_steps=1,
        )
    with pytest.raises(TypeError, match="source 0 must be OfflineDataset"):
        build_offline_dataloader(
            [object()],
            source_weights=[1],
            batch_size=1,
            num_replicas=1,
            rank=0,
            gradient_accumulation_steps=1,
        )


def test_builder_rejects_rank_local_empty_loader(tmp_path: Path) -> None:
    dataset = _offline_dataset(
        tmp_path,
        source_name="too-small",
        source_id=0,
        size=1,
    )

    with pytest.raises(ValueError, match="offline dataloader is empty on this rank"):
        _build(
            dataset,
            batch_size=2,
            num_replicas=2,
            rank=0,
            sampler_drop_last=True,
        )


def test_gradient_accumulation_tail_fails_without_extra_batches_or_implicit_flush(
    tmp_path: Path,
) -> None:
    dataset = _offline_dataset(
        tmp_path,
        source_name="accumulation",
        source_id=0,
        size=5,
    )

    with pytest.raises(ValueError) as exc_info:
        _build(
            dataset,
            batch_size=2,
            gradient_accumulation_steps=2,
            sampler_drop_last=False,
            batch_drop_last=False,
        )
    message = str(exc_info.value)
    assert "yields 3 batches" in message
    assert "does not add batches" in message
    assert "implicitly flush" in message

    aligned = _build(
        dataset,
        batch_size=2,
        gradient_accumulation_steps=2,
        sampler_drop_last=False,
        batch_drop_last=True,
    )
    assert len(aligned) == 2


def test_pin_memory_defaults_false_and_requires_explicit_opt_in(tmp_path: Path) -> None:
    dataset = _offline_dataset(
        tmp_path,
        source_name="pin-memory",
        source_id=0,
        size=2,
    )

    default_loader = _build(dataset)
    pinned_loader = _build(dataset, pin_memory=True)

    assert default_loader.pin_memory is False
    assert pinned_loader.pin_memory is True
