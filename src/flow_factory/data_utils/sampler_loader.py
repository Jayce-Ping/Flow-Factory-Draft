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

# src/flow_factory/data_utils/sampler_loader.py
from typing import Optional

from torch.utils.data import Sampler, Dataset
from accelerate import Accelerator

from .sampler import (
    DistributedKRepeatSampler,
    GroupContiguousSampler,
    GroupDistributedSampler,
)
from ..hparams import Arguments

SAMPLER_REGISTRY = {
    "distributed_k_repeat": DistributedKRepeatSampler,
    "group_contiguous": GroupContiguousSampler,
    "group_distributed": GroupDistributedSampler,
}


def get_data_sampler(
    dataset: Dataset,
    config: Arguments,
    accelerator: Accelerator,
    *,
    unique_sample_num: Optional[int] = None,
) -> Sampler:
    """
    Factory function to create the appropriate distributed sampler.

    The sampler strategy is determined by ``config.data_args.sampler_type``,
    which is resolved in ``Arguments._resolve_sampler_type()`` and aligned in
    ``Arguments._align_batch_geometry()`` during ``__post_init__``.

    Args:
        dataset: The dataset to sample from.
        config: The full ``Arguments`` configuration object.
        accelerator: HuggingFace Accelerator (provides world size + rank).
        unique_sample_num: Optional override for ``unique_sample_num_per_epoch``.
            When ``None`` (default), the sampler reads
            ``training_args.unique_sample_num_per_epoch`` — the legacy
            single-source / aggregate value.  Multi-source callers pass
            the per-source ``M_i`` from
            ``training_args._per_source_unique_sample_num`` so each
            per-source sampler emits exactly its allocated batches per
            epoch.  Must be a positive multiple of the relevant alignment
            step (``Arguments._partition_unique_sample_num`` guarantees
            this when set by the multi-source path).

    Returns:
        - GroupContiguousSampler when resolved type is ``"group_contiguous"``
          (keeps each group's samples on the same rank)
        - GroupDistributedSampler when resolved type is ``"group_distributed"``
          (split each group evenly across ranks)
        - DistributedKRepeatSampler when resolved type is ``"distributed_k_repeat"``
          (default behavior)
    """
    training_args = config.training_args
    sampler_type = config.data_args.sampler_type
    sampler_cls = SAMPLER_REGISTRY.get(sampler_type)
    if sampler_cls is None:
        raise ValueError(
            f"Unknown sampler_type={sampler_type!r}. Expected one of {sorted(SAMPLER_REGISTRY)}."
        )
    resolved_M = (
        unique_sample_num
        if unique_sample_num is not None
        else training_args.unique_sample_num_per_epoch
    )
    return sampler_cls(
        dataset=dataset,
        batch_size=training_args.per_device_batch_size,
        group_size=training_args.group_size,
        unique_sample_num=resolved_M,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=training_args.seed,
    )
