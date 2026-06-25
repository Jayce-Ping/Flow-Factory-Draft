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

# src/flow_factory/trainers/abc.py
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union, Literal, Iterator
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image
from diffusers.utils.outputs import BaseOutput
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration

from ..hparams import *
from ..models.abc import BaseAdapter
from ..models.model_bundle import ModelBundle, RoutedComponentProxy
from ..data_utils.dataset import METADATA_COLUMN
from ..data_utils.loader import (
    get_train_dataloader,
    get_eval_dataloaders,
)
from ..rewards import load_reward_model, BaseRewardModel, MultiRewardLoader, RewardProcessor, RewardBuffer
from ..advantage import AdvantageProcessor
from ..logger import load_logger, LogFormatter
from ..samples import BaseSample, StackedSampleBatch
from ..utils.logger_utils import setup_logger
from ..utils.base import create_generator, create_generator_by_prompt, filter_kwargs, json_default, visit_tensor_leaves

logger = setup_logger(__name__)


def _record_stream_on_batch(value: Any, stream: "torch.cuda.Stream") -> None:
    """Record ``stream`` on every CUDA tensor in a stacked batch.

    Required for the copy-stream prefetch: it stops the caching allocator from
    reusing copy-stream-produced tensors until the consuming stream is done.
    """
    visit_tensor_leaves(value, lambda t: t.record_stream(stream) if t.is_cuda else None)


class BaseTrainer(ABC):
    """
    Abstract Base Class for Flow-Factory trainers.
    """
    def __init__(
            self,
            accelerator: Accelerator,
            config : Arguments,
            adapter : BaseAdapter,
        ):
        self.accelerator = accelerator
        self.config = config
        self.log_args = config.log_args
        self.model_args = config.model_args

        self.training_args = config.training_args
        self.eval_args = config.eval_args

        self.reward_args = config.reward_args
        self.eval_reward_args = config.eval_reward_args or config.reward_args # If `eval_reward_args` is not given, use `reward_args`

        self.adapter = adapter
        self.epoch = 0
        self.step = 0

        self._initialization()
        self.adapter.post_init()
        self._init_logging_backend()

        self._patch_deepspeed_autocast(accelerator)
        self.autocast = partial(
            torch.autocast,
            device_type=accelerator.device.type,
            dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16
        )

        if self.accelerator.is_local_main_process:
            self.adapter.log_trainable_parameters()

    @property
    def show_progress_bar(self) -> bool:
        """Whether to show tqdm progress bars."""
        return self.log_args.verbose and self.accelerator.is_local_main_process

    def should_continue_training(self) -> bool:
        """Outer epoch loop: continue unless a finite ``max_epochs`` has been reached."""
        m = self.training_args.max_epochs
        if m is None or m < 0:
            return True
        return self.epoch < m

    def accumulate_gradients(self):
        """Context manager for gradient accumulation over the single prepared root.

        Centralizes ``accelerator.accumulate(self.model_bundle)`` so trainers do
        not couple to the prepared-root identity: ``self.model_bundle`` is the one
        object DDP/FSDP/DeepSpeed wraps, and accumulation must always target it.

        Usage::

            with self.accumulate_gradients():
                ...  # forward / loss / backward / step
        """
        return self.accelerator.accumulate(self.model_bundle)

    def log_data(self, data: Dict[str, Any], step: int):
        """Log data using the initialized logger."""
        if self.logger is not None:
            self.logger.log_data(data, step=step)
        
        # Print summary to console
        if self.accelerator.is_local_main_process:
            metrics = {k: v for k, v in ((k, LogFormatter.to_scalar(v)) for k, v in data.items()) if v is not None}
            if metrics:
                parts = [f"[Step {step:04d} | Epoch {self.epoch:03d}]"]
                parts.extend(
                    f"{k}={int(v)}" if isinstance(v, int) or (isinstance(v, float) and v.is_integer())
                    else f"{k}={v:.4f}"
                    for k, v in metrics.items()
                )
                logger.info(" ".join(parts))
    
    def _init_logging_backend(self):
        """Initialize logging backend if specified."""
        if self.accelerator.is_main_process:
            self.logger = load_logger(self.config)
        else:
            self.logger = None
        self.accelerator.wait_for_everyone()

    def _init_reward_model(self) -> Tuple[Dict[str, BaseRewardModel], Dict[str, BaseRewardModel]]:
        """Initialize reward model from configuration."""

        # If DeepSpeed ZeRO-3 is enabled, the reward model will be somehow sharded.
        # We need to disable ZeRO-3 init context when loading the model to avoid issues
        # NOTE: This bug persists even with this context manager. DONOT USE ZeRO-3.
        # A possible solution: use DeepSpeed GatherParamter manually in the reward_model's `forward`.

        # Collect training dataset names so MultiRewardLoader can pre-compute
        # the per-source reward routing used by the runtime reward gate
        # and any future trainer that needs "which rewards apply to source S?"
        # lookups.  Training is the primary path; eval names follow.
        training_dataset_names = (
            [td.name for td in self.config.data_args.training_datasets]
            if self.config.data_args.training_datasets
            else []
        )
        # Collect eval dataset names for per-eval-dataset reward routing
        # (mirror of the training-side bookkeeping).
        eval_dataset_names = (
            [ed.name for ed in self.config.data_args.eval_datasets]
            if self.config.data_args.eval_datasets
            else []
        )

        # Initialize all reward model instances
        self.reward_loader = MultiRewardLoader(
            reward_args=self.config.reward_args,
            accelerator=self.accelerator,
            training_dataset_names=training_dataset_names,
            eval_reward_args=self.config.eval_reward_args,
            eval_dataset_names=eval_dataset_names,
        ).load()
        # Get training & eval reward models
        self.reward_models = self.reward_loader.get_training_reward_models()
        self.eval_reward_models = self.reward_loader.get_eval_reward_models()
        train_reward_configs = self.reward_loader.get_reward_configs('train')
        # Initialize reward processor (training side only — eval-side
        # processors are per-dataset, built below).
        group_on_same_rank = self.config.data_args.sampler_type == "group_contiguous"
        self.reward_processor = RewardProcessor(
            accelerator=self.accelerator,
            reward_models=self.reward_models,
            reward_configs=train_reward_configs,
            tokenizer=self.adapter.tokenizer, # For prompt encoding/decoding,
            group_on_same_rank=group_on_same_rank,
            verbose=self.log_args.verbose,
        )
        # Initialize the training-side reward buffer.
        self.reward_buffer = RewardBuffer(
            self.reward_processor, self.training_args.group_size,
        )

        # Per-eval-dataset reward processors and buffers.  Eval is now
        # always per-dataset (the legacy single `eval_reward_buffer`
        # was retired with the unified `evaluate()` path); the loop
        # below builds one processor + buffer per eval-eligible entry,
        # which `evaluate()` then iterates.
        self.eval_dataset_reward_processors: Dict[str, RewardProcessor] = {}
        self.eval_dataset_reward_buffers: Dict[str, RewardBuffer] = {}
        self._eval_dataset_configs: Dict[str, "DatasetArguments"] = {}

        if self.config.data_args.eval_datasets:
            self._eval_dataset_configs = {ed.name: ed for ed in self.config.data_args.eval_datasets}
            for ed in self.config.data_args.eval_datasets:
                ds_models = self.reward_loader.get_eval_dataset_reward_models(ed.name)
                ds_configs = self.reward_loader.get_eval_dataset_reward_configs(ed.name)
                if ds_models:
                    ds_processor = RewardProcessor(
                        accelerator=self.accelerator,
                        reward_models=ds_models,
                        reward_configs=ds_configs,
                        tokenizer=self.adapter.tokenizer,
                        group_on_same_rank=group_on_same_rank,
                        verbose=self.log_args.verbose,
                    )
                    self.eval_dataset_reward_processors[ed.name] = ds_processor
                    self.eval_dataset_reward_buffers[ed.name] = RewardBuffer(
                        ds_processor, self.training_args.group_size,
                    )

        # Initialize advantage processor.
        # `cfg.weight` is a Dict[str, float] after `_resolve_reward_weights`,
        # so reward_weights is Dict[reward_name, Dict[dataset_name, float]].
        self.advantage_processor = AdvantageProcessor(
            accelerator=self.accelerator,
            reward_weights={
                name: cfg.weight
                for name, cfg in train_reward_configs.items()
            },
            group_size=self.training_args.group_size,
            global_std=getattr(self.training_args, 'global_std', True),
            sampler_type=self.config.data_args.sampler_type,
            verbose=self.log_args.verbose,
            source_id_to_name=self.config.data_args.source_id_to_name,
        )

        return self.reward_models, self.eval_reward_models

    def _init_dataloader(self) -> Tuple[Optional[Union[DataLoader, "MultiSourceTrainDataLoader"]], Dict[str, DataLoader]]:
        """Build train and eval dataloaders.

        Returns:
            Tuple of (train_dataloader, eval_dataloaders_by_name).
        """
        self.adapter.on_load_components(
            components=self.adapter.preprocessing_modules,
            device=self.accelerator.device
        )

        dataloader, train_dataloaders_by_source = get_train_dataloader(
            config=self.config,
            accelerator=self.accelerator,
            preprocess_func=self.adapter.preprocess_func,
        )
        self.train_dataloaders_by_source: Dict[str, DataLoader] = train_dataloaders_by_source

        eval_dataloaders = get_eval_dataloaders(
            eval_datasets=self.config.data_args.eval_datasets,
            config=self.config,
            accelerator=self.accelerator,
            preprocess_func=self.adapter.preprocess_func,
        )

        self.adapter.off_load_components(
            components=self.adapter.preprocessing_modules,
        )

        self.accelerator.wait_for_everyone()

        return dataloader, eval_dataloaders
    
    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer."""
        self.optimizer = torch.optim.AdamW(
            self.adapter.get_trainable_parameters(),
            lr=self.training_args.learning_rate,
            betas=self.training_args.adam_betas,
            weight_decay=self.training_args.adam_weight_decay,
            eps=self.training_args.adam_epsilon,
        )
        return self.optimizer

    def _load_inference_components(self, trainable_module_names: List[str]):
        """
        Load non-trainable components needed at runtime to the accelerator device.
        
        Trainable modules are already on-device via `accelerator.prepare()`.
        This loads the remaining modules required for inference and,
        when preprocessing is disabled, also loads encoding components
        that would otherwise stay offloaded.
        """
        prepared_names = set(trainable_module_names)
        
        modules_to_load = list(self.adapter.inference_modules)
        
        if not self.config.data_args.enable_preprocess:
            modules_to_load.extend(self.adapter.preprocessing_modules)
        
        # Resolve group names → concrete names, then deduplicate & exclude prepared
        resolved = self.adapter._resolve_component_names(modules_to_load)
        resolved = [m for m in resolved if m not in prepared_names]
        
        if resolved:
            self.adapter.on_load_components(
                components=resolved,
                device=self.accelerator.device,
            )

    def _initialization(self):
        # Fix for FSDP, synchronize frozen components like text encoder & VAE.
        # Otherwise they may be uninitialized on Rank > 0.
        if self.adapter._is_fsdp_cpu_efficient_loading():
            logger.info("FSDP CPU Efficient Loading detected. Synchronizing frozen components...")
            # self.adapter.on_load(self.accelerator.device)
            self._synchronize_frozen_components()

        # Init dataloader and optimizer
        self.dataloader, eval_dataloaders = self._init_dataloader()
        self.optimizer = self._init_optimizer()

        # Bundle ALL target components (trainable + frozen-but-shardable, e.g.
        # Wan2.2's inactive transformer) into ONE nn.Module so accelerate wraps a
        # single root. DeepSpeed (one engine) and FSDP2 (one root) cannot wrap
        # multiple models, so PPO (policy + critic) and Wan2.2 (shard both, train
        # one) require this. The optimizer/EMA/ref still operate on the
        # requires_grad subset via `get_trainable_parameters()`; frozen members
        # are sharded for memory but never receive gradient.
        bundle_names = list(self.adapter.target_module_map.keys())
        # Bundle the resolved trainable/frozen components. get_component returns the
        # LoRA PeftModel for LoRA training (apply_lora stores it via set_component,
        # NOT in-place on the pipeline), matching the pre-refactor membership.
        bundle_members = {name: self.adapter.get_component(name) for name in bundle_names}
        model_bundle = ModelBundle(bundle_members)

        eval_dataloader_names = list(eval_dataloaders.keys())
        eval_dataloader_list = [eval_dataloaders[n] for n in eval_dataloader_names]

        # One prepare call -> one DDP/FSDP/DeepSpeed root for the whole bundle.
        prepared = self.accelerator.prepare(model_bundle, self.optimizer, *eval_dataloader_list)
        self.model_bundle = prepared[0]
        self.optimizer = prepared[1]
        prepared_eval_dataloaders = prepared[2:]
        self.eval_dataloaders: Dict[str, DataLoader] = dict(
            zip(eval_dataloader_names, prepared_eval_dataloaders)
        )

        # Install routing proxies so adapter forwards (`self.transformer(...)`,
        # `self.transformer_2(...)`, ...) dispatch through the prepared root --
        # required for DDP's reducer / FSDP's gather / the DeepSpeed engine --
        # while attribute access delegates to the inner member.
        inner_bundle = self.accelerator.unwrap_model(self.model_bundle)
        for name in bundle_names:
            self.adapter.set_component(
                name, RoutedComponentProxy(self.model_bundle, name, inner_bundle.members[name])
            )

        # Load inference modules, excluding all bundle members (already prepared).
        self._load_inference_components(bundle_names)

        # Initialize reward model
        self._init_reward_model()

    def _synchronize_frozen_components(self):
        if self.accelerator.num_processes <= 1:
            return
        
        # Synchronize all non-prepared components
        all_names = self.adapter._resolve_component_names()
        for name in all_names:
            if self.adapter._should_manage_device(name):
                comp = self.adapter.get_component(name)
                if comp is not None:
                    for param in comp.parameters():
                        param.data = param.data.to(self.accelerator.device)
                        dist.broadcast(param.data, src=0)

        # Barrier to ensure everyone is done
        self.accelerator.wait_for_everyone()
        logger.info(f"[Rank {self.accelerator.process_index}] Frozen components synchronized.")

    @staticmethod
    def _patch_deepspeed_autocast(accelerator):
        """Patch DeepSpeed >=0.17.2 to allow external torch.autocast contexts.

        In v0.17.2+, engine.forward() calls validate_nested_autocast() which
        raises AssertionError if torch.autocast is active outside the engine,
        then wraps the forward with torch.autocast(enabled=torch_autocast_enabled).
        When torch_autocast is not configured (the default for bf16 built-in
        mixed-precision), this inner context uses enabled=False, which explicitly
        *disables* any outer autocast and causes dtype mismatches.

        This patch makes the engine transparent to an outer autocast context:
        validate_nested_autocast becomes a no-op, and torch_autocast_enabled /
        torch_autocast_dtype fall through to the active torch.autocast state so
        the engine re-enables (rather than disables) autocast during forward.
        """
        if getattr(accelerator.state, 'deepspeed_plugin', None) is None:
            return

        try:
            import deepspeed.runtime.torch_autocast as _ds_ac
            from deepspeed.runtime.engine import DeepSpeedEngine
        except ImportError:
            return

        if getattr(DeepSpeedEngine, '_ff_autocast_patched', False):
            return

        if hasattr(_ds_ac, 'validate_nested_autocast'):
            _ds_ac.validate_nested_autocast = lambda engine: None

        if hasattr(DeepSpeedEngine, 'torch_autocast_enabled'):
            _orig_enabled = DeepSpeedEngine.torch_autocast_enabled
            _orig_dtype = DeepSpeedEngine.torch_autocast_dtype

            def _patched_enabled(self):
                return _orig_enabled(self) or torch.is_autocast_enabled()

            def _patched_dtype(self):
                if not _orig_enabled(self) and torch.is_autocast_enabled():
                    return torch.get_autocast_gpu_dtype()
                return _orig_dtype(self)

            DeepSpeedEngine.torch_autocast_enabled = _patched_enabled
            DeepSpeedEngine.torch_autocast_dtype = _patched_dtype

        DeepSpeedEngine._ff_autocast_patched = True

    @abstractmethod
    def start(self, *args, **kwargs):
        """Start training process."""
        pass

    @abstractmethod
    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Stages 4--5: finalize rewards, compute advantages, and log metrics (no policy gradients).

        Algorithms that need extra batching before the loss (e.g. DPO chosen/rejected pairs) may
        perform that work in :meth:`optimize` after advantages are on each sample.
        """
        pass

    @abstractmethod
    def optimize(self, *args, **kwargs):
        """Update policy model"""
        pass

    def _order_samples_for_optimize(
        self, samples: List[BaseSample], inner_epoch: int
    ) -> List[BaseSample]:
        """Return the per-inner-epoch sample ordering for the optimize loop.

        When ``training_args.shuffle_samples`` is False, the rollout-pack order is
        preserved so each training micro-batch packs exactly the samples of its
        corresponding rollout ``inference`` pack. For adapters whose batched forward
        is pack-composition-dependent (e.g. Bagel/NaViT packing), this keeps the
        bf16 forward bit-identical between rollout and training (on-policy ratio==1).
        """
        if not self.training_args.shuffle_samples:
            return samples
        perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
        perm = torch.randperm(len(samples), generator=perm_gen)
        return [samples[i] for i in perm]

    def _maybe_offload_samples_to_cpu(self, samples: List[BaseSample]) -> None:
        """Offload each sample's tensors to pinned CPU when offload is enabled.

        Producer half of the CPU-offload pipeline; keeps the rollout buffer's GPU
        peak bounded. Must run BEFORE ``reward_buffer.add_samples`` so the recorded
        ``sync_event`` captures "D2H complete + data on CPU" for async reward
        workers. Uses pinned CPU + blocking D2H so the later per-micro-batch H2D
        reload (``_iter_prefetched_batches``) can be issued asynchronously. No-op
        when ``training_args.offload_samples_to_cpu`` is False (default).
        """
        if not self.training_args.offload_samples_to_cpu:
            return
        for sample in samples:
            sample.to('cpu', pin_memory=True)

    def _iter_prefetched_batches(
        self,
        samples: List[BaseSample],
        per_device_batch_size: int,
    ) -> Iterator[StackedSampleBatch]:
        """Yield device-resident stacked micro-batches for the optimize loop.

        Each yielded :class:`StackedSampleBatch` also exposes the moved per-sample
        objects it was stacked from via ``batch.samples`` -- callers that need
        per-sample access (e.g. OPD teacher routing / ``mu_teacher`` write-back)
        read that, with no second move or a redundant side index.

        When samples are CPU-offloaded (pinned), the next micro-batch's H2D copy
        runs on a dedicated copy stream to overlap the current batch's compute;
        ``wait_stream`` ensures the batch is fully copied before use and
        ``record_stream`` keeps it alive until the default stream is done.
        Otherwise (offload off, no CUDA, or a single batch) it is a plain blocking
        stack. Numerically equivalent either way; only data-movement timing changes.

        Yields:
            StackedSampleBatch: a stacked micro-batch (its source samples are at
            ``batch.samples``).
        """
        device = self.accelerator.device
        starts = list(range(0, len(samples), per_device_batch_size))

        use_prefetch = (
            torch.cuda.is_available()
            and self.training_args.offload_samples_to_cpu
            and len(starts) > 1
        )
        if not use_prefetch:
            for start in starts:
                batch_samples = [
                    sample.to(device)
                    for sample in samples[start:start + per_device_batch_size]
                ]
                yield BaseSample.stack(batch_samples)
            return

        copy_stream = torch.cuda.Stream(device)
        compute_stream = torch.cuda.current_stream(device)

        def _load(start: int) -> StackedSampleBatch:
            with torch.cuda.stream(copy_stream):
                moved = [
                    sample.to(device, non_blocking=True)
                    for sample in samples[start:start + per_device_batch_size]
                ]
                return BaseSample.stack(moved)

        next_batch = _load(starts[0])
        for i, _ in enumerate(starts):
            batch = next_batch
            compute_stream.wait_stream(copy_stream)  # batch H2D complete before use
            _record_stream_on_batch(batch, compute_stream)  # keep alive for compute stream
            if i + 1 < len(starts):
                next_batch = _load(starts[i + 1])  # prefetch next, overlaps compute
            yield batch

    def sample_batch(
        self,
        batch: Dict[str, Any],
        reward_buffer: Optional[RewardBuffer] = None,
        **extra_inference_kwargs,
    ) -> List[BaseSample]:
        """Unified single-batch sampling pipeline.

        Encapsulates the standard post-inference steps that every trainer
        repeats in its sampling loop:

            1. Merge training/eval args + batch + extra kwargs
            2. ``filter_kwargs`` → ``adapter.inference()``
            3. Inject dataset metadata into samples
            4. Optionally offload samples to CPU
            5. Optionally feed samples into a ``RewardBuffer``

        Subclasses may override this method to customize the per-batch
        pipeline (e.g. adding custom post-processing or using a different
        inference call). The default implementation is sufficient for most
        algorithms.

        Args:
            batch: DataLoader batch dict (contains prompt, metadata, etc.)
            reward_buffer: If provided, ``add_samples()`` is called automatically.
            **extra_inference_kwargs: Passed to ``adapter.inference()`` after
                filtering. Common keys: ``compute_log_prob``,
                ``trajectory_indices``, ``generator``.

        Returns:
            List of generated ``BaseSample`` instances with metadata injected.
        """
        sample_kwargs = {**self.training_args, **extra_inference_kwargs, **batch}
        sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
        sample_batch = self.adapter.inference(**sample_kwargs)

        # Defensively reset applicable_rewards on every newly produced sample.
        # The factory default is an empty set, but if any future trainer
        # reuses sample objects across epochs (e.g. a sample buffer), stale
        # bookkeeping from prior epochs would corrupt aggregation.  Cheap
        # to do unconditionally; makes the contract explicit.
        for s in sample_batch:
            s.applicable_rewards = set()

        # Inject dataset metadata (e.g. geneval_metadata) into samples' extra_kwargs
        self._inject_batch_metadata(sample_batch, batch)

        # Offload to CPU before reward buffer sees them
        self._maybe_offload_samples_to_cpu(sample_batch)

        # Feed into reward buffer for async/sync reward computation
        if reward_buffer is not None:
            reward_buffer.add_samples(sample_batch)

        return sample_batch

    @staticmethod
    def _augment_batch_with_source(
        batch: Dict[str, Any],
        source_name: str,
        source_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Stamp source routing keys onto a batch dict for downstream propagation.

        Plain DataLoaders (eval, future standalone sampling) lack the
        automatic ``__source__`` / ``__source_id__`` injection that
        ``MultiSourceTrainDataLoader`` provides.  Call this before
        ``sample_batch`` so ``_inject_batch_metadata`` can propagate
        source onto every generated sample via its existing K-repeat
        broadcast logic.
        """
        batch = dict(batch)
        B = len(batch["prompt"])
        batch["__source__"] = [source_name] * B
        if source_id is not None:
            batch["__source_id__"] = [source_id] * B
        return batch

    @staticmethod
    def _inject_batch_metadata(
        samples: List[BaseSample],
        batch: Dict[str, Any],
    ) -> None:
        """Inject dataset metadata into generated samples' extra_kwargs.

        Bridges the gap between dataset JSONL fields and reward model kwargs:
        non-preprocess fields from the dataloader batch are copied into each
        sample's ``extra_kwargs``, making them accessible to reward models via
        ``filter_kwargs(model.__call__, **sample)``.

        Convention: complex metadata values are stored as JSON strings in the
        JSONL for Arrow serialization safety. Reward models parse them with
        ``json.loads()`` as needed.

        Also propagates the per-batch ``__source__`` / ``__source_id__``
        (multi-source training only — populated by
        ``MultiSourceTrainDataLoader`` in ``data_utils/loader.py``) onto
        the typed ``BaseSample.source`` / ``BaseSample.source_id`` fields.
        Drives both the ``RewardProcessor`` gate and the
        ``AdvantageProcessor`` applicability mask.

        No-op when ``batch['metadata']``, ``batch['__source__']`` and
        ``batch['__source_id__']`` are all absent or empty.

        Args:
            samples: Generated samples from ``adapter.inference()``.
            batch: The dataloader batch dict (may contain ``metadata`` /
                ``__source__`` / ``__source_id__`` keys).
        """
        # Per-prompt ratio used for both metadata and __source__ broadcasting.
        # Some adapters generate K replicates per prompt (group_size > 1) so
        # one batch row maps to several samples.
        sources = batch.get('__source__')
        source_ids = batch.get('__source_id__')
        metadata_list = batch.get(METADATA_COLUMN)
        if not metadata_list and not sources and not source_ids:
            return
        if not samples:
            return

        # Pick a length-bearing reference for the broadcast ratio.
        if metadata_list:
            B = len(metadata_list)
        elif sources:
            B = len(sources)
        elif source_ids:
            B = len(source_ids)
        else:
            return
        samples_per_prompt = len(samples) // B
        if samples_per_prompt == 0:
            return

        for i, sample in enumerate(samples):
            batch_idx = i // samples_per_prompt
            if batch_idx >= B:
                continue
            if metadata_list:
                meta = metadata_list[batch_idx]
                if isinstance(meta, dict):
                    sample.extra_kwargs[METADATA_COLUMN] = json.dumps(meta, default=json_default)
            if sources:
                # Homogeneous within a batch in this PR; per-sample shape
                # leaves room for future PRs that may interleave within a
                # batch without a code change.
                sample.source = sources[batch_idx]
            if source_ids:
                sample.source_id = source_ids[batch_idx]

    # ============================ Public Sampling API ============================

    def generate_samples(
        self,
        reward_buffer: Optional[RewardBuffer] = None,
        compute_log_prob: bool = False,
        trajectory_indices: Optional[List[int]] = None,
        **extra_inference_kwargs,
    ) -> List[BaseSample]:
        """Complete one epoch of sample generation.

        Standard pipeline::

            adapter.rollout() → clear buffer → loop(dataloader) {
                sample_batch() → extend samples
            }

        Subclasses call this from their ``sample()`` method with
        algorithm-specific parameters. For fully custom sampling logic
        (e.g. paired generation), override this method directly.

        Args:
            reward_buffer: Buffer for reward computation. Cleared at start
                and fed after each batch automatically.
            compute_log_prob: Whether to store log-probabilities during inference.
            trajectory_indices: Which timestep positions to store in each sample.
                ``[-1]`` = final latent only (default for most algorithms).
                Full list = store all (GRPO needs this for PPO ratio).
                ``None`` = no trajectory recording (used during evaluation).
            **extra_inference_kwargs: Forwarded to ``adapter.inference()``
                after ``filter_kwargs``. Common keys: ``generator``.

        Returns:
            All generated samples for this epoch.

        Note:
            Trainers that override ``generate_samples`` instead of just
            ``sample()`` must still call :meth:`sample_batch` per batch
            so :meth:`_inject_batch_metadata` propagates ``__source__``
            onto every sample.  An end-of-loop runtime check verifies
            this in multi-source mode.
        """
        if self.dataloader is None:
            raise RuntimeError(
                "generate_samples() called but no training dataloader exists. "
                "`data.datasets` has no entry with `train: enabled` (eval-only "
                "config); a trainer should not enter the sampling loop here."
            )

        self.adapter.rollout()
        if reward_buffer is not None:
            reward_buffer.clear()

        # Multi-source: reseed the per-source schedule + every per-source
        # sampler so replays of the same epoch are reproducible. No-op
        # for the bare DataLoader (no `set_epoch`).
        if hasattr(self.dataloader, "set_epoch"):
            self.dataloader.set_epoch(self.epoch)

        samples: List[BaseSample] = []
        data_iter = iter(self.dataloader)

        with torch.no_grad(), self.autocast():
            for _ in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_batch = self.sample_batch(
                    batch,
                    reward_buffer=reward_buffer,
                    compute_log_prob=compute_log_prob,
                    trajectory_indices=trajectory_indices,
                    **extra_inference_kwargs,
                )
                samples.extend(sample_batch)

        # Multi-source invariant: when more than one training source is
        # active, batches flow through `MultiSourceTrainDataLoader`, which
        # injects `__source__` so every sample carries `source`. Single-source
        # configs use a bare DataLoader (no injection) and the reward gate
        # treats `source is None` as "applies to all" — so the check must NOT
        # fire there. This catches a trainer that overrode generate_samples
        # but bypassed sample_batch / _inject_batch_metadata.
        if len(self.train_dataloaders_by_source) > 1 and samples:
            missing = [
                i for i, s in enumerate(samples)
                if s.source is None
            ]
            if missing:
                raise RuntimeError(
                    f"Multi-source training: {len(missing)} sample(s) at indices "
                    f"{missing[:5]}{'...' if len(missing) > 5 else ''} are missing "
                    "`source`. Did a trainer override "
                    "`generate_samples` without going through `sample_batch` "
                    "(which calls `_inject_batch_metadata`)?"
                )

        return samples

    def evaluate(self) -> None:
        """Evaluation loop: a single, unified per-dataset path.

        For every eval-eligible entry in ``data.datasets`` (which now
        includes the canonicalized legacy ``data.dataset_dir`` when a
        ``test.jsonl`` exists):

        1. Generate samples using the dataset's DataLoader with per-dataset
           eval overrides (resolution, guidance_scale, num_inference_steps).
        2. Compute rewards via the dataset-specific RewardBuffer.
        3. Gather rewards across ranks.
        4. Log metrics under ``eval/{dataset_name}/reward_{name}_{stat}``.

        Logs are flushed per-dataset to avoid holding all generated samples
        in memory simultaneously.  Uses EMA parameters (if available) and
        eval-specific config (resolution, inference steps, guidance scale).

        No-op when ``self.eval_dataloaders`` is empty.
        """
        if not self.eval_dataloaders:
            return

        self.adapter.eval()

        with torch.no_grad(), self.autocast(), self.adapter.use_ema_parameters():
            for dataset_name, dataloader in self.eval_dataloaders.items():
                buffer = self.eval_dataset_reward_buffers.get(dataset_name)
                if buffer is None:
                    logger.warning(
                        f"No reward buffer for eval dataset '{dataset_name}', skipping."
                    )
                    continue
                buffer.clear()
                all_samples: List[BaseSample] = []

                # Merge per-dataset eval overrides with shared eval_args
                ed_config = self._eval_dataset_configs[dataset_name]
                eval_kwargs = ed_config.eval.get_merged_eval_kwargs(self.eval_args) if ed_config.eval else dict(self.eval_args)

                for batch in tqdm(
                    dataloader,
                    desc=f'Eval/{dataset_name}',
                    disable=not self.show_progress_bar,
                ):
                    batch = self._augment_batch_with_source(
                        batch, dataset_name, ed_config.source_id
                    )
                    generator = create_generator_by_prompt(
                        batch['prompt'], self.training_args.seed
                    )
                    samples = self.sample_batch(
                        batch,
                        reward_buffer=buffer,
                        compute_log_prob=False,
                        generator=generator,
                        trajectory_indices=None,
                        **eval_kwargs,
                    )
                    all_samples.extend(samples)

                rewards = buffer.finalize(store_to_samples=True, split='pointwise')

                # Gather across ranks
                rewards_tensors = {
                    k: torch.as_tensor(v).to(self.accelerator.device)
                    for k, v in rewards.items()
                }
                gathered_rewards = {
                    k: self.accelerator.gather(v).cpu().numpy()
                    for k, v in rewards_tensors.items()
                }

                # Log per-dataset immediately to avoid accumulating all samples in memory
                if self.accelerator.is_main_process:
                    log_data: Dict[str, Any] = {}
                    for k, v in gathered_rewards.items():
                        log_data[f'eval/{dataset_name}/reward_{k}_mean'] = np.mean(v)
                        log_data[f'eval/{dataset_name}/reward_{k}_std'] = np.std(v)
                    log_data[f'eval/{dataset_name}/samples'] = all_samples
                    self.log_data(log_data, step=self.step)

        self.accelerator.wait_for_everyone()

    def save_checkpoint(self, save_directory: str, epoch: Optional[int] = None):
        """Save trainer state to a specific path."""
        if epoch is not None:
            save_directory = os.path.join(save_directory, f"checkpoint-{epoch}")

        self.adapter.save_checkpoint(
            save_directory=save_directory,
            model_only=self.log_args.save_model_only,
        )

        self.accelerator.wait_for_everyone()

    def load_checkpoint(
            self,
            path: str,
            resume_type: Optional[Literal['lora', 'full', 'state']] = None,
        ):
        """Load trainer state from a specific path."""
        self.adapter.load_checkpoint(
            path=path,
            strict=True,
            resume_type=resume_type,
        )
        self.accelerator.wait_for_everyone()

    def cleanup(self) -> None:
        """Initiate non-blocking shutdown of async reward workers.

        Called on KeyboardInterrupt to cancel pending futures and signal
        executor threads to stop. This does NOT wait for threads to finish;
        the caller is expected to follow with os._exit() which will forcefully
        reclaim all resources including GPU memory.
        """
        # Training-side reward buffer.
        train_buf = getattr(self, 'reward_buffer', None)
        if train_buf is not None:
            train_buf.shutdown(wait=False, cancel_futures=True)

        # Per-eval-dataset reward buffers.
        for buf in getattr(self, 'eval_dataset_reward_buffers', {}).values():
            if buf is not None:
                buf.shutdown(wait=False, cancel_futures=True)
