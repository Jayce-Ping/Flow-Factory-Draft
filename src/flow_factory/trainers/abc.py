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
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union, Literal
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
from ..data_utils.loader import get_dataloader
from ..rewards import load_reward_model, BaseRewardModel, MultiRewardLoader, RewardProcessor, RewardBuffer
from ..advantage import AdvantageProcessor
from ..logger import load_logger, LogFormatter
from ..samples import BaseSample
from ..utils.logger_utils import setup_logger
from ..utils.base import filter_kwargs, create_generator_by_prompt

logger = setup_logger(__name__)

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

        # Initialize all reward model instances
        self.reward_loader = MultiRewardLoader(
            reward_args=self.config.reward_args,
            accelerator=self.accelerator,
            eval_reward_args=self.config.eval_reward_args,
        ).load()
        # Get training & eval reward models
        self.reward_models = self.reward_loader.get_training_reward_models()
        self.eval_reward_models = self.reward_loader.get_eval_reward_models()
        train_reward_configs = self.reward_loader.get_reward_configs('train')
        eval_reward_configs = self.reward_loader.get_reward_configs('eval')
        # Initialize reward processor
        group_on_same_rank = self.config.data_args.sampler_type == "group_contiguous"
        self.reward_processor = RewardProcessor(
            accelerator=self.accelerator,
            reward_models=self.reward_models,
            reward_configs=train_reward_configs,
            tokenizer=self.adapter.tokenizer, # For prompt encoding/decoding,
            group_on_same_rank=group_on_same_rank,
            verbose=self.log_args.verbose,
        )
        self.eval_reward_processor = RewardProcessor(
            accelerator=self.accelerator,
            reward_models=self.eval_reward_models,
            reward_configs=eval_reward_configs,
            tokenizer=self.adapter.tokenizer, # For prompt encoding/decoding
            group_on_same_rank=group_on_same_rank,
            verbose=self.log_args.verbose,
        )
        # Initialize reward buffers
        self.reward_buffer = RewardBuffer(
            self.reward_processor, self.training_args.group_size,
        )
        self.eval_reward_buffer = RewardBuffer(
            self.eval_reward_processor, self.training_args.group_size,
        )

        # Initialize advantage processor
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
        )

        return self.reward_models, self.eval_reward_models

    def _init_dataloader(self) -> Tuple[DataLoader, Union[None, DataLoader]]:
        # Move text-encoder & vae to GPU for dataloader encoding
        self.adapter.on_load_components(
            components=self.adapter.preprocessing_modules,
            device=self.accelerator.device
        )
        dataloader, test_dataloader = get_dataloader(
            config=self.config,
            accelerator=self.accelerator,
            preprocess_func=self.adapter.preprocess_func,
        )
        # Offload text-encoder after dataloader encoding
        self.adapter.off_load_components(
            components=self.adapter.preprocessing_modules,
        )

        self.accelerator.wait_for_everyone()

        return dataloader, test_dataloader
    
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
        self.dataloader, self.test_dataloader = self._init_dataloader()
        self.optimizer = self._init_optimizer()
        # Prepare everything with accelerator
        # Dynamically get all trainable modules from target_module_map
        trainable_module_names = list(self.adapter.target_module_map.keys())
        trainable_modules = [
            getattr(self.adapter, name) 
            for name in trainable_module_names 
            if hasattr(self.adapter, name) and getattr(self.adapter, name) is not None
        ]
        # Prepare trainable modules + optimizer + test_dataloader
        to_prepare = trainable_modules + [self.optimizer]
        if self.test_dataloader is not None:
            to_prepare.append(self.test_dataloader)

        prepared = self.accelerator.prepare(*to_prepare)
        # Here, `self.dataloader` is not prepared since it has been handled with DistributedKRepeatSampler
        for i, name in enumerate(trainable_module_names):
            if hasattr(self.adapter, name) and getattr(self.adapter, name) is not None:
                self.adapter.set_component(name, prepared[i])

        self.optimizer = prepared[len(trainable_modules)]
        if self.test_dataloader is not None:
            self.test_dataloader = prepared[len(trainable_modules) + 1]

        # Load inference modules, excluding already-prepared ones
        self._load_inference_components(trainable_module_names)
        
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

    def _maybe_offload_samples_to_cpu(self, samples: List[BaseSample]) -> None:
        """Move every sample's tensor fields to CPU when ``offload_samples_to_cpu`` is enabled.

        Producer-side half of the CPU-offload + lazy-reload pipeline: samples
        leave ``sample()`` already on CPU so that the GPU peak from the rollout
        buffer is bounded by a single batch worth of inference activations.

        Must be called BEFORE ``self.reward_buffer.add_samples(...)`` so that
        the buffer's recorded ``sync_event`` captures "D2H complete + data
        ready on CPU"; downstream reward workers (sync or async) then see a
        deterministic CPU-resident state and trigger their own H2D inside
        ``RewardProcessor`` (see ``move_tensors_to_device`` in
        ``utils/base.py``).

        No-op when ``training_args.offload_samples_to_cpu`` is False
        (default), preserving the legacy GPU-resident behaviour.

        Args:
            samples: Newly generated samples for the current sample loop iteration.
        """
        if not self.training_args.offload_samples_to_cpu:
            return
        for sample in samples:
            sample.to('cpu')

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

        # Inject dataset metadata (e.g. geneval_metadata) into samples' extra_kwargs
        self._inject_batch_metadata(sample_batch, batch)

        # Offload to CPU before reward buffer sees them
        self._maybe_offload_samples_to_cpu(sample_batch)

        # Feed into reward buffer for async/sync reward computation
        if reward_buffer is not None:
            reward_buffer.add_samples(sample_batch)

        return sample_batch

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

        No-op when ``batch['metadata']`` is absent or empty.

        Args:
            samples: Generated samples from ``adapter.inference()``.
            batch: The dataloader batch dict (may contain a ``metadata`` key).
        """
        metadata_list = batch.get('metadata')
        if not metadata_list or not samples:
            return
        batch_size = len(metadata_list)
        samples_per_prompt = len(samples) // batch_size
        if samples_per_prompt == 0:
            return
        for i, sample in enumerate(samples):
            batch_idx = i // samples_per_prompt
            if batch_idx < batch_size:
                meta = metadata_list[batch_idx]
                if isinstance(meta, dict):
                    sample.extra_kwargs.update(meta)

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
        """
        self.adapter.rollout()
        if reward_buffer is not None:
            reward_buffer.clear()

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

        return samples

    def evaluate(self) -> None:
        """Evaluation loop: generate samples with eval settings and log reward statistics.

        Uses EMA parameters (if available) and eval-specific config (resolution,
        inference steps, guidance scale). Rewards are gathered across all ranks
        and logged as mean/std.

        Subclasses can override for custom evaluation logic. This method is a
        no-op when ``self.test_dataloader`` is None.
        """
        if self.test_dataloader is None:
            return

        self.adapter.eval()
        self.eval_reward_buffer.clear()

        with torch.no_grad(), self.autocast(), self.adapter.use_ema_parameters():
            all_samples: List[BaseSample] = []

            for batch in tqdm(
                self.test_dataloader,
                desc='Evaluating',
                disable=not self.show_progress_bar,
            ):
                generator = create_generator_by_prompt(batch['prompt'], self.training_args.seed)
                samples = self.sample_batch(
                    batch,
                    reward_buffer=self.eval_reward_buffer,
                    compute_log_prob=False,
                    generator=generator,
                    trajectory_indices=None,
                    **self.eval_args,
                )
                all_samples.extend(samples)

            rewards = self.eval_reward_buffer.finalize(store_to_samples=True, split='pointwise')

            # Gather across ranks and log
            rewards = {k: torch.as_tensor(v).to(self.accelerator.device) for k, v in rewards.items()}
            gathered_rewards = {
                k: self.accelerator.gather(v).cpu().numpy()
                for k, v in rewards.items()
            }

            if self.accelerator.is_main_process:
                _log_data = {f'eval/reward_{k}_mean': np.mean(v) for k, v in gathered_rewards.items()}
                _log_data.update({f'eval/reward_{k}_std': np.std(v) for k, v in gathered_rewards.items()})
                _log_data['eval_samples'] = all_samples
                self.log_data(_log_data, step=self.step)
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
        for buf in (
            getattr(self, 'reward_buffer', None),
            getattr(self, 'eval_reward_buffer', None),
        ):
            if buf is not None:
                buf.shutdown(wait=False, cancel_futures=True)