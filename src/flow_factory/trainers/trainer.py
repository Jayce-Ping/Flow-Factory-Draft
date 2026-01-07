# src/flow_factory/models/trainer.py
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
from functools import partial
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from dataclasses import dataclass
from PIL import Image
from diffusers.utils.outputs import BaseOutput
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration

from ..hparams import *
from ..models.adapter import BaseAdapter
from ..data_utils.loader import get_dataloader
from ..rewards import load_reward_model, BaseRewardModel
from ..logger import load_logger
from ..utils.logger_utils import setup_logger

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
        self.reward_args = config.reward_args

        self.eval_args = config.eval_args
        self.eval_reward_args = config.eval_reward_args or config.reward_args # If `eval_reward_args` is not given, use `reward_args`

        self.adapter = adapter
        self.epoch = 0
        self.step = 0

        self._initialization()
        self.adapter.post_init()
        self._init_logging_backend()

        self.autocast = partial(
            torch.autocast,
            device_type=accelerator.device.type,
            dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16
        )

        if self.accelerator.is_local_main_process:
            self.adapter.log_trainable_parameters()


    def log_data(self, data: Dict[str, Any], step: int):
        """Log data using the initialized logger."""
        if self.logger is not None:
            self.logger.log_data(data, step=step)
    
    def _init_logging_backend(self):
        if not self.accelerator.is_main_process:
            self.logger = None
            return
        """Initialize logging backend if specified."""
        self.logger = load_logger(self.config)

    def _init_reward_model(self) -> BaseRewardModel:
        """Initialize reward model from configuration."""

        # If DeepSpeed ZeRO-3 is enabled, the reward model will be somehow sharded.
        # We need to disable ZeRO-3 init context when loading the model to avoid issues
        # NOTE: This bug persists even with this context manager. DONOT USE ZeRO-3.
        # A possible solution: use DeepSpeed GatherParamter manually in the reward_model's `forward`.
        self.reward_model = load_reward_model(
            reward_args=self.reward_args,
            accelerator=self.accelerator,
        )
        if (
            self.eval_reward_args is None # This should never happen since __init__ has handled this
            or self.eval_reward_args is self.reward_args # This happens when eval_reward_args is not given
            or self.eval_reward_args.reward_model == self.reward_args.reward_model # Same reward configuration specified
        ):
            self.eval_reward_model = self.reward_model
        else:
            self.eval_reward_model = load_reward_model(
                reward_args=self.eval_reward_args,
                accelerator=self.accelerator,
            )
        return self.reward_model

    def _init_dataloader(self) -> Tuple[DataLoader, Union[None, DataLoader]]:
        # Move text-encoder & vae to GPU for dataloader encoding

        self.adapter.on_load_text_encoders(self.accelerator.device)
        self.adapter.on_load_vae(self.accelerator.device)
        dataloader, test_dataloader = get_dataloader(
            config=self.config,
            accelerator=self.accelerator,
            preprocess_func=self.adapter.preprocess_func,
        )
        # Offload text-encoder after dataloader encoding
        self.adapter.off_load_text_encoders()
        self.adapter.off_load_vae()

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

    def _initialization(self):
        # Fix for FSDP, synchronize frozen components like text encoder & VAE.
        # Otherwise they may be uninitialized on Rank > 0.
        if self.adapter._is_fsdp_cpu_efficient_loading():
            logger.info("FSDP CPU Efficient Loading detected. Synchronizing frozen components...")
            self.adapter.on_load(self.accelerator.device)
            self._synchronize_frozen_components()
            self.adapter.off_load()

        # Init dataloader and optimizer
        self.dataloader, self.test_dataloader = self._init_dataloader()
        self.optimizer = self._init_optimizer()
        # Prepare everything with accelerator
        # Here, `self.dataloader` is not prepared since it has been handled with DistributedKRepeatSampler
        to_prepare = [self.adapter.transformer, self.optimizer, self.test_dataloader]
        to_prepare = [x for x in to_prepare if x is not None]
        prepared = self.accelerator.prepare(*to_prepare)
        self.adapter.transformer, self.optimizer = prepared[:2]
        if len(prepared) > 2:
            self.test_dataloader = prepared[2]

        # Load Vae for image decoding
        self.adapter.on_load_vae(self.accelerator.device)
        
        # Initialize reward model
        self._init_reward_model()

    def _synchronize_frozen_components(self):
        """
        Force broadcast frozen components (Text Encoder / VAE) from Rank 0 to all other ranks.
        This prevents Rank > 0 from having uninitialized (zero/nan) weights when they are NOT wrapped by FSDP.
        """
        if self.accelerator.num_processes <= 1:
            return

        logger.info(f"[Rank {self.accelerator.process_index}] Synchronizing frozen components...")
        
        # 1. Synchronize Text Encoders
        for i, encoder in enumerate(self.adapter.text_encoders):
            logger.info(f"Broadcasting Text Encoder {i} weights from Rank 0...")
            for param in encoder.parameters():
                # Ensure param is on the device for communication
                param.data = param.data.to(self.accelerator.device)
                dist.broadcast(param.data, src=0)
        
        # 2. Synchronize VAE (Optional, but recommended)
        if hasattr(self.adapter, 'vae') and self.adapter.vae is not None:
             logger.info(f"Broadcasting VAE weights from Rank 0...")
             for param in self.adapter.vae.parameters():
                param.data = param.data.to(self.accelerator.device)
                dist.broadcast(param.data, src=0)
        
        # Barrier to ensure everyone is done
        self.accelerator.wait_for_everyone()
        logger.info(f"[Rank {self.accelerator.process_index}] Frozen components synchronized.")

    @abstractmethod
    def start(self, *args, **kwargs):
        """Start training process."""
        pass

    @abstractmethod
    def optimize(self, *args, **kwargs):
        """Update policy model"""
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluation for one epoch."""
        pass

    def save_checkpoint(self, save_directory: str, epoch: Optional[int] = None):
        """Save trainer state to a specific path."""
        if epoch is not None:
            save_directory = os.path.join(save_directory, f"checkpoint-{epoch}")

        self.adapter.save_checkpoint(
            save_directory=save_directory,
            model_only=self.log_args.save_model_only,
        )

        self.accelerator.wait_for_everyone()

    def load_checkpoint(self, path: str):
        """Load trainer state from a specific path."""
        self.adapter.load_checkpoint(
            path=path,
            strict=True,
            model_only=not self.model_args.resume_training_state,
        )
        self.accelerator.wait_for_everyone()