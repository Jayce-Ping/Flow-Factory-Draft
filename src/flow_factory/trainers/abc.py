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
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from diffusers.utils.outputs import BaseOutput
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..acceleration import BaseAccelerator, build_accelerator, validate_accelerator
from ..advantage import AdvantageProcessor
from ..data_utils.dataset import METADATA_COLUMN
from ..data_utils.loader import (
    get_eval_dataloaders,
    get_train_dataloader,
)
from ..hparams import *
from ..logger import LogFormatter, load_logger
from ..models.abc import BaseAdapter
from ..models.model_bundle import ModelBundle, RoutedComponentProxy
from ..rewards import (
    BaseRewardModel,
    MultiRewardLoader,
    RewardBuffer,
    RewardProcessor,
    load_reward_model,
)
from ..samples import BaseSample, LatentState, NoisedState, StackedSampleBatch
from ..utils.base import (
    create_generator,
    create_generator_by_prompt,
    filter_kwargs,
    json_default,
    visit_tensor_leaves,
)
from ..utils.dist import reduce_loss_info
from ..utils.logger_utils import setup_logger
from ..utils.noise_schedule import TimeSampler
from .role_optimization import (
    OptimizationRole,
    RoleOptimizationCoordinator,
    RoleOptimizerConfig,
    RolePhase,
    RoleUpdatePlan,
)

logger = setup_logger(__name__)


def validate_supported_distributed_plan(accelerator: Accelerator) -> None:
    """Reject distributed plans this framework cannot train correctly.

    Supported plans are DDP, FSDP, and DeepSpeed ZeRO-1/2. ZeRO-3 shards
    parameters across ranks, which breaks reward-model loading and the
    frozen-component synchronization this framework relies on. Rejecting it here
    fails at startup rather than partway through a training step.

    Args:
        accelerator: Configured Accelerate accelerator.

    Raises:
        ValueError: If DeepSpeed ZeRO-3 is configured.
    """
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        return
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    if deepspeed_plugin is None:
        return
    if deepspeed_plugin.zero_stage == 3:
        raise ValueError(
            "DeepSpeed ZeRO-3 is not supported by Flow-Factory: parameter sharding breaks "
            "reward-model loading and frozen-component synchronization. Expected ZeRO-1/2, "
            "FSDP, or DDP; received DeepSpeed stage 3."
        )

_MULTIROLE_METADATA_FILENAME = "flow_factory_multirole_metadata.json"
_MULTIROLE_METADATA_VERSION = 1
_MULTIROLE_STATE_KEYS = {
    "version",
    "metadata",
    "coordinator",
    "trainer_step",
    "parameter_emas",
}


def _record_stream_on_batch(value: Any, stream: "torch.cuda.Stream") -> None:
    """Record ``stream`` on every CUDA tensor in a stacked batch.

    Required for the copy-stream prefetch: it stops the caching allocator from
    reusing copy-stream-produced tensors until the consuming stream is done.
    """
    visit_tensor_leaves(value, lambda t: t.record_stream(stream) if t.is_cuda else None)


class _MultiRoleCheckpointState:
    """Delegate Accelerate custom checkpoint state to one trainer."""

    def __init__(self, trainer: "BaseTrainer") -> None:
        self._trainer = trainer

    def state_dict(self) -> Dict[str, Any]:
        """Return multi-role counters and defensive compatibility metadata."""
        return self._trainer._multirole_state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore multi-role counters after Accelerate restores prepared state."""
        self._trainer._load_multirole_state_dict(state)

    def prepare_save(self, output_dir: str) -> None:
        """Validate a closed boundary and write metadata before Accelerate saves."""
        try:
            custom_state = self.state_dict()
        except RuntimeError as error:
            raise RuntimeError(
                "cannot checkpoint invalid multi-role training state before "
                f"model/optimizer save; validation reported: {error}"
            ) from error
        accelerator = self._trainer.accelerator
        save_on_each_node = accelerator.project_configuration.save_on_each_node
        should_write_metadata = accelerator.is_main_process or (
            save_on_each_node and accelerator.is_local_main_process
        )
        if should_write_metadata:
            os.makedirs(output_dir, exist_ok=True)
            metadata_path = os.path.join(output_dir, _MULTIROLE_METADATA_FILENAME)
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(custom_state["metadata"], metadata_file, indent=2, sort_keys=True)
                metadata_file.write("\n")

    def validate_load(self, input_dir: str) -> None:
        """Validate metadata before Accelerate can mutate prepared state."""
        metadata_path = os.path.join(input_dir, _MULTIROLE_METADATA_FILENAME)
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                "multi-role metadata compatibility gate expected file "
                f"{metadata_path!r}, received missing file"
            )
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        self._trainer._validate_multirole_metadata(metadata)


class BaseTrainer(ABC):
    """
    Abstract Base Class for Flow-Factory trainers.
    """

    # RL paradigm of this algorithm (``constraints.md`` #7). Read by the
    # acceleration validator to gate lossy rollout accelerators: only
    # 'decoupled' / 'distillation' trainers may use them. Concrete trainers
    # MUST override this; leaving it None disables lossy acceleration.
    paradigm: ClassVar[Optional[Literal["coupled", "decoupled", "distillation"]]] = None

    def __init__(
        self,
        accelerator: Accelerator,
        config: Arguments,
        adapter: BaseAdapter,
    ):
        validate_supported_distributed_plan(accelerator)

        self.accelerator = accelerator
        self.config = config
        self.log_args = config.log_args
        self.model_args = config.model_args

        self.training_args = config.training_args
        self.eval_args = config.eval_args

        self.reward_args = config.reward_args
        self.eval_reward_args = (
            config.eval_reward_args or config.reward_args
        )  # If `eval_reward_args` is not given, use `reward_args`

        self.adapter = adapter
        self.epoch = 0
        self.step = 0

        self._initialization()
        self._initialize_parameter_emas()
        self._register_multirole_checkpointing()
        self.adapter.post_init()
        # Apply persistent stage='both' accelerators last: after prepare, state-resume,
        # EMA, and reference-parameter setup, so e.g. torch.compile wraps the final
        # weights and keeps state_dict keys / parameter identity stable.
        self._apply_shared_acceleration()
        self._init_logging_backend()

        self._patch_deepspeed_autocast(accelerator)
        self.autocast = partial(
            torch.autocast,
            device_type=accelerator.device.type,
            dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16,
        )

        if self.accelerator.is_local_main_process:
            self.adapter.log_trainable_parameters()

    @property
    def show_progress_bar(self) -> bool:
        """Whether to show tqdm progress bars."""
        return self.log_args.verbose and self.accelerator.is_local_main_process

    def _initialize_parameter_emas(self) -> None:
        """Initialize optional trainer-owned parameter snapshots before state resume."""

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
            metrics = {
                k: v
                for k, v in ((k, LogFormatter.to_scalar(v)) for k, v in data.items())
                if v is not None
            }
            if metrics:
                parts = [f"[Step {step:04d} | Epoch {self.epoch:03d}]"]
                parts.extend(
                    (
                        f"{k}={int(v)}"
                        if isinstance(v, int) or (isinstance(v, float) and v.is_integer())
                        else f"{k}={v:.4f}"
                    )
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
        train_reward_configs = self.reward_loader.get_reward_configs("train")
        # Initialize reward processor (training side only — eval-side
        # processors are per-dataset, built below).
        group_on_same_rank = self.config.data_args.sampler_type == "group_contiguous"
        self.reward_processor = RewardProcessor(
            accelerator=self.accelerator,
            reward_models=self.reward_models,
            reward_configs=train_reward_configs,
            tokenizer=self.adapter.tokenizer,  # For prompt encoding/decoding,
            group_on_same_rank=group_on_same_rank,
            verbose=self.log_args.verbose,
        )
        # Initialize the training-side reward buffer.
        self.reward_buffer = RewardBuffer(
            self.reward_processor,
            self.training_args.group_size,
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
                        ds_processor,
                        self.training_args.group_size,
                    )

        # Initialize advantage processor.
        # `cfg.weight` is a Dict[str, float] after `_resolve_reward_weights`,
        # so reward_weights is Dict[reward_name, Dict[dataset_name, float]].
        self.advantage_processor = AdvantageProcessor(
            accelerator=self.accelerator,
            reward_weights={name: cfg.weight for name, cfg in train_reward_configs.items()},
            group_size=self.training_args.group_size,
            global_std=getattr(self.training_args, "global_std", True),
            sampler_type=self.config.data_args.sampler_type,
            verbose=self.log_args.verbose,
            source_id_to_name=self.config.data_args.source_id_to_name,
        )

        return self.reward_models, self.eval_reward_models

    def _init_dataloader(
        self,
    ) -> Tuple[Optional[Union[DataLoader, "MultiSourceTrainDataLoader"]], Dict[str, DataLoader]]:
        """Build train and eval dataloaders.

        Returns:
            Tuple of (train_dataloader, eval_dataloaders_by_name).
        """
        self.adapter.on_load_components(
            components=self.adapter.preprocessing_modules, device=self.accelerator.device
        )
        if self.adapter.uses_fsdp_cpu_efficient_loading():
            self._synchronize_frozen_components(self.adapter.preprocessing_modules)

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
        """Initialize one AdamW with one ordered parameter group per trainable role."""
        registry = self.adapter.model_role_registry
        trainable_role_names = tuple(
            role_name for role_name in registry.role_names if registry.get_spec(role_name).trainable
        )
        role_config_builder = getattr(self, "_role_optimizer_configs", None)
        if role_config_builder is None:
            role_configs = tuple(
                BaseTrainer._legacy_role_optimizer_config(self, role_name)
                for role_name in trainable_role_names
            )
        else:
            role_configs = role_config_builder()
        configured_role_names = tuple(config.role_name for config in role_configs)
        if configured_role_names != trainable_role_names:
            raise ValueError(
                "expected role optimizer configs to exactly match declared trainable roles "
                f"{trainable_role_names!r}, received {configured_role_names!r}"
            )

        parameter_groups = []
        self.optimization_roles = {}
        for group_id, config in enumerate(role_configs):
            parameters = registry.parameters(config.role_name)
            if not parameters:
                raise ValueError(
                    f"expected trainable role {config.role_name!r} to own optimizer "
                    "parameters, received none"
                )
            parameter_groups.append(
                {
                    "params": list(parameters),
                    "role_name": config.role_name,
                    "lr": config.learning_rate,
                    "betas": config.adam_betas,
                    "weight_decay": config.adam_weight_decay,
                    "eps": config.adam_epsilon,
                }
            )
            self.optimization_roles[config.role_name] = OptimizationRole(
                config=config,
                parameters=parameters,
                optimizer_group_ids=(group_id,),
            )

        self.optimizer = torch.optim.AdamW(parameter_groups)
        return self.optimizer

    def _role_optimizer_configs(self) -> Tuple[RoleOptimizerConfig, ...]:
        """Build role configs from nested arguments or legacy flat arguments."""
        required_roles = BaseTrainer._required_trainable_roles(self)
        if required_roles == ("generator",):
            return (BaseTrainer._legacy_role_optimizer_config(self, "generator"),)

        if getattr(self.training_args, "role_update_plan", None) is not None:
            update_plan = BaseTrainer._role_update_plan(self)
            plan_roles = {phase.role_name for phase in update_plan.phases}
            if plan_roles != set(required_roles):
                raise ValueError(
                    "expected role update plan roles to exactly match required trainable roles "
                    f"{required_roles!r}, received {tuple(plan_roles)!r}"
                )

        role_configs = []
        for role_name in required_roles:
            field_name = f"{role_name}_optimizer"
            if not hasattr(self.training_args, field_name):
                raise ValueError(
                    f"expected nested optimizer arguments train.{field_name} for "
                    f"required role {role_name!r}, received no such field"
                )
            optimizer_args = getattr(self.training_args, field_name)
            role_configs.append(
                RoleOptimizerConfig(
                    role_name=role_name,
                    learning_rate=optimizer_args.learning_rate,
                    adam_betas=optimizer_args.adam_betas,
                    adam_weight_decay=optimizer_args.adam_weight_decay,
                    adam_epsilon=optimizer_args.adam_epsilon,
                    max_grad_norm=optimizer_args.max_grad_norm,
                )
            )
        return tuple(role_configs)

    def _legacy_role_optimizer_config(self, role_name: str) -> RoleOptimizerConfig:
        """Build one role config from the existing flat training arguments."""
        return RoleOptimizerConfig(
            role_name=role_name,
            learning_rate=self.training_args.learning_rate,
            adam_betas=self.training_args.adam_betas,
            adam_weight_decay=self.training_args.adam_weight_decay,
            adam_epsilon=self.training_args.adam_epsilon,
            max_grad_norm=getattr(self.training_args, "max_grad_norm", 1.0),
        )

    def _finish_role_microbatch(self) -> bool:
        """Finish a role microbatch and advance public step for generator only."""
        role_name = self.role_optimization.active_role_name
        stepped = self.role_optimization.finish_microbatch()
        if stepped and role_name == "generator":
            self.step += 1
        return stepped

    def _rebind_prepared_optimization_roles(self) -> None:
        """Rebuild role ownership from prepared optimizer parameter identities."""
        optimizer_groups = self.optimizer.param_groups
        expected_group_ids = tuple(
            group_id
            for role in self.optimization_roles.values()
            for group_id in role.optimizer_group_ids
        )
        if tuple(sorted(expected_group_ids)) != tuple(range(len(optimizer_groups))):
            raise ValueError(
                "expected optimization roles to exhaust prepared optimizer groups "
                f"{tuple(range(len(optimizer_groups)))!r}, received {expected_group_ids!r}"
            )

        rebound_roles = {}
        for role_name, role in self.optimization_roles.items():
            prepared_parameters = []
            for group_id in role.optimizer_group_ids:
                group = optimizer_groups[group_id]
                prepared_role_name = group.get("role_name")
                if prepared_role_name != role_name:
                    raise ValueError(
                        f"prepared optimizer group {group_id} expected role_name "
                        f"{role_name!r}, received {prepared_role_name!r}"
                    )
                group_parameters = tuple(group["params"])
                if not group_parameters:
                    raise ValueError(
                        f"prepared optimizer group {group_id} for role {role_name!r} "
                        "expected at least one parameter, received none"
                    )
                prepared_parameters.extend(group_parameters)
            rebound_roles[role_name] = OptimizationRole(
                config=role.config,
                parameters=tuple(prepared_parameters),
                optimizer_group_ids=role.optimizer_group_ids,
                step=role.step,
                scheduler=role.scheduler,
            )
        self.optimization_roles = rebound_roles

    def _init_prepared_role_optimization(self) -> None:
        """Bind prepared identities and construct the role coordinator."""
        BaseTrainer._rebind_prepared_optimization_roles(self)
        self.role_optimization = RoleOptimizationCoordinator(
            accelerator=self.accelerator,
            model_bundle=self.model_bundle,
            optimizer=self.optimizer,
            roles=self.optimization_roles,
        )

    def _multirole_metadata(self) -> Dict[str, Any]:
        """Return deterministic metadata used to validate resume compatibility."""
        role_metadata = self.adapter.model_role_registry.training_state_dict()
        update_plan = self._role_update_plan()
        return {
            "version": _MULTIROLE_METADATA_VERSION,
            "roles": role_metadata["roles"],
            "optimizer_group_roles": [
                group.get("role_name") for group in self.optimizer.param_groups
            ],
            "update_plan": [
                {
                    "role_name": phase.role_name,
                    "repeats": phase.repeats,
                }
                for phase in update_plan.phases
            ],
        }

    def _role_update_plan(self) -> RoleUpdatePlan:
        """Return the ordered role plan used for checkpoint compatibility."""
        configured_plan_builder = getattr(self.training_args, "role_update_plan", None)
        if configured_plan_builder is not None:
            configured_plan = configured_plan_builder()
            if not isinstance(configured_plan, RoleUpdatePlan):
                raise TypeError(
                    "expected training_args.role_update_plan() to return RoleUpdatePlan, "
                    f"received {type(configured_plan).__name__}: {configured_plan!r}"
                )
            return configured_plan
        return RoleUpdatePlan(
            phases=tuple(RolePhase(role_name) for role_name in self.optimization_roles)
        )

    def _validate_multirole_metadata(self, state: Mapping[str, Any]) -> None:
        """Validate all metadata that must match before prepared-state mutation."""
        if not isinstance(state, Mapping):
            raise TypeError(
                "expected multi-role metadata as a mapping, "
                f"received {type(state).__name__}: {state!r}"
            )
        expected_keys = {"version", "roles", "optimizer_group_roles", "update_plan"}
        received_keys = set(state)
        if received_keys != expected_keys:
            raise ValueError(
                "multi-role metadata keys mismatch: expected "
                f"{tuple(sorted(expected_keys))!r}, "
                f"received {tuple(sorted(received_keys))!r}"
            )
        received_version = state.get("version")
        if (
            not isinstance(received_version, int)
            or isinstance(received_version, bool)
            or received_version != _MULTIROLE_METADATA_VERSION
        ):
            raise ValueError(
                "multi-role metadata version mismatch: expected "
                f"{_MULTIROLE_METADATA_VERSION}, received {received_version!r}"
            )

        self.adapter.model_role_registry.load_training_state_dict(
            {
                "version": received_version,
                "roles": state.get("roles"),
            }
        )
        expected = self._multirole_metadata()
        for field_name in ("optimizer_group_roles", "update_plan"):
            expected_value = expected[field_name]
            received_value = state.get(field_name)
            if received_value != expected_value:
                raise ValueError(
                    f"multi-role metadata {field_name} mismatch: expected "
                    f"{expected_value!r}, received {received_value!r}"
                )

    def _multirole_state_dict(self) -> Dict[str, Any]:
        """Return registered custom state without duplicating prepared state."""
        coordinator_state = self.role_optimization.state_dict()
        generator_step = coordinator_state["role_steps"]["generator"]
        if self.step != generator_step:
            raise RuntimeError(
                "multi-role checkpoint counter mismatch: expected trainer step to equal "
                f"generator role step {generator_step}, received trainer_step={self.step}"
            )
        return {
            "version": 1,
            "metadata": self._multirole_metadata(),
            "coordinator": coordinator_state,
            "trainer_step": self.step,
            "parameter_emas": self.adapter.model_role_registry.parameter_ema_state_dict(),
        }

    def _load_multirole_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore custom multi-role counters after complete validation."""
        if not isinstance(state, Mapping):
            raise TypeError(
                "expected registered multi-role state as a mapping, "
                f"received {type(state).__name__}: {state!r}"
            )
        received_keys = set(state)
        if received_keys != _MULTIROLE_STATE_KEYS:
            raise ValueError(
                "registered multi-role state keys mismatch: expected "
                f"{tuple(sorted(_MULTIROLE_STATE_KEYS))!r}, "
                f"received {tuple(sorted(received_keys))!r}"
            )
        state_version = state["version"]
        if (
            not isinstance(state_version, int)
            or isinstance(state_version, bool)
            or state_version != 1
        ):
            raise ValueError(
                f"registered multi-role state version mismatch: expected 1, "
                f"received {state_version!r}"
            )
        self._validate_multirole_metadata(state["metadata"])
        trainer_step = state["trainer_step"]
        if not isinstance(trainer_step, int) or isinstance(trainer_step, bool) or trainer_step < 0:
            raise ValueError(
                "expected non-negative int trainer_step in registered multi-role state, "
                f"received {trainer_step!r}"
            )
        coordinator_state = state["coordinator"]
        if not isinstance(coordinator_state, Mapping):
            raise TypeError(
                "expected coordinator in registered multi-role state as a mapping, "
                f"received {type(coordinator_state).__name__}: {coordinator_state!r}"
            )
        role_steps = coordinator_state.get("role_steps")
        if not isinstance(role_steps, Mapping) or role_steps.get("generator") != trainer_step:
            received_generator_step = (
                role_steps.get("generator") if isinstance(role_steps, Mapping) else role_steps
            )
            raise ValueError(
                "registered multi-role counter mismatch: expected trainer_step "
                f"{trainer_step} to equal generator role step, "
                f"received generator step {received_generator_step!r}"
            )
        self.role_optimization.load_state_dict(coordinator_state)
        self.adapter.model_role_registry.load_parameter_ema_state_dict(state["parameter_emas"])
        self.step = trainer_step
        self.adapter.model_role_registry.activate("generator")

    def _register_multirole_checkpointing(self) -> None:
        """Register Accelerate metadata gates and custom state for multi-role runs."""
        if len(self._required_trainable_roles()) <= 1:
            return
        if getattr(self, "_multirole_checkpoint_registered", False):
            raise RuntimeError(
                "cannot register multi-role checkpointing twice for trainer "
                f"{type(self).__name__}"
            )

        def save_metadata_hook(
            models: List[torch.nn.Module],
            weights: List[Dict[str, torch.Tensor]],
            output_dir: str,
        ) -> None:
            del models, weights
            checkpoint_state.prepare_save(output_dir)

        def load_metadata_hook(models: List[torch.nn.Module], input_dir: str) -> None:
            del models
            checkpoint_state.validate_load(input_dir)

        checkpoint_state = _MultiRoleCheckpointState(self)
        self._multirole_checkpoint_state = checkpoint_state
        self.adapter._multirole_checkpoint_state = checkpoint_state
        self.accelerator.register_save_state_pre_hook(save_metadata_hook)
        self.accelerator.register_load_state_pre_hook(load_metadata_hook)
        self.accelerator.register_for_checkpointing(checkpoint_state)
        self._multirole_checkpoint_registered = True

    def _required_trainable_roles(self) -> Tuple[str, ...]:
        """Return algorithm-required roles, defaulting legacy trainers to generator-only."""
        if not hasattr(self.training_args, "required_trainable_roles"):
            return ("generator",)
        return self.training_args.required_trainable_roles

    def _validate_multirole_backend(self) -> None:
        """Validate one-root backend semantics for multi-role trainers."""
        required_roles = self._required_trainable_roles()
        if len(required_roles) <= 1:
            return

        algorithm = self.training_args.trainer_type
        prepared_models = tuple(self.accelerator._models)
        if len(prepared_models) != 1:
            raise RuntimeError(
                f"algorithm {algorithm!r} with roles {required_roles!r} expected exactly "
                f"one prepared model root, received {len(prepared_models)}"
            )
        if prepared_models[0] is not self.model_bundle:
            raise RuntimeError(
                f"algorithm {algorithm!r} with roles {required_roles!r} expected the tracked "
                "prepared model root to be self.model_bundle, received different identities"
            )

        prepared_optimizers = tuple(self.accelerator._optimizers)
        if len(prepared_optimizers) != 1:
            raise RuntimeError(
                f"algorithm {algorithm!r} with roles {required_roles!r} expected exactly "
                f"one prepared optimizer, received {len(prepared_optimizers)}"
            )
        if prepared_optimizers[0] is not self.optimizer:
            raise RuntimeError(
                f"algorithm {algorithm!r} with roles {required_roles!r} expected the tracked "
                "prepared optimizer to be self.optimizer, received different identities"
            )

        prepared_group_roles = tuple(
            group.get("role_name") for group in self.optimizer.param_groups
        )
        if prepared_group_roles != self._unprepared_optimizer_group_roles:
            raise RuntimeError(
                f"algorithm {algorithm!r} optimizer group role mapping expected "
                f"{self._unprepared_optimizer_group_roles!r}, received {prepared_group_roles!r}"
            )

        deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if deepspeed_plugin is not None and deepspeed_plugin.zero_stage not in (1, 2):
            raise ValueError(
                f"DeepSpeed multi-role algorithm {algorithm!r} with roles "
                f"{required_roles!r} requires zero_stage in (1, 2), received "
                f"zero_stage={deepspeed_plugin.zero_stage!r}"
            )

        fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
        if fsdp_plugin is None or fsdp_plugin.fsdp_version != 2:
            return
        if fsdp_plugin.use_orig_params is False:
            raise ValueError(
                f"FSDP2 multi-role algorithm {algorithm!r} with roles {required_roles!r} "
                "requires use_orig_params=True, received False"
            )
        prepared_root_parameter_ids = {
            id(parameter)
            for parameter in self.accelerator.unwrap_model(self.model_bundle).parameters()
        }
        foreign_parameters = tuple(
            (group_id, group["role_name"], parameter_index, id(parameter))
            for group_id, group in enumerate(self.optimizer.param_groups)
            for parameter_index, parameter in enumerate(group["params"])
            if id(parameter) not in prepared_root_parameter_ids
        )
        if foreign_parameters:
            raise RuntimeError(
                "FSDP2 optimizer parameter identity must reference the prepared model root "
                f"for algorithm {algorithm!r} with roles {required_roles!r}; received foreign "
                f"(group_id, role_name, parameter_index, parameter_id) entries "
                f"{foreign_parameters!r}"
            )

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
            if self.adapter.uses_fsdp_cpu_efficient_loading():
                self._synchronize_frozen_components(resolved)

    def _validate_paradigm_dynamics(self) -> None:
        """Reject a scheduler whose dynamics the declared paradigm cannot use.

        A coupled algorithm differentiates a stochastic transition, so an ODE
        scheduler leaves it with no transition density and silently wrong policy
        gradients (``constraints.md`` #7). Only the coupled path was guarded, and
        only lazily at the point a transition scale was first needed, which is
        after a run has already started.
        """
        if type(self).paradigm != "coupled":
            return
        scheduler_group = getattr(self.adapter, "scheduler_group", None)
        if scheduler_group is None:
            return
        stochastic = ("Flow-SDE", "Dance-SDE", "CPS")
        for component in scheduler_group.names:
            dynamics_type = scheduler_group[component].dynamics_type
            if dynamics_type not in stochastic:
                raise ValueError(
                    f"coupled algorithm {type(self).__name__} requires stochastic dynamics, "
                    f"received dynamics_type={dynamics_type!r} for component {component!r}; "
                    f"expected one of {stochastic}. Either configure an SDE scheduler or use a "
                    "decoupled algorithm (see constraints #7)."
                )

    def _initialization(self):
        self._validate_paradigm_dynamics()

        # Fix for FSDP, synchronize frozen components like text encoder & VAE.
        # Otherwise they may be uninitialized on Rank > 0.
        if self.adapter.uses_fsdp_cpu_efficient_loading():
            logger.info("FSDP CPU Efficient Loading detected. Synchronizing frozen components...")
            # self.adapter.on_load(self.accelerator.device)
            self._synchronize_frozen_components()

        # Init dataloader, then materialize every live model role before optimizer
        # and distributed bundle construction.
        self.dataloader, eval_dataloaders = self._init_dataloader()
        required_trainable_roles = self._required_trainable_roles()
        self.adapter.configure_model_roles(required_trainable_roles)
        self.optimizer = self._init_optimizer()

        # Bundle ALL target components (trainable + frozen-but-shardable, e.g.
        # Wan2.2's inactive transformer) into ONE nn.Module so accelerate wraps a
        # single root. DeepSpeed (one engine) and FSDP2 (one root) cannot wrap
        # multiple models, so PPO (policy + critic) and Wan2.2 (shard both, train
        # one) require this. The optimizer/EMA/ref still operate on the
        # requires_grad subset via `get_trainable_parameters()`; frozen members
        # are sharded for memory but never receive gradient.
        canonical_bundle_names = list(self.adapter.target_module_map.keys())
        role_registry = self.adapter.model_role_registry
        bundle_members = role_registry.bundle_members()
        model_bundle = ModelBundle(bundle_members)
        self._unprepared_optimizer_group_roles = tuple(
            group["role_name"] for group in self.optimizer.param_groups
        )

        eval_dataloader_names = list(eval_dataloaders.keys())
        eval_dataloader_list = [eval_dataloaders[n] for n in eval_dataloader_names]

        # One prepare call -> one DDP/FSDP/DeepSpeed root for the whole bundle.
        # (Parameter dtypes -- incl. the FSDP2 uniform-fp32 requirement for sharded trained
        # components -- are already handled in the adapter's `_mix_precision`.)
        prepared = self.accelerator.prepare(model_bundle, self.optimizer, *eval_dataloader_list)
        self.model_bundle = prepared[0]
        self.optimizer = prepared[1]
        BaseTrainer._init_prepared_role_optimization(self)
        BaseTrainer._validate_multirole_backend(self)
        prepared_eval_dataloaders = prepared[2:]
        self.eval_dataloaders: Dict[str, DataLoader] = dict(
            zip(eval_dataloader_names, prepared_eval_dataloaders)
        )

        # Install routing proxies so adapter forwards (`self.transformer(...)`,
        # `self.transformer_2(...)`, ...) dispatch through the prepared root --
        # required for DDP's reducer / FSDP's gather / the DeepSpeed engine --
        # while attribute access delegates to the inner member.
        inner_bundle = self.accelerator.unwrap_model(self.model_bundle)
        for name in canonical_bundle_names:
            self.adapter.set_component(
                name,
                RoutedComponentProxy(
                    self.model_bundle,
                    name,
                    role_registry,
                    inner_bundle.members,
                ),
            )

        # Load inference modules, excluding all bundle members (already prepared).
        self._load_inference_components(canonical_bundle_names)

        # Build + validate acceleration plugins. Persistent stage='both' accelerators
        # are *applied* later via _apply_shared_acceleration(), after post_init()
        # finishes any state-resume / EMA / reference setup.
        self._init_acceleration()

        # Initialize reward model
        self._init_reward_model()

    def _init_acceleration(self):
        """Build and validate acceleration plugins from ``config.acceleration_args``.

        Two independent slots, each an **ordered list** (both empty by default).
        List order is the application order:

        * ``shared`` — persistent ``stage='both'`` accelerators (e.g.
          ``attention_backend`` then ``torch_compile``) applied to both rollout and
          the training forward. Only built/validated here; they are *applied* later by
          :meth:`_apply_shared_acceleration` (after ``post_init`` finishes
          state-resume / EMA / reference setup), so they transform the final weights.
        * ``rollout`` — accelerators applied per-epoch in :meth:`generate_samples`
          via :meth:`~BaseAccelerator.rollout_context`; may be lossy.

        Each accelerator is validated against this trainer's ``paradigm`` before
        use (fail-fast, ``constraints.md`` #26).
        """
        accel_args = self.config.acceleration_args
        self.shared_accelerators: List[BaseAccelerator] = []
        self.rollout_accelerators: List[BaseAccelerator] = []

        trainer_name = type(self).__name__
        paradigm = type(self).paradigm

        for spec in accel_args.shared:
            accelerator = build_accelerator(spec.name, spec.params)
            validate_accelerator(
                accelerator, slot="shared", paradigm=paradigm, trainer_name=trainer_name
            )
            self.shared_accelerators.append(accelerator)

        for spec in accel_args.rollout:
            accelerator = build_accelerator(spec.name, spec.params)
            validate_accelerator(
                accelerator, slot="rollout", paradigm=paradigm, trainer_name=trainer_name
            )
            self.rollout_accelerators.append(accelerator)
            if self.accelerator.is_main_process:
                logger.info(
                    "Acceleration: rollout accelerator '%s' (safety=%s) enabled.",
                    spec.name,
                    accelerator.safety,
                )

    def _apply_shared_acceleration(self) -> None:
        """Apply persistent ``stage='both'`` accelerators in config order.

        Called from ``__init__`` AFTER ``adapter.post_init()`` so transforms wrap the
        final weights — i.e. after ``accelerator.prepare``, any ``state`` checkpoint
        resume, and EMA / reference-parameter snapshotting.

        Each entry's ``setup`` runs in list order, so a config that lists
        ``attention_backend`` before ``torch_compile`` sets the backend first and
        then compiles the graph capturing it. In-place compilation
        (``nn.Module.compile`` / ``compile_repeated_blocks``) preserves parameter
        identity and ``state_dict`` keys, so checkpointing and the ``copy_``-based
        EMA / ref / named-parameter swaps stay correct.
        """
        for accelerator in self.shared_accelerators:
            accelerator.setup(self.adapter)
            if self.accelerator.is_main_process:
                logger.info(
                    "Acceleration: shared accelerator '%s' (safety=%s) applied to adapter.",
                    type(accelerator).__name__,
                    accelerator.safety,
                )

    @contextmanager
    def _rollout_acceleration(self) -> Iterator[None]:
        """Nest every rollout accelerator's context (first in list = outermost).

        A no-op when no rollout accelerator is configured.
        """
        with ExitStack() as stack:
            for accelerator in self.rollout_accelerators:
                stack.enter_context(accelerator.rollout_context(self.adapter))
            yield

    def _synchronize_frozen_components(
        self,
        components: Optional[Union[str, List[str]]] = None,
    ):
        if self.accelerator.num_processes <= 1:
            return

        # Synchronize all non-prepared components
        all_names = self.adapter._resolve_component_names(components)
        for name in all_names:
            if self.adapter._should_manage_device(name):
                comp = self.adapter.get_component(name)
                if isinstance(comp, nn.Module):
                    for param in comp.parameters():
                        param.data = param.data.to(self.accelerator.device)
                        dist.broadcast(param.data, src=0)
                    for buffer in comp.buffers():
                        buffer.data = buffer.data.to(self.accelerator.device)
                        dist.broadcast(buffer.data, src=0)

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
        if getattr(accelerator.state, "deepspeed_plugin", None) is None:
            return

        try:
            import deepspeed.runtime.torch_autocast as _ds_ac
            from deepspeed.runtime.engine import DeepSpeedEngine
        except ImportError:
            return

        if getattr(DeepSpeedEngine, "_ff_autocast_patched", False):
            return

        if hasattr(_ds_ac, "validate_nested_autocast"):
            _ds_ac.validate_nested_autocast = lambda engine: None

        if hasattr(DeepSpeedEngine, "torch_autocast_enabled"):
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

    def start(self) -> None:
        """Run the training loop until the configured budget is exhausted.

        Every algorithm drives the same epoch: reseed, save on ``save_freq``,
        evaluate on ``eval_freq``, then sample, score, optimize, and step EMA.
        Only the middle of that sequence is algorithm-specific, so the loop lives
        here and the variation is expressed through
        :meth:`sampling_context`, :meth:`_run_training_step` and
        :meth:`_after_optimizer_step` rather than by restating the loop.
        """
        while self.should_continue_training():
            self.adapter.set_trajectory_seed(self.epoch + self.training_args.seed)

            if (
                self.log_args.save_freq > 0
                and self.epoch % self.log_args.save_freq == 0
                and self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    "checkpoints",
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            if self.eval_args.eval_freq > 0 and self.epoch % self.eval_args.eval_freq == 0:
                self.evaluate()

            self._run_training_step()

            self.adapter.ema_step(step=self.epoch)
            self._after_optimizer_step()
            self.epoch += 1

    def _run_training_step(self) -> None:
        """Run one epoch's rollout, feedback and optimization.

        Every trainer supplies ``sample()``; what it stores follows from the
        paradigm, since a coupled algorithm needs the full trajectory and its log
        probabilities while a decoupled one needs only the terminal state.
        Distillation accumulates several dataloader batches before a single
        optimizer step, so the grouping is a hook rather than a fixed sequence.
        """
        with self.sampling_context():
            samples = self.sample()
        self.prepare_feedback(samples)
        self.optimize(samples)

    @contextmanager
    def sampling_context(self) -> Iterator[None]:
        """Parameter scope for rollout generation.

        On-policy sampling needs no swap; algorithms that roll out under EMA, a
        reference snapshot, or a separate sampling model override this.
        """
        yield

    def _after_optimizer_step(self) -> None:
        """Update algorithm-owned auxiliary weights after the optimizer step.

        EMA is handled by the loop; this is for extra snapshots an algorithm keeps
        alongside it, such as CRD's old model and sampling model.
        """

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Stages 4--5: finalize rewards, compute advantages, and log metrics.

        No policy gradients here. Distillation has no reward signal and overrides
        this with a no-op; algorithms that need extra batching before the loss
        (DPO's chosen/rejected pairing) do that work in :meth:`optimize`, after
        advantages are on each sample.
        """
        rewards = self.reward_buffer.finalize(store_to_samples=True, split="all")
        self.compute_advantages(samples, rewards, store_to_samples=True)
        adv_metrics = self.advantage_processor.pop_advantage_metrics()
        if adv_metrics:
            self.log_data(adv_metrics, step=self.step)

    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func: Optional[Union[Literal["sum", "gdpo"], Callable]] = None,
    ) -> torch.Tensor:
        """Turn per-sample rewards into advantages via the advantage processor.

        Args:
            samples: Samples this epoch's rewards belong to.
            rewards: Reward tensors by reward name, aligned with ``samples``.
            store_to_samples: Whether to write advantages back onto each sample.
            aggregation_func: Within-group aggregation, defaulting to the
                configured ``advantage_aggregation``.

        Returns:
            One advantage per sample.
        """
        aggregation_func = aggregation_func or self.training_args.advantage_aggregation
        return self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
        )

    @abstractmethod
    def optimize(self, *args, **kwargs):
        """Update policy model"""
        pass

    def _sample_timesteps(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample scheduler-scale training timesteps in ``[0, 1000]``.

        Decoupled algorithms draw a training coordinate rather than replaying a
        stored one, and the strategy is a configuration choice rather than an
        algorithmic one, so it lives here. An algorithm whose draw is part of its
        objective (DPO shares one draw across preference arms) overrides this.

        Args:
            batch_size: Size of the broadcast batch dimension.
            generator: Optional ``torch.Generator``. When supplied, the draw is
                deterministic and cross-rank-reproducible for any strategy, which
                is how a group-based algorithm shares one coordinate across ranks.

        Returns:
            Tensor of shape ``(num_train_timesteps, batch_size)``.

        Raises:
            ValueError: If the configured strategy is not recognized.
        """
        device = self.accelerator.device
        strategy = self.time_sampling_strategy.lower()
        available = [
            "logit_normal",
            "uniform",
            "discrete",
            "discrete_with_init",
            "discrete_wo_init",
        ]

        if strategy == "logit_normal":
            return TimeSampler.logit_normal_shifted(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
                stratified=True,
                generator=generator,
            )
        if strategy == "uniform":
            return TimeSampler.uniform(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
                generator=generator,
            )
        if strategy.startswith("discrete"):
            discrete_config = {
                "discrete": (True, False),
                "discrete_with_init": (True, True),
                "discrete_wo_init": (False, False),
            }
            if strategy not in discrete_config:
                raise ValueError(
                    f"Unknown time_sampling_strategy: {strategy!r}. Available: {available}"
                )
            include_init, force_init = discrete_config[strategy]
            return TimeSampler.discrete(
                batch_size=batch_size,
                num_train_timesteps=self.num_train_timesteps,
                scheduler_timesteps=self.adapter.scheduler.timesteps,
                timestep_range=self.timestep_range,
                include_init=include_init,
                force_init=force_init,
                generator=generator,
            )

        raise ValueError(f"Unknown time_sampling_strategy: {strategy!r}. Available: {available}")

    def _apply_optimizer_step(
        self,
        loss_info: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, List[torch.Tensor]]:
        """Clip, step, log the accumulated losses and start a fresh accumulation.

        Call this once ``accelerator.sync_gradients`` is true. Only the loss that
        reached ``backward`` is algorithm-specific; clipping, stepping and metric
        reduction are the same for every algorithm.

        Args:
            loss_info: Per-metric values accumulated since the last optimizer step.

        Returns:
            An empty accumulator for the next optimizer step.
        """
        grad_norm = self.accelerator.clip_grad_norm_(
            self.adapter.get_trainable_parameters(),
            self.training_args.max_grad_norm,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._after_gradient_step()

        reduced = reduce_loss_info(self.accelerator, loss_info)
        reduced["grad_norm"] = grad_norm
        self.log_data({f"train/{k}": v for k, v in reduced.items()}, step=self.step)
        self.step += 1
        return defaultdict(list)

    def _after_gradient_step(self) -> None:
        """Update per-optimizer-step auxiliary weights before metrics are logged.

        Distinct from :meth:`_after_optimizer_step`, which runs once per epoch;
        this runs on every optimizer step, which is the cadence DGPO's fast
        reference EMA needs.
        """

    def _velocity_kl(
        self,
        velocity: LatentState,
        other_velocity: LatentState,
        noised: NoisedState,
    ) -> torch.Tensor:
        """Compute the per-sample squared velocity gap against another policy.

        Under a fixed forward process the KL between two Gaussian transition
        kernels reduces to the squared gap between their velocity predictions, so
        every decoupled algorithm that regularizes towards a reference, an EMA, or
        an older snapshot needs exactly this quantity. Which policy supplies
        ``other_velocity`` is the algorithm's choice; the reduction is not.

        Args:
            velocity: Current-policy velocity per component.
            other_velocity: Reference, EMA or old-snapshot velocity per component.
            noised: Forward-noised state supplying per-sample reduction context.

        Returns:
            Per-sample KL surrogate of shape ``(B,)``.
        """
        errors = {
            name: (velocity.components[name] - other_velocity.components[name]) ** 2
            for name in self.adapter.trajectory_component_order
        }
        return self.adapter.reduce_latent_values(errors, state=noised.state)

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
            sample.to("cpu", pin_memory=True)

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
                    sample.to(device) for sample in samples[start : start + per_device_batch_size]
                ]
                yield BaseSample.stack(batch_samples)
            return

        copy_stream = torch.cuda.Stream(device)
        compute_stream = torch.cuda.current_stream(device)

        def _load(start: int) -> StackedSampleBatch:
            with torch.cuda.stream(copy_stream):
                moved = [
                    sample.to(device, non_blocking=True)
                    for sample in samples[start : start + per_device_batch_size]
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
        sources = batch.get("__source__")
        source_ids = batch.get("__source_id__")
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

        # Stage-3-only acceleration (e.g. feature caching) is scoped to this loop
        # so its state never leaks into the Stage-6 training forward.
        # The outer path stays no_grad; a compile accelerator re-enables gradients
        # only inside the transformer call to match the training compiled graph.
        with self._rollout_acceleration(), torch.no_grad(), self.autocast():
            for _ in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f"Epoch {self.epoch} Sampling",
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
            missing = [i for i, s in enumerate(samples) if s.source is None]
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
                    logger.warning(f"No reward buffer for eval dataset '{dataset_name}', skipping.")
                    continue
                buffer.clear()
                all_samples: List[BaseSample] = []

                # Merge per-dataset eval overrides with shared eval_args
                ed_config = self._eval_dataset_configs[dataset_name]
                eval_kwargs = (
                    ed_config.eval.get_merged_eval_kwargs(self.eval_args)
                    if ed_config.eval
                    else dict(self.eval_args)
                )

                for batch in tqdm(
                    dataloader,
                    desc=f"Eval/{dataset_name}",
                    disable=not self.show_progress_bar,
                ):
                    batch = self._augment_batch_with_source(
                        batch, dataset_name, ed_config.source_id
                    )
                    generator = create_generator_by_prompt(batch["prompt"], self.training_args.seed)
                    samples = self.sample_batch(
                        batch,
                        reward_buffer=buffer,
                        compute_log_prob=False,
                        generator=generator,
                        trajectory_indices=None,
                        **eval_kwargs,
                    )
                    all_samples.extend(samples)

                rewards = buffer.finalize(store_to_samples=True, split="pointwise")

                # Gather across ranks
                rewards_tensors = {
                    k: torch.as_tensor(v).to(self.accelerator.device) for k, v in rewards.items()
                }
                gathered_rewards = {
                    k: self.accelerator.gather(v).cpu().numpy() for k, v in rewards_tensors.items()
                }

                # Log per-dataset immediately to avoid accumulating all samples in memory
                if self.accelerator.is_main_process:
                    log_data: Dict[str, Any] = {}
                    for k, v in gathered_rewards.items():
                        log_data[f"eval/{dataset_name}/reward_{k}_mean"] = np.mean(v)
                        log_data[f"eval/{dataset_name}/reward_{k}_std"] = np.std(v)
                    log_data[f"eval/{dataset_name}/samples"] = all_samples
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
        resume_type: Optional[Literal["lora", "full", "state"]] = None,
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
        train_buf = getattr(self, "reward_buffer", None)
        if train_buf is not None:
            train_buf.shutdown(wait=False, cancel_futures=True)

        # Per-eval-dataset reward buffers.
        for buf in getattr(self, "eval_dataset_reward_buffers", {}).values():
            if buf is not None:
                buf.shutdown(wait=False, cancel_futures=True)
