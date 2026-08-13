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

import glob
import hashlib
import json
import logging

# src/flow_factory/models/abc.py
import os
import re
from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, fields
from functools import partial
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator, DistributedType
from accelerate.state import PartialState
from accelerate.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_PATTERN_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    WEIGHTS_PATTERN_NAME,
    clean_state_dict_for_safetensors,
    has_offloaded_params,
)
from accelerate.utils.modeling import (
    get_state_dict_offloaded_model,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.outputs import BaseOutput
from huggingface_hub import split_torch_state_dict_into_shards
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from safetensors.torch import load_file, save_file

from ..ema import EMAModuleWrapper
from ..hparams import *
from ..samples import (
    BaseSample,
    ComponentTimes,
    LatentState,
    MultiModalStepOutput,
    NoisedState,
    ReplayStep,
    StackedSampleBatch,
)
from ..scheduler import (
    SchedulerGroup,
    SDESchedulerMixin,
    SDESchedulerOutput,
)
from ..scheduler import load_scheduler as _load_scheduler
from ..utils.audio import MultiAudioBatch
from ..utils.base import filter_kwargs, is_tensor_list
from ..utils.checkpoint import (
    HF_PATH_PREFIX,
    download_hf_checkpoint,
    infer_lora_config,
    infer_target_modules,
    mapping_lora_state_dict,
    parse_hf_checkpoint_path,
)
from ..utils.image import MultiImageBatch
from ..utils.logger_utils import setup_logger
from ..utils.video import MultiVideoBatch
from . import trajectory_bridge as bridge
from .latent_geometry import LatentAxes, infer_latent_axes
from .model_bundle import RoutedComponentProxy
from .runtime import ClassicPipelineRuntime, ComponentRuntime
from .variants import ComponentVariantRegistry, ComponentVariantSpec

# Constants
CONFIG_NAME = "config.json"
DIFFUSION_WEIGHTS_NAME = "diffusion_pytorch_model.bin"
DIFFUSION_WEIGHTS_PATTERN_NAME = "diffusion_pytorch_model{suffix}.bin"
DIFFUSION_WEIGHTS_INDEX_NAME = f"{DIFFUSION_WEIGHTS_NAME}.index.json"
SAFE_DIFFUSION_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
SAFE_DIFFUSION_WEIGHTS_PATTERN_NAME = "diffusion_pytorch_model{suffix}.safetensors"
SAFE_DIFFUSION_WEIGHTS_INDEX_NAME = f"{SAFE_DIFFUSION_WEIGHTS_NAME}.index.json"
LORA_ADAPTER_CONFIG_NAME = "adapter_config.json"
LORA_ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"

logger = setup_logger(__name__)


@dataclass
class NamedParametersInfo:
    """Metadata for named parameters snapshot."""

    target_components: List[str]
    ema_wrapper: EMAModuleWrapper


class BaseAdapter(ABC):
    """
    Abstract Base Class for Flow-Factory models.
    """

    _DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

    lora_keys: List[str] = [
        "lora_A",
        "lora_B",
        "lora_magnitude_vector",  # DoRA
        "lora_embedding_A",
        "lora_embedding_B",  # Embedding LoRA
        "modules_to_save",  # Additional modules marked for saving
    ]

    # Names of ``preprocess_func`` output columns that must be surfaced in the HF
    # "python" format (returned as PIL, never tensorized) instead of the torch
    # format. They are persisted via the HuggingFace ``Image`` feature (PNG bytes)
    # rather than raw tensors, which lets the dataset store variable-size /
    # variable-count images (e.g. multi-reference I2I) that Arrow cannot serialize
    # as ragged tensors, and read them back as PIL -- see
    # ``data_utils.dataset._apply_torch_format``.
    #
    # MUST contain only genuine RGB image columns: each entry is run through PIL
    # canonicalization (``_to_pil_image_list``), so non-image data would break.
    # Empty by default and OPT-IN per adapter: only declare an output here when it
    # is a genuine RGB image that survives a PIL round-trip. Do NOT declare
    # preprocessed/non-RGB tensors (e.g. VAE-ready video tensors, latents) -- PIL
    # conversion would be lossy and break tensor consumers; non-image columns that
    # must stay python belong in ``dataset.EXTRA_PYTHON_FORMAT_COLUMNS``, not here.
    # The raw modality column ``images`` is always handled as images by the dataset
    # itself, independent of this declaration.
    python_format_columns: ClassVar[frozenset[str]] = frozenset()

    # Opt in only when every transformer forward branch runs inside a diffusers
    # ``cache_context``. The rollout cache accelerator rejects the default.
    supports_diffusers_cache: ClassVar[bool] = False
    supports_ordered_references: ClassVar[bool] = False
    preprocess_cache_fields: ClassVar[frozenset[str]] = frozenset()
    preprocess_cache_version: ClassVar[str] = ""
    trajectory_component_order: ClassVar[Tuple[str, ...]] = ("latent",)
    flow_velocity_direction: ClassVar[Literal["noise", "data"]] = "noise"

    # Resolution-invariant latent axis roles for the model-agnostic latent state
    # API (see `latent_geometry.py`). ``None`` means "infer from latent ndim" via
    # `resolve_latent_axes`, which covers every standard 3D/4D/5D layout. Adapters
    # with a non-standard layout may override this with an explicit `LatentAxes`.
    LATENT_AXES: ClassVar[Optional[LatentAxes]] = None

    # DDP-only knob (ignored under FSDP/DeepSpeed). Set True only for adapters
    # whose training step can leave some trainable parameters without a gradient
    # in a given iteration (e.g. Qwen-Image: guidance=None leaves the guidance
    # embedder unused). The default False lets DDP use static buckets and overlap
    # gradient all-reduce with backward; if an adapter actually needs True but
    # leaves it False, DDP fails fast at the first backward with an explicit
    # "didn't receive grad" error rather than silently producing wrong results.
    ddp_find_unused_parameters: ClassVar[bool] = False

    # Public wrappers that own a shared contract (argument ownership, validation)
    # and delegate the component-specific part to the protected hook of the same
    # name. Overriding one would silently bypass that contract, so subclasses are
    # rejected at class creation instead of at training time.
    _BOUNDARY_OWNING_METHODS: ClassVar[Tuple[str, ...]] = (
        "forward_state",
        "reduce_component_latent_values",
        "reduce_latent_values",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in BaseAdapter._BOUNDARY_OWNING_METHODS:
            if name in cls.__dict__:
                raise TypeError(
                    f"adapter {cls.__name__} must not override BaseAdapter.{name}: it owns a "
                    f"shared contract that an override would bypass ({name} validates its "
                    f"arguments and its result on behalf of every caller). Override the "
                    f"protected hook _{name} instead."
                )

    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__()
        self.config = config
        self.accelerator = accelerator
        self.model_args = config.model_args
        self.training_args = config.training_args
        self.eval_args = config.eval_args
        self._mode: str = "train"  # ['train', 'eval', 'rollout']
        self._named_parameters: Dict[str, NamedParametersInfo] = {}

        # Build the component runtime while preserving the public pipeline alias.
        self.component_runtime = self.build_component_runtime()
        self.pipeline = self.component_runtime.pipeline
        if (
            getattr(self.pipeline, "scheduler", None) is None
            and "scheduler" in self.component_runtime.declared_component_names
        ):
            self.component_runtime.materialize_components(["scheduler"])
        self.pipeline.scheduler = self.load_scheduler()
        self.scheduler_group = self.build_scheduler_group()
        if self.scheduler_group.names != self.trajectory_component_order:
            raise ValueError(
                "expected scheduler group names to match trajectory_component_order "
                f"{self.trajectory_component_order}, received {self.scheduler_group.names}"
            )
        if self.scheduler_group.primary is not self.pipeline.scheduler:
            raise ValueError(
                "expected SchedulerGroup.primary to be the canonical pipeline scheduler, "
                f"received primary component {self.scheduler_group.primary_name!r}"
            )

        # Compatibility alias: the runtime override mapping is the sole authoritative cache.
        self._components: Dict[str, torch.nn.Module] = cast(
            Dict[str, torch.nn.Module], self.component_runtime.override_components
        )

        # Cache target module mapping
        self.target_module_map = self._init_target_module_map()

        # Load checkpoint.
        # 'lora'/'full' load into the unwrapped pipeline modules here (before prepare()).
        # 'state' is deferred to post_init(): accelerator.load_state() only restores into
        # modules/optimizer registered by accelerator.prepare(), which the trainer runs later.
        if self.model_args.resume_path and self.model_args.resume_type != "state":
            self.load_checkpoint(
                self.model_args.resume_path, resume_type=self.model_args.resume_type
            )

        # Merge LoRA adapters into base model when transitioning to full fine-tuning
        if self.model_args.resume_path and self.model_args.finetune_type != "lora":
            self._merge_lora_if_needed()

        # Freeze non-trainable components
        self._freeze_components()

        # Apply LoRA if needed
        if self.model_args.finetune_type == "lora":
            self.apply_lora(
                target_modules=self.model_args.target_modules,
                components=self.model_args.target_components,
                overwrite=False,  # Do not overwrite existing adapters
            )

        # Set precision
        self._mix_precision()

        # NOTE: attention-backend selection is applied later by the trainer's
        # acceleration step (AttentionBackendAccelerator, configured as a `shared`
        # entry in the acceleration block), after accelerator.prepare()/post_init()
        # and before torch.compile — see BaseTrainer._apply_shared_acceleration().
        # It is intentionally NOT set here.

        # Enable gradient checkpointing if needed
        if self.training_args.enable_gradient_checkpointing:
            self.enable_gradient_checkpointing()

    # ================================== Post Init =================================
    def post_init(self):
        """Hook for additional initialization after main trainer's `accelerator.prepare`."""
        # Full training-state resume must happen here: accelerator.prepare() has now
        # registered the trainable modules and optimizer, so accelerator.load_state()
        # can actually restore model + optimizer + RNG (and any other prepared objects).
        if self.model_args.resume_path and self.model_args.resume_type == "state":
            self.load_checkpoint(self.model_args.resume_path, resume_type="state")
        self._init_ema()
        self._init_ref_parameters()

    # ============================== Latent Casting =================================
    @property
    def latent_storage_dtype(self) -> Optional[torch.dtype]:
        val = getattr(self.training_args, "latent_storage_dtype", None)
        return self._DTYPE_MAP.get(val) if val else None

    def cast_latents(
        self, latents: torch.Tensor, default_dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Cast latents to storage dtype with float16 overflow protection."""
        target = self.latent_storage_dtype or default_dtype
        if target is None or latents.dtype == target:
            return latents
        if target == torch.float16:
            abs_max = latents.abs().max().item()
            if abs_max > 65504.0:
                logger.warning(f"float16 overflow: abs_max={abs_max:.1f} > 65504, clamping.")
                latents = latents.clamp(-65504.0, 65504.0)
        return latents.to(target)

    def cast_latent_state(
        self, state: LatentState, default_dtype: Optional[torch.dtype] = None
    ) -> LatentState:
        """Cast every component latent of a state to the storage dtype.

        Multi-component adapters store and replay a whole ``LatentState`` rather than
        one tensor, and the storage dtype has to reach every component or the stored
        trajectory and the replayed one disagree on precision. Active masks are
        boolean selectors rather than latents, so they are carried over unchanged.

        Args:
            state: Structured multi-component latent state.
            default_dtype: Fallback dtype when no storage dtype is configured.

        Returns:
            State whose component latents use the storage dtype.

        Raises:
            TypeError: If ``state`` is not a ``LatentState``.
        """
        if not isinstance(state, LatentState):
            raise TypeError(
                f"expected LatentState for cast_latent_state, got {type(state).__name__}: {state!r}"
            )
        if self.latent_storage_dtype is None and default_dtype is None:
            return state
        components = {
            name: self.cast_latents(latents, default_dtype=default_dtype)
            for name, latents in state.components.items()
        }
        if all(components[name] is latents for name, latents in state.components.items()):
            return state
        return LatentState(components, active_masks=state.active_masks)

    # ============================== Loading Components ==============================
    @abstractmethod
    def load_pipeline(self) -> DiffusionPipeline:
        """Load and return the diffusion pipeline. Must be implemented by subclasses."""
        pass

    def build_component_runtime(self) -> ComponentRuntime:
        """Build the component runtime used by this adapter.

        Returns:
            Classic runtime wrapping the subclass's existing ``load_pipeline`` result.
        """
        return ClassicPipelineRuntime(self.load_pipeline())

    def load_scheduler(self) -> SDESchedulerMixin:
        """Load and return the scheduler."""
        scheduler = _load_scheduler(
            pipeline_scheduler=self.pipeline.scheduler,
            scheduler_args=self.config.scheduler_args,
        )
        return scheduler

    def build_scheduler_group(self) -> SchedulerGroup:
        """Build the default single-component scheduler group.

        Initialization calls this only after the canonical scheduler is installed.
        The scheduler setter therefore updates only the pipeline until this group exists.

        Returns:
            Scheduler group exposing the canonical scheduler as ``"latent"``.
        """
        return SchedulerGroup({"latent": self.scheduler}, primary_name="latent")

    # ============================== Component Accessors ==============================
    # ---------------------------------- Wrappers ----------------------------------
    def _unwrap(self, model: Union[torch.nn.Module, RoutedComponentProxy]) -> torch.nn.Module:
        """Get the unwrapped model.

        Peels a `RoutedComponentProxy` to its inner module first (the proxy is
        installed as a component after `accelerator.prepare`), then strips the
        accelerator wrapper (DDP/FSDP/DeepSpeed).
        """
        if isinstance(model, RoutedComponentProxy):
            model = model.inner
        return self.accelerator.unwrap_model(model)

    def set_component(self, name: str, module: Union[torch.nn.Module, RoutedComponentProxy]):
        """Install a component override for this name.

        Accepts either a real ``nn.Module`` (a LoRA wrapper, checkpoint replacement,
        accelerator-prepared module, etc.) or a transparent ``RoutedComponentProxy``
        installed after ``accelerator.prepare``. Every override is excluded from
        runtime device management. The proxy is stored as an ``nn.Module`` stand-in
        (it duck-types one), so the cast is the single, encapsulated acknowledgement
        of that contract; readers via ``get_component`` see a plain ``nn.Module``.
        """
        self.component_runtime.set_component_override(name, cast(torch.nn.Module, module))

    def get_component(self, name: str) -> torch.nn.Module:
        """Get a component, preferring the runtime override if available.

        Overrides also include LoRA/checkpoint replacement modules. After
        `accelerator.prepare`, trainable/bundled components resolve to a transparent
        `RoutedComponentProxy` (a duck-typed nn.Module stand-in) that routes forwards
        through the prepared root; use ``_unwrap`` to recover the real module.
        """
        return cast(torch.nn.Module, self.component_runtime.get_component(name))

    def has_component(self, name: str) -> bool:
        """Return whether the runtime declares a component under this canonical name.

        Component membership is owned by ``component_runtime``, not by Python
        attribute existence. ``hasattr(self, name)`` only agrees for backends whose
        components happen to be adapter properties; a modular or lazy runtime
        declares components that are never attributes, and probing with ``hasattr``
        silently drops them from lifecycle loops.
        """
        return name in self.component_runtime.declared_component_names

    def _require_component(self, name: str) -> torch.nn.Module:
        """Return a declared component, rejecting one the runtime cannot provide.

        ``get_component`` returns ``None`` for a declared-optional component that the
        backend left unset (a Wan 2.1 checkpoint has no ``transformer_2``, an
        image-only pipeline has no ``audio_vae``). Dereferencing that yields a bare
        ``AttributeError`` naming neither the component nor the loop, so callers that
        need a real module ask for it here instead.
        """
        component = self.get_component(name)
        if component is None:
            raise ValueError(
                f"expected component {name!r} of {type(self).__name__} to be available, "
                f"received None from {type(self.component_runtime).__name__}; it is "
                f"declared but unset on this checkpoint. Declared components: "
                f"{self.component_runtime.declared_component_names}"
            )
        return component

    def get_component_unwrapped(self, name: str) -> torch.nn.Module:
        """Get the original unwrapped component."""
        return cast(torch.nn.Module, self.component_runtime.get_canonical_component(name))

    def get_component_config(self, name: str):
        """Get the config of a component."""
        return self.component_runtime.get_canonical_component(name).config

    def prepare_components(self, accelerator: Accelerator, component_names: List[str]):
        """Prepare specified components with the accelerator."""
        components = [
            self.component_runtime.get_canonical_component(name) for name in component_names
        ]
        prepared = accelerator.prepare(*components)
        for name, module in zip(component_names, prepared):
            self.set_component(name, module)
        return prepared

    # ------------------------------ Text Encoders & Tokenizers ------------------------------
    @property
    def text_encoder_names(self) -> List[str]:
        """Get all text encoder component names from pipeline."""
        return self.component_runtime.text_encoder_names

    @property
    def text_encoders(self) -> List[torch.nn.Module]:
        """Collect all text encoders, preferring prepared versions."""
        return [self.get_component(name) for name in self.text_encoder_names]

    @property
    def text_encoder(self) -> torch.nn.Module:
        """Get the primary text encoder."""
        return self.get_component("text_encoder")

    @text_encoder.setter
    def text_encoder(self, module: torch.nn.Module):
        self.set_component("text_encoder", module)

    @property
    def tokenizer_names(self) -> List[str]:
        """Get all tokenizer names from pipeline."""
        names = [
            name
            for name, value in vars(self.pipeline).items()
            if "tokenizer" in name and not name.startswith("_")
        ]
        return sorted(names)

    @property
    def tokenizers(self) -> List[Any]:
        """Collect all tokenizers from pipeline."""
        return [getattr(self.pipeline, name) for name in self.tokenizer_names]

    @property
    def tokenizer(self) -> Any:
        """Get the primary tokenizer."""
        tokenizers = self.tokenizers
        if not tokenizers:
            raise ValueError("No tokenizer found in the pipeline.")
        return tokenizers[0]

    # -------------------------------------- VAE --------------------------------------
    @property
    def vae(self) -> torch.nn.Module:
        """Get VAE, preferring prepared version."""
        return self.get_component("vae")

    @vae.setter
    def vae(self, module: torch.nn.Module):
        self.set_component("vae", module)

    # ------------------------------------ Audio VAE ------------------------------------
    @property
    def audio_vae(self) -> Optional[torch.nn.Module]:
        """Get audio VAE if available in pipeline, preferring prepared version."""
        if "audio_vae" not in self.component_runtime.declared_component_names:
            return None
        return self.get_component("audio_vae")

    @audio_vae.setter
    def audio_vae(self, module: torch.nn.Module):
        self.set_component("audio_vae", module)

    # ---------------------------------- Transformers ----------------------------------
    @property
    def transformer_names(self) -> List[str]:
        """Get all transformer component names."""
        return self.component_runtime.transformer_names

    @property
    def transformers(self) -> List[torch.nn.Module]:
        """Collect all transformers, preferring prepared versions."""
        return [self.get_component(name) for name in self.transformer_names]

    @property
    def transformer(self) -> torch.nn.Module:
        return self.get_component("transformer")

    @transformer.setter
    def transformer(self, module: torch.nn.Module):
        self.set_component("transformer", module)

    @property
    def transformer_config(self):
        return self.get_component_config("transformer")

    # ------------------------------------ Scheduler ------------------------------------
    @property
    def scheduler(self) -> SDESchedulerMixin:
        return self.pipeline.scheduler

    @scheduler.setter
    def scheduler(self, scheduler: Union[SDESchedulerMixin, SchedulerMixin]):
        """Set the canonical scheduler and refresh an initialized scheduler group."""
        self.pipeline.scheduler = scheduler
        if hasattr(self, "scheduler_group"):
            schedulers = dict(self.scheduler_group)
            schedulers[self.scheduler_group.primary_name] = scheduler
            self.scheduler_group = SchedulerGroup(
                schedulers,
                primary_name=self.scheduler_group.primary_name,
            )

    # ---------------------------------- Device & Dtype ----------------------------------
    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    @property
    def _inference_dtype(self) -> torch.dtype:
        """Get inference dtype based on mixed precision setting."""
        if self.config.mixed_precision == "fp16":
            return torch.float16
        elif self.config.mixed_precision == "bf16":
            return torch.bfloat16
        return torch.float32

    # ============================== Mode Management ==============================

    @property
    def mode(self) -> str:
        """Get current mode."""
        return self._mode

    def eval(self):
        """Set all target components to evaluation mode."""
        self._mode = "eval"
        for name in self.trainable_component_names:
            self.get_component(name).eval()
        self.scheduler_group.eval()

    def rollout(self, *args, **kwargs):
        """Set model to rollout mode."""
        self._mode = "rollout"
        for name in self.trainable_component_names:
            self.get_component(name).eval()
        self.scheduler_group.rollout(*args, **kwargs)

    def train(self, mode: bool = True):
        """Set trainable components to training mode."""
        self._mode = "train" if mode else "eval"
        for name in self.trainable_component_names:
            self.get_component(name).train(mode)
        self.scheduler_group.train(mode=mode)

    def set_trajectory_seed(self, seed: int) -> None:
        """Set scheduler seeds in authoritative component order.

        Args:
            seed: Seed passed unchanged to every component scheduler.
        """
        self.scheduler_group.set_seed(seed)

    # ============================== Target Modules ==============================
    @property
    def default_target_modules(self) -> List[str]:
        """Default target modules for training."""
        return ["to_q", "to_k", "to_v", "to_out.0"]

    @property
    def preprocessing_modules(self) -> List[str]:
        """Modules that are requires for preprocessing"""
        return ["text_encoders", "vae"]

    @property
    def inference_modules(self) -> List[str]:
        """Modules that are required for inference and forward"""
        return ["transformer", "vae"]

    @property
    def trainable_component_names(self) -> List[str]:
        """Names of components with trainable parameters."""
        return [comp for comp, mods in self.target_module_map.items() if mods]

    @property
    def trainable_components(self) -> List[torch.nn.Module]:
        """Prepared model objects with trainable parameters."""
        return [self.get_component(name) for name in self.trainable_component_names]

    # ============================== Component Variants ==============================
    """
        A single-policy algorithm trains one copy of the trainable components, so
        `target_module_map` alone describes parameter ownership. Some algorithms need
        several copies live at the same time, each owning its own optimizer groups. The
        named-parameter snapshots below are a *temporal* mechanism (one set of weights
        installed at a time); variants are the *spatial* one.

        This adapter supplies the mechanism and holds no opinion about what a variant
        means. The caller chooses every name, decides which variant is trained when,
        and decides which one an export writes. Under `lora` each variant is a PEFT
        adapter on one shared base; under `full` each gets its own component copy.
    """

    def declare_component_variants(self, trainable_variants: Sequence[str]) -> None:
        """Create live copies of the trainable components, one per named variant.

        Must run before ``accelerator.prepare`` so the bundle sees every member,
        hence the refusal to reconfigure an existing registry.

        A frozen reference is not a variant: it is the same weights at another
        point in time, so an algorithm that needs one asks for
        ``use_ref_parameters()`` (the pre-finetune weights) or its own named
        snapshot instead of declaring a copy that would cost a bundle member.

        Args:
            trainable_variants: Every trainable variant name, base variant first.
                The base owns the canonical components and every later variant is
                layered on it; a single-policy algorithm passes one name.

        Raises:
            RuntimeError: If variants are already declared or frozen.
            ValueError: If the finetune type has no variant storage mode.
        """
        existing_registry = getattr(self, "component_variant_registry", None)
        if existing_registry is not None:
            state = "frozen" if existing_registry.is_frozen else "already declared"
            raise RuntimeError(
                f"cannot declare component variants: registry is {state}; "
                f"existing variants are {existing_registry.variant_names!r}"
            )

        finetune_type = self.model_args.finetune_type
        if finetune_type not in ("lora", "full"):
            raise ValueError(
                "expected model finetune_type to be 'lora' or 'full' for component variants, "
                f"received {finetune_type!r}"
            )
        trainable_variants = ComponentVariantRegistry._validate_required_variants(
            trainable_variants
        )
        base_variant = trainable_variants[0]
        registry = ComponentVariantRegistry(self)
        for variant_name in trainable_variants:
            is_base = variant_name == base_variant
            if finetune_type == "lora":
                component_routes = {
                    component_name: component_name
                    for component_name in self.trainable_component_names
                }
                adapter_name = "default" if is_base else variant_name
            else:
                component_routes = {
                    component_name: (
                        component_name if is_base else f"{variant_name}__{component_name}"
                    )
                    for component_name in self.trainable_component_names
                }
                adapter_name = None
            registry.declare(
                ComponentVariantSpec(
                    name=variant_name,
                    storage_mode=finetune_type,
                    component_routes=component_routes,
                    adapter_name=adapter_name,
                )
            )
        registry.materialize(trainable_variants)
        self.component_variant_registry = registry

    def _require_variant_registry(self, purpose: str) -> ComponentVariantRegistry:
        """Return the variant registry, naming the caller that needs it."""
        registry = getattr(self, "component_variant_registry", None)
        if registry is None:
            raise RuntimeError(
                f"adapter={type(self).__name__!r} has no component variant registry for "
                f"{purpose}; call declare_component_variants() before accelerator.prepare"
            )
        return registry

    @contextmanager
    def use_component_variant(self, variant_name: str) -> Iterator[None]:
        """Temporarily route component lookups and forwards to one variant.

        Args:
            variant_name: Declared variant to activate.

        Yields:
            Control while the requested variant is active.

        Raises:
            RuntimeError: If no component variants were declared.
        """
        registry = self._require_variant_registry(f"variant {variant_name!r}")
        with registry.use(variant_name):
            yield

    def variant_parameters(self, variant_name: str) -> Tuple[torch.nn.Parameter, ...]:
        """Return the parameters one variant owns, for a caller-built optimizer group.

        Args:
            variant_name: Declared variant.

        Returns:
            The variant's parameters, in registration order.

        Raises:
            RuntimeError: If no component variants were declared.
        """
        registry = self._require_variant_registry(f"variant {variant_name!r} parameters")
        return registry.parameters(variant_name)

    def _merge_module_pattern(
        self, current_pattern: Union[str, List[str], Set[str]], new_pattern: str
    ) -> Union[str, Set[str]]:
        """
        Resolve pattern and merge into current modules.

        Args:
            current: Current state ('all' or list of modules)
            pattern: New pattern to merge ('all', 'default', or module name)

        Returns:
            'all' or updated module list
        """
        # 'all' is absorbing - once set, stays 'all'
        if current_pattern == "all" or new_pattern == "all":
            return "all"

        # Resolve pattern to module list
        new_modules = self.default_target_modules if new_pattern == "default" else [new_pattern]
        new_pattern_set = set(current_pattern) | set(new_modules)
        return new_pattern_set

    def _parse_target_modules(
        self, target_modules: Union[str, List[str]], components: Union[str, List[str]]
    ) -> Dict[str, Union[List[str], None]]:
        """
        Parse target_modules config into component-specific mapping.

        Args:
            target_modules:
                - 'default': Use self.default_target_modules
                - 'all': Unfreeze all parameters
                - str: Single module pattern
                - List[str]: Module patterns with optional component prefix
            components: Union[str, List[str]]
                - Component(s) to apply target_modules to.

        Returns:
            Dict mapping component names to their target modules.
            Example: {
                'transformer': ['attn.to_q', 'attn.to_k'],
                'transformer_2': 'all',
                'transformer_3': None
            }
        """
        # Normalize components to list
        if isinstance(components, str):
            components = [components]
        if isinstance(target_modules, str):
            target_modules = [target_modules]

        component_map = {comp: set() for comp in components}

        for module in target_modules:
            parts = module.split(".", 1)
            if len(parts) == 2 and parts[0] in components:
                component_map[parts[0]] = self._merge_module_pattern(
                    component_map[parts[0]], parts[1]
                )
            else:
                for comp in components:
                    component_map[comp] = self._merge_module_pattern(component_map[comp], module)

        # Remove duplicates and handle empty lists
        component_map = {
            comp: (
                "all" if mods == "all" else sorted(mods) if mods else None
            )  # Keep None here, to enable `accelerator.prepare` for non-trainable module to save mem.
            for comp, mods in component_map.items()
        }

        return component_map

    def _init_target_module_map(self) -> Dict[str, Union[List[str], None]]:
        """
        Initialize and cache target module mapping from config.

        Returns:
            Dict mapping component names to their target modules.
        """
        component_map = self._parse_target_modules(
            target_modules=self.model_args.target_modules,
            components=self.model_args.target_components,
        )

        return component_map

    # ============================== EMA Management ==============================
    def _init_ema(self):
        """Initialize EMA wrapper for the transformer."""
        if self.training_args.ema_decay > 0:
            ema_device = (
                self.accelerator.device
                if self.training_args.ema_device == "cuda"
                else torch.device("cpu")
            )
            self.ema_wrapper = EMAModuleWrapper(
                parameters=self.get_trainable_parameters(),
                decay=self.training_args.ema_decay,
                update_step_interval=self.training_args.ema_update_interval,
                device=ema_device,
                decay_schedule=self.training_args.ema_decay_schedule,
                # Pass decay schedule params from training_args
                **self.training_args,
            )
        else:
            self.ema_wrapper = None

    def ema_step(self, step: int):
        """Update EMA parameters."""
        if hasattr(self, "ema_wrapper") and self.ema_wrapper is not None:
            self.ema_wrapper.step(self.get_trainable_parameters(), optimization_step=step)

    @contextmanager
    def use_ema_parameters(self):
        if hasattr(self, "ema_wrapper") and self.ema_wrapper is not None:
            trainable_params = self.get_trainable_parameters()
            with self.ema_wrapper.use_ema_parameters(trainable_params):
                yield
        else:
            yield

    @contextmanager
    def use_variant_snapshot(self, snapshot_name: str) -> Iterator[None]:
        """Temporarily install a variant-local parameter EMA snapshot.

        The caller names the snapshot; this adapter attaches no meaning to the
        name. An algorithm that keeps an EMA of one variant asks for it here and
        decides itself when an export should use it.

        Args:
            snapshot_name: Snapshot registered through the variant registry.

        Yields:
            Control while the snapshot's parameters are installed.

        Raises:
            RuntimeError: If no component variants were declared.
        """
        registry = self._require_variant_registry("parameter EMA")
        with registry.use_snapshot(snapshot_name):
            yield

    def get_variant_snapshot(self, snapshot_name: str) -> Tuple[torch.Tensor, ...]:
        """Return the raw tensors of a variant-local parameter EMA snapshot.

        Args:
            snapshot_name: Snapshot registered through the variant registry.

        Returns:
            The snapshot's parameter tensors, for a caller that exports them.

        Raises:
            RuntimeError: If no component variants were declared.
        """
        registry = self._require_variant_registry("parameter EMA")
        return registry.snapshot_tensors(snapshot_name)

    # ============================== Reference Parameters ==============================
    def _init_ref_parameters(self):
        """
        Initialize reference parameters for target components.
        Used for KL regularization during training.
        """
        if self.training_args.requires_ref_model and self.model_args.finetune_type in ["full"]:
            ref_param_device = (
                self.accelerator.device
                if self.training_args.ref_param_device == "cuda"
                else torch.device("cpu")
            )
            self._ref_ema = EMAModuleWrapper(
                parameters=self.get_trainable_parameters(),
                decay=0.0,  # No decay,
                update_step_interval=0,  # No updates, just store original weights
                device=ref_param_device,
            )
        else:
            self._ref_ema = None

    @contextmanager
    def use_ref_parameters(self):
        """Context manager to use reference parameters."""
        if self.model_args.finetune_type == "lora":
            # Use ExitStack to manage multiple context managers (one per component)
            with ExitStack() as stack:
                enabled_any = False
                for comp_name in self.target_module_map.keys():
                    component = (
                        self.get_component(comp_name) if self.has_component(comp_name) else None
                    )
                    if component is not None:
                        unwrapped = self._unwrap(component)

                        # Handle Compiled Models (torch.compile)
                        if hasattr(unwrapped, "_orig_mod"):
                            unwrapped = unwrapped._orig_mod

                        if isinstance(unwrapped, PeftModel):
                            # Enter disable_adapter context for each component
                            stack.enter_context(unwrapped.disable_adapter())
                            enabled_any = True
                if not enabled_any:
                    logger.warning("No LoRA adapters found to disable in use_ref_parameters")

                yield

        elif self._ref_ema is not None:
            trainable_params = self.get_trainable_parameters()
            # If ref_ema is on CPU, this line will be very slow!
            with self._ref_ema.use_ema_parameters(trainable_params):
                yield
        else:
            yield

    # ============================== Named Parameters Snapshot ==============================
    """
        These utilities help to snapshot and restore named parameters for target components.
        NOTE: `use_ref_parameters` always refers to the original model weights before any fine-tuning.

        A caller that refreshes its frozen reference during training, or that keeps more
        than one such copy, needs something more flexible than that single fixed reference.
        The functions below store, use, update, and remove named parameter snapshots, and
        attach no meaning to the names.
    """

    def _get_component_parameters(self, component_names: List[str]) -> List[torch.nn.Parameter]:
        """Get trainable parameters from specified components."""
        params = []
        for comp_name in component_names:
            if self.has_component(comp_name):
                component = self._require_component(comp_name)
                params.extend(p for p in component.parameters() if p.requires_grad)
            else:
                logger.warning(f"Component '{comp_name}' not found in the model. Skipping.")
        return params

    def add_named_parameters(
        self,
        name: str,
        target_components: Optional[Union[str, List[str]]] = None,
        device: Optional[Union[torch.device, str]] = None,
        overwrite: bool = True,
    ) -> None:
        """
        Store current trainable parameters snapshot under a name.

        Args:
            name: Identifier for this parameter snapshot
            target_components: Component names to store. Defaults to components with trainable params.
            device: Storage device (defaults to 'cpu')
            overwrite: Whether to overwrite existing snapshot
        """
        if name in self._named_parameters and not overwrite:
            raise KeyError(f"Named parameters '{name}' exists. Use overwrite=True.")

        # Normalize target components - keep only those with trainable params
        if target_components is None:
            target_components = [k for k, v in self.target_module_map.items() if v]
        elif isinstance(target_components, str):
            target_components = [target_components]

        # Validate
        invalid = set(target_components) - set(self.target_module_map.keys())
        if invalid:
            raise ValueError(
                f"expected target_components to name declared components "
                f"{list(self.target_module_map.keys())}, received unknown {sorted(invalid)}"
            )

        device = torch.device(device) if device else torch.device("cpu")
        params = self._get_component_parameters(target_components)

        if not params:
            raise ValueError(f"No trainable parameters found in {target_components}")

        self._named_parameters[name] = NamedParametersInfo(
            target_components=target_components,
            ema_wrapper=EMAModuleWrapper(
                parameters=params,
                decay=0.0,
                update_step_interval=0,
                device=device,
            ),
        )
        logger.info(f"Stored named parameters '{name}' for {target_components} on {device}")

    @contextmanager
    def use_named_parameters(self, name: str):
        """
        Context manager to temporarily use named parameters.

        Args:
            name: Name of stored parameters snapshot

        Usage:
            adapter.add_named_parameters('init')
            # ... training ...
            with adapter.use_named_parameters('init'):
                evaluate(model)  # Uses stored weights
            # Current weights restored
        """
        if name not in self._named_parameters:
            raise KeyError(f"'{name}' not found. Available: {self.list_named_parameters()}")

        info = self._named_parameters[name]
        params = self._get_component_parameters(info.target_components)

        with info.ema_wrapper.use_ema_parameters(params):
            yield

    def update_named_parameters(
        self,
        name: str,
        target_components: Optional[Union[str, List[str]]] = None,
        new_parameters: Optional[Iterable[torch.nn.Parameter]] = None,
    ) -> None:
        """
        Update existing named parameters with specified or current values.

        Args:
            name: Name of snapshot to update
            target_components: Components to update. Defaults to originally stored components.
            new_parameters: Parameters to copy from. Defaults to current model parameters.
        """
        if name not in self._named_parameters:
            raise KeyError(f"'{name}' not found.")

        info = self._named_parameters[name]

        # Resolve target components
        if target_components is None:
            target_components = info.target_components
        elif isinstance(target_components, str):
            target_components = [target_components]

        if not set(target_components).issubset(set(info.target_components)):
            raise ValueError(
                f"expected target_components for snapshot {name!r} to be a subset of the "
                f"components it stored {info.target_components!r}, received "
                f"{sorted(set(target_components) - set(info.target_components))!r}"
            )

        # Resolve parameters
        if new_parameters is None:
            new_parameters = self._get_component_parameters(target_components)
        else:
            new_parameters = list(new_parameters)

        # Validate param count
        if len(new_parameters) != len(info.ema_wrapper.ema_parameters):
            raise ValueError(
                f"Parameter count mismatch: got {len(new_parameters)}, "
                f"expected {len(info.ema_wrapper.ema_parameters)}"
            )

        # Update
        with torch.no_grad():
            for ema_param, param in zip(
                info.ema_wrapper.ema_parameters, new_parameters, strict=True
            ):
                ema_param.data.copy_(param.detach().to(ema_param.device))

        logger.info(f"Updated named parameters '{name}'")

    def remove_named_parameters(self, name: str) -> None:
        """Remove named parameters."""
        if name not in self._named_parameters:
            raise KeyError(f"'{name}' not found.")
        del self._named_parameters[name]
        logger.info(f"Removed named parameters '{name}'")

    def list_named_parameters(self) -> List[str]:
        """List all stored parameter names."""
        return list(self._named_parameters.keys())

    def get_named_parameters_info(self, name: str) -> Dict[str, Any]:
        """Get info about a named parameter snapshot."""
        if name not in self._named_parameters:
            raise KeyError(f"'{name}' not found.")
        info = self._named_parameters[name]
        return {
            "name": name,
            "target_components": info.target_components,
            "num_params": len(info.ema_wrapper.ema_parameters),
            "device": str(info.ema_wrapper.device),
        }

    def get_named_parameters(self, name: str) -> List[torch.nn.Parameter]:
        """
        Get the stored parameter tensors for a named snapshot.

        Args:
            name: Identifier of the stored snapshot.

        Returns:
            List[torch.nn.Parameter]: The stored parameter tensors.
        """
        if name not in self._named_parameters:
            raise KeyError(f"'{name}' not found. Available: {self.list_named_parameters()}")
        return self._named_parameters[name].ema_wrapper.ema_parameters

    # ============================== Gradient Checkpointing ==============================
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for target components."""
        for comp_name in self.model_args.target_components:
            if self.has_component(comp_name):
                component = self.get_component(comp_name)
                if hasattr(component, "enable_gradient_checkpointing"):
                    component.enable_gradient_checkpointing()
                    logger.info(f"Enabled gradient checkpointing for {comp_name}")
                else:
                    logger.warning(f"{comp_name} does not support gradient checkpointing")

    # ============================== Precision Management ==============================
    def _cast_module_mixed_precision(
        self,
        component: torch.nn.Module,
        train_dtype: torch.dtype,
        frozen_dtype: Optional[torch.dtype],
    ) -> int:
        """
        Set floating-point parameters/buffers without a trainable round-trip through frozen_dtype.

        Trainable parameters use ``train_dtype``. Frozen parameters and floating-point buffers use
        ``frozen_dtype`` when it is set, or are left at their loaded dtype when ``frozen_dtype`` is
        ``None`` (preserve). Integer/bool buffers are left unchanged (same as ``Module.to``).
        """
        n_trainable = 0
        for _, param in component.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(dtype=train_dtype)
                n_trainable += 1
            elif frozen_dtype is not None:
                param.data = param.data.to(dtype=frozen_dtype)
        for _, buf in component.named_buffers():
            if buf.is_floating_point() and frozen_dtype is not None:
                buf.data = buf.data.to(dtype=frozen_dtype)
        return n_trainable

    def _apply_component_precision_policy(
        self,
        name: str,
        component: torch.nn.Module,
    ) -> None:
        """Apply the configured original-dtype policy to one materialized module."""
        train_dtype = self.model_args.trainable_parameters_dtype
        frozen_dtype = self.model_args.frozen_parameters_dtype
        if name in self.model_args.target_components:
            if self._is_fsdp2() and self.accelerator.mixed_precision != "no":
                component.to(dtype=torch.float32)
            else:
                self._cast_module_mixed_precision(component, train_dtype, frozen_dtype)
        elif frozen_dtype is not None:
            component.to(dtype=frozen_dtype)

    def _mix_precision(self):
        """Set trainable params to ``trainable_parameters_dtype``; by default leave frozen params
        and floating-point buffers at their loaded (``from_pretrained``) dtype.

        This is the single place that decides every parameter's *original* dtype before
        ``accelerator.prepare``; the trainer only bundles + prepares.

        Frozen-dtype policy: ``frozen_parameters_dtype=None`` (default) preserves each frozen
        parameter/buffer's original dtype and never downcasts -- a released checkpoint deliberately
        ships components in different dtypes (e.g. Z-Image: transformer fp32, text encoder bf16), and
        forcing one uniform frozen dtype would override those choices. Set an explicit
        ``frozen_parameters_dtype`` to opt into casting every frozen param to that dtype.

        FSDP2 caveat: FSDP2 shards each unit with ONE original dtype, and accelerate upcasts the
        trainable params to an fp32 master when ``mixed_precision != 'no'``. So a trained component
        that also bundles frozen members (e.g. Wan2.2 trains ``transformer`` while ``transformer_2``
        is frozen-but-sharded) would otherwise mix fp32/low-precision within a unit and trip FSDP2's
        uniform-dtype assert. We therefore force the TRAINED components to a uniform fp32 original
        dtype here (compute stays low-precision via accelerate's ``MixedPrecisionPolicy``); untrained
        components are cast to ``frozen_parameters_dtype`` when set, else preserved.
        """
        train_dtype = self.model_args.trainable_parameters_dtype
        frozen_dtype = self.model_args.frozen_parameters_dtype  # None -> preserve loaded dtype

        target_set = frozenset(self.model_args.target_components)
        component_names = self._resolve_component_names(None)
        merged_names = list(dict.fromkeys([*component_names, *self.model_args.target_components]))

        # FSDP2: trained (sharded) components need a uniform fp32 original dtype.
        if self._is_fsdp2() and self.accelerator.mixed_precision != "no":
            for name in merged_names:
                if name in target_set:
                    self.get_component(name).to(dtype=torch.float32)
                elif frozen_dtype is not None:
                    self.get_component(name).to(dtype=frozen_dtype)
                # else: preserve the untrained component's loaded dtype
            logger.info(
                f"FSDP2: trained components -> fp32 (uniform orig dtype); "
                f"other components -> {frozen_dtype or 'preserved (loaded dtype)'}"
            )
            return

        # Explicit uniform dtype -> a single cast of every component suffices.
        if frozen_dtype is not None and train_dtype == frozen_dtype:
            for name in merged_names:
                self.get_component(name).to(dtype=frozen_dtype)
            return

        # Split: trainable -> train_dtype; frozen -> frozen_dtype, or preserved when None.
        trainable_count = 0
        for name in merged_names:
            component = self.get_component(name)
            if name in target_set:
                trainable_count += self._cast_module_mixed_precision(
                    component, train_dtype, frozen_dtype
                )
            elif frozen_dtype is not None:
                component.to(dtype=frozen_dtype)
            # else: preserve the fully-frozen component's loaded dtype

        if trainable_count > 0:
            logger.info(
                f"Set {trainable_count} trainable parameters to {train_dtype}; "
                f"frozen params -> {frozen_dtype or 'preserved (loaded dtype)'}"
            )

    # ============================== LoRA Management ==============================
    def apply_lora(
        self,
        target_modules: Union[str, List[str]],
        components: Union[str, List[str]] = "transformer",
        overwrite: bool = False,
    ) -> Union[PeftModel, Dict[str, PeftModel]]:
        """
        Apply LoRA adapters to specified components with prefix-based module targeting.

        Args:
            target_modules: Module patterns with optional component prefix
                - 'to_q': Apply to all components in `components`
                - 'transformer.to_q': Apply only to transformer
                - 'transformer_2.to_v': Apply only to transformer_2
                - ['to_q', 'transformer.to_k']: Mixed specification
            components: Component(s) to apply LoRA
            overwrite: When applying LoRA to a component that already has LoRA adapters:
                If True, delete existing 'default' adapter and create new one.
                If False, skip components that already have LoRA adapters.
        """
        # Normalize components to list
        if isinstance(components, str):
            components = [components]

        # Parse with explicit target_modules
        component_modules = self._parse_target_modules(target_modules, components)
        # Apply LoRA to each component
        results = {}
        for comp in components:
            modules = component_modules.get(comp)

            # Handle special cases
            if modules == "default":
                modules = self.default_target_modules
            elif modules == "all":
                modules = "all"  # Keep as 'all' for PEFT
            elif not modules:
                logger.warning(f"No target modules for {comp}, skipping LoRA")
                continue

            lora_config = LoraConfig(
                r=self.model_args.lora_rank,
                lora_alpha=self.model_args.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=modules,
            )

            model_component = self.get_component(comp)

            if isinstance(model_component, PeftModel):
                # Already a PeftModel, check for existing adapter
                has_default = "default" in model_component.peft_config
                if has_default and not overwrite:
                    logger.info(
                        f"Component {comp} already has 'default' adapter. Skipping initialization but enabling gradients."
                    )
                    # We must unfreeze the lora parameters because `_freeze_components` might have frozen them!
                    for name, param in model_component.named_parameters():
                        if any(k in name for k in self.lora_keys):
                            param.requires_grad = True
                    results[comp] = model_component
                    continue

                if has_default and overwrite:
                    # Overwrite: delete existing adapter and reinitialize
                    logger.info(f"Overwriting existing 'default' adapter for {comp}")
                    model_component.delete_adapter("default")

                # Add `default` adapter to existing PeftModel
                model_component.add_adapter("default", lora_config)
            else:
                # Not a PeftModel, initialize directly
                lora_config = LoraConfig(
                    r=self.model_args.lora_rank,
                    lora_alpha=self.model_args.lora_alpha,
                    init_lora_weights="gaussian",
                    target_modules=modules,
                )
                model_component = get_peft_model(model_component, lora_config)
                # Set back to attribute
                self.set_component(comp, model_component)

            # Activate the adapter
            model_component.set_adapter("default")
            results[comp] = model_component

            logger.info(f"Applied LoRA to {comp} with modules: {modules}")

        if not results:
            logger.warning("No LoRA adapters were applied")
            return {}

        return next(iter(results.values())) if len(results) == 1 else results

    # ============================== Distributed Utils ==================================

    # ------------------------------ Dist Types -----------------------------------------
    @property
    def _distributed_type(self) -> DistributedType:
        """Get current distributed type."""
        return self.accelerator.distributed_type

    def _is_deepspeed(self) -> bool:
        """Check if DeepSpeed is enabled."""
        return self._distributed_type == DistributedType.DEEPSPEED

    def _is_fsdp(self) -> bool:
        """Check if FSDP (v1) is enabled."""
        return self._distributed_type == DistributedType.FSDP

    def _is_fsdp2(self) -> bool:
        """Check if FSDP2 is enabled."""
        return getattr(self.accelerator, "is_fsdp2", False)

    def uses_fsdp_cpu_efficient_loading(self) -> bool:
        """Return whether FSDP defers weight materialization to rank zero.

        Public because the trainer must know it: under this mode only rank zero
        holds real weights before ``prepare``, so frozen-component broadcasts and
        preprocessing loads are ordered around it.
        """
        if not self._is_fsdp():
            return False
        fsdp_plugin = self.accelerator.state.fsdp_plugin
        return fsdp_plugin is not None and getattr(fsdp_plugin, "cpu_ram_efficient_loading", False)

    # ------------------------------ Shard Strategies ---------------------------------
    def _is_zero3(self) -> bool:
        """Check if DeepSpeed ZeRO Stage 3 (parameter sharding) is enabled."""
        if not self._is_deepspeed():
            return False
        ds_plugin = self.accelerator.state.deepspeed_plugin
        return ds_plugin is not None and ds_plugin.zero_stage == 3

    def _is_fsdp_param_sharded(self) -> bool:
        """Check if FSDP shards parameters across ranks (FULL_SHARD or HYBRID)."""
        if not self._is_fsdp():
            return False
        fsdp_plugin = self.accelerator.state.fsdp_plugin
        if fsdp_plugin is None:
            return False
        from torch.distributed.fsdp import ShardingStrategy

        return fsdp_plugin.sharding_strategy in (
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.HYBRID_SHARD,
            ShardingStrategy._HYBRID_SHARD_ZERO2,
        )

    # ------------------------------ FSDP Views ----------------------------------------
    def _fsdp_state_dict_type(self):
        """Get FSDP state_dict_type, returns None if not FSDP."""
        if not self._is_fsdp():
            return None
        fsdp_plugin = self.accelerator.state.fsdp_plugin
        return fsdp_plugin.state_dict_type if fsdp_plugin else None

    def _is_fsdp_collective_state_dict(self) -> bool:
        """Check if FSDP state_dict_type requires collective operations."""
        from torch.distributed.fsdp import StateDictType

        state_dict_type = self._fsdp_state_dict_type()
        if state_dict_type is None:
            return False
        # LOCAL_STATE_DICT does not requires communication while others do
        return state_dict_type != StateDictType.LOCAL_STATE_DICT

    def _is_param_sharded(self) -> bool:
        """Check if parameters are sharded across ranks."""
        return self._is_zero3() or self._is_fsdp2() or self._is_fsdp_param_sharded()

    def _requires_collective_state_dict(self) -> bool:
        """
        Check if state_dict gathering requires all ranks to participate.

        This is True when:
        - DeepSpeed ZeRO-3 (parameters sharded)
        - FSDP2 (always uses collective ops)
        - FSDP with FULL/SHARDED_STATE_DICT (collective save)
        - FSDP with FULL_SHARD (parameters sharded, must gather)
        """
        if self._is_zero3():
            return True
        if self._is_fsdp2():
            return True
        if self._is_fsdp() and (
            self._is_fsdp_param_sharded() or self._is_fsdp_collective_state_dict()
        ):
            return True
        return False

    # ============================== Checkpoint Management ==============================

    # ------------------------------ State Dict ------------------------------------------

    def get_state_dict(
        self,
        model,
        unwrap=True,
        state_dict_keys: Optional[Iterable[str]] = None,
        ignore_frozen_params: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        **Modified from `Accelerator.get_state_dict`**
        Returns the state dictionary of a model sent through [`Accelerator.prepare`] potentially without full
        precision.

        Args:
            model (`torch.nn.Module`):
                A PyTorch model sent through [`Accelerator.prepare`]
            unwrap (`bool`, *optional*, defaults to `True`):
                Whether to return the original underlying state_dict of `model` or to return the wrapped state_dict
                (e.g. for DeepSpeed or FSDP models).
            state_dict_keys (`List[str]`, *optional*):
                If provided, only return the parameters with these keys in the state dict. This is useful for saving with FSDP
                when you only want to save the trainable parameters.
            ignore_frozen_params (`bool`, *optional*, defaults to `False`):
                For FSDP2 only. If `True`, frozen parameters (i.e., those with `requires_grad=False`) will be ignored when saving the state dict.

        Returns:
            `dict`: The state dictionary of the model potentially without full precision.
        ```
        """

        def is_param_match_key(name, keys, strict=True):
            if keys is None:
                return not strict  # strict: no keys → no match; non-strict: no keys → match all
            if strict:
                return name in keys
            return any(k in name for k in keys)

        state_dict_keys = set(state_dict_keys) if state_dict_keys is not None else None

        from accelerate.utils import compare_versions

        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            zero3_sharding = self.accelerator.deepspeed_config["zero_optimization"]["stage"] == 3
            tp_sharding = (
                self.accelerator.deepspeed_config.get("tensor_parallel", {}).get("autotp_size", 0)
                > 1
            )
            if zero3_sharding or tp_sharding:
                if model.zero_gather_16bit_weights_on_model_save():
                    ver_min_required = "0.16.4"
                    if tp_sharding and not compare_versions("deepspeed", ">=", ver_min_required):
                        raise ImportError(
                            f"Deepspeed TP requires deepspeed>={ver_min_required}. Please update DeepSpeed via `pip install deepspeed -U`."
                        )
                    state_dict = (
                        model._consolidated_16bit_state_dict()
                        if tp_sharding
                        else model._zero3_consolidated_16bit_state_dict()
                    )
                else:
                    raise ValueError(
                        "Cannot get 16bit model weights because `stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False. "
                        "To save the model weights in 16bit, set `stage3_gather_16bit_weights_on_model_save` to True in DeepSpeed config file or "
                        "set `zero3_save_16bit_model` to True when using `accelerate config`. "
                        "To save the full checkpoint, run `model.save_checkpoint(save_dir)` and use `zero_to_fp32.py` to recover weights."
                    )
            else:
                from deepspeed.checkpoint.utils import clone_tensors_for_torch_save

                state_dict = clone_tensors_for_torch_save(self._unwrap(model).state_dict())
        elif self.accelerator.is_fsdp2:
            # FSDP2: gather the full (unsharded) params to rank0 via the DTensor-aware API.
            # NOTE: the previous `state_dict_keys` path toggled `requires_grad` at runtime to
            # sub-select params, but FSDP2's `ignore_frozen_params` is keyed off the trainability
            # captured at `fully_shard` time -- the runtime toggle is a no-op, so it yielded an
            # EMPTY adapter. Gather straight through (LoRA params are exactly the trainable subset,
            # so `ignore_frozen_params=True` returns them) and let the shared key-filter below
            # narrow to `state_dict_keys` when provided.
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
            )

            options = StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                cpu_offload=True,
                ignore_frozen_params=ignore_frozen_params,
            )
            state_dict = get_model_state_dict(model, options=options)
        elif self.accelerator.distributed_type == DistributedType.FSDP:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.checkpoint.state_dict import (
                get_state_dict as fsdp_get_state_dict,
                get_model_state_dict as fsdp_get_model_state_dict,
            )

            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = model.state_dict()
        else:
            if unwrap:
                model = self._unwrap(model)
            state_dict = model.state_dict()

        # Filter by keys.
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if is_param_match_key(k, state_dict_keys, strict=False)
        }

        return state_dict

    @classmethod
    def _filter_lora_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        adapter_name: str = "default",
    ) -> Dict[str, torch.Tensor]:
        """
        Filter state dict to only include LoRA parameters.

        Args:
            state_dict: Full model state dict
            adapter_name: Name of the LoRA adapter (default: "default")

        Returns:
            State dict containing only LoRA-related weights
        """
        return {k: v for k, v in state_dict.items() if any(lk in k for lk in cls.lora_keys)}

    # -------------------------------------------- Save ------------------------------------
    def _save_lora(
        self,
        model: torch.nn.Module,
        save_directory: str,
    ) -> None:
        """Save LoRA adapter with distributed training support."""
        unwrapped = self._unwrap(model)

        if not isinstance(unwrapped, PeftModel):
            logger.warning(f"Model is not a PeftModel, falling back to full save.")
            self._save_full_model(
                model,
                save_directory,
                safe_serialization=True,
            )
            return

        # With variants declared, several adapters live on one shared base and
        # save_pretrained would write every one of them, so the save is scoped to the
        # adapter of the variant the caller asked for.
        registry = getattr(self, "component_variant_registry", None)
        selected_adapters = None
        if registry is not None:
            adapter_name = registry.get_spec(registry.active_variant).adapter_name
            if adapter_name is None:
                raise ValueError(
                    f"expected LoRA variant {registry.active_variant!r} to declare an "
                    "adapter_name, received None"
                )
            selected_adapters = [adapter_name]

        # If not sharded save, use standard save_pretrained
        if self._requires_collective_state_dict():
            # Handle sharded save
            # Gather all params before saving
            state_dict = self.get_state_dict(
                model,
                unwrap=True,
                state_dict_keys=self.lora_keys,
                ignore_frozen_params=True,
            )
            if self.accelerator.is_main_process:
                unwrapped.save_pretrained(
                    save_directory,
                    state_dict=state_dict,
                    selected_adapters=selected_adapters,
                )
        else:
            if self.accelerator.is_main_process:
                unwrapped.save_pretrained(
                    save_directory,
                    selected_adapters=selected_adapters,
                )

        self.accelerator.wait_for_everyone()

    def _save_full_model(
        self,
        model: torch.nn.Module,
        save_directory: str,
        max_shard_size: str = "10GB",
        safe_serialization: bool = True,
        dtype: Optional[Union[torch.dtype, str]] = None,
    ) -> None:
        """
        **Modified from `Accelerator.save_model`**
        Save full model weights with distributed training support.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        # Normalize dtype
        if isinstance(dtype, str):
            dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }.get(dtype.lower(), torch.bfloat16)

        unwrapped = self._unwrap(model)

        # Check if casting is needed
        cast_needed = False
        if dtype is not None:
            # Try to get model dtype, falling back to parameter inspection
            model_dtype = getattr(unwrapped, "dtype", None)
            if model_dtype is None:
                try:
                    model_dtype = next(unwrapped.parameters()).dtype
                except StopIteration:
                    # Empty model, assume no cast needed
                    model_dtype = dtype

            if model_dtype != dtype:
                cast_needed = True

        # Check offload
        is_offloaded = any(has_offloaded_params(module) for module in unwrapped.modules())

        # No shard, no casting, no offload, save directyly
        if not self._requires_collective_state_dict() and not cast_needed and not is_offloaded:
            # Standard save
            if self.accelerator.is_main_process:
                unwrapped.save_pretrained(
                    save_directory,
                    max_shard_size=max_shard_size,
                    safe_serialization=safe_serialization,
                )
            self.accelerator.wait_for_everyone()
            return

        # Get the state_dict of the model
        if is_offloaded:
            state_dict = get_state_dict_offloaded_model(model)
        else:
            if any(param.device == torch.device("meta") for param in model.parameters()):
                raise RuntimeError(
                    "You can't save the model since some parameters are on the meta device."
                )
            state_dict = self.get_state_dict(model, unwrap=True, ignore_frozen_params=False)

        # Case: DeepSpeed zero3 gets gathered and `state_dict` is empty
        if state_dict is None:
            return

        # Dtype casting
        if dtype is not None:
            for k in state_dict.keys():
                state_dict[k] = state_dict[k].to(device="cpu", dtype=dtype)

        os.makedirs(save_directory, exist_ok=True)

        if safe_serialization:
            state_dict = clean_state_dict_for_safetensors(state_dict)

        weights_name = SAFE_DIFFUSION_WEIGHTS_NAME if safe_serialization else DIFFUSION_WEIGHTS_NAME
        filename_pattern = (
            SAFE_DIFFUSION_WEIGHTS_PATTERN_NAME
            if safe_serialization
            else DIFFUSION_WEIGHTS_PATTERN_NAME
        )

        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
        )

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "")

            # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
            filename_no_suffix = filename.replace(".bin", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in state_dict_split.filename_to_tensors.keys()
                and reg.fullmatch(filename_no_suffix) is not None
                and PartialState().is_main_process
            ):
                os.remove(full_filename)

        # Save the model
        for filename, tensors in state_dict_split.filename_to_tensors.items():
            shard = {tensor: state_dict[tensor] for tensor in tensors}
            self.accelerator.save(
                shard, os.path.join(save_directory, filename), safe_serialization=safe_serialization
            )

        # Save the config file
        if hasattr(unwrapped, "config") and unwrapped.config is not None:
            config_save_file = os.path.join(save_directory, CONFIG_NAME)
            if hasattr(unwrapped.config, "save_pretrained"):
                unwrapped.config.save_pretrained(save_directory)
            else:
                # Handle dict-like configs (e.g., FrozenDict from diffusers)
                with open(config_save_file, "w", encoding="utf-8") as f:
                    json.dump(dict(unwrapped.config), f, indent=2, sort_keys=True)

            if self.accelerator.is_main_process:
                logger.info(f"Model config saved in {config_save_file}")

        # Save index if sharded
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = (
                SAFE_DIFFUSION_WEIGHTS_INDEX_NAME
                if safe_serialization
                else DIFFUSION_WEIGHTS_INDEX_NAME
            )
            save_index_file = os.path.join(save_directory, save_index_file)
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            if self.accelerator.is_main_process:
                logger.info(
                    f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                    f"split in {len(state_dict_split.filename_to_tensors)} checkpoint shards. You can find where each parameters has been saved in the "
                    f"index located at {save_index_file}."
                )
        else:
            path_to_weights = os.path.join(save_directory, weights_name)
            if self.accelerator.is_main_process:
                logger.info(f"Model weights saved in {path_to_weights}")

    def save_checkpoint(
        self,
        save_directory: str,
        max_shard_size: str = "10GB",
        dtype: Union[torch.dtype, str] = torch.bfloat16,
        save_ema: bool = True,
        model_only: bool = True,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        **kwargs,
    ):
        """Save a checkpoint for the target components.

        Args:
            save_directory: Destination directory.
            max_shard_size: Shard size for a sharded full-weight save.
            dtype: Storage dtype for full weights.
            save_ema: Whether to install the EMA parameters while saving.
            model_only: Whether to write model weights rather than training state.
            safe_serialization: Whether to write safetensors.
            variant: Component variant to write. ``None`` writes the base
                components, which is what a single-policy algorithm always wants.
            **kwargs: Forwarded to ``accelerator.save_state`` for a training state.
        """
        # Normalize dtype
        if isinstance(dtype, str):
            dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }.get(dtype.lower(), torch.bfloat16)

        # 1. Save the training state if not model_only
        if not model_only:
            if self.accelerator.is_main_process:
                logger.info(f"Saving training state (resume-ready) to {save_directory}...")

            # Variant metadata must be written before accelerate mutates optimizer
            # state, or a resume cannot tell which optimizer group owns which variant.
            multirole_checkpoint_state = getattr(self, "_multirole_checkpoint_state", None)
            if multirole_checkpoint_state is not None:
                multirole_checkpoint_state.prepare_save(save_directory)
            self.accelerator.save_state(
                save_directory, safe_serialization=safe_serialization, **kwargs
            )

            if self.accelerator.is_main_process:
                logger.info(f"Training state saved.")
            return

        # 2. Save the model only. The scope is the caller's: `variant=None` writes the
        # base components, and an algorithm that keeps training-only variants asks for
        # the one it ships rather than relying on a rule spelled out here.
        save_context = self.use_ema_parameters if save_ema else nullcontext
        registry = getattr(self, "component_variant_registry", None)
        if registry is None:
            variant_context = nullcontext
        elif variant is None:
            variant_context = registry.use_base_variant
        else:
            variant_context = partial(registry.use, variant)

        with variant_context(), save_context():
            for comp_name, target_modules in self.target_module_map.items():
                if not self.has_component(comp_name):
                    logger.warning(f"Component {comp_name} not found, skipping save")
                    continue

                if not target_modules:
                    logger.info(f"No target modules applied to {comp_name}, skip saving")
                    continue

                # Peel the RoutedComponentProxy to the inner module so the save
                # path operates on the real diffusers/PeftModel (the prepared root
                # is the ModelBundle; FSDP/DeepSpeed gathering happens inside
                # get_state_dict on this member's params).
                component = self._unwrap(self._require_component(comp_name))

                # Determine save path
                comp_path = (
                    os.path.join(save_directory, comp_name)
                    if len(self.model_args.target_components) > 1
                    else save_directory
                )

                os.makedirs(comp_path, exist_ok=True)

                # Dispatch to appropriate save method
                if self.model_args.finetune_type == "lora":
                    if self.accelerator.is_main_process:
                        logger.info(f"Saving LoRA weights for {comp_name} to {comp_path}")
                    self._save_lora(component, comp_path)
                else:
                    if self.accelerator.is_main_process:
                        logger.info(f"Saving full weights for {comp_name} to {comp_path}")
                    self._save_full_model(
                        component,
                        comp_path,
                        max_shard_size=max_shard_size,
                        safe_serialization=safe_serialization,
                        dtype=dtype,
                    )

            # Sync after saving
            self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            logger.info(f"Checkpoint saved successfully to {save_directory}")

    # -------------------------------------------- Load -------------------------------------------
    def _resolve_checkpoint_path(self, path: str) -> str:
        """
        Resolve `path` to a local directory, downloading from Hugging Face Hub when needed.

        Resolution order:
            1. If `path` starts with ``hf://``, strip the prefix and force HF download
               (lets users override a colliding local directory).
            2. Otherwise, if `path` exists locally, return it as-is.
            3. Otherwise, parse as ``owner/repo[/subfolder][@revision]`` and download
               via Hugging Face Hub.

        Multi-node-safe: all ranks call ``snapshot_download`` directly. Hugging
        Face Hub's per-blob ``WeakFileLock`` serializes concurrent calls within
        each filesystem domain (cross-node on POSIX-locking shared FS, per-node
        on non-shared FS), so exactly one rank per filesystem domain actually
        transfers bytes. Un-gated (rather than ``is_local_main_process`` plus a
        barrier) so a failed download raises uniformly on every affected rank
        instead of leaving siblings deadlocked at a barrier the failing rank
        never reaches. Residual hazard: a rare single-rank transient failure
        (e.g. one node's network blip) can produce asymmetric progress, in
        which case the surviving ranks will eventually trip the NCCL watchdog
        on the final barrier below.

        Args:
            path: Local filesystem path or HF spec (with or without ``hf://`` prefix).

        Returns:
            Absolute local directory path ready for the existing checkpoint loaders.

        Raises:
            FileNotFoundError: When the spec is neither a local path nor a reachable HF repo.
        """
        # Normalize leading ``~`` for local-path inputs; no-op for HF specs since
        # ``expanduser`` only acts on a leading ``~``.
        path = os.path.expanduser(path)
        force_hf = path.startswith(HF_PATH_PREFIX)

        # Local path wins unless an explicit ``hf://`` prefix forces remote.
        if not force_hf and os.path.exists(path):
            return path

        # ``parse_hf_checkpoint_path`` handles the ``hf://`` prefix internally.
        repo_id, subfolder, revision = parse_hf_checkpoint_path(path)

        try:
            local_path = download_hf_checkpoint(repo_id, subfolder, revision)
        except (RepositoryNotFoundError, HfHubHTTPError) as e:
            raise FileNotFoundError(
                f"Checkpoint {path!r} not found locally and could not be fetched "
                f"from Hugging Face Hub (repo={repo_id!r}, subfolder={subfolder!r}, "
                f"revision={revision!r}). For private repos, ensure HF_TOKEN is set "
                f"on ALL nodes."
            ) from e

        # Sync after download so downstream loaders enter the lockstep dispatch
        # together. On symmetric failure every rank raises above before this
        # barrier is reached, so no deadlock; the residual asymmetric-failure
        # case is documented in the docstring.
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_local_main_process:
            logger.info(
                f"[local rank 0 / global rank {self.accelerator.process_index}] "
                f"resolved checkpoint '{path}' -> {local_path}"
            )

        return local_path

    @staticmethod
    def load_sharded_checkpoint(checkpoint_dir: str, index_file: str) -> Dict[str, torch.Tensor]:
        """Load sharded safetensors checkpoint."""
        with open(index_file, "r") as f:
            index = json.load(f)

        state_dict = {}
        loaded_files = set()

        for param_name, filename in index["weight_map"].items():
            if filename not in loaded_files:
                shard_path = os.path.join(checkpoint_dir, filename)
                shard = load_file(shard_path)
                state_dict.update(shard)
                loaded_files.add(filename)

        return state_dict

    def _load_lora(self, path: str) -> None:
        """Load LoRA adapters for target components with auto-format detection."""
        # Iterate only components that actually carry target modules. Frozen
        # bundled members (e.g. Wan2.2's `transformer_2`, kept in
        # `target_components` solely to be FSDP-sharded for memory) map to None
        # in `target_module_map` and are skipped by `save_checkpoint`; iterating
        # `target_components` here would instead look for a `.../transformer_2/`
        # subdir that was never written and log a spurious error on every resume.
        for comp_name in self.trainable_component_names:
            if not self.has_component(comp_name):
                logger.warning(f"Component {comp_name} not found, skipping")
                continue

            component = self._require_component(comp_name)
            comp_path = (
                os.path.join(path, comp_name)
                if len(self.model_args.target_components) > 1
                else path
            )

            unwrapped = self._unwrap(component)

            # Auto-detect checkpoint format
            adapter_config_path = os.path.join(comp_path, LORA_ADAPTER_CONFIG_NAME)
            has_config_file = os.path.exists(adapter_config_path)

            if has_config_file:
                # Standard PeftModel format
                if not isinstance(unwrapped, PeftModel):
                    unwrapped = PeftModel.from_pretrained(unwrapped, comp_path, is_trainable=True)
                    unwrapped.set_adapter("default")
                    self.set_component(comp_name, unwrapped)
                else:
                    unwrapped.load_adapter(comp_path, unwrapped.active_adapter)
            else:
                # No config file found, manual `state_dict` loading with key mapping
                # Detect `safetensors` or `bin` format with `safetensors` preferred
                safetensors_files = glob.glob(os.path.join(comp_path, "*.safetensors"))
                if safetensors_files:
                    state_dict_path = sorted(safetensors_files)[0]
                    state_dict = load_file(state_dict_path)
                else:
                    bin_files = glob.glob(os.path.join(comp_path, "*.bin"))
                    if bin_files:
                        state_dict_path = sorted(bin_files)[0]
                        state_dict = torch.load(state_dict_path, map_location="cpu")
                    else:
                        logger.error(
                            f"No checkpoint file (.safetensors or .bin) found at {comp_path}"
                        )
                        continue

                if self.accelerator.is_main_process:
                    logger.info(
                        f"Loaded LoRA `state_dict` from: {state_dict_path}. "
                        f"If this is not wanted, please make sure the directory contains only single checkpoint file. "
                    )

                # Apply key mapping for legacy format
                state_dict = mapping_lora_state_dict(state_dict)

                # Infer LoRA configuration from state_dict
                lora_rank, lora_alpha = infer_lora_config(state_dict)
                lora_alpha = self.model_args.lora_alpha or lora_alpha  # Use model arg if given
                if self.model_args.target_modules in [None, "default"]:
                    # If default, infer target modules
                    target_modules = infer_target_modules(state_dict)
                else:
                    target_modules = self.model_args.target_modules

                if self.accelerator.is_main_process:
                    logger.info(
                        f"Inferred LoRA config for {comp_name}: "
                        f"rank={lora_rank}, alpha={lora_alpha}, target_modules={target_modules[:5]}..."
                    )

                # Create PeftModel if not already
                if not isinstance(unwrapped, PeftModel):
                    lora_config = LoraConfig(
                        r=lora_rank,
                        lora_alpha=lora_alpha,
                        init_lora_weights="gaussian",
                        target_modules=target_modules,
                    )

                    unwrapped = get_peft_model(unwrapped, lora_config)
                    unwrapped.set_adapter("default")

                # Load mapped state_dict
                missing, unexpected = unwrapped.load_state_dict(state_dict, strict=False)

                # Filter missing keys to LoRA only
                missing = [k for k in missing if any(lk in k for lk in self.lora_keys)]

                if self.accelerator.is_main_process:
                    if missing:
                        logger.warning(f"Missing keys: {missing[:5]}...")
                    if unexpected:
                        logger.warning(f"Unexpected keys: {unexpected[:5]}...")

                self.set_component(comp_name, unwrapped)

            if self.accelerator.is_main_process:
                logger.info(f"LoRA adapter loaded for {comp_name} from {comp_path}")

    def _load_full_model(self, path: str, strict: bool = True) -> None:
        """Load full model weights for target components."""
        # Match `save_checkpoint`: only components with target modules are
        # written, so frozen bundled members (`target_module_map[name] is None`)
        # must be skipped here rather than iterating all `target_components`.
        for comp_name in self.trainable_component_names:
            if not self.has_component(comp_name):
                logger.warning(f"Component {comp_name} not found, skipping")
                continue

            component = self._require_component(comp_name)
            comp_path = (
                os.path.join(path, comp_name)
                if len(self.model_args.target_components) > 1
                else path
            )

            unwrapped = self._unwrap(component)
            component_class = unwrapped.__class__

            # `from_pretrained` is the fast path; a checkpoint written by the manual
            # saver has no model index, so only that lookup is guarded. Installing the
            # result must stay outside the guard, or a rejected component name would be
            # downgraded to a debug line and the loaded weights silently discarded.
            new_component = None
            try:
                new_component = component_class.from_pretrained(comp_path)
            except (OSError, ValueError) as e:
                if self.accelerator.is_main_process:
                    logger.debug(
                        f"from_pretrained failed for {comp_name}: {e}, trying manual load..."
                    )

            if new_component is not None:
                self.set_component(comp_name, new_component)
                if self.accelerator.is_main_process:
                    logger.info(f"Loaded {comp_name} via from_pretrained from {comp_path}")
                continue

            # Detect the checkpoint type
            index_file = os.path.join(comp_path, SAFE_DIFFUSION_WEIGHTS_INDEX_NAME)
            weights_file = os.path.join(comp_path, SAFE_DIFFUSION_WEIGHTS_NAME)

            if os.path.exists(index_file):
                state_dict = self.load_sharded_checkpoint(comp_path, index_file)
            elif os.path.exists(weights_file):
                state_dict = load_file(weights_file)
            else:
                logger.error(f"No valid checkpoint found for {comp_name} at {comp_path}")
                continue

            # Load state_dict
            missing, unexpected = unwrapped.load_state_dict(state_dict, strict=strict)

            if self.accelerator.is_main_process:
                if missing:
                    logger.warning(f"Missing keys for {comp_name}: {missing[:5]}...")
                if unexpected:
                    logger.warning(f"Unexpected keys for {comp_name}: {unexpected[:5]}...")
                logger.info(f"Full model weights loaded for {comp_name} from {comp_path}")

    def _load_training_state(self, path: str) -> None:
        """Load full training state for resuming training."""
        if self.accelerator.is_main_process:
            logger.info(f"Loading training state from {path}...")

        # Reject a role layout that disagrees with this run before accelerate
        # restores optimizer state onto the wrong groups.
        multirole_checkpoint_state = getattr(self, "_multirole_checkpoint_state", None)
        if multirole_checkpoint_state is not None:
            multirole_checkpoint_state.validate_load(path)
        self.accelerator.load_state(path)

        if self.accelerator.is_main_process:
            logger.info("Training state loaded successfully.")

    def _detect_checkpoint_type(self, path: str) -> Literal["lora", "full"]:
        """
        Auto-detect checkpoint format by inspecting directory contents.

        Checks whether the checkpoint directory (or component subdirectories)
        contains LoRA adapter files (adapter_config.json). Falls back to 'full'
        if no LoRA signature files are found.
        """
        paths_to_check = (
            [os.path.join(path, comp_name) for comp_name in self.model_args.target_components]
            if len(self.model_args.target_components) > 1
            else [path]
        )
        for check_path in paths_to_check:
            if os.path.exists(os.path.join(check_path, LORA_ADAPTER_CONFIG_NAME)):
                if self.accelerator.is_main_process:
                    logger.info(f"Auto-detected LoRA checkpoint at {check_path}")
                return "lora"

        if self.accelerator.is_main_process:
            logger.info(f"Auto-detected full model checkpoint at {path}")
        return "full"

    def load_checkpoint(
        self,
        path: str,
        strict: bool = True,
        resume_type: Optional[Literal["lora", "full", "state"]] = None,
    ) -> None:
        """
        Load checkpoint for target components.

        Args:
            path: Checkpoint directory path.
            strict: Whether to strictly enforce state_dict key matching (only for full model).
            resume_type: Type of checkpoint to load.
                - 'lora': Load LoRA adapters only
                - 'full': Load full model weights
                - 'state': Load full training state (model + optimizer + RNG)
                - None: Auto-detect based on checkpoint directory contents
        """
        path = self._resolve_checkpoint_path(path)

        # Auto-detect if not specified
        if resume_type is None:
            resume_type = self._detect_checkpoint_type(path)

        if resume_type == "state":
            self._load_training_state(path)
        elif resume_type == "lora":
            self._load_lora(path)
        elif resume_type == "full":
            self._load_full_model(path, strict=strict)
        else:
            raise ValueError(
                f"Invalid resume_type: {resume_type}. Available: ['lora', 'full', 'state']."
            )

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            logger.info(f"Checkpoint loaded successfully from {path} (type={resume_type})")

    def _merge_lora_if_needed(self) -> None:
        """
        Merge LoRA adapters into base model weights when transitioning from
        LoRA checkpoint to full fine-tuning.

        Ensures the model is a plain nn.Module (not PeftModel) before entering
        the full training pipeline. The LoRA weights are permanently fused into
        the base model via merge_and_unload().
        """
        for comp_name in self.model_args.target_components:
            component = self.get_component(comp_name)
            unwrapped = self._unwrap(component)

            if isinstance(unwrapped, PeftModel):
                merged = unwrapped.merge_and_unload()
                self.set_component(comp_name, merged)
                if hasattr(self.pipeline, comp_name):
                    setattr(self.pipeline, comp_name, merged)

                if self.accelerator.is_main_process:
                    logger.info(f"Merged LoRA adapter into base model for {comp_name}")

    # ============================== Freezing Components ==============================
    def _freeze_text_encoders(self):
        """Freeze all text encoders."""
        for i, encoder in enumerate(self.text_encoders):
            encoder.requires_grad_(False)
            encoder.eval()

    def _freeze_vae(self):
        """Freeze video VAE and audio VAE (if present)."""
        self.vae.requires_grad_(False)
        self.vae.eval()
        if self.audio_vae is not None:
            self.audio_vae.requires_grad_(False)
            self.audio_vae.eval()

    def _freeze_transformers(self):
        """Freeze transformer components (e.g., UNet, ControlNets)."""
        for name in self.transformer_names:
            comp = self.get_component(name)
            if comp is None:
                continue
            comp.requires_grad_(False)
            comp.eval()

    def _freeze_components(self):
        """Freeze strategy using cached target_module_map."""
        # Freeze everything first
        self._freeze_text_encoders()
        self._freeze_vae()
        self._freeze_transformers()

        # Selectively unfreeze target components
        for comp_name in self.model_args.target_components:
            if not self.has_component(comp_name):
                logger.warning(f"Component {comp_name} not found, skipping freeze")
                continue

            trainable_modules = self.target_module_map.get(comp_name)

            if self.model_args.finetune_type == "lora":
                trainable_modules = None

            self._freeze_component(comp_name, trainable_modules=trainable_modules)

            # Restore train mode for components that have trainable parameters
            if trainable_modules:
                component = self._require_component(comp_name)
                component.train()

    def _freeze_component(
        self, component_name: str, trainable_modules: Optional[Union[str, List[str]]] = None
    ):
        """Freeze a specific component with optional selective unfreezing."""
        component = self._require_component(component_name)

        if trainable_modules == "all":
            logger.info(f"Unfreezing ALL {component_name} parameters")
            component.requires_grad_(True)
            return

        if isinstance(trainable_modules, str):
            if trainable_modules == "default":
                trainable_modules = self.default_target_modules
            else:
                trainable_modules = [trainable_modules]

        # Freeze all first
        component.requires_grad_(False)

        if not trainable_modules:
            logger.info(f"Froze ALL {component_name} parameters")
            return

        # Selectively unfreeze
        trainable_count = 0
        for name, param in component.named_parameters():
            if any(target in name for target in trainable_modules):
                param.requires_grad = True
                trainable_count += 1

        if trainable_count == 0:
            logger.warning(f"No parameters in {component_name} matched: {trainable_modules}")
        else:
            logger.info(f"Unfroze {trainable_count} parameters in {component_name}")

    # ============================== Trainable Parameters ==============================
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get trainable parameters from all target components."""
        params = []
        for comp_name in self.model_args.target_components:
            if self.has_component(comp_name):
                component = self._require_component(comp_name)
                params.extend(filter(lambda p: p.requires_grad, component.parameters()))
        return params

    def log_trainable_parameters(self):
        """Log trainable parameter statistics for all target components."""
        for comp_name in self.model_args.target_components:
            if not self.has_component(comp_name):
                continue

            component = self.get_component(comp_name)
            if component is None:
                continue
            total_params = 0
            trainable_params = 0
            total_size_bytes = 0
            trainable_size_bytes = 0

            for param in component.parameters():
                param_count = param.numel()
                param_size = param.element_size() * param_count

                total_params += param_count
                total_size_bytes += param_size

                if param.requires_grad:
                    trainable_params += param_count
                    trainable_size_bytes += param_size

            total_size_gb = total_size_bytes / (1024**3)
            trainable_size_gb = trainable_size_bytes / (1024**3)
            trainable_percentage = 100 * trainable_params / total_params if total_params > 0 else 0

            logger.info("=" * 70)
            logger.info(f"{comp_name.capitalize()} Trainable Parameters:")
            logger.info(f"  Total:      {total_params:>15,d} ({total_size_gb:>6.2f} GB)")
            logger.info(f"  Trainable:  {trainable_params:>15,d} ({trainable_size_gb:>6.2f} GB)")
            logger.info(f"  Percentage: {trainable_percentage:>14.2f}%")
            logger.info("=" * 70)

    # ============================== Device Management ==============================

    def _should_manage_device(self, name: str) -> bool:
        """
        Check if a component's device should be manually managed.
        Runtime overrides (prepared/proxied, LoRA, or checkpoint replacements)
        are never manually moved.
        """
        return not self.component_runtime.has_component_override(name)

    def _resolve_component_names(
        self, components: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        """
        Resolve component specifiers into concrete pipeline attribute names. `None` means all components.

        Handles group names ('text_encoders', 'transformers') by expanding them,
        and passes through concrete names ('text_encoder', 'vae', 'transformer_2') as-is.
        """
        return self.component_runtime.resolve_component_names(components)

    def on_load_components(
        self,
        components: Optional[Union[str, List[str]]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Load specified components to device, skipping prepared (accelerator-managed) ones.

        Args:
            components: Component name(s) or group names ('text_encoders', 'transformers').
                        None loads all components.
            device: Target device. Defaults to accelerator device.
        """
        names = self._resolve_component_names(components)
        materialized_before = set(self.component_runtime.materialized_component_names)
        self.component_runtime.materialize_components(components)
        materialized_after = set(self.component_runtime.materialized_component_names)
        for name in names:
            if name in materialized_after and name not in materialized_before:
                component = self.component_runtime.get_canonical_component(name)
                if isinstance(component, torch.nn.Module):
                    self._apply_component_precision_policy(name, component)
        self.component_runtime.load_stage_components(components, device=device or self.device)

    def off_load_components(self, components: Optional[Union[str, List[str]]] = None):
        """
        Off-load specified components to CPU, skipping prepared (accelerator-managed) ones.

        Args:
            components: Component name(s) or group names ('text_encoders', 'transformers').
                        None off-loads all components.
        """
        self.component_runtime.unload_stage_components(components)

    def on_load(self, device: Optional[Union[torch.device, str]] = None):
        """Load all components to device."""
        self.on_load_components(components=None, device=device)

    def off_load(self):
        """Off-load all components to CPU."""
        self.off_load_components(components=None)

    # Keep convenience aliases for backward compat, all delegate to unified methods
    def on_load_text_encoders(self, device: Optional[Union[torch.device, str]] = None):
        self.on_load_components("text_encoders", device)

    def off_load_text_encoders(self):
        self.off_load_components("text_encoders")

    def on_load_vae(self, device: Optional[Union[torch.device, str]] = None):
        self.on_load_components("vae", device)

    def off_load_vae(self):
        self.off_load_components("vae")

    def on_load_transformers(self, device: Optional[Union[torch.device, str]] = None):
        self.on_load_components("transformers", device)

    def off_load_transformers(self):
        self.off_load_components("transformers")

    # ============================== Preprocessing ==============================
    def preprocess_func(
        self,
        prompt: Optional[List[str]] = None,
        images: Optional[List[Union[Image.Image, List[Image.Image]]]] = None,
        videos: Optional[List[Union[List[Image.Image], List[List[Image.Image]]]]] = None,
        audios: Optional[List[Union[torch.Tensor, List[torch.Tensor]]]] = None,
        **kwargs,
    ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        """
        Preprocess input prompt, image, video, and audio into model-compatible embeddings/tensors.
        Always process a batch of inputs.
        Args:
            prompt: List of text prompts. A batch of text inputs.
            images:
                - None: no image input.
                - List[Image.Image]: list of images (a batch of single images)
                - List[List[Image.Image]]: list of list of images (a batch of a list images, each image list can be empty)
            videos:
                - None: no video input.
                - List[Video]: list of videos (a batch of single videos)
                - List[List[Video]]: list of list of videos (a batch of a list videos, each video list can be empty)
            audios:
                - None: no audio input.
                - List[torch.Tensor]: list of audio waveforms (a batch of single audios)
                - List[List[torch.Tensor]]: list of list of audio waveforms (a batch of a list audios, each audio list can be empty)
            **kwargs: Additional keyword arguments for encoder methods.

        """
        results = {}

        for input, encoder_method in [
            (prompt, self.encode_prompt),
            (images, self.encode_image),
            (videos, self.encode_video),
            (audios, self.encode_audio),
        ]:
            if input is not None:
                res = encoder_method(input, **(filter_kwargs(encoder_method, **kwargs)))

                if res is None:
                    # No preprocess needed
                    continue

                if (
                    isinstance(res, dict)
                    and res
                    and all(isinstance(v, (list, torch.Tensor, np.ndarray)) for v in res.values())
                ):
                    results.update(res)
                else:
                    raise ValueError(
                        f"Encoder method {encoder_method.__name__} should return a non-empty dict and each key maps to a list or tensor, "
                        f"but got {type(res)} with values types {[type(v) for v in res.values()]}"
                    )

        return results

    def encode_prompt(
        self,
        prompt: List[str],
        **kwargs,
    ) -> Optional[Dict[str, Union[List[Any], torch.Tensor]]]:
        """Encode a batch of text prompts into model-compatible embeddings.

        Default implementation is a no-op (returns ``None``). Subclasses
        override this when the model needs text conditioning.
        ``preprocess_func`` skips integration when the return value is
        ``None``, so adapters that don't need text encoding can simply
        inherit this default.

        Args:
            prompt: Batch of text prompts produced by
                ``dataset.py._preprocess_batch``.
            **kwargs: Adapter-specific encoding kwargs.

        Returns:
            Mapping from output key to encoded tensor/list, or ``None`` when
            the adapter does not perform prompt encoding.
        """
        pass

    def encode_image(
        self,
        images: MultiImageBatch,
        **kwargs,
    ) -> Optional[Dict[str, Union[List[Any], torch.Tensor]]]:
        """Encode a batch of (multi-)image inputs into latent representations.

        Default implementation is a no-op (returns ``None``). Subclasses
        override this when the model uses image conditioning.
        ``preprocess_func`` skips integration when the return value is
        ``None``, so adapters that don't need image encoding can simply
        inherit this default.

        Args:
            images: ``MultiImageBatch`` produced by
                ``dataset.py._preprocess_batch`` — a ``List[ImageBatch]``
                (ragged) or a uniform-shape tensor/array. Each batch slot
                is itself a list of images (``[]`` for empty samples).
            **kwargs: Adapter-specific encoding kwargs.

        Returns:
            Mapping from output key to encoded tensor/list (e.g.,
            ``condition_images``), or ``None`` when the adapter does not
            perform image encoding.
        """
        pass

    def encode_video(
        self,
        videos: MultiVideoBatch,
        **kwargs,
    ) -> Optional[Dict[str, Union[List[Any], torch.Tensor]]]:
        """Encode a batch of (multi-)video inputs into latent representations.

        Default implementation is a no-op (returns ``None``). Subclasses
        override this when the model uses video conditioning.
        ``preprocess_func`` skips integration when the return value is
        ``None``, so adapters that don't need video encoding can simply
        inherit this default.

        Args:
            videos: ``MultiVideoBatch`` produced by
                ``dataset.py._preprocess_batch`` — a ``List[VideoBatch]``
                (ragged) or a uniform-shape tensor/array. Each batch slot
                is itself a list of videos (``[]`` for empty samples).
            **kwargs: Adapter-specific encoding kwargs.

        Returns:
            Mapping from output key to encoded tensor/list (e.g.,
            ``condition_videos``), or ``None`` when the adapter does not
            perform video encoding.
        """
        pass

    def encode_audio(
        self,
        audios: MultiAudioBatch,
        **kwargs,
    ) -> Optional[Dict[str, Union[List[Any], torch.Tensor]]]:
        """Encode a batch of (multi-)audio inputs into latent representations.

        Default implementation is a no-op (returns ``None``). Subclasses
        override this when the model uses audio conditioning.
        ``preprocess_func`` skips integration when the return value is
        ``None``, so adapters that don't need audio encoding can simply
        inherit this default.

        Args:
            audios: ``MultiAudioBatch`` produced by
                ``dataset.py._preprocess_batch`` — a ``List[AudioBatch]``
                (ragged) or a uniform-shape tensor/array. Each batch slot
                is itself a list of audio waveforms (``[]`` for empty
                samples).
            **kwargs: Adapter-specific encoding kwargs.

        Returns:
            Mapping from output key to encoded tensor/list (e.g.,
            ``condition_audios``), or ``None`` when the adapter does not
            perform audio encoding.
        """
        pass

    # ======================================= Postprocessing =======================================
    @abstractmethod
    def decode_latents(
        self,
        latents: torch.Tensor,
        **kwargs,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Decodes latent representations back into images/videos if applicable.
        """
        pass

    # ======================================= Latent Geometry =======================================
    def resolve_latent_axes(self, latents: torch.Tensor) -> LatentAxes:
        """Resolve the :class:`LatentAxes` (axis roles) for ``latents``.

        Returns the adapter's static ``LATENT_AXES`` override when set, otherwise
        infers from the latent ndim (3=packed, 4=conv, 5=video). Resolution-invariant.

        Args:
            latents: A batched latent tensor.

        Returns:
            The :class:`LatentAxes` describing ``latents``.
        """
        if self.LATENT_AXES is not None:
            return self.LATENT_AXES
        return infer_latent_axes(latents.ndim)

    def resolve_component_latent_axes(self, component: str, latents: torch.Tensor) -> LatentAxes:
        """Resolve latent axes for a trajectory component.

        Args:
            component: Component identifier.
            latents: Batched component latent tensor.

        Returns:
            Latent axes resolved by the legacy single-latent adapter API.
        """
        return bridge.resolve_component_latent_axes(self, component, latents)

    def get_terminal_state(self, batch: StackedSampleBatch) -> LatentState:
        """Read each trajectory component's terminal stored state.

        Args:
            batch: Collated sample batch containing structured or legacy trajectory data.

        Returns:
            Terminal latent state keyed by component.
        """
        return bridge.get_terminal_state(self, batch)

    def get_replay_step(self, batch: StackedSampleBatch, step_index: int) -> ReplayStep:
        """Read one replay transition without coupling component schedules.

        Args:
            batch: Collated sample batch containing structured or legacy trajectory data.
            step_index: Global rollout transition index.

        Returns:
            Current/next states, component times, and optional stored log probability.
        """
        return bridge.get_replay_step(self, batch, step_index)

    def get_replay_callback(
        self, batch: StackedSampleBatch, step_index: int, field: str
    ) -> LatentState:
        """Read one stored rollout callback trajectory for every component.

        Args:
            batch: Collated sample batch containing structured or legacy callback data.
            step_index: Global rollout transition index.
            field: Callback field name, e.g. ``"velocity"`` or ``"next_latents_mean"``.

        Returns:
            Stored callback state keyed by component.
        """
        return bridge.get_replay_callback(self, batch, step_index, field)

    def get_state_active_numel(self, state: LatentState) -> Mapping[str, int]:
        """Count each component's active stochastic degrees of freedom.

        The default counts every non-batch element. Adapters whose components carry
        masked or conditioning positions may override this.

        Args:
            state: Batched latent state in ``trajectory_component_order``.

        Returns:
            Positive per-component element counts in component order.
        """
        return bridge.get_state_active_numel(self, state)

    def get_train_step_indices(self) -> torch.Tensor:
        """Return the primary scheduler's positional training step indices.

        Every scheduler-group member must declare the same positional indices;
        their numeric timestep values may differ.

        Returns:
            One-dimensional tensor of rollout positions to train on.
        """
        return bridge.get_train_step_indices(self)

    def get_state_active_numel_per_sample(
        self,
        state: LatentState,
    ) -> Mapping[str, torch.Tensor]:
        """Count each sample's active stochastic degrees of freedom by component.

        This validated wrapper supports variable geometry across a batch. Adapters
        with custom packing or reduction semantics override
        :meth:`_get_state_active_numel_per_sample`, not this method.

        Args:
            state: Batched latent state in ``trajectory_component_order``.

        Returns:
            One positive signed-integer ``(B,)`` tensor per component, on that
            component's device and in component order.
        """
        bridge.validate_state_active_numel_per_sample_input(self, state)
        active_numel = self._get_state_active_numel_per_sample(state)
        return bridge.validate_state_active_numel_per_sample(self, state, active_numel)

    def _get_state_active_numel_per_sample(
        self,
        state: LatentState,
    ) -> Mapping[str, torch.Tensor]:
        """Return adapter-owned per-sample active counts before validation.

        A custom packing adapter must override this hook together with its latent
        reduction hook so both describe exactly the same active elements.
        """
        return bridge.get_state_active_numel_per_sample(self, state)

    def replay_generator_boundary(
        self,
        batch: StackedSampleBatch,
        boundary_index: int,
        *,
        return_fields: Tuple[str, ...] = ("velocity", "next_latents"),
        rtol: float,
        atol: float,
        **forward_kwargs: Any,
    ) -> MultiModalStepOutput:
        """Recompute and validate the transition preceding a generated boundary.

        Args:
            batch: Collated batch owning trajectory states and conditioning.
            boundary_index: Generated state boundary to recompute; at least one.
            return_fields: Scheduler output fields requested from ``forward_state``.
            rtol: Explicit relative tolerance for stored-transition validation.
            atol: Explicit absolute tolerance for stored-transition validation.
            **forward_kwargs: Explicit conditioning not already owned by ``batch``.

        Returns:
            Recomputed one-step output after validating its next state.
        """
        return bridge.replay_generator_boundary(
            self,
            batch,
            boundary_index,
            return_fields=return_fields,
            rtol=rtol,
            atol=atol,
            forward_kwargs=forward_kwargs,
        )

    def build_training_component_times(
        self,
        primary_timesteps: torch.Tensor,
        *,
        batch: Optional[StackedSampleBatch] = None,
    ) -> ComponentTimes:
        """Map one primary scheduler coordinate onto every component's times.

        Trainers sample a single shared coordinate per sample; this hook turns it
        into component timesteps and sigmas without consuming randomness, so
        heterogeneous components can run on their own schedules. The default maps
        the legacy single ``"latent"`` component with a zero next coordinate.

        Args:
            primary_timesteps: Primary scheduler coordinates of shape ``(B,)``.
            batch: Optional collated batch supplying per-component geometry.

        Returns:
            Component times whose sigma follows the flow-matching schedule.
        """
        return bridge.build_training_component_times(self, primary_timesteps, batch=batch)

    def add_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> NoisedState:
        """Draw forward-process noise, then apply it to the clean state.

        This is the RNG-owning hook: it draws once per component in
        ``trajectory_component_order`` and delegates the noise application to
        :meth:`apply_forward_process_noise`. The default draws the legacy single
        ``"latent"`` tensor with diffusers ``randn_tensor``; heterogeneous adapters
        override only the ordered draw.

        Args:
            clean_state: Clean latent state containing exactly ``"latent"``.
            times: Component times including the current ``"latent"`` sigma.
            generator: Optional generator used for the single random draw.

        Returns:
            Noised state, target velocity, and sampled noise.
        """
        return bridge.add_forward_process_noise(
            self,
            clean_state,
            times,
            generator=generator,
        )

    def apply_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        noise: LatentState,
    ) -> NoisedState:
        """Apply already-drawn noise to every clean component.

        Consumes no randomness, so the same noise can be reused across preference
        arms, precomputed passes, and reference passes. Each component is
        interpolated with its own sigma as ``(1 - sigma) * x0 + sigma * noise``.

        Args:
            clean_state: Clean latent state in ``trajectory_component_order``.
            times: Component times including each component's current sigma.
            noise: Noise state matching the clean order, shapes, dtypes, devices.

        Returns:
            Noised state, target velocity, and the supplied noise state.
        """
        return bridge.apply_forward_process_noise(self, clean_state, times, noise)

    def project_velocity_to_clean_state(
        self,
        state: LatentState,
        times: ComponentTimes,
        velocity: LatentState,
    ) -> LatentState:
        """Project a noised state to ``x0`` under this adapter's flow convention.

        ``flow_velocity_direction="noise"`` is the standard
        ``velocity = noise - clean`` convention and uses ``x0 = xt - sigma * velocity``.
        ``"data"`` is the MiniMax H3 convention ``velocity = clean - noise`` and uses
        ``x0 = xt + sigma * velocity``.
        """
        return bridge.project_velocity_to_clean_state(self, state, times, velocity)

    def project_velocity_to_score_state(
        self,
        state: LatentState,
        times: ComponentTimes,
        velocity: LatentState,
    ) -> LatentState:
        """Project adapter-directed velocity through clean state to diffusion score.

        Distillation objectives are written against the score function, while every
        adapter predicts a velocity under its own direction convention. This
        non-overridable wrapper first applies that convention, then delegates the
        schedule-specific clean-to-score conversion to
        :meth:`_project_clean_to_score_state`.

        Args:
            state: Current noised state in ``trajectory_component_order``.
            times: Component times including each current sigma.
            velocity: Adapter-directed velocity prediction matching ``state``.

        Returns:
            Score state in component order with the input active masks preserved.
        """
        bridge.validate_score_projection_inputs(self, state, times, velocity)
        clean_state = self.project_velocity_to_clean_state(state, times, velocity)
        bridge.validate_score_projection_state(self, clean_state, field="clean_state")
        score_state = self._project_clean_to_score_state(state, times, clean_state)
        return bridge.validate_projected_score_state(self, state, score_state)

    def _project_clean_to_score_state(
        self,
        state: LatentState,
        times: ComponentTimes,
        clean_state: LatentState,
    ) -> LatentState:
        """Convert clean prediction to score under the single-latent flow schedule."""
        return bridge.project_clean_to_score_state(self, state, times, clean_state)

    def _project_flow_match_clean_to_score_state(
        self,
        state: LatentState,
        times: ComponentTimes,
        clean_state: LatentState,
    ) -> LatentState:
        """Convert clean prediction with each declared flow-match component schedule."""
        return bridge.project_flow_match_clean_to_score_state(self, state, times, clean_state)

    def forward_state(
        self,
        *,
        batch: StackedSampleBatch,
        state: LatentState,
        times: ComponentTimes,
        next_state: Optional[LatentState] = None,
        compute_log_prob: bool = False,
        return_fields: Tuple[str, ...] = (),
        noise_level: Optional[float] = None,
        **kwargs: Any,
    ) -> MultiModalStepOutput:
        """Run one state through the model, owning the forward-argument boundary.

        This wrapper resolves the arguments the model may see and then dispatches
        to :meth:`_forward_state`. Override the hook, not this method, so a
        component-specific implementation cannot bypass the shared boundary:

        - explicit kwargs naming a state-owned argument (``t``, ``latents``, ...)
          or a bridge-owned one (trajectory storage, trainer metadata) raise;
        - batch-level state-owned keys are dropped in favor of the bridge-owned
          state, times, return fields and noise level;
        - trajectory storage and trainer metadata (e.g. ``advantage``) are read
          from the batch by the bridge and never forwarded to the model.

        Args:
            batch: Collated sample batch supplying conditioning arguments.
            state: Current state in ``trajectory_component_order``.
            times: Current and next times in ``trajectory_component_order``.
            next_state: Optional stored next state in ``trajectory_component_order``.
            compute_log_prob: Whether to compute transition log probability.
            return_fields: Existing scheduler output fields requested from ``forward``.
            noise_level: Existing scheduler noise-level override.
            **kwargs: Explicit adapter/training arguments that do not own state.

        Returns:
            Multi-modal wrapper around the unchanged legacy scheduler output.
        """
        return self._forward_state(
            batch=batch,
            state=state,
            times=times,
            next_state=next_state,
            compute_log_prob=compute_log_prob,
            return_fields=return_fields,
            noise_level=noise_level,
            forward_kwargs=bridge.build_forward_state_kwargs(self, batch, kwargs),
        )

    def _forward_state(
        self,
        *,
        batch: StackedSampleBatch,
        state: LatentState,
        times: ComponentTimes,
        next_state: Optional[LatentState],
        compute_log_prob: bool,
        return_fields: Tuple[str, ...],
        noise_level: Optional[float],
        forward_kwargs: Mapping[str, Any],
    ) -> MultiModalStepOutput:
        """Pack one state for this adapter's ``forward`` and unpack the output.

        Default single-component behavior: require exactly ``"latent"``, map the
        state and times onto the legacy ``forward`` arguments, and wrap the
        scheduler output. Heterogeneous adapters override this hook to pack and
        unpack their own components; ``forward_kwargs`` is already stripped of
        state-owned, trajectory-storage and trainer-metadata keys, so an override
        may forward it as-is (after the usual signature filtering).

        Args:
            batch: Collated sample batch, for conditioning an override needs to
                derive per component; storage fields must not be forwarded.
            state: Current state in ``trajectory_component_order``.
            times: Current and next times in ``trajectory_component_order``.
            next_state: Optional stored next state in ``trajectory_component_order``.
            compute_log_prob: Whether to compute transition log probability.
            return_fields: Existing scheduler output fields requested from ``forward``.
            noise_level: Existing scheduler noise-level override.
            forward_kwargs: Model-conditioning arguments resolved by the wrapper.

        Returns:
            Multi-modal wrapper around this adapter's scheduler output.
        """
        return bridge.default_forward_state(
            self,
            batch=batch,
            state=state,
            times=times,
            next_state=next_state,
            compute_log_prob=compute_log_prob,
            return_fields=return_fields,
            noise_level=noise_level,
            forward_kwargs=forward_kwargs,
        )

    def reduce_component_latent_values(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        state: Optional[LatentState] = None,
    ) -> Mapping[str, torch.Tensor]:
        """Reduce each component separately to one scalar per sample.

        Validated wrapper around :meth:`_reduce_component_latent_values`. Override
        the hook, not this method, so an override cannot bypass input or output
        validation.

        Args:
            values: Raw per-element component tensors in ``trajectory_component_order``.
            state: Optional replay state supplying component masks to an override.

        Returns:
            One ``(B,)`` tensor per component, in ``trajectory_component_order``.
        """
        batch_size = bridge.validate_reduction_inputs(self, values, state)
        reduced = self._reduce_component_latent_values(values, state=state)
        return bridge.validate_reduced_component_values(self, reduced, batch_size)

    def _reduce_component_latent_values(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        state: Optional[LatentState] = None,
    ) -> Mapping[str, torch.Tensor]:
        """Per-component reduction hook; averages every non-batch element.

        Adapters whose components carry masked or conditioning positions override
        this to average only the active positions, which a global element sum
        cannot recover. ``state`` carries per-sample context such as a dynamic mask.

        Args:
            values: Raw per-element component tensors in ``trajectory_component_order``.
            state: Optional replay state supplying component masks.

        Returns:
            One ``(B,)`` tensor per component, in ``trajectory_component_order``.
        """
        return bridge.default_reduce_component_latent_values(self, values, state=state)

    def reduce_latent_values(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        active_numel: Optional[Mapping[str, int]] = None,
        state: Optional[LatentState] = None,
    ) -> torch.Tensor:
        """Reduce component values to one globally element-weighted scalar per sample.

        Validated wrapper around :meth:`_reduce_latent_values`. Override the hook,
        not this method, so an override cannot bypass input or output validation.

        Args:
            values: Component tensors in ``trajectory_component_order``.
            active_numel: Optional partial mapping of positive weights for
                already-reduced ``(B,)`` component scalars.
            state: Optional replay state supplying component masks to an override.

        Returns:
            One globally element-weighted scalar per batch item.
        """
        batch_size = bridge.validate_reduction_inputs(self, values, state)
        reduced = self._reduce_latent_values(values, active_numel=active_numel, state=state)
        return bridge.validate_reduced_latent_values(self, reduced, batch_size)

    def _reduce_latent_values(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        active_numel: Optional[Mapping[str, int]] = None,
        state: Optional[LatentState] = None,
    ) -> torch.Tensor:
        """Global reduction hook; weights every element of every component once.

        Prefer passing raw per-element tensors whenever only a global result is
        needed. ``active_numel`` is for values already reduced per component, which
        a global sum cannot recover. A single-component ``(B,)`` value is returned
        unchanged to preserve the legacy scalar scale and tensor identity. The
        default ignores ``state``; dynamic mask adapters override this to weight
        only the elements the state marks active.

        Args:
            values: Component tensors in ``trajectory_component_order``.
            active_numel: Optional partial mapping of positive weights for
                already-reduced ``(B,)`` component scalars.
            state: Optional replay state supplying component masks.

        Returns:
            One globally element-weighted scalar per batch item.
        """
        return bridge.default_reduce_latent_values(
            self,
            values,
            active_numel=active_numel,
            state=state,
        )

    # ======================================= Sampling & Training =======================================
    @abstractmethod
    def forward(
        self,
        *args,
        **kwargs,
    ) -> SDESchedulerOutput:
        """
        Calculates the log-probability of the action (image/latent) given inputs.
        """
        pass

    @abstractmethod
    def inference(
        self,
        *args,
        **kwargs,
    ) -> List[BaseSample]:
        """
        Execute the generation process (Integration/Sampling).
        Returns a list of BaseSample instances.
        """
        pass
