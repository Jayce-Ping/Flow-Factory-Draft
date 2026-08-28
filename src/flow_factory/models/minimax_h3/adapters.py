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

"""Define the three public MiniMax H3 workflow adapters."""

from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple, Union

import torch

from ...samples import (
    ComponentTimes,
    LatentState,
    MultiModalStepOutput,
    NoisedState,
    StackedSampleBatch,
)
from ...scheduler import MiniMaxH3SDEScheduler, SchedulerGroup
from ..abc import BaseAdapter
from ..runtime import ModularPipelineRuntime
from ._common import apply_forward_process_noise, draw_forward_process_noise
from .workflow import (
    build_h3_component_runtime,
    build_h3_replay_forward_kwargs,
    build_h3_scheduler,
    build_h3_scheduler_group,
    decode_h3_adapter_latents,
    forward_h3_adapter,
    infer_h3_workflow,
    init_h3_target_module_map,
    load_h3_workflow_pipeline,
    map_h3_training_component_times,
    preprocess_h3_workflow,
)

_H3_PREPROCESS_CACHE_FIELDS = frozenset({"height", "width", "num_frames"})
_H3_PREPROCESS_CACHE_VERSION = "minimax-h3-v1"


class _MiniMaxH3WorkflowAdapter:
    """Share the workflow-invariant behavior of the MiniMax H3 adapters.

    Concrete adapters remain direct ``BaseAdapter`` subclasses and differ only in
    workflow identity, canonical transformer name, and component lists.
    """

    trajectory_component_order: ClassVar[Tuple[str, ...]] = ("video", "audio")
    flow_velocity_direction: ClassVar[Literal["data"]] = "data"
    preprocess_cache_fields: ClassVar[frozenset[str]] = _H3_PREPROCESS_CACHE_FIELDS
    preprocess_cache_version: ClassVar[str] = _H3_PREPROCESS_CACHE_VERSION
    supports_fsdp2_cpu_efficient_loading: ClassVar[bool] = True
    # Official Diffusers recipe: load at BF16 and let each ModelMixin preserve
    # its declared FP32 islands (including both H3 autoencoders).
    component_load_dtype_defaults: ClassVar[torch.dtype] = torch.bfloat16

    def load_pipeline(self) -> Any:
        """Load this modular workflow from a local directory or Hugging Face repo."""
        return load_h3_workflow_pipeline(
            self.model_args.model_name_or_path,
            workflow=self.workflow,
        )

    def build_component_runtime(self) -> ModularPipelineRuntime:
        """Build the workflow-pruned modular runtime."""
        return build_h3_component_runtime(self)

    def load_scheduler(self) -> MiniMaxH3SDEScheduler:
        """Build the canonical shift-12 video scheduler."""
        return build_h3_scheduler(self.config.scheduler_args, shift=12.0)

    def build_scheduler_group(self) -> SchedulerGroup:
        """Build ordered fresh video/audio schedulers."""
        return build_h3_scheduler_group(self)

    def _init_target_module_map(self) -> Dict[str, Union[List[str], None]]:
        return init_h3_target_module_map(self)

    def preprocess_func(self, **kwargs: Any) -> Dict[str, Any]:
        return preprocess_h3_workflow(self, **kwargs)

    def build_training_component_times(
        self,
        primary_timesteps: torch.Tensor,
        *,
        batch: Optional[StackedSampleBatch] = None,
    ) -> ComponentTimes:
        return map_h3_training_component_times(self, primary_timesteps)

    def add_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> NoisedState:
        return draw_forward_process_noise(clean_state, times, generator=generator)

    def apply_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        noise: LatentState,
    ) -> NoisedState:
        return apply_forward_process_noise(clean_state, times, noise)

    def decode_latents(self, latents: Any, **kwargs: Any) -> Any:
        return decode_h3_adapter_latents(self, latents, **kwargs)

    def empty_decoded_media(self, batch_size: int) -> Any:
        """Preserve H3's video/audio/sample-rate decode structure without decoding."""
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError(
                f"MiniMax H3 expected positive int batch_size, received {batch_size!r}"
            )
        return ([None] * batch_size, [None] * batch_size, None)

    def inference(self, **kwargs: Any) -> List[Any]:
        return infer_h3_workflow(self, **kwargs)

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
        """Replay one stored transition through the public rollout entry point."""
        return self.forward(
            state=state,
            times=times,
            next_state=next_state,
            compute_log_prob=compute_log_prob,
            return_fields=return_fields,
            noise_level=noise_level,
            **build_h3_replay_forward_kwargs(forward_kwargs),
        )

    def forward(self, **kwargs: Any) -> MultiModalStepOutput:
        return forward_h3_adapter(self, **kwargs)


class MiniMaxH3T2VAAdapter(_MiniMaxH3WorkflowAdapter, BaseAdapter):
    """Load the workflow-pruned MiniMax H3 text-to-video-audio partition."""

    workflow: ClassVar[str] = "t2va"
    transformer_component_name: ClassVar[str] = "transformer"
    preprocessing_modules: ClassVar[List[str]] = ["text_encoder", "tokenizer", "processor"]
    inference_modules: ClassVar[List[str]] = [
        "transformer",
        "vae",
        "video_processor",
        "audio_vae",
    ]


class MiniMaxH3FL2VAAdapter(_MiniMaxH3WorkflowAdapter, BaseAdapter):
    """Load the workflow-pruned MiniMax H3 first/last-frame partition."""

    workflow: ClassVar[str] = "fl2va"
    transformer_component_name: ClassVar[str] = "transformer"
    preprocessing_modules: ClassVar[List[str]] = [
        "image_processor",
        "text_encoder",
        "tokenizer",
        "processor",
        "vae",
    ]
    inference_modules: ClassVar[List[str]] = [
        "transformer",
        "vae",
        "video_processor",
        "audio_vae",
    ]


class MiniMaxH3Ref2VAAdapter(_MiniMaxH3WorkflowAdapter, BaseAdapter):
    """Load the workflow-pruned MiniMax H3 omni-reference partition."""

    workflow: ClassVar[str] = "ref2va"
    transformer_component_name: ClassVar[str] = "transformer_ref"
    supports_ordered_references: ClassVar[bool] = True
    preprocessing_modules: ClassVar[List[str]] = [
        "image_processor",
        "text_encoder",
        "tokenizer",
        "processor",
        "vae",
        "audio_vae",
    ]
    inference_modules: ClassVar[List[str]] = [
        "transformer_ref",
        "vae",
        "video_processor",
        "audio_vae",
    ]
