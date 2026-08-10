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

from typing import Any, ClassVar, Dict, List, Tuple, Union

from ...scheduler import MiniMaxH3SDEScheduler, SchedulerGroup
from ..abc import BaseAdapter
from ..runtime import ModularPipelineRuntime
from .workflow import (
    build_h3_component_runtime,
    build_h3_scheduler,
    build_h3_scheduler_group,
    freeze_h3_setup_components,
    init_h3_target_module_map,
    load_h3_workflow_pipeline,
)


class MiniMaxH3T2VAAdapter(BaseAdapter):
    """Load the workflow-pruned MiniMax H3 text-to-video-audio partition."""

    workflow: ClassVar[str] = "t2va"
    transformer_component_name: ClassVar[str] = "transformer"
    trajectory_component_order: ClassVar[Tuple[str, ...]] = ("video", "audio")
    preprocessing_modules: ClassVar[List[str]] = ["text_encoder", "tokenizer", "processor"]
    inference_modules: ClassVar[List[str]] = [
        "transformer",
        "vae",
        "video_processor",
        "audio_vae",
    ]

    def load_pipeline(self) -> Any:
        """Load only the T2VA modular workflow."""
        return load_h3_workflow_pipeline(
            self.model_args.model_name_or_path,
            workflow=self.workflow,
            transformer_component_name=self.transformer_component_name,
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

    def _freeze_components(self) -> None:
        freeze_h3_setup_components(self)

    def decode_latents(self, latents: Any, **kwargs: Any) -> Any:
        """Defer target decoding to Task 5C Commit 3."""
        raise NotImplementedError("MiniMax H3 target decoding is implemented in Task 5C Commit 3")

    def inference(self, **kwargs: Any) -> List[Any]:
        """Defer workflow rollout to Task 5C Commit 3."""
        raise NotImplementedError("MiniMax H3 inference is implemented in Task 5C Commit 3")

    def forward(self, **kwargs: Any) -> Any:
        """Defer training forward to Task 5C Commit 3."""
        raise NotImplementedError("MiniMax H3 forward is implemented in Task 5C Commit 3")


class MiniMaxH3FL2VAAdapter(BaseAdapter):
    """Load the workflow-pruned MiniMax H3 first/last-frame partition."""

    workflow: ClassVar[str] = "fl2va"
    transformer_component_name: ClassVar[str] = "transformer"
    trajectory_component_order: ClassVar[Tuple[str, ...]] = ("video", "audio")
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

    def load_pipeline(self) -> Any:
        """Load only the FL2VA modular workflow."""
        return load_h3_workflow_pipeline(
            self.model_args.model_name_or_path,
            workflow=self.workflow,
            transformer_component_name=self.transformer_component_name,
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

    def _freeze_components(self) -> None:
        freeze_h3_setup_components(self)

    def decode_latents(self, latents: Any, **kwargs: Any) -> Any:
        """Defer target decoding to Task 5C Commit 3."""
        raise NotImplementedError("MiniMax H3 target decoding is implemented in Task 5C Commit 3")

    def inference(self, **kwargs: Any) -> List[Any]:
        """Defer workflow rollout to Task 5C Commit 3."""
        raise NotImplementedError("MiniMax H3 inference is implemented in Task 5C Commit 3")

    def forward(self, **kwargs: Any) -> Any:
        """Defer training forward to Task 5C Commit 3."""
        raise NotImplementedError("MiniMax H3 forward is implemented in Task 5C Commit 3")


class MiniMaxH3Ref2VAAdapter(BaseAdapter):
    """Load the workflow-pruned MiniMax H3 omni-reference partition."""

    workflow: ClassVar[str] = "ref2va"
    transformer_component_name: ClassVar[str] = "transformer_ref"
    trajectory_component_order: ClassVar[Tuple[str, ...]] = ("video", "audio")
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

    def load_pipeline(self) -> Any:
        """Load only the Ref2VA modular workflow."""
        return load_h3_workflow_pipeline(
            self.model_args.model_name_or_path,
            workflow=self.workflow,
            transformer_component_name=self.transformer_component_name,
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

    def _freeze_components(self) -> None:
        freeze_h3_setup_components(self)

    def decode_latents(self, latents: Any, **kwargs: Any) -> Any:
        """Defer target decoding to Task 5C Commit 3."""
        raise NotImplementedError("MiniMax H3 target decoding is implemented in Task 5C Commit 3")

    def inference(self, **kwargs: Any) -> List[Any]:
        """Defer workflow rollout to Task 5C Commit 3."""
        raise NotImplementedError("MiniMax H3 inference is implemented in Task 5C Commit 3")

    def forward(self, **kwargs: Any) -> Any:
        """Defer training forward to Task 5C Commit 3."""
        raise NotImplementedError("MiniMax H3 forward is implemented in Task 5C Commit 3")
