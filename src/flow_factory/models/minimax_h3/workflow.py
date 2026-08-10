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

"""Own MiniMax H3 workflow loading and setup contracts."""

from typing import Any, Dict, List, Union

from ...scheduler import MiniMaxH3SDEScheduler, SchedulerGroup
from ..runtime import ModularPipelineRuntime
from .dependency import require_minimax_h3_support

_COMMON_REQUIRED_COMPONENTS = (
    "scheduler",
    "text_encoder",
    "tokenizer",
    "processor",
    "vae",
    "audio_vae",
)


def load_h3_workflow_pipeline(
    model_name_or_path: str,
    *,
    workflow: str,
    transformer_component_name: str,
) -> Any:
    """Load and validate one workflow-pruned MiniMax H3 pipeline."""
    symbols = require_minimax_h3_support()
    pipeline = symbols.MiniMaxH3ModularPipeline.from_pretrained(
        model_name_or_path,
        workflow=workflow,
    )
    declared_specs = ModularPipelineRuntime(pipeline).canonical_components

    opposite_component_name = (
        "transformer_ref" if transformer_component_name == "transformer" else "transformer"
    )
    required_names = (*_COMMON_REQUIRED_COMPONENTS, transformer_component_name)
    missing_names = [name for name in required_names if declared_specs.get(name) is None]
    opposite_names = [opposite_component_name] if opposite_component_name in declared_specs else []
    if missing_names or opposite_names:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} expected required components "
            f"{required_names!r} and opposite transformer partition {opposite_component_name!r} "
            f"to be absent, received missing={missing_names!r}, "
            f"opposite_present={opposite_names!r}, declared={tuple(declared_specs)!r}"
        )
    return pipeline


def build_h3_component_runtime(adapter: Any) -> ModularPipelineRuntime:
    """Wrap one pruned pipeline and materialize only its training transformer."""
    validate_h3_target_components(adapter)
    runtime = ModularPipelineRuntime(adapter.load_pipeline())
    runtime.materialize_components([adapter.transformer_component_name])
    return runtime


def build_h3_scheduler(scheduler_args: Any, *, shift: float) -> MiniMaxH3SDEScheduler:
    """Build one independent H3 scheduler from Flow-Factory scheduler arguments."""
    return MiniMaxH3SDEScheduler(
        shift=shift,
        noise_level=scheduler_args.noise_level,
        sde_steps=scheduler_args.sde_steps,
        num_sde_steps=scheduler_args.num_sde_steps,
        seed=scheduler_args.seed,
        dynamics_type=scheduler_args.dynamics_type,
    )


def build_h3_scheduler_group(adapter: Any) -> SchedulerGroup:
    """Build fresh shift-12/shift-3 schedulers in video/audio order."""
    adapter.audio_scheduler = build_h3_scheduler(adapter.config.scheduler_args, shift=3.0)
    return SchedulerGroup(
        {"video": adapter.scheduler, "audio": adapter.audio_scheduler},
        primary_name="video",
    )


def init_h3_target_module_map(
    adapter: Any,
) -> Dict[str, Union[List[str], None]]:
    """Validate the sole workflow transformer before checkpoint and LoRA setup."""
    validate_h3_target_components(adapter)
    return adapter._parse_target_modules(
        target_modules=adapter.model_args.target_modules,
        components=adapter.model_args.target_components,
    )


def validate_h3_target_components(adapter: Any) -> None:
    """Reject an invalid training partition before loading model weights."""
    expected_targets = [adapter.transformer_component_name]
    received_targets = adapter.model_args.target_components
    if received_targets != expected_targets:
        raise ValueError(
            f"MiniMax H3 workflow={adapter.workflow!r} expected target_components "
            f"{expected_targets!r}, received {received_targets!r}"
        )


def freeze_h3_setup_components(adapter: Any) -> None:
    """Freeze or unfreeze only the explicitly materialized training transformer."""
    target_name = adapter.transformer_component_name
    trainable_modules = adapter.target_module_map[target_name]
    if adapter.model_args.finetune_type == "lora":
        trainable_modules = None
    adapter._freeze_component(target_name, trainable_modules=trainable_modules)
    if trainable_modules:
        adapter.get_component(target_name).train()
