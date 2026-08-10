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

"""Own MiniMax H3 workflow loading, setup, and execution contracts."""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch

from ...samples import (
    ComponentTimes,
    LatentState,
    MiniMaxH3FL2VASample,
    MiniMaxH3Ref2VASample,
    MiniMaxH3T2VASample,
    StructuredTrajectory,
)
from ...scheduler import MiniMaxH3SDEScheduler, SchedulerGroup
from ..runtime import ModularPipelineRuntime
from ._common import (
    build_structured_trajectories,
)
from ._common import build_training_component_times as build_h3_component_times
from .blocks import encode_h3_workflow_inputs, prepare_h3_rollout_state
from .decoding import decode_h3_targets
from .denoise import forward_h3_state
from .dependency import require_minimax_h3_support
from .layout import build_h3_schedule_plan

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


def map_h3_training_component_times(
    adapter: Any, primary_timesteps: torch.Tensor
) -> ComponentTimes:
    """Map primary video coordinates onto the independent audio shift."""
    return build_h3_component_times(
        primary_timesteps,
        video_shift=adapter.scheduler.shift,
        audio_shift=adapter.audio_scheduler.shift,
    )


def preprocess_h3_workflow(adapter: Any, **kwargs: Any) -> Dict[str, Any]:
    """Encode one Arrow-safe B=1 workflow input with pinned blocks."""
    prompt = _single_outer_value(kwargs.get("prompt"), "prompt", adapter.workflow)
    values: Dict[str, Any] = {
        "prompt": prompt,
        "height": kwargs["height"],
        "width": kwargs["width"],
        "num_frames": kwargs["num_frames"],
    }
    if adapter.workflow == "fl2va":
        images = _single_outer_value(
            kwargs.get("images", kwargs.get("condition_images")),
            "images",
            adapter.workflow,
        )
        if not isinstance(images, (list, tuple)) or not 1 <= len(images) <= 2:
            raise ValueError(
                "MiniMax H3 workflow='fl2va' field='images' expected one or two ordered "
                f"images, received {images!r}"
            )
        values["image"] = images[0]
        if len(images) == 2:
            values["last_image"] = images[1]
    elif adapter.workflow == "ref2va":
        references = _single_outer_value(kwargs.get("references"), "references", adapter.workflow)
        values["references"] = _build_pinned_references(references)

    encoded = encode_h3_workflow_inputs(adapter.pipeline, values, workflow=adapter.workflow)
    result = dict(encoded)
    if adapter.workflow == "ref2va":
        manifest = kwargs.get("reference_manifest")
        _single_outer_value(manifest, "reference_manifest", adapter.workflow)
        result["reference_manifest"] = manifest
    if any(_is_upstream_reference(value) for value in result.values()):
        raise TypeError(
            f"MiniMax H3 workflow={adapter.workflow!r} preprocessing output contains "
            "an upstream reference object"
        )
    return result


def infer_h3_workflow(adapter: Any, **kwargs: Any) -> List[Any]:
    """Run one B=1 target-only rollout and return a structured sample."""
    prompt = kwargs.get("prompt")
    prompt_value = _single_outer_value(prompt, "prompt", adapter.workflow)
    if kwargs.get("negative_prompt") is not None or kwargs.get("guidance_scale", 1.0) != 1.0:
        raise ValueError(
            f"MiniMax H3 workflow={adapter.workflow!r} does not support classifier-free guidance"
        )
    prompt_embeds = kwargs["prompt_embeds"]
    if not isinstance(prompt_embeds, torch.Tensor) or prompt_embeds.shape[0] != 1:
        raise ValueError(
            f"MiniMax H3 workflow={adapter.workflow!r} field='prompt_embeds' requires B=1, "
            f"received {getattr(prompt_embeds, 'shape', None)}"
        )
    layout = _execution_mapping(
        kwargs,
        "layout",
        (
            "position_ids",
            "token_tags",
            "video_indices",
            "audio_indices",
            "text_indices",
            "num_condition_video_rows",
            "num_condition_audio_rows",
        ),
    )
    geometry = _execution_mapping(
        kwargs,
        "geometry",
        (
            "height",
            "width",
            "num_frames",
            "num_latent_frames",
            "latent_height",
            "latent_width",
            "num_audio_latents",
        ),
    )
    generator = kwargs.get("generator")
    transformer = adapter.get_component(adapter.transformer_component_name)
    state, condition_prefixes = prepare_h3_rollout_state(
        adapter.pipeline,
        kwargs,
        workflow=adapter.workflow,
        generator=generator,
    )
    plan = build_h3_schedule_plan(
        adapter.scheduler,
        adapter.audio_scheduler,
        kwargs.get("num_inference_steps", 40),
        layout,
        state.components["video"].device,
        keyframe_noise_aug=getattr(transformer, "keyframe_noise_aug", 0.999),
    )
    num_transitions = len(plan.schedules["video"][0]) - 1
    trajectory_indices = kwargs.get("trajectory_indices", "all")
    state_positions = _selected_positions(trajectory_indices, num_transitions + 1)
    transition_positions = _selected_positions(trajectory_indices, num_transitions)
    collected_states: Optional[Dict[str, List[torch.Tensor]]] = None
    collected_log_probs: Optional[List[torch.Tensor]] = None
    collected_component_log_probs: Optional[Dict[str, List[torch.Tensor]]] = None
    callback_fields = tuple(kwargs.get("extra_call_back_kwargs", ()))
    collected_callbacks: Optional[Dict[str, Dict[str, List[torch.Tensor]]]] = None
    if trajectory_indices is not None and state_positions:
        collected_states = {"video": [], "audio": []}
        if 0 in state_positions:
            _append_state(collected_states, state)
        if kwargs.get("compute_log_prob", True) and transition_positions:
            collected_log_probs = []
            collected_component_log_probs = {"video": [], "audio": []}
        if callback_fields and transition_positions:
            collected_callbacks = {field: {"video": [], "audio": []} for field in callback_fields}

    for index in range(num_transitions):
        times = _component_times_at(plan.schedules, index)
        output = adapter.forward(
            state=state,
            times=times,
            condition_prefixes=condition_prefixes,
            prompt_embeds=prompt_embeds,
            layout=layout,
            generator=generator,
            compute_log_prob=kwargs.get("compute_log_prob", True),
        )
        if output.next_state is None:
            raise ValueError(
                f"MiniMax H3 workflow={adapter.workflow!r} transition {index} "
                "expected target next_state, received None"
            )
        state = output.next_state
        if collected_states is not None and index + 1 in state_positions:
            _append_state(collected_states, state)
        if collected_log_probs is not None and index in transition_positions:
            collected_log_probs.append(output.log_prob)
            for component in ("video", "audio"):
                collected_component_log_probs[component].append(
                    output.component_log_probs[component]
                )
        if collected_callbacks is not None and index in transition_positions:
            for field in callback_fields:
                value = getattr(output, field, None)
                if not isinstance(value, LatentState):
                    raise TypeError(
                        f"MiniMax H3 workflow={adapter.workflow!r} callback field={field!r} "
                        f"expected LatentState, received {type(value).__name__}"
                    )
                for component in ("video", "audio"):
                    collected_callbacks[field][component].append(
                        value.components[component].detach()
                    )

    video, audio, sample_rate = decode_h3_targets(
        adapter.pipeline,
        state,
        geometry,
        output_type=kwargs.get("output_type", "pt"),
        workflow=adapter.workflow,
    )
    trajectory: Optional[StructuredTrajectory] = None
    if collected_states is not None:
        state_map = _index_map(num_transitions + 1, state_positions)
        log_map = (
            None
            if collected_log_probs is None
            else _index_map(num_transitions, transition_positions)
        )
        trajectory = build_structured_trajectories(
            states={
                component: torch.stack(values, dim=1)
                for component, values in collected_states.items()
            },
            state_index_map=state_map,
            schedule=plan.schedules,
            log_probs=(
                None if collected_log_probs is None else torch.stack(collected_log_probs, dim=1)
            ),
            component_log_probs=(
                None
                if collected_component_log_probs is None
                else {
                    component: torch.stack(values, dim=1)
                    for component, values in collected_component_log_probs.items()
                }
            ),
            log_prob_index_map=log_map,
            callbacks=(
                None
                if collected_callbacks is None
                else {
                    field: {
                        component: torch.stack(values, dim=1)
                        for component, values in component_values.items()
                    }
                    for field, component_values in collected_callbacks.items()
                }
            ),
            callback_index_map=(
                None
                if collected_callbacks is None
                else _index_map(num_transitions, transition_positions)
            ),
        )[0]
    sample_classes = {
        "t2va": MiniMaxH3T2VASample,
        "fl2va": MiniMaxH3FL2VASample,
        "ref2va": MiniMaxH3Ref2VASample,
    }
    sample_kwargs = {
        "prompt": prompt_value,
        "prompt_embeds": prompt_embeds[0],
        "video": _decoded_video_sample(video),
        "audio": audio[0],
        "audio_sample_rate": sample_rate,
        "height": kwargs.get("height"),
        "width": kwargs.get("width"),
        "trajectory": trajectory,
        "extra_kwargs": {
            "condition_prefixes": {
                component: values[0] for component, values in condition_prefixes.items()
            },
            "layout": layout,
            "geometry": geometry,
        },
    }
    if adapter.workflow == "ref2va":
        sample_kwargs["reference_manifest"] = _single_outer_value(
            kwargs.get("reference_manifest"), "reference_manifest", adapter.workflow
        )
    return [sample_classes[adapter.workflow](**sample_kwargs)]


def forward_h3_adapter_state(
    adapter: Any,
    *,
    state: LatentState,
    times: ComponentTimes,
    next_state: Optional[LatentState],
    compute_log_prob: bool,
    noise_level: Optional[float],
    forward_kwargs: Mapping[str, Any],
) -> Any:
    """Run the prepared workflow transformer through the common H3 path."""
    prompt_embeds = forward_kwargs["prompt_embeds"]
    condition_prefixes = forward_kwargs["condition_prefixes"]
    layout = {name: _remove_b1_collation(value) for name, value in forward_kwargs["layout"].items()}
    return forward_h3_state(
        adapter.get_component(adapter.transformer_component_name),
        state,
        condition_prefixes,
        prompt_embeds,
        times,
        layout,
        adapter.scheduler,
        adapter.audio_scheduler,
        next_state=next_state,
        generator=forward_kwargs.get("generator"),
        noise_level=noise_level,
        compute_log_prob=compute_log_prob,
        workflow=adapter.workflow,
    )


def forward_h3_adapter(adapter: Any, **kwargs: Any) -> Any:
    """Route rollout and bridge calls through one adapter forward boundary."""
    if "batch" in kwargs:
        return adapter.forward_state(**kwargs)
    state = kwargs.pop("state")
    times = kwargs.pop("times")
    return forward_h3_adapter_state(
        adapter,
        state=state,
        times=times,
        next_state=kwargs.pop("next_state", None),
        compute_log_prob=kwargs.pop("compute_log_prob", False),
        noise_level=kwargs.pop("noise_level", None),
        forward_kwargs=kwargs,
    )


def decode_h3_adapter_latents(adapter: Any, latents: Any, **kwargs: Any) -> Any:
    """Decode one target-only state without condition-prefix concatenation."""
    if not isinstance(latents, LatentState):
        raise TypeError(
            f"MiniMax H3 workflow={adapter.workflow!r} decode expected LatentState, "
            f"received {type(latents).__name__}"
        )
    return decode_h3_targets(
        adapter.pipeline,
        latents,
        kwargs["geometry"],
        output_type=kwargs.get("output_type", "pt"),
        workflow=adapter.workflow,
    )


def _single_outer_value(value: Any, field: str, workflow: str) -> Any:
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} field={field!r} requires outer B=1, "
            f"received {value!r}"
        )
    return value[0]


def _build_pinned_references(entries: Sequence[Mapping[str, Any]]) -> List[Any]:
    symbols = require_minimax_h3_support()
    references = []
    for entry in entries:
        kind = entry["kind"]
        if kind == "image":
            references.append(symbols.ImageReference(image=entry["media"]))
        elif kind == "video":
            reference_kwargs = {"frames": entry["frames"], "fps": entry["fps"]}
            if entry.get("audio") is not None:
                reference_kwargs.update(audio=entry["audio"], sample_rate=entry["sample_rate"])
            references.append(symbols.VideoReference(**reference_kwargs))
        elif kind == "audio":
            references.append(
                symbols.AudioReference(audio=entry["media"], sample_rate=entry["sample_rate"])
            )
        else:
            raise ValueError(f"expected image/video/audio reference kind, received {kind!r}")
    return references


def _is_upstream_reference(value: Any) -> bool:
    return type(value).__name__.startswith("MiniMaxH3") and type(value).__name__.endswith(
        "Reference"
    )


def _component_times_at(
    schedules: Mapping[str, Sequence[torch.Tensor]], index: int
) -> ComponentTimes:
    timesteps = {name: values[0][index : index + 1] for name, values in schedules.items()}
    next_timesteps = {name: values[0][index + 1 : index + 2] for name, values in schedules.items()}
    sigmas = {name: values[1][index : index + 1] for name, values in schedules.items()}
    next_sigmas = {name: values[1][index + 1 : index + 2] for name, values in schedules.items()}
    return ComponentTimes(timesteps, next_timesteps, sigmas, next_sigmas)


def _selected_positions(indices: Any, length: int) -> List[int]:
    if indices == "all":
        return list(range(length))
    if indices is None:
        return []
    selected = set()
    for raw_index in indices:
        index = raw_index + length if raw_index < 0 else raw_index
        if 0 <= index < length:
            selected.add(index)
    return sorted(selected)


def _index_map(length: int, positions: Sequence[int]) -> torch.Tensor:
    result = torch.full((length,), -1, dtype=torch.long)
    for stored_index, position in enumerate(positions):
        result[position] = stored_index
    return result


def _append_state(storage: Dict[str, List[torch.Tensor]], state: LatentState) -> None:
    for component in ("video", "audio"):
        storage[component].append(state.components[component].detach())


def _execution_mapping(
    values: Mapping[str, Any], nested_name: str, fields: Sequence[str]
) -> Dict[str, Any]:
    nested = values.get(nested_name)
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise TypeError(
                f"expected mapping for MiniMax H3 {nested_name}, "
                f"received {type(nested).__name__}"
            )
        return {name: _remove_b1_collation(value) for name, value in nested.items()}
    return {field: _remove_b1_collation(values[field]) for field in fields if field in values}


def _remove_b1_collation(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim > 1 and value.shape[0] == 1:
        return value[0]
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _decoded_video_sample(video: Any) -> Any:
    if isinstance(video, torch.Tensor) and video.shape[0] == 1:
        return video[0]
    if isinstance(video, list) and len(video) == 1 and isinstance(video[0], list):
        return video[0]
    return video
