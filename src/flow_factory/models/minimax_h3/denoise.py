# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the local gradient-bearing MiniMax H3 transformer and component steps."""

import inspect
from typing import Any, List, Mapping, Optional, Sequence, Union

import torch

from ...samples import ComponentTimes, LatentState, MultiModalStepOutput
from ._common import (
    MINIMAX_H3_COMPONENT_ORDER,
    build_component_step_output,
    validate_target_state,
)
from .layout import build_row_timesteps


def run_h3_joint_transformer(
    transformer: Any,
    target_state: LatentState,
    condition_prefixes: Mapping[str, torch.Tensor],
    prompt_embeds: torch.Tensor,
    times: ComponentTimes,
    layout: Mapping[str, Any],
    *,
    attention_kwargs: Optional[Mapping[str, Any]] = None,
    workflow: str = "t2va",
) -> LatentState:
    """Run one supplied transformer component over conditions and target rows.

    Args:
        transformer: Resolved prepared transformer component.
        target_state: Generated target-only video/audio rows.
        condition_prefixes: Immutable leading condition rows by component.
        prompt_embeds: Encoded prompt rows.
        times: Current component scheduler coordinates.
        layout: Packed H3 layout tensors and condition counts.
        attention_kwargs: Optional transformer attention settings.
        workflow: Workflow identifier used in diagnostics.

    Returns:
        Generated target-only video/audio data-ward velocities.
    """
    validate_target_state(target_state)
    batch_size = target_state.components["video"].shape[0]
    if batch_size != 1:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} packed boundary requires B=1, received B={batch_size}"
        )
    if not isinstance(prompt_embeds, torch.Tensor) or prompt_embeds.ndim != 3:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} field='prompt_embeds' expected shape (B,N,C), "
            f"received {getattr(prompt_embeds, 'shape', None)}"
        )
    if prompt_embeds.shape[0] != 1:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} field='prompt_embeds' requires B=1, "
            f"received B={prompt_embeds.shape[0]}"
        )
    full_state = {}
    for component in MINIMAX_H3_COMPONENT_ORDER:
        prefix = condition_prefixes.get(component)
        target = target_state.components[component]
        expected_width = target.shape[-1]
        if (
            not isinstance(prefix, torch.Tensor)
            or prefix.ndim != 3
            or prefix.shape[0] != 1
            or prefix.shape[-1] != expected_width
        ):
            raise ValueError(
                f"MiniMax H3 workflow={workflow!r} component={component!r} condition prefix "
                f"expected shape (B=1,N,{expected_width}), received "
                f"{getattr(prefix, 'shape', None)}"
            )
        expected_count = layout.get(f"num_condition_{component}_rows")
        if expected_count != prefix.shape[1]:
            raise ValueError(
                f"MiniMax H3 workflow={workflow!r} component={component!r} prefix row count "
                f"expected {expected_count}, received {prefix.shape[1]}"
            )
        full_state[component] = torch.cat([prefix, target], dim=1)
    video_time, audio_time = _current_model_times(times, workflow)
    unique_timesteps, timestep_indices = build_row_timesteps(
        layout,
        video_time,
        audio_time,
        getattr(transformer, "keyframe_noise_aug", 0.999),
        device=target_state.components["video"].device,
    )
    forward_parameters = inspect.signature(transformer.forward).parameters
    layout_kwargs = {
        field: value
        for field, value in layout.items()
        if field in forward_parameters and isinstance(value, torch.Tensor)
    }
    call_kwargs = {
        "hidden_states": full_state["video"],
        "audio_hidden_states": full_state["audio"],
        "encoder_hidden_states": prompt_embeds,
        "timestep": unique_timesteps,
        "timestep_indices": timestep_indices,
        "attention_kwargs": attention_kwargs,
        **layout_kwargs,
    }
    if "return_dict" in forward_parameters:
        call_kwargs["return_dict"] = False
    outputs = transformer(**call_kwargs)
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} transformer expected two tensor outputs, "
            f"received {type(outputs).__name__}"
        )
    video_output, audio_output = outputs
    if not isinstance(video_output, torch.Tensor) or not isinstance(audio_output, torch.Tensor):
        raise TypeError(
            f"MiniMax H3 workflow={workflow!r} transformer outputs expected tensors, received "
            f"{type(video_output).__name__}/{type(audio_output).__name__}"
        )
    expected_shapes = (full_state["video"].shape, full_state["audio"].shape)
    if video_output.shape != expected_shapes[0] or audio_output.shape != expected_shapes[1]:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} transformer output shapes expected "
            f"{tuple(expected_shapes[0])}/{tuple(expected_shapes[1])}, received "
            f"{tuple(video_output.shape)}/{tuple(audio_output.shape)}"
        )
    return LatentState(
        {
            "video": video_output[:, condition_prefixes["video"].shape[1] :],
            "audio": audio_output[:, condition_prefixes["audio"].shape[1] :],
        }
    )


def step_h3_components(
    state: LatentState,
    velocity: LatentState,
    times: ComponentTimes,
    video_scheduler: Any,
    audio_scheduler: Any,
    *,
    next_state: Optional[LatentState] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    noise_level: Optional[Union[int, float, torch.Tensor]] = None,
    compute_log_prob: bool = True,
    return_kwargs: Optional[Sequence[str]] = None,
) -> MultiModalStepOutput:
    """Step generated video then audio targets with explicit component sigmas.

    Args:
        state: Current target-only component state.
        velocity: Target-only predicted data-ward velocities.
        times: Current and next component coordinates.
        video_scheduler: Resolved video scheduler.
        audio_scheduler: Resolved audio scheduler.
        next_state: Optional stored replay target.
        generator: Shared ordered random generator.
        noise_level: Optional SDE noise scale.
        compute_log_prob: Whether to compute component and joint log probability.
        return_kwargs: Optional scheduler return-field selection.

    Returns:
        Component-preserving Flow-Factory step output.
    """
    validate_target_state(state)
    validate_target_state(velocity)
    if next_state is not None:
        validate_target_state(next_state)
    outputs = {}
    schedulers = {"video": video_scheduler, "audio": audio_scheduler}
    for component in MINIMAX_H3_COMPONENT_ORDER:
        # The state dtype is the trajectory's storage precision; the model velocity is
        # aligned to it so the scheduler sees one precision and replay reproduces rollout.
        component_velocity = velocity.components[component].to(state.components[component].dtype)
        outputs[component] = schedulers[component].step(
            component_velocity,
            times.timestep[component],
            state.components[component],
            next_latents=None if next_state is None else next_state.components[component],
            timestep_next=times.next_timestep[component],
            generator=generator,
            noise_level=noise_level,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            return_kwargs=return_kwargs,
            sigma=times.sigma[component],
            sigma_next=times.next_sigma[component],
        )
    return build_component_step_output(
        outputs["video"],
        outputs["audio"],
        reference_state=state,
    )


def forward_h3_state(
    transformer: Any,
    state: LatentState,
    condition_prefixes: Mapping[str, torch.Tensor],
    prompt_embeds: torch.Tensor,
    times: ComponentTimes,
    layout: Mapping[str, Any],
    video_scheduler: Any,
    audio_scheduler: Any,
    *,
    next_state: Optional[LatentState] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    noise_level: Optional[Union[int, float, torch.Tensor]] = None,
    compute_log_prob: bool = True,
    velocity_only: bool = False,
    attention_kwargs: Optional[Mapping[str, Any]] = None,
    return_kwargs: Optional[Sequence[str]] = None,
    workflow: str = "t2va",
) -> Union[LatentState, MultiModalStepOutput]:
    """Compose one gradient-bearing transformer call and optional scheduler step.

    Args:
        transformer: Resolved prepared transformer component.
        state: Current target-only state.
        condition_prefixes: Immutable condition rows.
        prompt_embeds: Cached prompt embeddings.
        times: Component scheduler coordinates.
        layout: Cached packed-row layout.
        video_scheduler: Resolved video scheduler.
        audio_scheduler: Resolved audio scheduler.
        next_state: Optional replay next state.
        generator: Shared random generator.
        noise_level: Optional SDE noise scale.
        compute_log_prob: Whether to compute log probability.
        velocity_only: Return velocity without scheduler dispatch.
        attention_kwargs: Optional attention settings.
        return_kwargs: Optional scheduler output fields.
        workflow: H3 workflow identifier.

    Returns:
        Target velocity or multimodal scheduler output.
    """
    velocity = run_h3_joint_transformer(
        transformer,
        state,
        condition_prefixes,
        prompt_embeds,
        times,
        layout,
        attention_kwargs=attention_kwargs,
        workflow=workflow,
    )
    if velocity_only:
        return velocity
    return step_h3_components(
        state,
        velocity,
        times,
        video_scheduler,
        audio_scheduler,
        next_state=next_state,
        generator=generator,
        noise_level=noise_level,
        compute_log_prob=compute_log_prob,
        return_kwargs=return_kwargs,
    )


def _current_model_times(times: ComponentTimes, workflow: str) -> tuple:
    if not isinstance(times, ComponentTimes) or times.sigma is None:
        raise TypeError(
            f"MiniMax H3 workflow={workflow!r} field='times' expected ComponentTimes with sigma"
        )
    values = []
    for component in MINIMAX_H3_COMPONENT_ORDER:
        sigma = times.sigma[component]
        if not isinstance(sigma, torch.Tensor) or sigma.shape != (1,):
            raise ValueError(
                f"MiniMax H3 workflow={workflow!r} component={component!r} sigma expected "
                f"shape (B=1,), received {getattr(sigma, 'shape', None)}"
            )
        values.append(float(1 - sigma.item()))
    return tuple(values)
