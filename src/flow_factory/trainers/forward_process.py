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

# src/flow_factory/trainers/forward_process.py
"""Forward-process helpers shared by the decoupled trainers.

Decoupled algorithms (DiffusionNFT, AWM, DPO) train on a freshly noised terminal
state instead of a stored rollout transition, so they share one velocity-only
forward contract rather than the coupled replay contract in ``grpo.py``.
"""

from typing import Any, Dict

import torch

from ..samples import ComponentTimes, LatentState, MultiModalStepOutput, StackedSampleBatch


def require_component_sigmas(trainer: Any, times: ComponentTimes) -> Dict[str, torch.Tensor]:
    """Return per-component sigmas in authoritative component order.

    Args:
        trainer: Trainer owning the adapter that declares the component order.
        times: Component times produced for a forward-noised state.

    Returns:
        Sigma tensor per component.
    """
    expected_names = trainer.adapter.trajectory_component_order
    sigmas = times.sigma
    if sigmas is None or tuple(sigmas) != expected_names:
        received = None if sigmas is None else tuple(sigmas)
        raise ValueError(
            f"expected component sigmas in order {expected_names} for "
            f"{type(trainer).__name__}, received {received}"
        )
    return dict(sigmas)


def require_latent_state(trainer: Any, state: Any, identifier: str) -> LatentState:
    """Return a latent state validated against the declared component order.

    Args:
        trainer: Trainer owning the adapter that declares the component order.
        state: Candidate latent state.
        identifier: State name reported by validation errors.

    Returns:
        The validated latent state.
    """
    expected_names = trainer.adapter.trajectory_component_order
    if not isinstance(state, LatentState):
        raise TypeError(
            f"expected LatentState for {identifier} on {type(trainer).__name__}, "
            f"received {type(state).__name__}"
        )
    if state.component_names != expected_names:
        raise ValueError(
            f"expected {identifier} on {type(trainer).__name__} in component order "
            f"{expected_names}, received {state.component_names}"
        )
    return state


def state_batch_size(trainer: Any, state: LatentState, identifier: str = "latent state") -> int:
    """Return the batch size of a latent state's primary component.

    Args:
        trainer: Trainer owning the adapter that declares the component order.
        state: Batched latent state in ``trajectory_component_order``.
        identifier: State name reported by validation errors.

    Returns:
        Leading dimension of the primary component.
    """
    primary = trainer.adapter.trajectory_component_order[0]
    component = require_latent_state(trainer, state, identifier).components[primary]
    if component.ndim < 2:
        raise ValueError(
            f"expected {identifier} component {primary!r} on {type(trainer).__name__} to be a "
            f"batched tensor with a leading batch dimension and shape (B, ...), received shape "
            f"{tuple(component.shape)}"
        )
    return component.shape[0]


def training_forward_kwargs(trainer: Any, batch: StackedSampleBatch) -> Dict[str, Any]:
    """Return the training arguments the batch does not already carry.

    Legacy decoupled trainers unpacked ``batch`` after ``training_args``, so
    batch-level values win on shared keys.

    Args:
        trainer: Trainer whose ``training_args`` supply the forward defaults.
        batch: Collated sample batch supplying conditioning arguments.

    Returns:
        Training arguments that do not collide with batch-level keys.
    """
    return {key: value for key, value in {**trainer.training_args}.items() if key not in batch}


def require_velocity_state(
    trainer: Any,
    output: MultiModalStepOutput,
    source: str,
    expected_state: LatentState,
) -> LatentState:
    """Return a velocity state validated against the state it was predicted for.

    Every component must carry the exact shape and device of its input state:
    a ``(B, 1, ...)`` velocity would broadcast silently against a ``(B, C, ...)``
    state and quietly change every downstream matching loss.

    Dtype is checked for a floating-point type shared by all components rather
    than for equality with ``expected_state``: ``latent_storage_dtype`` (fp16 by
    default) deliberately decouples stored latents from the autocast compute
    dtype, and adapters that predict in fp32 (e.g. Z-Image) legitimately return a
    velocity in neither. A per-component dtype split, however, breaks the shared
    reduction contract.

    Args:
        trainer: Trainer owning the adapter that declares the component order.
        output: Forward output to read the velocity from.
        source: Pass identifier reported by validation errors.
        expected_state: Latent state the forward pass consumed.

    Returns:
        Velocity state keyed by component.
    """
    expected_names = trainer.adapter.trajectory_component_order
    reference = require_latent_state(trainer, expected_state, f"{source} forward state")
    velocity = output.velocity
    if velocity is None or velocity.component_names != expected_names:
        received = None if velocity is None else velocity.component_names
        raise ValueError(
            f"expected {source} velocity for {type(trainer).__name__} in component order "
            f"{expected_names}, received {received}; request 'velocity' through return_fields"
        )
    primary_name = expected_names[0]
    primary_dtype = velocity.components[primary_name].dtype
    for name in expected_names:
        component = velocity.components[name]
        component_reference = reference.components[name]
        if component.shape != component_reference.shape:
            raise ValueError(
                f"expected {source} velocity component {name!r} for {type(trainer).__name__} to "
                f"match its forward state shape {tuple(component_reference.shape)}, received "
                f"shape {tuple(component.shape)}"
            )
        if component.device != component_reference.device:
            raise ValueError(
                f"expected {source} velocity component {name!r} for {type(trainer).__name__} on "
                f"the forward state device {component_reference.device}, received "
                f"{component.device}"
            )
        if not component.is_floating_point():
            raise ValueError(
                f"expected {source} velocity component {name!r} for {type(trainer).__name__} to "
                f"be a floating point tensor, received {component.dtype}"
            )
        if component.dtype != primary_dtype:
            raise ValueError(
                f"expected {source} velocity component {name!r} for {type(trainer).__name__} to "
                f"share the dtype of component {primary_name!r} ({primary_dtype}), received "
                f"{component.dtype}"
            )
    return velocity


def forward_velocity_state(
    trainer: Any,
    batch: StackedSampleBatch,
    state: LatentState,
    times: ComponentTimes,
    *,
    source: str,
    noise_level: float = 0.0,
    **overrides: Any,
) -> LatentState:
    """Predict velocity for a forward-noised state through the active parameters.

    The caller owns the autocast / ``no_grad`` / parameter-swap scope, so this
    helper never opens one; it only bridges the state into ``forward_state`` and
    validates the returned velocity.

    Args:
        trainer: Trainer supplying the adapter and training arguments.
        batch: Collated sample batch supplying conditioning arguments.
        state: Forward-noised latent state to evaluate.
        times: Component times for the noised state.
        source: Pass identifier reported by validation errors.
        noise_level: Scheduler noise-level override; decoupled training uses ``0.0``.
        **overrides: Explicit forward arguments taking precedence over training args.

    Returns:
        Predicted velocity state keyed by component.
    """
    forward_kwargs = training_forward_kwargs(trainer, batch)
    forward_kwargs.update(overrides)
    # Validate the batch rank before the forward: the velocity check below only
    # compares shapes, so an unbatched state would pass unnoticed if the adapter
    # mirrored it.
    state_batch_size(trainer, state, f"{source} forward state")
    output = trainer.adapter.forward_state(
        batch=batch,
        state=state,
        times=times,
        compute_log_prob=False,
        return_fields=("velocity",),
        noise_level=noise_level,
        **forward_kwargs,
    )
    return require_velocity_state(trainer, output, source, state)
