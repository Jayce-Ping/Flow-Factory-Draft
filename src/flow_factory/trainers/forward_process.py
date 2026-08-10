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


def state_batch_size(trainer: Any, state: LatentState) -> int:
    """Return the batch size of a latent state's primary component.

    Args:
        trainer: Trainer owning the adapter that declares the component order.
        state: Batched latent state in ``trajectory_component_order``.

    Returns:
        Leading dimension of the primary component.
    """
    primary = trainer.adapter.trajectory_component_order[0]
    return state.components[primary].shape[0]


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
    batch_size: int,
) -> LatentState:
    """Return a required velocity state in authoritative component order.

    Args:
        trainer: Trainer owning the adapter that declares the component order.
        output: Forward output to read the velocity from.
        source: Pass identifier reported by validation errors.
        batch_size: Batch size every velocity component must use.

    Returns:
        Velocity state keyed by component.
    """
    expected_names = trainer.adapter.trajectory_component_order
    velocity = output.velocity
    if velocity is None or velocity.component_names != expected_names:
        received = None if velocity is None else velocity.component_names
        raise ValueError(
            f"expected {source} velocity for {type(trainer).__name__} in component order "
            f"{expected_names}, received {received}; request 'velocity' through return_fields"
        )
    for name in expected_names:
        component = velocity.components[name]
        if component.ndim < 2 or component.shape[0] != batch_size:
            raise ValueError(
                f"expected {source} velocity component {name!r} for {type(trainer).__name__} "
                f"to use batch size {batch_size}, received shape {tuple(component.shape)}"
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
    output = trainer.adapter.forward_state(
        batch=batch,
        state=state,
        times=times,
        compute_log_prob=False,
        return_fields=("velocity",),
        noise_level=noise_level,
        **forward_kwargs,
    )
    return require_velocity_state(trainer, output, source, state_batch_size(trainer, state))
