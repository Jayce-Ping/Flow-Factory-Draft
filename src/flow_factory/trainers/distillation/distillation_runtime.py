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

"""Provide shared media-free and score-query runtime plumbing for distillation."""

from __future__ import annotations

import inspect
import math
import zlib
from contextlib import contextmanager
from numbers import Real
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
from tqdm import tqdm

from ...models.abc import BaseAdapter
from ...samples import (
    BaseSample,
    ComponentTimes,
    LatentState,
    MultiModalStepOutput,
    StackedSampleBatch,
)

if TYPE_CHECKING:
    from ..rewards import RewardBuffer

_REPLAY_FORWARD_KEYS: tuple[str, ...] = ("guidance_scale", "stg_scale", "true_cfg_scale")

UNSUPPORTED_MEDIA_FREE_PREFIX_REASONS: Mapping[str, str] = {
    "flow_factory.models.bagel.": "inference has an unsupported media decode contract",
}


def validate_media_free_rollout(adapter: BaseAdapter, *, algorithm_name: str) -> None:
    """Require inference to expose the adapter media reconstruction seam.

    Args:
        adapter: Model adapter whose inference decoder seam is inspected.
        algorithm_name: User-facing trainer name included in diagnostics.
    """
    adapter_type = type(adapter)
    decoder = getattr(adapter, "decode_latents", None)
    for module_prefix, reason in UNSUPPORTED_MEDIA_FREE_PREFIX_REASONS.items():
        if adapter_type.__module__.startswith(module_prefix):
            raise ValueError(
                f"{algorithm_name} media-free rollout cannot use "
                f"adapter={adapter_type.__name__!r}: {reason}, so media reconstruction "
                "cannot be disabled"
            )
    if getattr(adapter_type, "decode_latents", None) is BaseAdapter.decode_latents or not callable(
        decoder
    ):
        raise ValueError(
            f"{algorithm_name} media-free rollout cannot use "
            f"adapter={adapter_type.__name__!r}: inference must route media reconstruction "
            "through adapter.decode_latents, and this adapter is unsupported until it exposes "
            "that seam"
        )


@contextmanager
def without_media_decoding(
    adapter: BaseAdapter,
    *,
    algorithm_name: str,
) -> Iterator[None]:
    """Replace rollout media reconstruction with shape-preserving empty outputs.

    Args:
        adapter: Adapter whose decoder is temporarily replaced.
        algorithm_name: User-facing trainer name included in diagnostics.

    Yields:
        Control while media decoding is suppressed.
    """
    adapter_attributes = vars(adapter)
    had_instance_override = "decode_latents" in adapter_attributes
    previous_override = adapter_attributes.get("decode_latents")
    decoder = adapter.decode_latents
    decoder_signature = inspect.signature(decoder)
    adapter_name = type(adapter).__name__

    def empty_media(*latent_inputs: Any, **kwargs: Any) -> Any:
        try:
            bound = decoder_signature.bind(*latent_inputs, **kwargs)
        except TypeError as error:
            raise TypeError(
                f"{algorithm_name} media-free decoder adapter={adapter_name!r} could not bind "
                f"decode_latents signature={decoder_signature}; received positional types="
                f"{[type(value).__name__ for value in latent_inputs]}, keyword types="
                f"{ {name: type(value).__name__ for name, value in kwargs.items()} }"
            ) from error

        latent_arguments = {
            name: value
            for name, value in bound.arguments.items()
            if name == "latents" or name.endswith("_latents")
        }
        batch_sizes = {}
        for name, value in latent_arguments.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                batch_sizes[name] = value.shape[0]
                continue
            if isinstance(value, LatentState):
                component_sizes = {
                    component: tensor.shape[0]
                    for component, tensor in value.components.items()
                    if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1
                }
                if len(component_sizes) != len(value.components) or len(
                    set(component_sizes.values())
                ) != 1:
                    raise ValueError(
                        f"{algorithm_name} media-free decoder adapter={adapter_name!r}, "
                        f"signature={decoder_signature} received invalid LatentState batch "
                        f"sizes for {name!r}: {component_sizes!r}"
                    )
                batch_sizes[name] = next(iter(component_sizes.values()))
                continue
            raise TypeError(
                f"{algorithm_name} media-free decoder adapter={adapter_name!r}, "
                f"signature={decoder_signature} expected tensor or LatentState argument "
                f"named 'latents' or '*_latents', received {name}={type(value).__name__}"
            )
        if not batch_sizes:
            received = ", ".join(
                f"{name}={type(value).__name__}" for name, value in latent_arguments.items()
            )
            raise TypeError(
                f"{algorithm_name} media-free decoder adapter={adapter_name!r}, "
                f"signature={decoder_signature} expected tensor argument named "
                "'latents' or '*_latents' with a batch dimension; "
                f"received {received or 'no recognized latent argument'}"
            )
        if len(set(batch_sizes.values())) != 1:
            sizes = ", ".join(f"{name}={size}" for name, size in batch_sizes.items())
            raise ValueError(
                f"{algorithm_name} media-free decoder adapter={adapter_name!r}, "
                f"signature={decoder_signature} received ambiguous batch sizes: {sizes}"
            )

        batch_size = next(iter(batch_sizes.values()))
        empty_factory = getattr(adapter, "empty_decoded_media", None)
        if callable(empty_factory):
            return empty_factory(batch_size)
        return BaseAdapter.empty_decoded_media(adapter, batch_size)

    adapter.decode_latents = empty_media
    try:
        yield
    finally:
        if had_instance_override:
            adapter.decode_latents = previous_override
        else:
            delattr(adapter, "decode_latents")


def detach_state(state: LatentState) -> LatentState:
    """Detach every component while preserving active masks.

    Args:
        state: Latent state to detach.

    Returns:
        Detached state with the same component order and masks.
    """
    return LatentState(
        {name: component.detach() for name, component in state.components.items()},
        active_masks=state.active_masks,
    )


def require_velocity(
    output: MultiModalStepOutput,
    *,
    algorithm_name: str,
    role_name: str,
) -> LatentState:
    """Return a required role velocity with trainer context.

    Args:
        output: Adapter step output expected to contain velocity.
        algorithm_name: User-facing trainer name included in diagnostics.
        role_name: Active model role included in diagnostics.

    Returns:
        Structured velocity prediction.
    """
    if output.velocity is None:
        raise ValueError(f"{algorithm_name} {role_name} forward expected velocity, received None")
    return output.velocity


def query_score_velocity(
    adapter: BaseAdapter,
    batch: StackedSampleBatch,
    state: LatentState,
    times: ComponentTimes,
    *,
    role_name: Literal["reference", "fake", "surrogate"],
    autocast: Callable[[], ContextManager[Any]],
    forward_kwargs: Dict[str, object],
    algorithm_name: str,
) -> LatentState:
    """Query one detached score role in fresh role and autocast scopes.

    Args:
        adapter: Role-aware model adapter.
        batch: Collated conditioning batch.
        state: Shared detached perturbed state.
        times: Shared component scheduler coordinates.
        role_name: Frozen score role to query.
        autocast: Factory producing one fresh autocast context.
        forward_kwargs: Trainer configuration not already owned by the batch.
        algorithm_name: User-facing trainer name included in diagnostics.

    Returns:
        Detached structured velocity prediction.
    """
    # The reference score is the pre-finetune teacher, i.e. the same components at
    # an earlier point in time. That is a parameter snapshot, not a variant: it
    # holds no gradients and no optimizer state, so it must not cost a bundle
    # member. Only the trainable roles are declared variants.
    weights = (
        adapter.use_ref_parameters()
        if role_name == "reference"
        else adapter.use_component_variant(role_name)
    )
    with weights:
        with torch.no_grad():
            with autocast():
                output = adapter.forward_state(
                    batch=batch,
                    state=state,
                    times=times,
                    compute_log_prob=False,
                    return_fields=("velocity",),
                    **forward_kwargs,
                )
    return detach_state(
        require_velocity(
            output,
            algorithm_name=algorithm_name,
            role_name=role_name,
        )
    )


def replay_forward_kwargs(training_args: Any, batch: Mapping[str, Any]) -> Dict[str, object]:
    """Return allow-listed adapter forward arguments not already owned by the batch.

    Args:
        training_args: Trainer configuration that may carry guidance knobs.
        batch: Collated sample batch whose keys take precedence.

    Returns:
        Keyword arguments safe to splat into adapter forwards.
    """
    kwargs: Dict[str, object] = {}
    for key in _REPLAY_FORWARD_KEYS:
        if hasattr(training_args, key) and key not in batch:
            kwargs[key] = getattr(training_args, key)
    extra_kwargs = getattr(training_args, "extra_kwargs", None)
    if extra_kwargs:
        if not isinstance(extra_kwargs, Mapping):
            raise TypeError(
                "expected train.extra_kwargs as a mapping, "
                f"received {type(extra_kwargs).__name__}: {extra_kwargs!r}"
            )
        for key, value in extra_kwargs.items():
            if key not in batch:
                kwargs[key] = value
    return kwargs


def reject_training_rewards(trainer: Any, *, algorithm_name: str) -> tuple:
    """Build the shared feedback runtime and assert the training side stayed empty.

    A reward-free loss is a property of training, not of evaluation. Zeroing the whole
    runtime to express it takes eval monitoring down with it, so instead build the shared
    one -- `Arguments` already rejects training rewards for these trainers, so the
    training side comes back empty on its own -- and check that it did.

    Written as a function rather than a base method because these trainers are siblings
    under `BaseTrainer`, not a chain: a zero-argument `super()` inside one of them
    resolves against the wrong class when another calls it.

    Args:
        trainer: Distillation trainer being initialized.
        algorithm_name: User-facing name included in diagnostics.

    Returns:
        Training and eval reward models; the training mapping is always empty.

    Raises:
        RuntimeError: If a training reward survived configuration validation.
    """
    from ..abc import BaseTrainer

    training_models, eval_models = BaseTrainer._init_reward_model(trainer)
    if training_models:
        raise RuntimeError(
            f"{algorithm_name} is data-free and must not train against rewards, received "
            f"{sorted(training_models)}"
        )
    return training_models, eval_models


def reference_forward_kwargs(training_args: Any, batch: Mapping[str, Any]) -> Dict[str, object]:
    """Return forward arguments for the real score, the only role that may be guided.

    The real score defines the target distribution, so classifier-free guidance on it
    is what the match is worth. The generator rolls out CFG-free and the fake score has
    to model what the generator actually produces, so both stay on
    ``train.guidance_scale``. Leaving ``train.real_guidance_scale`` unset keeps every
    role on one scale, which is what these algorithms did before.

    Args:
        training_args: Trainer configuration carrying the guidance knobs.
        batch: Collated sample batch whose keys take precedence.

    Returns:
        Keyword arguments for the reference query.
    """
    kwargs = replay_forward_kwargs(training_args, batch)
    real_guidance_scale = getattr(training_args, "real_guidance_scale", None)
    if real_guidance_scale is not None and "guidance_scale" not in batch:
        kwargs["guidance_scale"] = real_guidance_scale
    return kwargs


def as_role_microbatches(
    samples: Sequence[Any],
    *,
    batch_size: int,
    accumulation_steps: int,
    algorithm_name: str,
) -> List[List[Any]]:
    """Normalize a flat batch or list-of-batches into GAS role microbatches.

    Args:
        samples: One dataloader batch, or a sequence of batches matching GAS.
        batch_size: Required ``per_device_batch_size`` of every microbatch.
        accumulation_steps: Required number of same-role microbatches.
        algorithm_name: User-facing trainer name included in diagnostics.

    Returns:
        ``accumulation_steps`` lists, each of length ``batch_size``.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError(
            f"{algorithm_name} expected per_device_batch_size >= 1 as an int, "
            f"received {type(batch_size).__name__}: {batch_size!r}"
        )
    if (
        not isinstance(accumulation_steps, int)
        or isinstance(accumulation_steps, bool)
        or accumulation_steps < 1
    ):
        raise ValueError(
            f"{algorithm_name} expected gradient_accumulation_steps >= 1 as an int, "
            f"received {type(accumulation_steps).__name__}: {accumulation_steps!r}"
        )
    if not samples:
        raise ValueError(f"{algorithm_name} optimize expected a non-empty sample collection")

    first = samples[0]
    if isinstance(first, (list, tuple)):
        microbatches = [list(batch) for batch in samples]
    else:
        microbatches = [list(samples)]

    if len(microbatches) != accumulation_steps:
        raise ValueError(
            f"{algorithm_name} optimize expected {accumulation_steps} role microbatches "
            "matching gradient_accumulation_steps, "
            f"received {len(microbatches)}"
        )
    for index, microbatch in enumerate(microbatches):
        if not microbatch:
            raise ValueError(
                f"{algorithm_name} optimize expected a non-empty microbatch at index {index}"
            )
        if len(microbatch) != batch_size:
            raise ValueError(
                f"{algorithm_name} optimize expected microbatch index {index} to have "
                f"per_device_batch_size={batch_size} samples, received {len(microbatch)}"
            )
    return microbatches


def generate_one_rollout_batch(
    trainer: Any,
    *,
    reward_buffer: Optional["RewardBuffer"] = None,
    compute_log_prob: bool = False,
    trajectory_indices: Optional[List[int]] = None,
    algorithm_name: str,
) -> List[BaseSample]:
    """Generate exactly one dataloader batch for one distillation outer iteration.

    Args:
        trainer: Distillation trainer owning the prompt dataloader iterator.
        reward_buffer: Optional reward buffer; DMD2/TDM pass ``None``.
        compute_log_prob: Whether sample_batch should store log-probabilities.
        trajectory_indices: Stored trajectory positions for the rollout.
        algorithm_name: User-facing trainer name included in diagnostics.

    Returns:
        Samples from one dataloader batch.
    """
    if trainer.dataloader is None:
        raise RuntimeError(
            f"{algorithm_name} generate_one_rollout_batch() called but no training "
            "dataloader exists. `data.datasets` has no entry with `train: enabled` "
            "(eval-only config); a trainer should not enter the sampling loop here."
        )
    if not hasattr(trainer, "_rollout_dataloader_epoch"):
        trainer._rollout_dataloader_epoch = 0
    if not hasattr(trainer, "_rollout_data_iter"):
        trainer._rollout_data_iter = None

    trainer.adapter.rollout()
    # Each outer iteration scores exactly the batch it just rolled out, matching
    # BaseTrainer.generate_samples. Left uncleared, the buffer carries the previous
    # iteration's samples into this one, and the reward count stops matching the
    # sample count as soon as an epoch accumulates more than one batch.
    if reward_buffer is not None:
        reward_buffer.clear()
    if trainer._rollout_data_iter is None:
        if hasattr(trainer.dataloader, "set_epoch"):
            trainer.dataloader.set_epoch(trainer._rollout_dataloader_epoch)
        trainer._rollout_data_iter = iter(trainer.dataloader)
    try:
        batch = next(trainer._rollout_data_iter)
    except StopIteration:
        trainer._rollout_dataloader_epoch += 1
        if hasattr(trainer.dataloader, "set_epoch"):
            trainer.dataloader.set_epoch(trainer._rollout_dataloader_epoch)
        trainer._rollout_data_iter = iter(trainer.dataloader)
        batch = next(trainer._rollout_data_iter)

    with trainer._rollout_acceleration(), torch.no_grad(), trainer.autocast():
        return trainer.sample_batch(
            batch,
            reward_buffer=reward_buffer,
            compute_log_prob=compute_log_prob,
            trajectory_indices=trajectory_indices,
        )


def role_repeat_progress(trainer: Any, *, role_name: str, repeats: int) -> Iterator[int]:
    """Report progress through a role's repeated phases.

    TTUR runs the fake role several times per generator step, so the repeat count --
    not the microbatch window -- is what a reader watching the log wants to see move.

    Args:
        trainer: Trainer owning the epoch counter and progress-bar preference.
        role_name: Role being repeated, shown in the bar description.
        repeats: Number of phases to run.

    Returns:
        Iterator over the repeat indices, wrapped in a progress bar.
    """
    return tqdm(
        range(repeats),
        desc=f"Epoch {trainer.epoch} {role_name.capitalize()}",
        position=1,
        leave=False,
        disable=not trainer.show_progress_bar,
    )


def record_distillation_metric(trainer: Any, name: str, value: Any) -> None:
    """Buffer one scalar for the next sync-boundary log.

    Buffered rather than logged on the spot because a role phase runs once per TTUR
    repeat and once per microbatch inside that, so logging directly would emit many
    values against a single ``step`` and let the last writer win.

    Args:
        trainer: Trainer owning the buffer.
        name: Metric key, already carrying its ``train/`` prefix.
        value: Scalar or 0-dim tensor to average into the metric.

    Raises:
        TypeError: If ``name`` is not a non-empty string.
        ValueError: If ``value`` is not a finite scalar.
    """
    if not isinstance(name, str) or not name:
        raise TypeError(
            f"expected a non-empty metric name, received {type(name).__name__}: {name!r}"
        )
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"expected a scalar for metric {name!r}, received a tensor of "
                f"shape {tuple(value.shape)}"
            )
        value = value.detach().float().item()
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"expected a finite scalar for metric {name!r}, received {value!r}")

    buffer: Dict[str, Tuple[float, float]] = trainer.__dict__.setdefault(
        "_distillation_metrics", {}
    )
    total, count = buffer.get(name, (0.0, 0.0))
    buffer[name] = (total + float(value), count + 1.0)


def record_state_statistics(trainer: Any, prefix: str, state: Any) -> None:
    """Buffer the mean and standard deviation of one latent state.

    These are the drift monitors. A generator whose clean latents wander out of the
    VAE's range decodes as dark or washed-out images while its loss keeps falling, so
    the losses alone never show it; a reference state whose std strays far from 1 means
    the teacher is being queried at the wrong time.

    Args:
        trainer: Trainer owning the buffer.
        prefix: Metric key prefix, already carrying its ``train/`` prefix.
        state: Latent state whose components are pooled into one mean and std.
    """
    total = 0.0
    total_square = 0.0
    count = 0.0
    for component in state.components.values():
        values = component.detach().float()
        total += float(values.sum())
        total_square += float(values.square().sum())
        count += float(values.numel())
    if count == 0:
        raise ValueError(f"expected a non-empty state for metric prefix {prefix!r}, received none")

    mean = total / count
    variance = max(total_square / count - mean * mean, 0.0)
    record_distillation_metric(trainer, f"{prefix}_mean", mean)
    record_distillation_metric(trainer, f"{prefix}_std", math.sqrt(variance))


def pop_distillation_metrics(trainer: Any) -> Dict[str, float]:
    """Average the buffered metrics across ranks and clear the buffer.

    Args:
        trainer: Trainer owning the buffer.

    Returns:
        Metric name to its process-group mean, empty when nothing was buffered.

    Raises:
        RuntimeError: If the ranks buffered different metrics, which would otherwise
            hang the reduction below instead of failing.
    """
    buffer: Dict[str, Tuple[float, float]] = (
        trainer.__dict__.pop("_distillation_metrics", None) or {}
    )
    names = sorted(buffer)
    accelerator = trainer.accelerator

    # A rank-dependent key set makes the reduction below deadlock rather than raise,
    # and a deadlock 40 minutes into a 64-GPU run costs far more than this check.
    signature = torch.tensor(
        [float(len(names)), float(sum(zlib.crc32(name.encode()) for name in names))],
        device=accelerator.device,
        dtype=torch.float64,
    )
    # Compared against the group maximum rather than a min/max pair because Accelerate
    # maps every reduction other than "max" onto a sum, so "min" would silently return
    # one. Any rank that disagrees sits below the maximum and raises, which is enough to
    # stop the run.
    highest = accelerator.reduce(signature.clone(), reduction="max")
    if not torch.equal(highest, signature):
        raise RuntimeError(
            "expected every rank to buffer the same distillation metrics; this rank "
            f"buffered {len(names)}: {names}. Metrics are recorded on code paths that "
            "must be rank-uniform, so a mismatch means one rank skipped a role phase."
        )
    if not names:
        return {}

    packed = torch.tensor(
        [buffer[name][0] for name in names] + [buffer[name][1] for name in names],
        device=accelerator.device,
        dtype=torch.float64,
    )
    reduced = accelerator.reduce(packed, reduction="sum")
    totals = reduced[: len(names)]
    counts = reduced[len(names) :]
    return {name: totals[i].item() / counts[i].item() for i, name in enumerate(names)}


def run_role_phase(
    trainer: Any,
    role_name: str,
    microbatches: Sequence[Sequence[Any]],
    loss_fn: Callable[[Sequence[Any]], torch.Tensor],
) -> None:
    """Run one exclusive role phase over same-role GAS microbatches.

    Args:
        trainer: Trainer owning the role coordinator.
        role_name: Active trainable role for this phase.
        microbatches: Same-role accumulation window; one optimizer step at the end.
        loss_fn: Maps one microbatch to a scalar loss.
    """
    if not microbatches:
        raise ValueError(f"{role_name} phase expected at least one microbatch, received none")
    with trainer.role_optimization.phase(role_name):
        # A single microbatch would render a 1/1 bar once per TTUR repeat, which is
        # noise; the role's own progress is already carried by the caller's bar.
        for microbatch in tqdm(
            microbatches,
            desc=f"Epoch {trainer.epoch} {role_name.capitalize()}",
            position=2,
            leave=False,
            disable=not trainer.show_progress_bar or len(microbatches) < 2,
        ):
            with trainer.role_optimization.microbatch():
                loss = loss_fn(microbatch)
                record_distillation_metric(trainer, f"train/{role_name}_loss", loss)
                trainer.role_optimization.backward(loss)
                trainer._finish_role_microbatch()

    grad_norm = trainer.role_optimization.roles[role_name].last_grad_norm
    if grad_norm is not None:
        record_distillation_metric(trainer, f"train/{role_name}_grad_norm", grad_norm)


def run_distillation_training_step(trainer: Any) -> None:
    """Run one distillation epoch: accumulate ``GAS`` rollouts, then optimize once.

    This is the only way a distillation epoch differs from any other. Everything
    around it - reseeding, checkpointing, evaluation, the EMA step - is the shared
    loop in :meth:`BaseTrainer.start`, so a distillation trainer overrides
    :meth:`BaseTrainer._run_training_step` with this and inherits the rest. That is
    what gives these algorithms the same eval-time reward monitoring every other
    trainer has.

    Args:
        trainer: DMD2, TDM, or TDM-R1 trainer.

    Raises:
        ValueError: If the configured accumulation step count is not a positive int.
    """
    accumulation_steps = trainer.training_args.gradient_accumulation_steps
    if (
        not isinstance(accumulation_steps, int)
        or isinstance(accumulation_steps, bool)
        or accumulation_steps < 1
    ):
        raise ValueError(
            "expected gradient_accumulation_steps >= 1 as an int, received "
            f"{type(accumulation_steps).__name__}: {accumulation_steps!r}"
        )
    microbatches: List[List[BaseSample]] = []
    for _ in tqdm(
        range(accumulation_steps),
        desc=f"Epoch {trainer.epoch} Sampling",
        position=0,
        disable=not trainer.show_progress_bar,
    ):
        with trainer.sampling_context():
            samples = trainer.sample()
        trainer.prepare_feedback(samples)
        microbatches.append(samples)
    trainer.optimize(microbatches)

    metrics = pop_distillation_metrics(trainer)
    if metrics:
        trainer.log_data(metrics, step=trainer.step)
