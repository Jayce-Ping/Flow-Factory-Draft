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

# src/flow_factory/advantage/advantage_processor.py
"""
Communication-aware Advantage Processor.

Extracts advantage computation logic from GRPOTrainer into a standalone,
reusable component.  Automatically selects the communication strategy based
on the resolved sampler type:

- ``distributed_k_repeat``: gather rewards + unique_ids across ranks →
  global grouping → scatter back to local rank.
- ``group_contiguous``: all K copies already reside on the same rank →
  skip all cross-rank communication for advantage computation.  Training log
  metrics are computed via mode-aware ``_metric_*`` helpers that transparently
  select between plain NumPy (post-gather global arrays) and ``utils.dist``
  reductions (local shards) so logging always reflects global statistics.
"""
from typing import List, Dict, Optional, Union, Literal, Callable, Tuple, Any
import numpy as np
import torch
from accelerate import Accelerator

from ..samples import BaseSample
from ..rewards import RewardProcessor
from ..utils.dist import global_zero_std_ratio, global_tensor_stats_batch
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class AdvantageProcessor:
    """Communication-aware advantage computation processor.

    Parameters
    ----------
    accelerator : Accelerator
        HuggingFace Accelerator instance for distributed ops.
    reward_weights : dict[str, float]
        Mapping from reward name to its aggregation weight.
    group_size : int
        Number of repeated samples per unique prompt (K).
    global_std : bool
        If ``True``, normalise advantages using the global std across all
        groups; otherwise use per-group std.
    sampler_type : str
        One of ``"distributed_k_repeat"`` or ``"group_contiguous"``.
        Determines whether cross-rank communication is needed.
    verbose : bool
        Whether to emit progress information.

    Notes
    -----
    After :meth:`compute_advantages` with ``'sum'`` or ``'gdpo'``, call
    :meth:`pop_advantage_metrics` once to retrieve training metrics (including
    ``train_samples``) for ``log_data``. Custom callables leave an empty metrics
    snapshot. This class does not perform logging itself.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        reward_weights: Dict[str, float],
        group_size: int,
        global_std: bool = True,
        sampler_type: str = "distributed_k_repeat",
        verbose: bool = True,
    ):
        self.accelerator = accelerator
        self.reward_weights = reward_weights
        self.group_size = group_size
        self.global_std = global_std
        self.sampler_type = sampler_type
        self.verbose = verbose

        self.group_on_same_rank = sampler_type == "group_contiguous"
        self._pending_advantage_metrics: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pop_advantage_metrics(self) -> Dict[str, Any]:
        """Return and clear metrics from the last ``sum`` / ``gdpo`` advantage pass.

        Call once per :meth:`compute_advantages` when using built-in aggregation.
        Returns an empty dict if nothing was produced (e.g. custom callable only,
        or no prior computation).
        """
        out = dict(self._pending_advantage_metrics or {})
        self._pending_advantage_metrics = None
        return out

    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func: Optional[Union[Literal["sum", "gdpo"], Callable]] = None,
    ) -> torch.Tensor:
        """Compute per-sample advantages.

        Parameters
        ----------
        samples : list[BaseSample]
            Samples on the current rank.
        rewards : dict[str, Tensor]
            Per-reward-model reward tensors aligned with *samples*.
        store_to_samples : bool
            Write computed advantages into ``sample.extra_kwargs['advantage']``.
        aggregation_func : str or callable
            ``'sum'`` for weighted-sum GRPO, ``'gdpo'`` for GDPO-style, or a
            custom ``callable(processor, samples, rewards, store_to_samples)``.

        Returns
        -------
        Tensor
            Advantages for the local rank, shape ``(len(samples),)``.
        """
        self._pending_advantage_metrics = None
        aggregation_func = aggregation_func or "gdpo"
        if aggregation_func == "sum":
            return self.compute_weighted_sum(samples, rewards, store_to_samples)
        elif aggregation_func == "gdpo":
            return self.compute_gdpo(samples, rewards, store_to_samples)
        elif callable(aggregation_func):
            adv = aggregation_func(self, samples, rewards, store_to_samples)
            if self._pending_advantage_metrics is None:
                self._pending_advantage_metrics = {}
            return adv
        else:
            raise ValueError(
                f"Unsupported advantage aggregation method: {aggregation_func}. "
                "Supported: ['sum', 'gdpo'] "
                "or a callable function that takes (processor, samples, rewards, store_to_samples) as inputs."
            )

    # ------------------------------------------------------------------
    # Communication layer
    # ------------------------------------------------------------------

    def collect_group_rewards(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Collect rewards and group indices, respecting sampler topology.

        Automatically selects between two code paths based on the sampler type:

        - ``group_contiguous``: no cross-rank communication.  Rewards are
          converted to NumPy locally and group indices are derived from
          ``sample.unique_id``.  Returned arrays have shape ``(B,)`` (local).
        - ``distributed_k_repeat``: all per-reward tensors and the
          ``unique_id`` vector are packed into a single ``(B, N+1)`` tensor
          and gathered with one ``accelerator.gather()`` call.  Returned
          arrays have shape ``(W*B,)`` (global, ordered by rank index).

        Whether the returned arrays are local or global is an internal detail
        handled by :meth:`_to_local`.  Callers should not branch on it.

        Parameters
        ----------
        samples : list[BaseSample]
            Samples on the current rank.  Only ``sample.unique_id`` is read.
        rewards : dict[str, Tensor]
            Mapping from reward name to a 1-D tensor of reward values,
            aligned with *samples*.

        Returns
        -------
        collected_rewards : dict[str, np.ndarray]
            Mapping from reward name to a NumPy array of reward values.
        group_indices : np.ndarray
            Integer array mapping each element to its prompt group
            (contiguous integers starting from 0).
        """
        if self.group_on_same_rank:
            # group_contiguous: all K copies on same rank, no communication needed.
            # Rewards arrive as cpu tensors; convert directly to numpy.
            collected_rewards = {
                key: torch.as_tensor(value).cpu().numpy() for key, value in rewards.items()
            }
            unique_ids = np.array([s.unique_id for s in samples], dtype=np.int64)
            _unique_ids, group_indices = np.unique(unique_ids, return_inverse=True)
            return collected_rewards, group_indices
        else:
            # distributed_k_repeat: move to device for accelerator.gather()
            rewards = {
                key: torch.as_tensor(value).to(self.accelerator.device)
                for key, value in rewards.items()
            }
            reward_keys = list(rewards.keys())
            unique_ids = torch.tensor(
                [s.unique_id for s in samples],
                dtype=torch.int64,
                device=self.accelerator.device,
            )
            columns = [rewards[k].view(-1).float() for k in reward_keys]
            columns.append(unique_ids.float())
            packed = torch.stack(columns, dim=1)  # (B, N+1)

            gathered = self.accelerator.gather(packed).cpu().numpy()  # (W*B, N+1)

            collected_rewards = {
                key: gathered[:, i] for i, key in enumerate(reward_keys)
            }
            gathered_ids = gathered[:, -1].astype(np.int64)
            _unique_ids, group_indices = np.unique(gathered_ids, return_inverse=True)
            return collected_rewards, group_indices

    def collect_applicability_mask(
        self,
        samples: List[BaseSample],
        reward_keys: List[str],
    ) -> np.ndarray:
        """Gather the per-(reward, sample) applicability matrix.

        Mirrors :meth:`collect_group_rewards` topology:

        - ``group_contiguous`` (local): no communication; matrix shape
          ``(R, B_local)``.
        - ``distributed_k_repeat``: all per-rank ``(R, B)`` matrices are
          packed and gathered, returning ``(R, W*B)`` ordered the same
          way as :meth:`collect_group_rewards` so positions align.

        Used by aggregation as the *authoritative* source of which
        positions a reward applies to (NOT ``np.isnan(rewards)``, which
        would silently mask in-model NaN bugs).

        Legacy back-compat: when a sample's ``applicable_rewards`` set
        is empty AND the sample carries no ``source`` / ``source_id``
        (i.e. came from the legacy single-source path that never updates
        the bookkeeping), every reward is treated as applicable.

        TODO(perf): the distributed gather here is independent of
        :meth:`collect_group_rewards` — both run once per advantage
        compute.  Future optimisation: pack ``source_id`` as an extra
        column in :meth:`collect_group_rewards`'s payload and recompute
        the mask LOCALLY on every rank from the gathered ``source_id``s
        + ``cfg._datasets_resolved`` (item 6's frozenset[int] cache).
        Saves one collective per epoch; not transformative on its own
        but composes with future per-source logging which would otherwise
        re-gather the same data.
        """
        R = len(reward_keys)
        B = len(samples)
        local_mask = np.zeros((R, B), dtype=bool)
        for j, s in enumerate(samples):
            applicable = s.applicable_rewards
            has_source = s.source is not None or s.source_id is not None
            if not applicable and not has_source:
                # Legacy single-source path: no source bookkeeping at all.
                # Honour the original "every reward applies" invariant so
                # existing configs are byte-identical.
                local_mask[:, j] = True
            else:
                for i, name in enumerate(reward_keys):
                    local_mask[i, j] = (name in applicable)

        if self.group_on_same_rank:
            return local_mask

        # distributed_k_repeat: pack as float (gather requires tensor),
        # transpose to (B, R) so row-i-after-gather still corresponds
        # to sample i (gather concatenates along dim 0 = ranks).
        packed = torch.from_numpy(local_mask.T.astype(np.float32)).to(self.accelerator.device)
        gathered = self.accelerator.gather(packed).cpu().numpy()  # (W*B, R)
        return gathered.T.astype(bool)  # (R, W*B)

    def _to_local(
        self,
        values: np.ndarray,
    ) -> torch.Tensor:
        """Convert collected values back to a local-rank tensor.

        When ``group_on_same_rank`` is ``True`` the array is already local and
        is simply converted.  Otherwise the array spans all ranks and is sliced
        to this rank's portion.
        """
        if not self.group_on_same_rank:
            values = torch.as_tensor(values).reshape(
                self.accelerator.num_processes, -1, *values.shape[1:]
            )[self.accelerator.process_index].to(self.accelerator.device)
        else:
            values = torch.as_tensor(values).to(self.accelerator.device)
        return values

    def _global_mean_std(self, values: np.ndarray) -> tuple:
        """Compute global mean and std for *values*.

        When ``group_on_same_rank`` is ``True`` the array only contains
        local-rank data, so we all-reduce ``(count, sum, sum_sq)`` in a
        single call to obtain the true global statistics.  Otherwise the
        array already spans all ranks (post-gather) and we compute
        directly with NumPy — no communication needed.
        """
        if self.group_on_same_rank:
            t = torch.tensor(
                [float(len(values)), float(np.sum(values)), float(np.sum(values ** 2))],
                device=self.accelerator.device,
            )
            t = self.accelerator.reduce(t, reduction="sum")  # 1 call, 3 scalars
            n, s, ss = t[0].item(), t[1].item(), t[2].item()
            mean = s / n
            std = max((ss / n - mean ** 2) ** 0.5, 1e-6)
        else:
            mean = float(np.mean(values))
            std = max(float(np.std(values)), 1e-6)
        return mean, std

    # ------------------------------------------------------------------
    # Batched metric reduction (mode-aware)
    # ------------------------------------------------------------------

    def _batch_reduce_stats(
        self, arrays: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Compute global ``{min, max, mean, std}`` for each named array.

        When ``group_on_same_rank`` the arrays are local shards and require
        cross-rank reduction via :func:`dm.global_tensor_stats_batch` (3
        all-reduce calls total, regardless of the number of arrays).

        Otherwise the arrays already span all ranks (post-gather) and stats
        are computed locally with plain NumPy.
        """
        if self.group_on_same_rank:
            tensors = {
                k: torch.from_numpy(np.asarray(v, dtype=np.float64))
                for k, v in arrays.items()
            }
            return global_tensor_stats_batch(self.accelerator, tensors)

        out: Dict[str, Dict[str, float]] = {}
        for k, v in arrays.items():
            v = np.asarray(v, dtype=np.float64)
            if len(v) == 0:
                out[k] = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
            else:
                out[k] = {
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                    "mean": float(np.mean(v)),
                    "std": max(float(np.std(v)), 1e-8),
                }
        return out

    def _metric_zero_std_ratio(
        self, rewards: np.ndarray, group_indices: np.ndarray
    ) -> float:
        """Fraction of groups with near-zero std — global-reduced when ``group_on_same_rank``."""
        if self.group_on_same_rank:
            return global_zero_std_ratio(self.accelerator, rewards, group_indices)
        return RewardProcessor.compute_group_zero_std_ratio(rewards, group_indices)

    # ------------------------------------------------------------------
    # Strategy: weighted sum (default GRPO)
    # ------------------------------------------------------------------

    def compute_weighted_sum(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool,
    ) -> torch.Tensor:
        """Compute advantages using the weighted-sum GRPO strategy.

        This is the standard GRPO advantage computation.  Each reward model's
        scores are multiplied by its configured weight and summed into a single
        aggregated reward per sample.  Advantages are then group-normalised
        (subtract per-group mean, divide by std).

        **Source-aware aggregation** (plan §6.4): the per-sample
        applicability matrix from :meth:`collect_applicability_mask` is
        the authoritative source of truth.  NaN at applicable positions
        is asserted to be a model bug (loud failure); NaN at
        non-applicable positions is honored as "this reward doesn't
        contribute to this sample".  Samples with NO applicable reward
        raise -- a misconfigured `RewardArguments.applicable_datasets` shouldn't
        silently produce zero advantages.

        **Algorithm**:

        1. **Collect** — call :meth:`collect_group_rewards` to obtain
           reward arrays and group assignments.
        2. **Aggregate** — compute
           ``r_agg[i] = sum_k(reward_k[i] * weight_k * applicable_k_i)``.
           NaN values at non-applicable positions are zero-weighted; NaN
           at applicable positions raises.
        3. **Group-normalise** — for each group *g*:
           ``advantage[i] = (r_agg[i] - mean(r_agg[g])) / std``
           where *std* is either the global std across all samples (when
           ``global_std=True``) or the per-group std (when ``global_std=False``).
        4. **To-local** — convert back to local-rank tensor via
           :meth:`_to_local`.
        5. **Store** — optionally write advantages into each sample's
           ``extra_kwargs['advantage']``.
        """
        gathered_rewards, group_indices = self.collect_group_rewards(samples, rewards)
        reward_keys = list(gathered_rewards.keys())
        # Authoritative applicability matrix, shape (R, S) where S matches
        # gathered_rewards' shape.
        applicable = self.collect_applicability_mask(samples, reward_keys)

        S = len(next(iter(gathered_rewards.values())))
        weights = np.array(
            [self.reward_weights[k] for k in reward_keys], dtype=np.float64
        )

        # Bug-detection: NaN at applicable position == reward-model bug.
        stack = np.stack(
            [gathered_rewards[k].astype(np.float64) for k in reward_keys], axis=0
        )  # (R, S)
        nan_mask = ~np.isfinite(stack)
        bug_positions = nan_mask & applicable
        if bug_positions.any():
            r_idx, s_idx = np.where(bug_positions)
            offenders = sorted({reward_keys[i] for i in r_idx})
            raise RuntimeError(
                f"NaN/Inf reward at APPLICABLE positions for reward(s) "
                f"{offenders} (sample indices {sorted(set(s_idx.tolist()))[:10]}{'...' if len(s_idx) > 10 else ''}). "
                "This is a reward-model bug, not a routing miss; "
                "aggregation refuses to silently mask it."
            )

        # Aggregate: weighted sum over applicable rewards only.
        # Non-applicable positions: contribute 0 (mask is False).
        contrib = np.where(applicable, stack, 0.0) * weights[:, None]
        aggregated_rewards = contrib.sum(axis=0)  # (S,)

        # Per-sample applicable weight sum -> sanity check.
        weight_per_s = (applicable * weights[:, None]).sum(axis=0)  # (S,)
        if (weight_per_s == 0).any():
            bad = np.where(weight_per_s == 0)[0].tolist()
            raise RuntimeError(
                "AdvantageProcessor: samples at indices "
                f"{bad[:10]}{'...' if len(bad) > 10 else ''} have NO applicable "
                "reward (weight_sum == 0). Check that "
                "`RewardArguments.applicable_datasets` covers every training source — "
                "at least one reward must apply to every source."
            )

        # Group-normalise
        _unique_ids, _counts = np.unique(group_indices, return_counts=True)
        advantages = np.zeros_like(aggregated_rewards, dtype=np.float64)

        if self.global_std:
            _, std = self._global_mean_std(aggregated_rewards)

        for group_id in np.unique(group_indices):
            mask = group_indices == group_id
            group_rewards = aggregated_rewards[mask]
            if len(group_rewards) != self.group_size:
                raise RuntimeError(
                    f"Group size mismatch: expected {self.group_size}, got {len(group_rewards)} "
                    f"for group {group_id} in rank {self.accelerator.process_index}"
                )
            mean = np.mean(group_rewards, axis=0, keepdims=True)
            if not self.global_std:
                std = max(np.std(group_rewards, axis=0, keepdims=True), 1e-6)
            advantages[mask] = (group_rewards - mean) / std

        self._pending_advantage_metrics = self._build_weighted_sum_log_data(
            gathered_rewards, group_indices, aggregated_rewards, advantages, samples,
            applicable=applicable, reward_keys=reward_keys,
        )

        # Scatter & store
        advantages = self._to_local(advantages)
        if store_to_samples:
            for sample, adv in zip(samples, advantages):
                sample.extra_kwargs["advantage"] = adv
        return advantages

    # ------------------------------------------------------------------
    # Strategy: GDPO
    # ------------------------------------------------------------------

    def compute_gdpo(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool,
    ) -> torch.Tensor:
        """Compute advantages using the GDPO (Group-wise DPO) strategy.

        Unlike :meth:`compute_weighted_sum`, which first aggregates all
        rewards into a single scalar then normalises, GDPO normalises each
        reward **independently** within its group before combining.  This
        prevents a single high-variance reward from dominating the advantage
        signal.

        **Source-aware aggregation**: per-reward group statistics are
        computed only over applicable group members.  Under the
        homogeneous-batch design (plan §6.7) a reward is either
        applicable to ALL K samples of a group or to NONE — so GDPO's
        per-(reward, group) normalisation either fires or is skipped
        entirely for that pair.  Mixed applicability within a group is
        an asserted error (caught upstream in
        ``_compute_groupwise_group``).

        **Algorithm**:

        1. **Collect** — call :meth:`collect_group_rewards` to obtain
           reward arrays and group assignments; also gather the
           per-(reward, sample) applicability matrix.
        2. **Per-reward, per-group, per-applicable normalisation**.
        3. **Combine** — sum per-reward normalised contributions.
        4. **Batch normalisation** — compute global mean and std and
           normalise.
        5. **To-local** — convert back to local-rank tensor.
        6. **Store** — optionally write advantages into each sample's
           ``extra_kwargs['advantage']``.
        """
        gathered_rewards, group_indices = self.collect_group_rewards(samples, rewards)
        reward_keys = list(gathered_rewards.keys())
        applicable = self.collect_applicability_mask(samples, reward_keys)
        S = len(group_indices)

        # Bug-detection: NaN at applicable position == reward-model bug.
        stack = np.stack(
            [gathered_rewards[k].astype(np.float64) for k in reward_keys], axis=0
        )
        nan_mask = ~np.isfinite(stack)
        bug_positions = nan_mask & applicable
        if bug_positions.any():
            r_idx, _s_idx = np.where(bug_positions)
            offenders = sorted({reward_keys[i] for i in r_idx})
            raise RuntimeError(
                f"GDPO: NaN/Inf reward at APPLICABLE positions for reward(s) "
                f"{offenders}. This is a reward-model bug, not a routing miss."
            )

        # Per-reward group-wise normalisation, restricted to applicable samples.
        all_reward_advantages = []
        for r_idx, key in enumerate(reward_keys):
            reward_array = gathered_rewards[key]
            r_applicable = applicable[r_idx]
            reward_adv = np.zeros_like(reward_array, dtype=np.float64)
            for group_id in np.unique(group_indices):
                gmask = group_indices == group_id
                in_group_applicable = gmask & r_applicable
                if not in_group_applicable.any():
                    # Reward doesn't apply to this entire group; contribute 0.
                    continue
                # Group homogeneity invariant: applicable_in_group is either
                # all-True (and equals gmask) or all-False (already returned).
                # The asserted invariant is enforced upstream; if it slips
                # we still proceed with whatever applicable subset exists.
                group_rewards = reward_array[in_group_applicable]
                mean = np.mean(group_rewards)
                std = max(np.std(group_rewards), 1e-6)
                # Write normalised values only at applicable positions in this group.
                reward_adv[in_group_applicable] = (group_rewards - mean) / std
            all_reward_advantages.append(reward_adv * self.reward_weights[key])

        # Combine and batch normalise. Samples with no applicable reward
        # would aggregate to 0 here -- guard upfront so misconfigs are loud.
        weight_per_s = (applicable * np.array(
            [self.reward_weights[k] for k in reward_keys], dtype=np.float64
        )[:, None]).sum(axis=0)
        if (weight_per_s == 0).any():
            bad = np.where(weight_per_s == 0)[0].tolist()
            raise RuntimeError(
                "GDPO: samples at indices "
                f"{bad[:10]}{'...' if len(bad) > 10 else ''} have NO applicable "
                "reward. Check `RewardArguments.applicable_datasets` coverage."
            )

        combined_advantages = np.sum(all_reward_advantages, axis=0)
        bn_mean, bn_std = self._global_mean_std(combined_advantages)
        advantages = (combined_advantages - bn_mean) / bn_std

        self._pending_advantage_metrics = self._build_gdpo_log_data(
            gathered_rewards, group_indices, advantages, bn_mean, bn_std, samples,
            applicable=applicable, reward_keys=reward_keys,
        )

        # Scatter & store
        advantages = self._to_local(advantages)
        if store_to_samples:
            for sample, adv in zip(samples, advantages):
                sample.extra_kwargs["advantage"] = adv
        return advantages

    # ------------------------------------------------------------------
    # Log payloads (trainers pass to ``log_data``)
    # ------------------------------------------------------------------

    def _build_base_log_stats(
        self,
        gathered_rewards: Dict[str, np.ndarray],
        group_indices: np.ndarray,
        applicable: Optional[np.ndarray],
        reward_keys: Optional[List[str]],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, bool]]]:
        """Shared boilerplate for both log-data builders.

        Returns (stat_arrays, r_applicable) where stat_arrays is ready
        for ``_batch_reduce_stats`` and r_applicable maps each reward
        key to its boolean mask over gathered samples.
        """
        keys_sorted = sorted(gathered_rewards.keys())
        if applicable is not None and reward_keys is not None:
            r_applicable = {k: applicable[reward_keys.index(k)] for k in keys_sorted}
        else:
            r_applicable = {k: np.ones(len(gathered_rewards[k]), dtype=bool) for k in keys_sorted}

        stat_arrays: Dict[str, np.ndarray] = {}
        for key in keys_sorted:
            mask_k = r_applicable[key]
            stat_arrays[f"reward_{key}"] = gathered_rewards[key][mask_k]

        for key in keys_sorted:
            mask_k = r_applicable[key]
            group_means, group_stds = RewardProcessor.compute_group_reward_stats(
                gathered_rewards[key][mask_k], group_indices[mask_k]
            )
            stat_arrays[f"reward_{key}_g_stds"] = group_stds
            stat_arrays[f"reward_{key}_g_means"] = group_means

        return stat_arrays, r_applicable

    def _unpack_per_reward_log_data(
        self,
        all_stats: Dict[str, Dict[str, float]],
        gathered_rewards: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Unpack per-reward stats common to both log-data builders."""
        _log_data: Dict[str, Any] = {}
        keys_sorted = sorted(gathered_rewards.keys())
        for key in keys_sorted:
            reward_stats = all_stats[f"reward_{key}"]
            _log_data[f"train/reward_{key}_mean"] = reward_stats["mean"]
            _log_data[f"train/reward_{key}_std"] = reward_stats["std"]

        for key in keys_sorted:
            group_std_stats = all_stats[f"reward_{key}_g_stds"]
            group_mean_stats = all_stats[f"reward_{key}_g_means"]
            _log_data[f"train/reward_{key}_group_std_mean"] = group_std_stats["mean"]
            _log_data[f"train/reward_{key}_group_std_max"] = group_std_stats["max"]
            _log_data[f"train/reward_{key}_group_std_min"] = group_std_stats["min"]
            _log_data[f"train/reward_{key}_group_mean_std"] = group_mean_stats["std"]
        return _log_data

    def _build_weighted_sum_log_data(
        self,
        gathered_rewards: Dict[str, np.ndarray],
        group_indices: np.ndarray,
        aggregated_rewards: np.ndarray,
        advantages: np.ndarray,
        samples: List[BaseSample],
        applicable: Optional[np.ndarray] = None,
        reward_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        stat_arrays, r_applicable = self._build_base_log_stats(
            gathered_rewards, group_indices, applicable, reward_keys
        )

        stat_arrays["reward_agg"] = aggregated_rewards
        agg_group_means, agg_group_stds = RewardProcessor.compute_group_reward_stats(
            aggregated_rewards, group_indices
        )
        stat_arrays["reward_agg_g_stds"] = agg_group_stds
        stat_arrays["reward_agg_g_means"] = agg_group_means
        stat_arrays["adv"] = advantages
        stat_arrays["adv_abs"] = np.abs(advantages)

        all_stats = self._batch_reduce_stats(stat_arrays)

        _log_data = self._unpack_per_reward_log_data(all_stats, gathered_rewards)
        _log_data["train/reward_mean"] = all_stats["reward_agg"]["mean"]
        _log_data["train/reward_std"] = all_stats["reward_agg"]["std"]

        agg_group_std_stats = all_stats["reward_agg_g_stds"]
        agg_group_mean_stats = all_stats["reward_agg_g_means"]
        _log_data["train/reward_group_std_mean"] = agg_group_std_stats["mean"]
        _log_data["train/reward_group_std_max"] = agg_group_std_stats["max"]
        _log_data["train/reward_group_mean_std"] = agg_group_mean_stats["std"]

        # Zero-std ratio (count-based; requires a separate all-reduce)
        _log_data["train/reward_zero_std_ratio"] = self._metric_zero_std_ratio(
            aggregated_rewards, group_indices
        )

        # Unpack advantage stats
        adv_stats = all_stats["adv"]
        _log_data["train/adv_min"] = adv_stats["min"]
        _log_data["train/adv_max"] = adv_stats["max"]
        _log_data["train/adv_abs_mean"] = all_stats["adv_abs"]["mean"]

        _log_data["train_samples"] = samples[:30]
        return _log_data

    def _build_gdpo_log_data(
        self,
        gathered_rewards: Dict[str, np.ndarray],
        group_indices: np.ndarray,
        advantages: np.ndarray,
        bn_mean: float,
        bn_std: float,
        samples: List[BaseSample],
        applicable: Optional[np.ndarray] = None,
        reward_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        stat_arrays, r_applicable = self._build_base_log_stats(
            gathered_rewards, group_indices, applicable, reward_keys
        )

        stat_arrays["adv"] = advantages
        stat_arrays["adv_abs"] = np.abs(advantages)

        all_stats = self._batch_reduce_stats(stat_arrays)

        _log_data = self._unpack_per_reward_log_data(all_stats, gathered_rewards)

        keys_sorted = sorted(gathered_rewards.keys())
        for key in keys_sorted:
            mask_k = r_applicable[key]
            _log_data[f"train/reward_{key}_zero_std_ratio"] = self._metric_zero_std_ratio(
                gathered_rewards[key][mask_k], group_indices[mask_k]
            )

        adv_stats = all_stats["adv"]
        _log_data.update({
            "train/batch_norm_mean": bn_mean,
            "train/batch_norm_std": bn_std,
            "train/adv_min": adv_stats["min"],
            "train/adv_max": adv_stats["max"],
            "train/adv_abs_mean": all_stats["adv_abs"]["mean"],
            "train_samples": samples[:30],
        })
        return _log_data
