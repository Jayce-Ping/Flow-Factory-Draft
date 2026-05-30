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

# src/flow_factory/hparams/args.py
"""
Main arguments class that encapsulates all configurations.

Supports loading from YAML files with nested structure.
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Optional
import yaml
from datetime import datetime
import math

from .abc import ArgABC
from .data_args import DataArguments
from .model_args import ModelArguments
from .scheduler_args import SchedulerArguments
from .training_args import TrainingArguments, EvaluationArguments, get_training_args_class
from .reward_args import RewardArguments, MultiRewardArguments
from .log_args import LogArguments
from ..utils.logger_utils import setup_logger
from ..utils.dist import get_world_size

logger = setup_logger(__name__, rank_zero_only=True)


@dataclass
class Arguments(ArgABC):
    """
    Main arguments class encapsulating all configurations.
    """
    
    launcher: Literal['accelerate'] = field(
        default='accelerate',
        metadata={"help": "Distributed launcher to use."},
    )
    config_file: str | None = field(
        default=None,
        metadata={"help": "Path to distributed configuration file."},
    )
    num_processes: int = field(
        default=1,
        metadata={"help": "Number of processes for distributed training."},
    )
    main_process_port: int = field(
        default=29500,
        metadata={"help": "Main process port for distributed training."},
    )
    mixed_precision: Optional[Literal['no', 'fp16', 'bf16']] = field(
        default='bf16',
        metadata={"help": "Mixed precision setting for training."},
    )
    # Runtime distributed fields (populated by reconcile_config after accelerator creation)
    process_index: Optional[int] = field(
        default=None,
        metadata={"help": "Global process index (set at runtime by reconcile_config)."},
    )
    local_process_index: Optional[int] = field(
        default=None,
        metadata={"help": "Local process index on this node (set at runtime)."},
    )
    num_machines: Optional[int] = field(
        default=None,
        metadata={"help": "Number of machines detected from environment."},
    )
    gpus_per_node: Optional[int] = field(
        default=None,
        metadata={"help": "GPUs per node (num_processes // num_machines)."},
    )
    machine_rank: Optional[int] = field(
        default=None,
        metadata={"help": "Rank of the current machine (set at runtime)."},
    )
    main_process_ip: Optional[str] = field(
        default=None,
        metadata={"help": "IP of the master node (set at runtime from env vars)."},
    )
    # Nested argument groups
    data_args: DataArguments = field(
        default_factory=DataArguments,
        metadata={"help": "Arguments for data configuration."},
    )
    model_args: ModelArguments = field(
        default_factory=ModelArguments,
        metadata={"help": "Arguments for model configuration."},
    )
    scheduler_args: SchedulerArguments = field(
        default_factory=SchedulerArguments,
        metadata={"help": "Arguments for scheduler configuration."},
    )
    training_args: TrainingArguments = field(
        default_factory=TrainingArguments,
        metadata={"help": "Arguments for training configuration."},
    )
    eval_args: EvaluationArguments = field(
        default_factory=EvaluationArguments,
        metadata={"help": "Arguments for evaluation configuration."},
    )
    log_args: LogArguments = field(
        default_factory=LogArguments,
        metadata={"help": "Arguments for logging configuration."},
    )
    reward_args: MultiRewardArguments = field(
        default_factory=MultiRewardArguments,
        metadata={"help": "Arguments for multiple reward configurations."},
    )
    eval_reward_args: Optional[MultiRewardArguments] = field(
        default=None,
        metadata={"help": "Arguments for multiple evaluation reward configurations."},
    )

    def __post_init__(self):
        if self.log_args.run_name is None:
            time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_args.run_name = f"{self.model_args.model_type}_{self.model_args.finetune_type}_{self.training_args.trainer_type}_{time_stamp}"

        # Canonicalize legacy `data.dataset_dir` into a synthetic 1-entry
        # `data.datasets` list so the rest of the pipeline only handles
        # one shape (collapses single/multi-source dispatch downstream).
        # This is a NO-OP when the user already configured `data.datasets`.
        self._canonicalize_legacy_dataset_dir()

        self._validate_dataset_routing()
        # Resolve `RewardArguments.datasets is None` -> concrete list of
        # applicable dataset names. Must run AFTER validation (so the
        # unknown-name check is against the user's raw input, not the
        # expanded list) and BEFORE any consumer reads the field.
        self._resolve_reward_dataset_routing()
        # With routing concrete, fail-fast when a training source has NO
        # applicable training reward — the user almost certainly meant
        # to add the source name to some reward's `datasets` list.
        self._validate_every_source_has_a_reward()
        # Stamp every `data.datasets[*]` entry with a stable monotonic
        # `source_id` matching its list position. Read by the dataloader
        # / trainer / gate everywhere a transport-friendly form of the
        # source string is wanted. Populates `data_args.source_id_to_name`.
        self._assign_source_ids()
        # Translate every `RewardArguments.datasets: List[str]` into the
        # `_datasets_resolved: frozenset[int]` cache used by the hot-path
        # gate. Trivially deterministic on every rank because IDs come
        # from the same shared config.
        self._resolve_reward_dataset_ids()
        self._resolve_scheduler_sde_defaults()
        self._resolve_sampler_type()
        self._align_batch_geometry()
        self._adjust_gradient_accumulation()

    def _canonicalize_legacy_dataset_dir(self) -> None:
        """Promote legacy ``data.dataset_dir`` into a synthetic ``data.datasets``.

        After this method returns, ``data.datasets`` is the single source
        of truth for everywhere downstream — single- and multi-source
        configurations flow through identical code paths.

        Rules:

        * If ``data.datasets`` is already populated, do nothing (the
          mutual-exclusion validator runs next and will reject any
          conflict).
        * Otherwise build a 1-entry list with ``name="default"``,
          ``dataset_dir`` from the legacy field, and:
            - ``train`` enabled (every legacy run had a training dataset).
            - ``eval`` enabled iff ``test.jsonl`` / ``test.txt`` exists
              under the legacy ``dataset_dir``; otherwise ``eval=None``.

        We set ``self._legacy_canonicalized = True`` so the mutex
        validator knows this synthetic list does NOT conflict with the
        user's legacy ``dataset_dir`` (they're the same source described
        twice — by definition).
        """
        self._legacy_canonicalized = False
        if self.data_args.datasets:
            return  # explicit user config wins
        from .dataset_args import DatasetArguments, DatasetTrainSpec, DatasetEvalSpec
        from dataclasses import fields as _fields
        default_dataset_dir = next(
            (f.default for f in _fields(self.data_args.__class__) if f.name == "dataset_dir"),
            None,
        )
        if self.data_args.dataset_dir == default_dataset_dir:
            return  # no legacy field -> let downstream handle the no-data error
        # Probe for a test split so eval-enabled is set correctly for
        # legacy runs that DO have a test.jsonl.
        from ..data_utils.dataset import GeneralDataset
        eval_spec = (
            DatasetEvalSpec()
            if GeneralDataset.check_exists(self.data_args.dataset_dir, "test")
            else None
        )
        self.data_args.datasets = [
            DatasetArguments(
                name="default",
                dataset_dir=self.data_args.dataset_dir,
                image_dir=self.data_args.image_dir,
                video_dir=self.data_args.video_dir,
                audio_dir=self.data_args.audio_dir,
                train=DatasetTrainSpec(weight=1),
                eval=eval_spec,
            )
        ]
        self._legacy_canonicalized = True

    def _assign_source_ids(self) -> None:
        """Stamp each ``data.datasets[*]`` entry with a stable monotonic id.

        The id is just the entry's index in ``data.datasets``; with YAML
        being deterministic and every rank receiving the same config dict,
        this gives identical id assignments on every rank with no extra
        sync.

        No-op when ``data.datasets`` is unset (legacy single-source mode);
        consumers fall back to the legacy "source is None -> applies to
        all" path.
        """
        if not self.data_args.datasets:
            return
        for source_id, ds in enumerate(self.data_args.datasets):
            ds.source_id = source_id

    def _resolve_reward_dataset_ids(self) -> None:
        """Populate ``RewardArguments._datasets_resolved`` (``frozenset[int]``).

        Hot-path callers (the reward gate) do an ``int in frozenset[int]``
        membership check instead of an ``str in List[str]`` scan plus a
        string allocation for ``__source__``. Builds the cache once at
        config-load time so the runtime gate stays O(1).

        Consumers that bypass ``Arguments`` entirely (e.g. tests
        constructing a stand-alone ``RewardProcessor``) leave the cache
        as ``None`` and the gate falls back to the string form — same
        observable behavior.
        """
        if not self.data_args.datasets:
            return
        name_to_id = self.data_args.source_name_to_id
        for rc in list(self.reward_args) + list(self.eval_reward_args or []):
            # `datasets` was resolved to a concrete `List[str]` upstream by
            # `_resolve_reward_dataset_routing`; the names are guaranteed to
            # be in the registry because they passed `_validate_dataset_routing`.
            assert rc.datasets is not None, (
                "Internal error: RewardArguments.datasets not resolved before "
                "_resolve_reward_dataset_ids; check Arguments.__post_init__ ordering."
            )
            rc._datasets_resolved = frozenset(name_to_id[n] for n in rc.datasets)

    def _resolve_reward_dataset_routing(self) -> None:
        """Replace ``RewardArguments.datasets is None`` with the explicit list.

        After this method returns, every reward's ``datasets`` field is a
        concrete ``List[str]``:

        * ``datasets=None`` (user wrote ``null`` or omitted the field) is
          replaced with the full list of applicable dataset names for that
          reward's side (training rewards -> training-source names; eval
          rewards -> eval-source names).
        * ``datasets=[]`` (explicit empty list) is left as-is and a warning
          is emitted, because the reward will never fire — almost certainly
          a misconfiguration but the user is allowed to express it.
        * ``datasets=[name1, ...]`` (explicit non-empty list) is left as-is
          (validation already ran).

        Consumers (RewardProcessor, MultiRewardLoader,
        AdvantageProcessor, log builders) can therefore read ``datasets``
        as a concrete list without any ``is None`` short-circuit.
        """
        train_names = [d.name for d in self.data_args.training_datasets]
        eval_names = [d.name for d in self.data_args.eval_datasets]

        def _resolve_one_side(reward_args, applicable_universe, side: str) -> None:
            if not reward_args:
                return
            for rc in reward_args:
                if rc.datasets is None:
                    # Eager copy: rewards never share their `datasets` list
                    # so mutating one cannot bleed into another.
                    rc.datasets = list(applicable_universe)
                elif len(rc.datasets) == 0:
                    logger.warning(
                        f"{side} reward '{rc.name}' has `datasets: []` — "
                        "it will never fire. If you intended 'apply to every "
                        "dataset', omit the field or set it to null."
                    )
                # else: explicit non-empty list, leave alone (validation
                # already confirmed every name is in the universe).

        _resolve_one_side(self.reward_args, train_names, side="Training")
        _resolve_one_side(self.eval_reward_args, eval_names, side="Eval")

    def _validate_every_source_has_a_reward(self) -> None:
        """Inverse routing check: every training/eval source must be covered.

        ``_resolve_reward_dataset_routing`` already turned every reward's
        ``datasets`` into a concrete list.  Now, for each declared
        training source, check that AT LEAST ONE training reward routes
        to it; same for the eval side.  A source with no applicable
        reward would silently produce all-NaN advantages at runtime
        (caught only later by ``AdvantageProcessor``'s ``weight_sum == 0``
        guard) — much better to fail at config-load time with a clear
        message naming the missing source.

        Skipped when no rewards are configured on the relevant side
        (legitimate for eval-only or sample-only runs).
        """
        train_names = [d.name for d in self.data_args.training_datasets]
        eval_names = [d.name for d in self.data_args.eval_datasets]

        def _check_side(reward_args, source_names, side: str) -> None:
            if not reward_args or not source_names:
                return
            covered: set[str] = set()
            for rc in reward_args:
                # `rc.datasets` is concrete post-resolver.
                covered.update(rc.datasets or [])
            uncovered = [n for n in source_names if n not in covered]
            if uncovered:
                raise ValueError(
                    f"{side} source(s) {uncovered!r} have NO applicable {side.lower()} "
                    f"reward — every reward's `datasets` field excludes them. Either "
                    f"add at least one of these names to a reward's `datasets`, set "
                    f"that reward's `datasets` to `null` (= apply to every source), "
                    f"or drop the dataset entry from `data.datasets`."
                )

        _check_side(self.reward_args, train_names, side="Training")
        _check_side(self.eval_reward_args, eval_names, side="Eval")

    def _validate_dataset_routing(self) -> None:
        """Validate the unified ``data.datasets`` schema and reward routing.

        Single source of truth for cross-cutting checks involving the new
        :class:`DatasetArguments` list.  Covers:

        * Mutual exclusion with the legacy single ``data.dataset_dir`` path
          when the user has actually customized that field (we cannot
          reject the dataclass default ``"data"`` itself, as many configs
          set it explicitly).
        * Uniqueness of dataset names across the ``data.datasets`` list.
        * ``train.weight > 0`` for every training participant.
        * Cross-validation of every ``RewardArguments.datasets`` entry
          against the union of declared training / eval dataset names.
          Training rewards must reference training-source names; eval
          rewards must reference eval-source names.
        """
        tds_unified = self.data_args.datasets or []
        if not tds_unified:
            return

        # Mutual exclusion with the legacy single-source `data.dataset_dir`.
        # Detect "user actually customized dataset_dir" by comparing with
        # the dataclass default; this lets us tolerate configs that always
        # write `dataset_dir: data` (the default) as a noop.
        # Skip this check when we synthesised `data.datasets` ourselves
        # in `_canonicalize_legacy_dataset_dir` — the legacy `dataset_dir`
        # IS the source for the synthetic single entry, not a conflict.
        from dataclasses import fields as _fields
        default_dataset_dir = next(
            (f.default for f in _fields(self.data_args.__class__) if f.name == "dataset_dir"),
            None,
        )
        if (
            not getattr(self, "_legacy_canonicalized", False)
            and default_dataset_dir is not None
            and self.data_args.dataset_dir != default_dataset_dir
        ):
            raise ValueError(
                "Both `data.dataset_dir` (custom value: "
                f"{self.data_args.dataset_dir!r}) and `data.datasets` are set. "
                "These are mutually exclusive — pick one. If you intended "
                "multi-source training, remove `data.dataset_dir`."
            )

        # Unique names within the unified list.
        names = [d.name for d in tds_unified]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"Duplicate `data.datasets` names: {dupes}. "
                "Each dataset entry must have a unique name."
            )

        # Weight > 0 for every training participant.
        bad_weights: list[tuple[str, float | None]] = []
        for d in tds_unified:
            if not d.is_training_source:
                continue
            assert d.train is not None  # narrows for type-checkers
            w = d.train.weight
            if w is None or w <= 0:
                bad_weights.append((d.name, w))
        if bad_weights:
            raise ValueError(
                f"All training datasets must have train.weight > 0; got: {bad_weights}. "
                "Set `train.weight: 1.0` for uniform mixing."
            )

        # Cross-validate reward `datasets` references.
        train_names = {d.name for d in tds_unified if d.is_training_source}
        eval_names = {d.name for d in tds_unified if d.is_eval_source}

        # Training rewards: `datasets` must reference TRAINING-source names.
        for rc in self.reward_args:
            if rc.datasets is None:
                continue
            if not train_names:
                raise ValueError(
                    f"Reward '{rc.name}' has datasets={rc.datasets!r} but no training "
                    "dataset is configured under `data.datasets`. Either remove "
                    "`datasets` from this reward or define training datasets it can route to."
                )
            unknown = set(rc.datasets) - train_names
            if unknown:
                raise ValueError(
                    f"Reward '{rc.name}' references unknown training dataset(s): "
                    f"{sorted(unknown)}. Valid training dataset names: {sorted(train_names)}."
                )

        # Eval rewards: `datasets` must reference EVAL-source names.
        if self.eval_reward_args and eval_names:
            for rc in self.eval_reward_args:
                if rc.datasets is None:
                    continue
                unknown = set(rc.datasets) - eval_names
                if unknown:
                    raise ValueError(
                        f"Eval reward '{rc.name}' references unknown eval dataset(s): "
                        f"{sorted(unknown)}. Valid eval dataset names: {sorted(eval_names)}."
                    )

    def _resolve_sampler_type(self) -> None:
        """Choose the distributed sampler strategy.

        Writes the resolved value back to ``data_args.sampler_type`` so all
        downstream consumers (``get_data_sampler``, ``RewardProcessor``,
        ``AdvantageProcessor``) read a concrete choice — never ``"auto"``.

        Rules:
        - DGPO is always forced to ``group_distributed``.
        - For non-DGPO trainers, explicit user choice is respected, unless
          ``distributed_k_repeat`` / ``group_distributed`` conflicts with async
          rewards (hard override to ``group_contiguous``).
        - ``"auto"`` prefers ``group_contiguous`` (minimal communication) and
          falls back to ``distributed_k_repeat`` only when the stricter
          geometric constraints cannot be satisfied without padding
          ``unique_sample_num_per_epoch``.
        """

        # 1. Detect async rewards
        all_configs = list(self.reward_args or [])
        if self.eval_reward_args:
            all_configs += list(self.eval_reward_args)

        self._has_async_rewards = any(getattr(cfg, 'async_reward', False) for cfg in all_configs)

        # 2. Resolve sampler type
        ta = self.training_args
        user_choice = self.data_args.sampler_type

        trainer_type = str(ta.trainer_type).lower()

        if (
            user_choice in {"distributed_k_repeat", "group_distributed"}
            and self._has_async_rewards
            and trainer_type != "dgpo"
        ):
            # Hard override to `group_contiguous` for async rewards
            # In fact, only group-wise async rewards require `group_contiguous` sampler
            # For pointwise async rewards, distributed_k_repeat is still valid
            # but for simplicity, we enforce `group_contiguous` for all async rewards
            logger.warning(
                "Async rewards require 'group_contiguous' sampler. "
                f"Overriding '{user_choice}' → 'group_contiguous'."
            )
            self.data_args.sampler_type = "group_contiguous"
        
        if user_choice == "auto" and trainer_type != "dgpo":
            # auto: prefer `group_contiguous` (all K copies on same rank → no cross-rank all-gather for rewards/advantages),
            # fall back to `distributed_k_repeat` (all K copies scattered across ranks → cross-rank all-gather for rewards/advantages)
            # There are two geometric constraints:
            #   - `groups_per_rank_ok`: unique_sample_num_per_epoch % num_replicas == 0
            #   - `local_batch_tiling_ok`: (unique_sample_num_per_epoch // num_replicas) * group_size % per_device_batch_size == 0
            world_size = get_world_size()
            m = ta.unique_sample_num_per_epoch
            groups_per_rank_ok = (m % world_size == 0)
            local_batch_tiling_ok = (m // world_size * ta.group_size % ta.per_device_batch_size == 0)
            # GroupContiguousSampler's requires both while DistributedKRepeatSampler's only requires the local batch tiling constraint.
            # If `groups_per_rank_ok` is not satisfied but `local_batch_tiling_ok` is satisfied,
            # use `distributed_k_repeat` to satisfy the constraint.
            if not groups_per_rank_ok and local_batch_tiling_ok:
                self.data_args.sampler_type = "distributed_k_repeat"
            else:
                # Otherwise, use `group_contiguous`
                # and later `_align_batch_geometry()` will adjust `unique_sample_num_per_epoch` to satisfy the geometric constraints above.
                self.data_args.sampler_type = "group_contiguous"

        if trainer_type == "dgpo" and self.data_args.sampler_type != "group_distributed":
            logger.warning(
                "DGPO requires sampler_type='group_distributed'. "
                f"Overriding '{self.data_args.sampler_type}' -> 'group_distributed'."
            )
            self.data_args.sampler_type = "group_distributed"


    def _align_batch_geometry(self) -> None:
        """Align ``unique_sample_num_per_epoch`` (and, for ``group_distributed``,
        ``group_size``) to sampler constraints, then recompute derived batch
        quantities.

        Must run after ``_resolve_sampler_type()`` so the sampler choice is
        finalised.  Overwrites placeholder values set in
        ``TrainingArguments.__post_init__``.

        Each sampler enforces slightly different divisibility constraints on
        ``unique_sample_num_per_epoch * group_size`` vs.
        ``num_replicas * per_device_batch_size`` (and, optionally,
        ``gradient_step_per_epoch``); the actual logic lives in the
        per-sampler helpers below.  This method only dispatches to the right
        one and then updates derived quantities (``num_batches_per_epoch`` +
        ``gradient_accumulation_steps``).
        """
        sampler_type = self.data_args.sampler_type
        if sampler_type == "distributed_k_repeat":
            self._align_for_distributed_k_repeat()
        elif sampler_type == "group_contiguous":
            self._align_for_group_contiguous()
        elif sampler_type == "group_distributed":
            self._align_for_group_distributed()
        else:
            raise ValueError(
                f"Unknown sampler_type={sampler_type!r}; "
                "expected one of {'distributed_k_repeat', 'group_contiguous', 'group_distributed'}."
            )
        self._recompute_derived_batch_quantities()

    # ---------------------------------------------------------------------
    # Shared alignment primitives (used by the three per-sampler helpers).
    # ---------------------------------------------------------------------
    @staticmethod
    def _round_up_to_step(value: int, step: int) -> int:
        """Smallest multiple of ``step`` that is ``>= value``."""
        return ((value + step - 1) // step) * step

    def _base_unique_sample_step(self) -> int:
        """Minimum alignment step for ``unique_sample_num_per_epoch``.

        Ensures::

            unique_sample_num_per_epoch * group_size
                ≡ 0  (mod  num_replicas * per_device_batch_size [* gradient_step_per_epoch])

        i.e. ``unique_sample_num_per_epoch`` must be a multiple of::

            (num_replicas * per_device_batch_size [* gradient_step_per_epoch])
            / gcd(group_size, num_replicas * per_device_batch_size)

        This same base step powers all three samplers; ``group_contiguous``
        further ``lcm``-s it with ``num_replicas``.
        """
        ta = self.training_args
        sample_num_per_iteration = get_world_size() * ta.per_device_batch_size
        base = sample_num_per_iteration // math.gcd(ta.group_size, sample_num_per_iteration)
        if not ta._manual_gradient_accumulation_steps:
            base *= ta.gradient_step_per_epoch
        return base

    def _warn_and_assign_unique_sample_num(
        self,
        new_unique_sample_num: int,
        sampler_name: str,
        extra_line: str = "",
    ) -> None:
        """Emit a standardised "adjusted unique_sample_num_per_epoch" warning
        and assign the new value.

        ``extra_line`` lets a caller prepend a sampler-specific constraint line
        (e.g. ``group_contiguous``'s ``unique_sample_num_per_epoch %
        num_replicas == 0`` requirement) above the shared constraint
        description.
        """
        ta = self.training_args
        world_size = get_world_size()
        constraint_suffix = (
            f" * gradient_step_per_epoch({ta.gradient_step_per_epoch}))"
            if not ta._manual_gradient_accumulation_steps
            else ")"
        )
        prefix = f"{extra_line}\n  " if extra_line else "  "
        logger.warning(
            f"{sampler_name}: adjusted `unique_sample_num_per_epoch` "
            f"from {ta.unique_sample_num_per_epoch} to {new_unique_sample_num} to satisfy:\n"
            f"{prefix}unique_sample_num_per_epoch({new_unique_sample_num}) "
            f"* group_size({ta.group_size}) "
            f"% (num_replicas({world_size}) "
            f"* per_device_batch_size({ta.per_device_batch_size})"
            + constraint_suffix
            + " == 0"
        )
        ta.unique_sample_num_per_epoch = new_unique_sample_num

    def _recompute_derived_batch_quantities(self) -> None:
        """Recompute ``num_batches_per_epoch`` and (in ``auto`` grad-accum
        mode) ``gradient_accumulation_steps`` from the now-aligned
        ``(unique_sample_num_per_epoch, group_size)`` pair.

        Called by the dispatcher after any per-sampler alignment; each of
        the three helpers thus only needs to adjust the inputs, not the
        outputs.
        """
        ta = self.training_args
        sample_num_per_iteration = get_world_size() * ta.per_device_batch_size
        ta.num_batches_per_epoch = (
            (ta.unique_sample_num_per_epoch * ta.group_size)
            // sample_num_per_iteration
        )
        if not ta._manual_gradient_accumulation_steps:
            ta.gradient_accumulation_steps = ta.compute_gradient_accumulation_steps(
                ta.num_batches_per_epoch,
            )

    # ---------------------------------------------------------------------
    # Per-sampler alignment (identical shape: adjust inputs, then unique_sample_num).
    # ---------------------------------------------------------------------
    def _align_for_distributed_k_repeat(self) -> None:
        """``DistributedKRepeatSampler``: only the base
        ``unique_sample_num_per_epoch * group_size`` divisibility constraint.
        """
        self._align_unique_sample_num(
            sampler_name="DistributedKRepeatSampler",
            base_step_func=self._base_unique_sample_step,
        )

    def _align_for_group_contiguous(self) -> None:
        """``GroupContiguousSampler``: base constraint + ``unique_sample_num_per_epoch % num_replicas == 0``."""
        world_size = get_world_size()

        def _step() -> int:
            return math.lcm(self._base_unique_sample_step(), world_size)

        # Per-sampler `extra_line` so the warning explains why the alignment
        # bumps `M` further than the base step.  Lazily formatted inside the
        # primitive so the per-sampler-specific text only fires on the
        # legacy (single-source) path; the multi-source path includes a
        # source-breakdown line of its own.
        def _extra(new_M: int) -> str:
            return (
                f"  1) unique_sample_num_per_epoch({new_M}) "
                f"% num_replicas({world_size}) == 0\n  2)"
            )

        self._align_unique_sample_num(
            sampler_name="GroupContiguousSampler",
            base_step_func=_step,
            extra_line_for_legacy=_extra,
        )

    def _align_for_group_distributed(self) -> None:
        """``GroupDistributedSampler``: first align ``group_size`` so that
        ``group_size % num_replicas == 0`` **and**
        ``(num_replicas * per_device_batch_size) % group_size == 0``; then
        do the base ``unique_sample_num_per_epoch`` alignment with the
        (possibly bumped) ``group_size``.

        **group_size alignment** — any valid ``group_size`` has the form
        ``num_replicas * d`` where ``d`` is a divisor of
        ``per_device_batch_size``, so we enumerate divisors of
        ``per_device_batch_size`` in O(√per_device_batch_size) and pick the
        smallest ``d`` with ``d >= ceil(group_size / num_replicas)``.
        When ``group_size > num_replicas * per_device_batch_size`` no
        solution exists.
        """
        ta = self.training_args
        if ta.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {ta.group_size}.")

        world_size = get_world_size()
        per_device_batch_size = ta.per_device_batch_size
        sample_num_per_iteration = world_size * per_device_batch_size
        original_group_size = ta.group_size

        if original_group_size > sample_num_per_iteration:
            raise ValueError(
                "sampler_type='group_distributed' requires "
                "`group_size <= num_replicas * per_device_batch_size`; "
                f"got group_size={original_group_size}, "
                f"num_replicas * per_device_batch_size={sample_num_per_iteration}."
            )

        # Smallest new group_size = num_replicas * d  where
        # d divides per_device_batch_size  and  d >= ceil(group_size / num_replicas).
        min_copies_per_rank = -(-original_group_size // world_size)  # ceil division
        best_copies_per_rank = per_device_batch_size                 # fallback (always valid)
        i = 1
        while i * i <= per_device_batch_size:
            if per_device_batch_size % i == 0:
                for d in (i, per_device_batch_size // i):
                    if min_copies_per_rank <= d < best_copies_per_rank:
                        best_copies_per_rank = d
            i += 1
        new_group_size = world_size * best_copies_per_rank

        if new_group_size != original_group_size:
            logger.warning(
                "sampler_type='group_distributed' requires `group_size %% num_replicas == 0` "
                "and `(num_replicas * per_device_batch_size) %% group_size == 0`; "
                "auto-adjusting group_size from %d to %d "
                "(num_replicas=%d, per_device_batch_size=%d).",
                original_group_size,
                new_group_size,
                world_size,
                per_device_batch_size,
            )
            ta.group_size = new_group_size

        # Now do the shared unique_sample_num_per_epoch alignment with the aligned group_size.
        self._align_unique_sample_num(
            sampler_name="GroupDistributedSampler",
            base_step_func=self._base_unique_sample_step,
        )

    # ---------------------------------------------------------------------
    # Shared alignment primitive (legacy + multi-source dispatch)
    # ---------------------------------------------------------------------
    def _align_unique_sample_num(
        self,
        *,
        sampler_name: str,
        base_step_func,
        extra_line_for_legacy=None,
    ) -> None:
        """Shared core for the three per-sampler ``_align_for_*`` helpers.

        Dispatches on whether ``data.datasets`` declares more than one
        training-eligible source:

        * **Legacy / single-source** — round ``unique_sample_num_per_epoch``
          up to the next multiple of ``base_step_func()``; emit the
          standard "adjusted" warning when the value changes.  Behaviour
          is byte-identical to the pre-refactor code paths.

        * **Multi-source (N >= 2)** — exact allocator: round the total
          up to a multiple of ``step * sum(weights)`` so per-source
          ``M_i = M_total * w_i / sum(weights)`` is itself a positive
          integer multiple of ``step``.  Geometric consequence: every
          batch comes from a single source (no per-batch homogeneity
          contract to defend at runtime; it falls out of the math).

        ``extra_line_for_legacy`` is a one-shot text producer for the
        legacy path's per-sampler constraint description; it's only
        invoked on the single-source code path so the multi-source
        warning can supply its own per-source breakdown without
        duplicating the sampler-specific blurb.
        """
        ta = self.training_args
        step = base_step_func()
        tds = self.data_args.training_datasets  # property; List[DatasetArguments]
        N = len(tds)

        if N <= 1:
            new_M = self._round_up_to_step(ta.unique_sample_num_per_epoch, step)
            if new_M != ta.unique_sample_num_per_epoch:
                extra = extra_line_for_legacy(new_M) if extra_line_for_legacy else ""
                self._warn_and_assign_unique_sample_num(new_M, sampler_name, extra)
            # Stash a 1-entry partition when single-source multi-dataset
            # mode is configured (N == 1) — keeps the data layer's
            # downstream lookup uniform.  None when fully legacy.
            ta._per_source_unique_sample_num = (
                {tds[0].name: ta.unique_sample_num_per_epoch}
                if N == 1
                else None
            )
            # Also stamp the resolved M_i onto each training spec so
            # `print(config)` reflects the final geometry.
            if N == 1:
                tds[0].train.unique_sample_num_per_epoch = ta.unique_sample_num_per_epoch  # type: ignore[union-attr]
            return

        # Multi-source partition (N >= 2).
        # Step 1: target total = M rounded up so that
        #   (a) M_total is a multiple of `step` (per-source sampler constraint), and
        #   (b) M_total is a multiple of `step * sum(weights)` so the per-source
        #       allocation `M_i = M_total * w_i / sum(weights)` is itself a positive
        #       integer multiple of `step` -- no remainder loop, no rounding drift,
        #       and every batch is geometrically guaranteed to come from a single
        #       source (`num_batches_per_epoch % sum(weights) == 0` -> the scheduler
        #       can place each source's quota exactly).
        original_M = ta.unique_sample_num_per_epoch
        weights = [int(d.train.weight) for d in tds]   # type: ignore[union-attr]
        W_sum = sum(weights)
        if W_sum <= 0:
            # _validate_dataset_routing should have caught this; assert defensively.
            raise ValueError(f"sum(train.weight) must be > 0; got weights={weights}.")
        partition_step = step * W_sum
        target_total = max(self._round_up_to_step(original_M, partition_step), partition_step)
        # Exact allocation: M_i = (target_total / sum(w)) * w_i = step * j * w_i.
        per_source_unit = target_total // W_sum   # multiple of `step` by construction
        partition = {d.name: per_source_unit * w for d, w in zip(tds, weights)}
        final_total = sum(partition.values())
        assert final_total == target_total, (final_total, target_total, partition)

        if final_total != original_M:
            breakdown = ", ".join(f"{n}={v}" for n, v in sorted(partition.items()))
            extra = (
                f"  multi-source partition ({len(tds)} sources, "
                f"sum(weight)={W_sum}): {breakdown}\n  "
            )
            self._warn_and_assign_unique_sample_num(final_total, sampler_name, extra)
        # Always stash the partition for the data layer to consume.
        ta._per_source_unique_sample_num = partition
        # Also stamp resolved M_i onto each training spec so the printed
        # config shows the final geometry per-source.
        for d in tds:
            assert d.train is not None
            d.train.unique_sample_num_per_epoch = partition[d.name]

    @staticmethod
    def _partition_unique_sample_num(
        target_total: int,
        step: int,
        training_datasets,
    ) -> "dict[str, int]":
        """Partition ``target_total`` across training datasets by ``train.weight``.

        Algorithm (deterministic, total over all valid inputs):

        1. Raw allocation:   ``M_i_raw = target_total * w_i / sum(w_j)``.
        2. Step-floor:       ``M_i = max(step, floor(M_i_raw / step) * step)``.
           The ``max(step, ...)`` floor prevents ``weight=1`` from rounding
           to zero (which would silently drop the source).
        3. Distribute the deficit ``target_total - sum(M_i)`` by ``±step``
           bumps, ranked by largest fractional remainder
           ``(M_i_raw - M_i) * sign(deficit)``.  Names break ties so the
           allocation is reproducible across runs / ranks.

        Edge cases (all flow through the same path; no try/except):

        * ``N == 1`` — caller short-circuits before reaching here.
        * Highly uneven weights (e.g. ``(1, 999)``) — the small source
          still gets the ``step`` minimum.
        * Total too small (``target_total < N * step``) — caller already
          bumped ``target_total`` to ``N * step`` before calling.
        """
        weights = {d.name: float(d.train.weight) for d in training_datasets}  # type: ignore[union-attr]
        total_w = sum(weights.values())
        # Step 1+2: raw allocation -> step-floor with `step` minimum.
        raw = {n: target_total * w / total_w for n, w in weights.items()}
        partition = {n: max(step, (int(r) // step) * step) for n, r in raw.items()}

        # Step 3: distribute deficit by largest fractional remainder.
        deficit = target_total - sum(partition.values())
        # Sort by `name` first so deterministic across runs; then by remainder.
        names_sorted = sorted(weights.keys())

        while deficit != 0:
            sign = 1 if deficit > 0 else -1
            # Sort sources by remainder relevance:
            #   - sign > 0 (need to add)    -> largest positive remainder first
            #   - sign < 0 (need to remove) -> largest "overshoot" first (most-negative remainder)
            ordered = sorted(
                names_sorted,
                key=lambda n: (raw[n] - partition[n]) * sign,
                reverse=True,
            )
            progressed = False
            for n in ordered:
                new_val = partition[n] + sign * step
                if new_val < step:
                    continue          # never go below the step minimum
                partition[n] = new_val
                deficit -= sign * step
                progressed = True
                if deficit == 0:
                    break
            if not progressed:
                # Mathematically unreachable when target_total >= N*step,
                # but guard against pathological inputs (e.g., negative
                # target) so we never spin forever.
                raise RuntimeError(
                    "Internal error: could not partition "
                    f"unique_sample_num_per_epoch={target_total} across "
                    f"{len(weights)} sources at step={step}; "
                    f"current partition={partition}, deficit={deficit}."
                )
        return partition



    def _adjust_gradient_accumulation(self) -> None:
        """Adjust gradient accumulation for per-timestep losses.

        Must run AFTER `_align_batch_geometry()` which finalises the base
        gradient_accumulation_steps from the aligned M.
        Skipped when gradient_accumulation_steps is manually set — the user
        value is treated as final.
        """
        if not self.training_args._manual_gradient_accumulation_steps:
            num_train_timesteps = self.training_args.get_num_train_timesteps(self)
            self.training_args.gradient_accumulation_steps *= num_train_timesteps
        else:
            logger.info(
                f"`gradient_accumulation_steps` manually set to "
                f"{self.training_args.gradient_accumulation_steps}. "
                f"`gradient_step_per_epoch` will not be used for "
                f"gradient accumulation computation."
            )

    def _resolve_scheduler_sde_defaults(self) -> None:
        """Fill `sde_steps` / `num_sde_steps` when YAML uses null.

        Matches runtime SDE schedulers: default step indices are
        ``0 .. num_inference_steps-2`` (all steps except the last). When
        ``num_sde_steps`` is null, use the full resolved pool (same as the
        scheduler property default).

        Skipped for ODE dynamics (no stochastic steps).
        """
        sched = self.scheduler_args
        if sched.dynamics_type == 'ODE':
            return

        n_inf = self.training_args.num_inference_steps
        if sched.sde_steps is None:
            sched.sde_steps = list(range(max(0, n_inf - 1)))
        if sched.num_sde_steps is None:
            sched.num_sde_steps = len(sched.sde_steps)
        if sched.num_sde_steps <= 0:
            raise ValueError(
                "scheduler.num_sde_steps must be positive after resolving nulls; "
                f"got `num_sde_steps`={sched.num_sde_steps!r}, `sde_steps`={sched.sde_steps!r}, "
                f"`num_inference_steps`={n_inf!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {}

        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if isinstance(value, ArgABC):
                # Remove '_args' suffix for nested configs
                key = f.name.replace('_args', '')
                result[key] = value.to_dict()
            elif isinstance(value, list) and value and isinstance(value[0], ArgABC):
                # List of ArgABC instances (e.g. eval_datasets)
                result[f.name] = [item.to_dict() for item in value]
            else:
                result[f.name] = value

        extras = result.pop("extra_kwargs", {})
        result.update(extras)
        return result

    @classmethod
    def from_dict(cls, args_dict: dict[str, Any]) -> Arguments:
        """Create Arguments instance from dictionary.

        Side effects: a top-level ``eval_datasets:`` key (legacy schema
        introduced earlier on this branch) is migrated into
        ``data.datasets[*].eval`` with a :class:`DeprecationWarning`.
        The migration shim is removed in a future release; in the
        meantime callers should prefer the unified ``data.datasets:``
        list (each entry opts into training and/or evaluation via its
        ``train`` / ``eval`` sub-blocks).
        """
        # 0. Top-level eval_datasets deprecation shim
        # ------------------------------------------------
        # Migrate `eval_datasets: [...]` (top level) into
        # `data.datasets: [{name, dataset_dir, ..., eval: {...}}]`.
        # We keep the migrated dict shape consistent with the new schema
        # so the rest of `from_dict` (and `Arguments.__post_init__`) does
        # not need to know about the legacy location.
        args_dict = cls._migrate_legacy_eval_datasets(args_dict)

        # 1. Resolve TrainingArguments subclass based on trainer_type
        train_dict = args_dict.get('train', {})
        trainer_type = train_dict.get('trainer_type', 'grpo')
        training_args_cls = get_training_args_class(trainer_type)

        # 2. Nested arguments map
        nested_map = {
            'data': ('data_args', DataArguments),
            'model': ('model_args', ModelArguments),
            'scheduler': ('scheduler_args', SchedulerArguments),
            'train': ('training_args', training_args_cls),
            'eval': ('eval_args', EvaluationArguments),
            'log': ('log_args', LogArguments),
            'rewards': ('reward_args', MultiRewardArguments),
            'eval_rewards': ('eval_reward_args', MultiRewardArguments),
        }

        # 3. Build init kwargs
        init_kwargs = {}
        extras = {}

        valid_field_names = {f.name for f in fields(cls)}

        for k, v in args_dict.items():
            if k in nested_map:
                arg_name, arg_cls = nested_map[k]
                init_kwargs[arg_name] = arg_cls.from_dict(v)

            elif k in valid_field_names:
                init_kwargs[k] = v

            else:
                extras[k] = v

        if extras:
            expected_top_level_keys = sorted(
                set(nested_map.keys()) | (valid_field_names - {"extra_kwargs"})
            )
            logger.warning(
                f"{cls.__name__}.from_dict captured {len(extras)} unknown top-level key(s) into extra_kwargs: "
                f"{sorted(extras.keys())}. "
                "Verify these are intentional (expected top-level keys are "
                f"{expected_top_level_keys}); typos will be silently accepted otherwise."
            )

        # 4. Handle explicit 'extra_kwargs' if present in YAML and merge
        if "extra_kwargs" in init_kwargs:
            extras.update(init_kwargs["extra_kwargs"])
        
        init_kwargs["extra_kwargs"] = extras
        
        return cls(**init_kwargs)

    @classmethod
    def load_from_yaml(cls, yaml_file: str) -> Arguments:
        """
        Load Arguments from a YAML configuration file.
        Example: args = Arguments.load_from_yaml("config.yaml")
        """
        with open(yaml_file, 'r', encoding='utf-8') as f:
            args_dict = yaml.safe_load(f)

        return cls.from_dict(args_dict)

    @staticmethod
    def _migrate_legacy_eval_datasets(args_dict: dict[str, Any]) -> dict[str, Any]:
        """Auto-migrate the legacy top-level ``eval_datasets:`` YAML key.

        Converts each legacy ``EvalDatasetArguments``-shaped dict into a
        ``DatasetArguments`` entry under ``data.datasets`` whose
        ``eval:`` sub-block carries the same overrides.  Top-level
        ``eval_datasets`` is removed from the input dict so the rest of
        ``from_dict`` sees the canonical schema only.

        Idempotent for fully-migrated configs.  We deep-copy the input
        before mutating so the caller's dict (e.g. a YAML round-trip
        cache held by a test harness) is never modified — defensive, since
        we mutate nested ``data.datasets`` lists below.

        Notes:
        - Eval-only datasets (no ``train:`` block) are perfectly valid
          under the unified schema, so no fabricated ``train:`` field is
          inserted.
        - When a ``data.datasets`` entry with the same ``name`` is
          already present, the migration MERGES the legacy ``eval:``
          fields into it (rather than creating a duplicate, which
          would trip the duplicate-name validator).
        """
        if 'eval_datasets' not in args_dict:
            return args_dict

        legacy = args_dict['eval_datasets']
        if not isinstance(legacy, list) or not legacy:
            # Empty list or weird shape — drop and let the validator complain.
            new_dict = dict(args_dict)
            new_dict.pop('eval_datasets', None)
            return new_dict

        import warnings as _warnings
        import copy as _copy
        _warnings.warn(
            "Top-level `eval_datasets:` is deprecated and will be removed in a "
            "future release. Move each entry under `data.datasets:` and put "
            "its eval-specific fields (split / num_inference_steps / "
            "guidance_scale / resolution / max_dataset_size) inside an `eval:` "
            "sub-block. The shared `dataset_dir` / `image_dir` / `video_dir` / "
            "`audio_dir` move to the parent dataset entry.",
            DeprecationWarning,
            stacklevel=3,
        )

        # Deep-copy so we never mutate the caller's dict (or its
        # nested data/datasets list, which we may extend in place below).
        new_dict = _copy.deepcopy(args_dict)
        data_dict = new_dict.get('data') or {}
        if not isinstance(data_dict, dict):
            data_dict = {}
        existing_datasets: list = list(data_dict.get('datasets') or [])
        # Index existing entries by name for in-place merge below.
        by_name: dict[str, dict] = {}
        for i, d in enumerate(existing_datasets):
            if isinstance(d, dict) and 'name' in d:
                by_name[d['name']] = d

        # Field categorisation: the legacy EvalDatasetArguments dataclass
        # carries both parent-level fields (name / dataset_dir / media
        # roots) and DatasetEvalSpec-level fields. Split accordingly.
        _PARENT_KEYS = {"name", "dataset_dir", "image_dir", "video_dir", "audio_dir"}
        _EVAL_SPEC_KEYS = {
            "split", "max_dataset_size",
            "resolution", "num_inference_steps", "guidance_scale",
        }

        for entry in legacy:
            if not isinstance(entry, dict):
                raise TypeError(
                    f"Legacy `eval_datasets:` entries must be dicts, got "
                    f"{type(entry).__name__}."
                )
            name = entry.get('name')
            if not name:
                raise ValueError(
                    "Legacy `eval_datasets:` entry is missing a `name`. "
                    "Migrate manually to the unified `data.datasets:` schema."
                )

            parent_fields = {k: v for k, v in entry.items() if k in _PARENT_KEYS}
            eval_fields = {k: v for k, v in entry.items() if k in _EVAL_SPEC_KEYS}
            unknown = set(entry) - _PARENT_KEYS - _EVAL_SPEC_KEYS
            if unknown:
                # Surface unknown keys as DatasetArguments extras (extra_kwargs)
                # rather than silently dropping them.  Same shape as ArgABC.
                parent_fields.setdefault('extra_kwargs', {}).update(
                    {k: entry[k] for k in unknown}
                )

            if name in by_name:
                # Merge eval-spec fields into existing entry's `eval` block.
                target = by_name[name]
                eval_block = target.get('eval')
                if eval_block is None:
                    target['eval'] = eval_fields
                elif isinstance(eval_block, dict):
                    # Newer schema wins on conflict (defensive — should be rare).
                    eval_block_merged = {**eval_fields, **eval_block}
                    target['eval'] = eval_block_merged
                # Don't overwrite parent fields when merging — the unified
                # entry is authoritative.
            else:
                # Create a fresh dataset entry from the legacy fields.
                migrated = {**parent_fields, 'eval': eval_fields}
                existing_datasets.append(migrated)
                by_name[name] = migrated

        data_dict['datasets'] = existing_datasets
        new_dict['data'] = data_dict
        new_dict.pop('eval_datasets', None)
        return new_dict
    
    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)
    
    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()