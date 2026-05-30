# Changelog

## Unreleased — `feat/multi-eval-dataset`

**Range:** `652f7315..15c3943f` (28 commits, all 2026-05-26 … 2026-05-30)
**Repo:** [X-GenGroup/Flow-Factory](https://github.com/X-GenGroup/Flow-Factory)
**Branch:** `feat/multi-eval-dataset`
**PR:** _TBD — to be filled in when opened_

### Commit ranges by phase

| Phase | Range | Date | What |
|-------|-------|------|------|
| Pre-existing multi-eval foundation (merged before this stack) | `8e573cc..33d678b` | 2026-05-26 | Initial multi-eval support + per-dataset reward routing + per-dataset overrides |
| Plan steps 1–12 (multi-training-dataset infra) | `f2b2100..cf3a3f7` | 2026-05-30 | Unified `data.datasets` schema, exact partitioning, source-aware gate, NaN-pad transport, multi_source.yaml example |
| Review items 1, 4, 5, 6, 7, 8, 9 (post-implementation tightening) | `56c1b67..3d1d098` | 2026-05-30 | Naming / ordering / typed source fields / integer weights / eager resolution / cleanup |
| Eval-path unification (merge steps 1, 3, 4, 5, 6) | `5a00d9b..15c3943` | 2026-05-30 | Single eval implementation; rename `get_dataloader` → `get_train_dataloader` |

### Commit list

| Hash | Date | Subject |
|------|------|---------|
| `15c3943` | 2026-05-30 | `[review,merge-6]` CHANGELOG.md: document eval-merge breaking changes |
| `9e38246` | 2026-05-30 | `[review,merge-5]` drop self.test_dataloader and self.eval_reward_buffer |
| `b4717f4` | 2026-05-30 | `[review,merge-4]` delete _evaluate_single_dataset; unify into evaluate() |
| `b360984` | 2026-05-30 | `[review,merge-3]` route legacy test through get_eval_dataloaders; drop _build_legacy_test_dataloader |
| `5a00d9b` | 2026-05-30 | `[review,merge-1]` rename: get_dataloader -> get_train_dataloader; train-only return |
| `3d1d098` | 2026-05-30 | `[review,item-8]` cleanups: explicit batch_size, deepcopy shim, inverse routing check |
| `94f28ba` | 2026-05-30 | `[review,items-2+3]` feat: per-source spec fields + unified single/multi path |
| `933beae` | 2026-05-30 | `[review,item-7]` feat: integer weights + exact-divisibility partition |
| `0a612bb` | 2026-05-30 | `[review,item-6]` feat: promote source/source_id to first-class BaseSample fields |
| `6be080f` | 2026-05-30 | `[review,item-9]` feat: eager `RewardArguments.datasets` resolution |
| `fd2e2bd` | 2026-05-30 | `[review,item-4]` reorder: training-before-eval parameters and accessors |
| `521a9bb` | 2026-05-30 | `[review,item-1]` rename: dl -> loader (consistent with project naming) |
| `56c1b67` | 2026-05-30 | `[review,item-5]` cleanup: drop object.__getattribute__ paranoia in gate/aggregation |
| `cf3a3f7` | 2026-05-30 | `[examples]` feat: multi_source.yaml smoke config (step 12) |
| `610a80f` | 2026-05-30 | `[trainers]` feat: BaseTrainer __source__ injection + set_epoch propagation (step 11) |
| `da62f47` | 2026-05-30 | `[advantage]` feat: applicability-aware aggregation (step 10) |
| `993d144` | 2026-05-30 | `[rewards,samples]` feat: source-aware reward gate with NaN-padded transport (step 9) |
| `de40eec` | 2026-05-30 | `[rewards,trainers]` feat: MultiRewardLoader training_dataset_names plumbing (step 8) |
| `b3e74c7` | 2026-05-30 | `[data_utils,trainers]` feat: multi-source train dataloader infra (step 7) |
| `4662279` | 2026-05-30 | `[data_utils]` feat: get_data_sampler unique_sample_num override (step 6, later removed by review-2+3) |
| `e072d1c` | 2026-05-30 | `[hparams]` feat: shared `_align_unique_sample_num` + multi-source partition (step 5) |
| `33a9349` | 2026-05-30 | `[hparams,trainers,data_utils]` feat: switch eval consumers to data.datasets (step 4) |
| `5f84e05` | 2026-05-30 | `[hparams]` feat: top-level eval_datasets deprecation shim (step 3) |
| `c01909e` | 2026-05-30 | `[hparams]` feat: data.datasets field + per-split properties + validators (step 2) |
| `f2b2100` | 2026-05-30 | `[hparams]` feat: unified DatasetArguments schema (step 1) |
| `33d678b` | 2026-05-26 | refactor: simplify multi-eval implementation after code review |
| `cb3c434` | 2026-05-26 | feat: add per-dataset eval generation overrides |
| `8e573cc` | 2026-05-26 | feat: support multiple eval datasets with per-dataset reward routing |

---

### Breaking changes

#### Eval metric key rename

The legacy single-test eval path now flows through the same per-dataset
pipeline as the multi-eval path. Configs that previously used the
legacy `data.dataset_dir` (with a `test.jsonl` alongside) are
auto-promoted by `Arguments._canonicalize_legacy_dataset_dir` to a
1-entry `data.datasets` list whose entry is named `default`, so the
unified `evaluate()` routes them through the same per-dataset machinery
as any explicit `data.datasets` config.

Consequence — eval metric keys move from:

```
eval/reward_<name>_mean
eval/reward_<name>_std
eval_samples
```

to:

```
eval/default/reward_<name>_mean
eval/default/reward_<name>_std
eval/default/samples
```

W&B / TensorBoard dashboards / alerts / aggregations targeting the old
keys must be updated (find-and-replace `eval/reward_` →
`eval/default/reward_`; `eval_samples` → `eval/default/samples`).

Landed in: `b4717f4` (step 4 of eval merge).

#### Eval cache one-time reprocess

The unified eval path adds an `eval_<name>` token (here `eval_default`)
to the preprocessing-cache fingerprint. Existing
`~/.cache/flow_factory/datasets/...` entries from the old code path
do not match the new fingerprint, so the test split is reprocessed
once on the next run. Training caches are unaffected.

Landed in: `b360984` (step 3 of eval merge).

#### Removed `BaseTrainer` attributes

These were used only by the deleted legacy single-eval path; subclasses
(GRPO / CRD / DGPO / NFT / AWM / DPO) never referenced them. Forks
that did need to migrate to the per-dataset shape:

- `self.test_dataloader` → `self.eval_dataloaders` (`Dict[str, DataLoader]`).
- `self.eval_reward_buffer` → `self.eval_dataset_reward_buffers[name]`.
- `self.eval_reward_processor` → `self.eval_dataset_reward_processors[name]`.
- Private methods `_evaluate_single_dataset` / `_evaluate_multi_dataset`
  are gone; `evaluate()` is the single eval entry point.

Landed in: `9e38246` (step 5 of eval merge).

#### Renamed `data_utils.get_dataloader` → `get_train_dataloader`

The function now returns a 2-tuple `(train_loader, train_loaders_by_source)`
— the test-loader return slot has moved to `get_eval_dataloaders`.
The eval / test path is fully owned by `get_eval_dataloaders`.

```python
# Before
from flow_factory.data_utils.loader import get_dataloader
train, test, by_source = get_dataloader(config, accelerator, ...)

# After
from flow_factory.data_utils.loader import get_train_dataloader, get_eval_dataloaders
train, by_source = get_train_dataloader(config, accelerator, ...)
eval_dict = get_eval_dataloaders(config.data_args.eval_datasets, ...)
```

Landed in: `5a00d9b` (step 1 of eval merge).

#### Top-level `eval_datasets:` YAML key deprecated

YAMLs using the brief-lived top-level `eval_datasets:` field (introduced
on this same branch in `8e573cc..33d678b`) are auto-migrated to
`data.datasets[*].eval` with a single `DeprecationWarning` per config
load. The shim is scheduled for removal one release after this PR
ships.

Landed in: `5f84e05` (step 3 of plan).

#### `RewardArguments.datasets` semantic change

`None` (the user-supplied default) is now eagerly resolved at config
load time to the explicit list of applicable side names — so
`print(config)` shows a concrete `List[str]` instead of `null`. Empty
list `[]` is honored as "this reward never fires" with a warning.

Landed in: `6be080f` (review item 9).

#### Integer `train.weight` required

`DatasetTrainSpec.weight` must be a positive integer (float values that
are integer-valued, e.g. `1.0`, are silently coerced; non-integer
floats raise). Combined with `num_batches_per_epoch %
sum(weights) == 0` enforcement, this geometrically guarantees every
batch comes from a single source.

Landed in: `933beae` (review item 7).

---

### Non-breaking additive features

Everything in the rest of this branch lands as additive features — no
existing code paths regress, and the legacy single-source training
flow is byte-identical for configs that don't opt into multi-source:

- **Unified `data.datasets:` schema** with per-entry `train:` / `eval:`
  sub-blocks (legacy `data.dataset_dir` and top-level `eval_datasets:`
  are auto-canonicalized with deprecation warnings).
- **Multi-source training:** weight-based interleaving with exact
  per-batch source homogeneity, weighted scheduler, per-source
  DataLoaders.
- **Source-aware reward routing:** `RewardArguments.datasets` extends
  to training rewards (was eval-only before this branch).
- **Source bookkeeping on samples:** `BaseSample.source: Optional[str]`
  + `BaseSample.source_id: Optional[int]` first-class typed fields;
  `_datasets_resolved: frozenset[int]` cache on `RewardArguments` for
  hot-path comparison.
- **NaN-padded reward transport:** every reward call yields a
  full-length tensor with NaN at non-applicable positions, so
  cross-rank `accelerator.gather` participants are uniform regardless
  of which sources each rank processed (no deadlock).
- **Applicability-aware aggregation:** `AdvantageProcessor` reads
  `sample.applicable_rewards` as the source of truth (not
  `np.isnan`), so an in-model NaN at an applicable position raises
  loudly instead of silently masking. Samples with no applicable
  reward also raise at config-load time
  (`Arguments._validate_every_source_has_a_reward`).
- **Future OPD readiness:** `self.train_dataloaders_by_source: Dict[str,
  DataLoader]` is exposed on every trainer for the upcoming
  DiffusionOPD trainer to drive its own balanced per-teacher sampling
  without re-deriving the partition.

---

### Notable internal refactors

- **One eval implementation** (`evaluate()`); `_evaluate_single_dataset`
  / `_evaluate_multi_dataset` deleted (`b4717f4`).
- **One train data factory** (`get_train_dataloader`) + one eval data
  factory (`get_eval_dataloaders`); legacy single-test path
  (`_build_legacy_test_dataloader`) deleted (`b360984`).
- **Single config canonicalization pass**
  (`Arguments._canonicalize_legacy_dataset_dir`) in `__post_init__` so
  every downstream consumer sees only the unified schema (`94f28ba`).
- **Latent bug fix**: `_evaluate_multi_dataset` was calling
  `get_merged_eval_kwargs` on the parent `DatasetArguments` (carrying
  over from the legacy `EvalDatasetArguments` shape where the method
  lived on the parent). Method actually lives on `DatasetEvalSpec`
  (the inner sub-block) — so per-dataset eval overrides would have
  raised `AttributeError`. Fixed during the eval-merge unification
  (`b4717f4`).
- **Pre-PR latent-cache compatibility**: when `len(training_datasets)
  == 1` the per-source loader skips the `train_source:{name}` token
  in `extra_hash_strs`, so caches built by the legacy single-source
  code path remain hits after the canonicalization upgrade
  (`94f28ba`). One-time reprocess only applies to the eval/test side.
