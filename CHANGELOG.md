# Changelog

## Unreleased — `feat/multi-eval-dataset`

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

#### Eval cache one-time reprocess

The unified eval path adds an `eval_<name>` token (here `eval_default`)
to the preprocessing-cache fingerprint. Existing
`~/.cache/flow_factory/datasets/...` entries from the old code path
do not match the new fingerprint, so the test split is reprocessed
once on the next run. Training caches are unaffected.

#### Removed `BaseTrainer` attributes

These were used only by the deleted legacy single-eval path; subclasses
(GRPO / CRD / DGPO / NFT / AWM / DPO) never referenced them. Forks
that did need to migrate to the per-dataset shape:

- `self.test_dataloader` → `self.eval_dataloaders` (`Dict[str, DataLoader]`).
- `self.eval_reward_buffer` → `self.eval_dataset_reward_buffers[name]`.
- `self.eval_reward_processor` → `self.eval_dataset_reward_processors[name]`.
- Private methods `_evaluate_single_dataset` / `_evaluate_multi_dataset`
  are gone; `evaluate()` is the single eval entry point.

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

### Non-breaking changes (multi-source training)

Everything in the rest of this branch (steps 1–12 of the original
plan plus review items 1–9) lands as additive features:

- Unified `data.datasets:` schema with per-entry `train:` / `eval:`
  sub-blocks (legacy `data.dataset_dir` and top-level `eval_datasets:`
  are auto-canonicalized with deprecation warnings).
- `RewardArguments.datasets`: `None` is eagerly resolved at config-load
  to the explicit list of applicable side names; `[]` means "never
  fires" (warned).
- Integer `train.weight` + exact-divisibility partitioning — every
  batch geometrically comes from a single source.
- `BaseSample.source` and `BaseSample.source_id` first-class typed
  fields; `_datasets_resolved: frozenset[int]` cache on
  `RewardArguments` for hot-path gate.
- `RewardProcessor` source-aware gate + NaN-padded transport for
  cross-rank gather safety.
- `AdvantageProcessor` applicability-aware aggregation; raises loudly
  on NaN at applicable positions and on samples with no applicable
  reward.
