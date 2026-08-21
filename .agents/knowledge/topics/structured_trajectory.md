# Structured Trajectory

**Read when**: Changing rollout collection, replay bridges, index maps, callbacks, active masks, or
multi-component scheduler/noise order.

## Authority and shape

- `BaseSample.trajectory is None`: legacy trajectory fields are authoritative.
- `BaseSample.trajectory` is present: `StructuredTrajectory` is authoritative and all five legacy
  trajectory fields stay `None`. LTX2 is the reference multi-component adapter.
- `component_order` defines scheduler, RNG, replay, and reduction order.
- Each component retains independent state shape, schedule, state/transition maps, callbacks,
  component log probabilities, and active masks.

For T transitions:

| Collection | Rollout positions | Index-map length |
|---|---:|---:|
| States | initial plus post-transition states | T + 1 |
| Joint/component log probabilities | transitions | T |
| Transition callbacks | transitions | T |

`-1` means a rollout position was not collected. A whole absent category uses a `None` tensor/map
container, never an all-`-1` placeholder. Sparse terminal-only state collection is valid.

## Both storage formats, one trainer-facing API

Twelve of the fourteen adapters still emit legacy trajectories; only LTX2 emits structured ones.
That dual track is deliberately invisible above the bridge: no trainer reads `all_latents`,
`latent_index_map` or `log_prob_index_map`, and the format branch exists only inside
`trajectory_bridge`. Keep it that way — a trainer that indexes storage directly re-couples every
algorithm to the migration state.

Trainers consume both formats through adapter-owned bridge APIs:

- `get_terminal_state`
- `get_replay_step`
- `get_replay_callback`
- `add_forward_process_noise` / `apply_forward_process_noise`
- `reduce_component_latent_values` / `reduce_latent_values`

Bridge-owned state/noise arguments have one source and must not be forwarded again from batch
conditioning. Active masks exclude immutable conditioning positions from noising, log-probability
reduction, and losses.

## Trainer matrix

| Trainer | Structured consumption |
|---|---|
| GRPO / GRPO-Guard / DPPO | Coupled replay; transition statistics and joint/component log probabilities |
| DiffusionNFT / AWM / DPO | Terminal state plus ordered forward-process noise |
| DGPO | Deterministic per-UID component noise in `component_order` |
| CRD | Global two-pass order; pass two rebuilds state from stored component noise |
| DiffusionOPD | Requires homogeneous scheduler-group dynamics; no reward/advantage stage |

Single-component adapters retain legacy bit-parity: generalizing a reduction to N components
changes floating-point association order, so the one-component case keeps its original arithmetic
rather than routing through the general path (see `trainers/distillation/opd/common.py`). Multi-component
reductions preserve one scalar per sample while averaging only active degrees of freedom, so
sequence length or conditioning tokens cannot change objective scale.

Implementation: `src/flow_factory/samples/trajectory.py`,
`src/flow_factory/models/trajectory_bridge/`, and trainer `optimize()` paths.

Shared assembly, rather than per-family copies:

| Helper | Location | Owns |
|---|---|---|
| `unstack_structured_trajectories()` | `samples/trajectory.py` | Inverse of `StructuredTrajectory.stack`: batched component tensors -> one trajectory per sample. Families keep only their own validation and any concatenated-tensor splitting. |
| `reduce_component_log_probs()` | `models/component_reduction.py` | Degrees-of-freedom-weighted joint log probability. Shape-agnostic, so a per-step `(B,)` and a stored `(B, T)` both reduce identically. |

`trajectory_bridge/` is split by responsibility: `replay.py` (read stored trajectories out of a
collated batch), `noising.py` (forward-process noise), `dispatch.py` (resolve conditioning and
dispatch one replayed step), `score.py` (velocity-to-score projection for distillation),
`masks.py` (active-mask geometry), `reduction.py` (component reduction entry points).
`__init__.py` re-exports the public surface; `BaseAdapter` imports the package under the alias
`bridge`, so each delegation reads as `bridge.get_replay_step(...)`.

## Review checklist

- [ ] State maps use T + 1; transition maps use T.
- [ ] Uncollected positions use `-1`; wholly absent categories use `None`.
- [ ] Component order, schedules, callbacks, log probabilities, and active masks remain aligned.
- [ ] Trainers use adapter bridge APIs rather than direct legacy indexing.
- [ ] Single-component behavior remains bit-parity compatible.
- [ ] Multi-component reductions count active degrees of freedom only.
- [ ] DiffusionOPD rejects mixed dynamics.

## Cross-refs

- UP: [`constraints.md` #14](../constraints.md#14-sample-dataclass-hierarchy), [`architecture.md` Sample Dataclass Hierarchy](../architecture.md#sample-dataclass-hierarchy)
- PEER: [Component Runtime](component_runtime.md), [Component Variants](component_variants.md), [Train/Inference Consistency](train_inference_consistency.md)
