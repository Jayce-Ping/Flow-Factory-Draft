# Structured Trajectory

**Read when**: Changing rollout collection, replay bridges, index maps, callbacks, active masks, or
multi-component scheduler/noise order.

## Authority and shape

- `BaseSample.trajectory is None`: legacy trajectory fields are authoritative.
- `BaseSample.trajectory` is present: `StructuredTrajectory` is authoritative; H3 and LTX2 leave
  all five legacy trajectory fields `None`.
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

## Replay bridges

Trainers consume both storage formats through adapter-owned bridge APIs:

- `get_terminal_state`
- `get_replay_step`
- `get_replay_callback`
- `build_forward_process_state`
- `reduce_component_log_probs`

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

Single-component adapters retain legacy bit-parity. Multi-component reductions preserve one scalar
per sample while averaging only active degrees of freedom, preventing sequence length or
conditioning tokens from changing objective scale. H3 under DiffusionOPD accepts neutral
guidance `1.0` only for teacher and student calls.

Implementation: `src/flow_factory/samples/structured_trajectory.py`,
`src/flow_factory/models/trajectory_bridge.py`, and trainer `optimize()` paths.

## Review checklist

- [ ] State maps use T + 1; transition maps use T.
- [ ] Uncollected positions use `-1`; wholly absent categories use `None`.
- [ ] Component order, schedules, callbacks, log probabilities, and active masks remain aligned.
- [ ] Trainers use adapter bridge APIs rather than direct legacy indexing.
- [ ] Single-component behavior remains bit-parity compatible.
- [ ] Multi-component reductions count active degrees of freedom only.
- [ ] DiffusionOPD rejects mixed dynamics and non-neutral H3 guidance.

## Cross-refs

- UP: [`constraints.md` #14](../constraints.md#14-sample-dataclass-hierarchy), [`architecture.md` Sample Dataclass Hierarchy](../architecture.md#sample-dataclass-hierarchy)
- PEER: [Component Runtime](component_runtime.md), [MiniMax H3](minimax_h3.md), [Train/Inference Consistency](train_inference_consistency.md)
