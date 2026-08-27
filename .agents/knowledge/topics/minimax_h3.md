# MiniMax H3

**Read when**: Changing H3 adapters, workflows, schedules, references, dependency pins, examples, or
verification claims.

## Dependency and workflows

Required dependency: `diffusers>=0.40.0`.

| Registry key | Workflow input | Trainable component |
|---|---|---|
| `minimax-h3-t2va` | prompt | `transformer` |
| `minimax-h3-fl2va` | prompt plus first frame, optionally last frame | `transformer` |
| `minimax-h3-ref2va` | prompt plus ordered heterogeneous references | `transformer_ref` |

The feature probe must validate public symbols, workflow maps, block call surfaces,
row-timestep construction, reference constructors, and modular APIs. A no-weight
`from_config`/component-spec/workflow build proves API compatibility only.

## Adapter structure

The three adapters differ only in workflow identity, canonical transformer name, and component
lists. Everything workflow-invariant lives on the `_MiniMaxH3WorkflowAdapter` mixin, so each
method has exactly one implementation (`test_workflow_adapters_share_one_implementation_per_method`
enforces this). Each adapter is still a direct `BaseAdapter` subclass and the mixin is not one, so
the registry's single-base contract holds; tests assert `BaseAdapter in cls.__bases__` plus no
second `BaseAdapter` subclass in the MRO, rather than an exact `__bases__` tuple.

`_forward_state` replays a stored transition through the public `forward()` instead of a private
path, which keeps rollout and replay on one entry point. Replay receives every collated batch
field, so `build_h3_replay_forward_kwargs()` selects the conditioning arguments `forward()` accepts
and lets that boundary stay strict.

## Hard execution contract

- Batch size is B=1 for preprocessing, rollout, and evaluation.
- H3 has no CFG: guidance is `1.0`, and negative prompts are empty or absent.
- Rollout is structured-only with separate video and audio states.
- Authoritative scheduler/component order is `("video", "audio")`.
- Video uses shift 12; audio uses shift 3.
- Model output is data-ward velocity. Adapters declare `flow_velocity_direction="data"` so both
  scheduler conversion and trainer `x0` projection use the correct sign.
- `num_inference_steps=N` means N transitions and N + 1 state coordinates.

## Input contracts

- T2VA accepts prompt-only workflow input.
- FL2VA accepts one first image or two images ordered first then last.
- Ref2VA preserves and hashes ordered image/video/audio manifests.
- Ref2VA declares `supports_ordered_references=True`; all H3 adapters explicitly declare hidden
  geometry cache fields and a preprocessing cache version.
- Reference paths are dataset-relative. Positive finite `fps` and `sample_rate` overrides follow
  `samples/references.py`.
- PyAV >=18.0.0 decodes video/audio references, including embedded or separate soundtracks.

## Verification boundary

All workflows have pinned API/schema/no-weight verification. T2VA additionally completed
real-weight LoRA rollout, decode, reward, replay, backward, checkpoint, and resume tests on one
GPU and with FSDP2 on 16 GPUs. The native-resolution path completed initialization, checkpoint,
decode, and evaluation. FL2VA and Ref2VA remain no-weight validated. Do not claim long-run reward
improvement, convergence, or numerical parity.

## Upgrade checklist

- [ ] Update the project pin and H3 dependency constant together.
- [ ] Run `require_minimax_h3_support()`.
- [ ] Rerun the real public-symbol and no-weight component-spec/workflow probes.
- [ ] Run H3 scheduler/runtime/registry/reference tests in the pinned environment.
- [ ] Parse all H3 examples through `Arguments.load_from_yaml`.
- [ ] Rerun the documented T2VA real-weight smoke before changing support or memory claims.

## Cross-refs

- UP: [`constraints.md` #5](../constraints.md#5-adapter-component-runtime-contract), [`constraints.md` #14](../constraints.md#14-sample-dataclass-hierarchy), [`architecture.md` MiniMax H3](../architecture.md#minimax-h3)
- PEER: [Component Runtime](component_runtime.md), [Structured Trajectory](structured_trajectory.md), [Parity Testing](parity_testing.md)
