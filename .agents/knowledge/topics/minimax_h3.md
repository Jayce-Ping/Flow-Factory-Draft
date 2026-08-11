# MiniMax H3

**Read when**: Changing H3 adapters, workflows, schedules, references, dependency pins, examples, or
verification claims.

## Dependency and workflows

Required diffusers commit:
`f53d552036a0d1bd5570782a39cd40cfabf112bc`.

| Registry key | Workflow input | Trainable component |
|---|---|---|
| `minimax-h3-t2va` | prompt | `transformer` |
| `minimax-h3-fl2va` | prompt plus first frame, optionally last frame | `transformer` |
| `minimax-h3-ref2va` | prompt plus ordered heterogeneous references | `transformer_ref` |

The feature probe must validate pinned public symbols, workflow maps, block call surfaces,
row-timestep construction, reference constructors, and modular APIs. A no-weight
`from_config`/component-spec/workflow build proves API compatibility only.

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

The examples and adapters have pinned API/schema/no-weight workflow verification. The 61 GB
checkpoint was not downloaded. Do not claim real-weight generation/training, GPU/distributed
execution, memory fit, throughput, reward improvement, convergence, or numerical parity.

## Upgrade checklist

- [ ] Update the project pin and H3 dependency constant together.
- [ ] Run `require_minimax_h3_support()`.
- [ ] Rerun the real public-symbol and no-weight component-spec/workflow probes.
- [ ] Run H3 scheduler/runtime/registry/reference tests in the pinned environment.
- [ ] Parse all H3 examples through `Arguments.load_from_yaml`.
- [ ] Run a separately documented real-weight smoke before making support or memory claims.

## Cross-refs

- UP: [`constraints.md` #5](../constraints.md#5-adapter-component-runtime-contract), [`constraints.md` #14](../constraints.md#14-sample-dataclass-hierarchy), [`architecture.md` MiniMax H3](../architecture.md#minimax-h3)
- PEER: [Component Runtime](component_runtime.md), [Structured Trajectory](structured_trajectory.md), [Parity Testing](parity_testing.md)
