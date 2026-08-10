# Task 4B LTX2 Structured Trajectory Report

## Status

DONE

Baseline: `21cec8d`

Commit: `73e8cb683f219680f2d7b9e5ac47337022da902c`

## Scope implemented

- Rewrote `LTX2_T2AV_Adapter.inference` and `LTX2_I2AV_Adapter.inference` to collect
  per-component rollout data and publish one authoritative `StructuredTrajectory` per sample.
- Added the shared builder and its helpers to `models/ltx2/_common.py`:
  - `LTX2_STRUCTURED_CALLBACK_FIELDS`
  - `build_ltx2_full_component_schedule`
  - `build_ltx2_legacy_callback_view`
  - `build_ltx2_sparse_transition_map`
  - `build_ltx2_rollout_log_probs`
  - `split_ltx2_callback_results`
  - `build_ltx2_structured_trajectories`
- Legacy sample fields (`timesteps`, `all_latents`, `latent_index_map`, `log_probs`,
  `log_prob_index_map`) are now `None` on LTX2 samples; the structured trajectory is authoritative.
- Non-latent callbacks stay in `extra_kwargs` together with their `callback_index_map`, which is
  emitted only when such a callback was actually collected.
- I2AV attaches a video `active_mask` derived from `~conditioning_mask`, so the conditioning frame
  is excluded from every reduction, log-prob weighting and forward-process noising.
- Added inference-capable fakes, an independent legacy-loop oracle, and four test modules
  (builder contracts, rollout parity, stack/replay + condition frame, algorithm interface matrix).
- Updated `.agents/knowledge/architecture.md` and `topics/adapter_conventions.md` for the new
  adapter-level contract.
- Did not modify any other adapter, any trainer, the collectors, or the trajectory bridge.

## Test modules

- `tests/models/ltx2/ltx2_inference_fakes.py` — inference-capable scheduler/VAE/vocoder/pipeline
  twins with a real shifted flow-matching schedule, a sparse SDE window, and the Flow-SDE KL
  denominator. Kept separate from `ltx2_fakes.py` so the pre-Task-4A golden generator stays stable.
- `tests/models/ltx2/ltx2_inference_oracle.py` — independent transcription of the pre-collector
  LTX2 denoising loop, used as the parity oracle.
- `tests/models/ltx2/test_ltx2_structured_builder.py` — builder contracts: dense/sparse states,
  joint and per-component log probabilities, callback splitting, active masks, fail-fast paths.
- `tests/models/ltx2/test_ltx2_inference_parity.py` — full T2AV/I2AV rollout parity against the
  oracle: latents, timesteps, log probabilities, callbacks, decoded video/audio, RNG stream, and
  scheduler/model dispatch order, including an explicit single `torch.Generator` and non-default
  decode timestep/noise settings.
- `tests/models/ltx2/test_ltx2_structured_replay.py` — collation, terminal state, replay steps,
  replay callbacks, replay-equals-legacy-forward, and the I2AV condition-frame invariants.
- `tests/models/ltx2/test_ltx2_algorithm_interfaces.py` — algorithm interface matrix executed with
  a real LTX2 adapter instance, a real rollout batch and real replay states/times/masks.

## Algorithm interface matrix coverage

Every case below calls the production formula helper on an actual LTX2 adapter instance and
compares it against an independently written expectation, for both T2AV (no mask) and I2AV
(conditioning-frame mask):

| Algorithm | Production entry point exercised |
|---|---|
| GRPO | `_require_replay_log_prob`, `_require_policy_log_prob`, `_reference_kl_divergence` (v/x) |
| GRPO-Guard | `_guard_ratio` (per-component log-prob, mean, std/dt, DOF reduction) |
| DPPO | `_trust_region_kl` (v-based and x-based with per-scheduler effective sigma) |
| DiffusionNFT | `_matching_losses` (normalized x0 matching), `_velocity_reference_kl` |
| AWM | `_matching_log_prob` (per-component sigma weighting), `_velocity_kl` |
| DPO | shared component noise via `apply_forward_process_noise`, `_arm_velocity_error`, `_preference_loss` |
| DGPO | `_shared_group_noise` (per-`unique_id`, per component), `_compute_dsm_loss` |
| CRD | `_implicit_reward` for `adaptive_logp` both off and on |
| DiffusionOPD | `_component_kl_denominators`, `project_distillation_target_state` (xt/v/x0), `compute_structured_distillation_loss` |

The matrix also asserts that a poisoned conditioning frame cannot influence any reduction, that
the video and audio denominators differ (each component uses its own scheduler), and that the
joint log probability is the DOF-weighted combination of the two component log probabilities.

## Parity guarantees verified

- Latents, per-component timesteps, joint log probabilities, callbacks and decoded video/audio
  match the independent legacy-loop oracle exactly (`torch.equal`).
- The rollout consumes the same RNG stream and issues the same scheduler/transformer/VAE dispatch
  sequence as the oracle, so no extra draw or model call was introduced.
- Explicit public-API generators use independent same-seed instances for the oracle and adapter;
  their post-run state and subsequent draw match, as do the global RNG state and subsequent draw
  when non-default `decode_timestep` and `decode_noise_scale` are active.
- Replaying a stored transition through `forward_state` reproduces the legacy concatenated
  `forward` for `next_latents`, `next_latents_mean`, `velocity`, `log_prob`, `std_dev_t` and `dt`,
  and recovers the stored joint and component log probabilities bit-for-bit (unit PPO/Guard ratio).
- The I2AV conditioning frame is identical at every replay position, unchanged by a replayed step,
  unchanged by forward-process noising (with zero target velocity), and excluded from
  `get_state_active_numel` and both reducers.

## Intentional behavior differences versus the legacy fields

1. `log_prob_index_map` is now sparse. The final rollout transition leaves the SDE window and stores
   no log probability, so the structured map is `[0, 1, -1]` instead of the dense legacy
   `[0, 1, 2]`. The legacy identity map incorrectly claimed that the final transition had been
   collected; a replay through the bridge would explicitly fail because compact index `2` is out
   of bounds. The structured `[0, 1, -1]` contract marks that transition as unavailable.
2. `callback_index_map` is present in `extra_kwargs` only when a non-latent callback was actually
   collected. The legacy loop emitted it unconditionally, including for evaluation-only rollouts.

Both are covered by explicit tests.

## Verification commands and outputs

1. New/changed LTX2 tests:
   `.scratch/test-venv/bin/python -m pytest -q tests/models/ltx2/` — `290 passed`.
2. Task 1-4B suites:
   `.scratch/test-venv/bin/python -m pytest -q tests/` — `760 passed`.
3. Compile smoke: `compileall` over `src/flow_factory/models/ltx2` — exit 0.
4. Formatting: Black and isort over the Task-owned LTX2 source and test files — clean. The one
   remaining Black complaint on `_common.py` is the pre-existing "blank line after module
   docstring" rule of the newer local Black version; it already failed at `21cec8d` and was left
   untouched to keep the diff scoped.
5. Diff scope: `git status --porcelain` shows only the three LTX2 source files plus the two
   knowledge documents; no other adapter, trainer, collector or bridge file changed.

## Concerns

1. The parity oracle and the fakes model the LTX2 loop, not the real LTX2 checkpoints. A real
   end-to-end LTX2 RL run on GPU is still the only way to confirm wall-clock/memory behavior and
   the audio VAE/vocoder geometry against upstream.
2. `build_ltx2_legacy_callback_view` reconstructs the concatenated view by exposing the video
   component's `std_dev_t`/`dt`, matching the legacy loop. Any future LTX2 configuration where the
   two schedulers diverge in those statistics would need the callback consumer to move fully to the
   structured per-component fields.
3. The trainers themselves were out of scope, so this task verifies their formula helpers against a
   real LTX2 trajectory but does not run a full optimizer loop for LTX2.
