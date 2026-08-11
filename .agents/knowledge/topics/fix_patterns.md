# Fix Patterns

**Read when**: After completing a bug fix.

---

This document defines the recording template and archival rules for fix experiences.

## Fix Entry Template

Each fix record uses the following format:

```markdown
### [Short Title]
- **Date**: YYYY-MM-DD
- **Symptom**: What the user observed (error message / abnormal behavior)
- **Root Cause**: Root cause analysis (one sentence)
- **Fix**: What was changed (files involved and key modifications)
- **Lesson**: Implications for future development (why this happened, how to prevent it)
- **Related Constraint**: If a new hard constraint was created, reference the constraint number (N/A if none)
```

## Archival Location Decision Table

Based on the fix type, write the fix entry to the appropriate document:

| Fix Type | Archival Location | Example |
|----------|------------------|---------|
| Violated an existing constraint | `constraints.md` — add "common violation case" under the relevant entry | Forgot to update registry path |
| Discovered a new hard constraint | `constraints.md` — new entry | Found ZeRO-2 + EMA incompatibility |
| Architecture / data-flow misunderstanding | `architecture.md` — relevant module section | Misunderstood preprocess_func call timing |
| Subsystem-specific pitfall | `topics/<topic>.md` — corresponding topic | Sampler boundary condition |
| Does not fit any of the above | This document's "Recorded Fix Patterns" section below | Append as a new record |

**Decision flow**: Check whether the fix matches the first four rows; if none match, fall back to this document.

## Recorded Fix Patterns

<!-- This section accumulates over time. Append new records at the end using the template above. -->

### Multi-modal batch homogeneity (R6)
- **Date**: 2026-04
- **Symptom**: Silent HF `Dataset.map` errors and inconsistent per-sample types in the `audios` column (sometimes `None`, sometimes `Tensor`, sometimes `List[Tensor]`); image/video columns had a latent batch-length mismatch when a sample contributed zero items.
- **Root Cause**: `_preprocess_batch` returned a mix of `None`, `Tensor`, and `List[Tensor]` for the same modality column, breaking Arrow's homogeneous-column requirement and forcing every downstream consumer to handle three input shapes.
- **Fix**: `data_utils/dataset.py:_preprocess_batch` now always emits `List[List[Media]]` per modality (`[]` for empty samples, `[item]` for single-item samples, multi as-is) and appends to BOTH `xx_args[xx]` and `batch[xx]` for every sample so the columns stay length-aligned. Mirrored the same shape on `models/abc.py:preprocess_func` (`audios` parameter) and `utils/audio.py` (`MultiAudioBatch` type alias).
- **Lesson**: HF Arrow demands homogeneous columns, and downstream consumers benefit from a single canonical type. When a column has variable cardinality per row, always represent it as `List[...]` even when the row is empty or has exactly one element. Never special-case "single item" by unwrapping.
- **Related Constraint**: N/A (codified in `topics/adapter_conventions.md` Gotcha #6 and the new "Multi-media batch homogeneity" bullet under Batch Dimension Convention).

### Non-abstract encoder defaults (R7)
- **Date**: 2026-04
- **Symptom**: Adding `encode_audio` as `@abstractmethod` on `BaseAdapter` would force one-line `pass` stubs on 11 existing concrete adapters, none of which consume audio. The first iteration of R6 actually shipped this — and the resulting "noise" diff dwarfed the real change.
- **Root Cause**: Incorrect default-discoverability assumption — abstract methods force every subclass to acknowledge a feature, even when the subclass doesn't use it.
- **Fix**: `models/abc.py` dropped `@abstractmethod` from all 4 encoders (`encode_prompt`, `encode_image`, `encode_video`, `encode_audio`); default body is `pass` returning `None`; `preprocess_func` skips integration when the called encoder returns `None`. The Round-6 stub overrides on 11 concrete adapters were reverted, leaving them byte-identical to `origin/main`.
- **Lesson**: When extending a base contract for a partial-coverage feature (where only some subclasses will participate), no-op default + opt-in override beats forcing every subclass to acknowledge it. Reserve `@abstractmethod` for invariants that ALL subclasses must implement (e.g. `load_pipeline`, `decode_latents`, `forward`, `inference`).
- **Related Constraint**: #12 (post-update text codifies "Optional encoder overrides (no-op default)").

### Preference-arm conditioning ownership
- **Date**: 2026-08-11
- **Symptom**: DPO evaluated the rejected H3 state with the chosen sample's prompt/reference conditioning.
- **Root Cause**: Shared-noise logic was incorrectly extended to the conditioning batch, even though only the forward-process noise is shared between arms.
- **Fix**: Rejected policy/reference forwards now receive `rejected_batch`, with a production-path regression using distinct prompt embeddings.
- **Lesson**: Preference arms may share stochastic coordinates but never model conditioning; replay state and conditioning must come from the same sample.
- **Related Constraint**: #7

### Dynamics and velocity conventions belong to adapters/schedulers
- **Date**: 2026-08-11
- **Symptom**: GRPO-Guard applied Flow-SDE `sqrt(-dt)` scaling to CPS, while NFT and OPD reconstructed H3 `x0` with the standard flow sign.
- **Root Cause**: Trainer-local formulas assumed one dynamics type and one velocity direction.
- **Fix**: Coupled trainers share a dynamics-aware transition-scale helper that rejects zero/non-finite variance, and `x0` projection routes through `adapter.project_velocity_to_clean_state()` using model-compute dtype.
- **Lesson**: Trainers should consume declared process semantics rather than infer them from historical single-model formulas.
- **Related Constraint**: #7

### Lazy components must re-enter lifecycle policy
- **Date**: 2026-08-11
- **Symptom**: Modular preprocessing modules materialized after adapter initialization missed explicit frozen dtype casting and the FSDP synchronization point.
- **Root Cause**: `_mix_precision()` and frozen synchronization enumerated only modules materialized at construction time.
- **Fix**: `on_load_components()` applies precision policy to newly materialized modules; preprocessing and inference synchronize explicit module parameters/buffers before use while skipping tokenizers/processors.
- **Lesson**: Laziness changes timing, not ownership or lifecycle obligations.
- **Related Constraint**: #19, #20

### Preprocessing cache identity must be explicit
- **Date**: 2026-08-11
- **Symptom**: Ref2VA could skip ordered-reference decoding or reuse a merged cache after source/geometry/helper changes.
- **Root Cause**: The real adapter lacked its capability flag, merged cache paths omitted source bytes, and reflective signature filtering could not see semantic fields hidden by `**kwargs`.
- **Fix**: Ref2VA opts into ordered references; cache paths hash dataset root, source bytes, and adapter versions; adapters declare hidden semantic fields via `preprocess_cache_fields`.
- **Lesson**: Reflection is only a default. Dynamic preprocessors need an explicit cache contract.
- **Related Constraint**: N/A

## Cross-refs

- `constraints.md` (archival target for constraint violations)
- `architecture.md` (archival target for data-flow misunderstandings)
- `ff-debug/SKILL.md` Phase 5 (knowledge capture workflow)
