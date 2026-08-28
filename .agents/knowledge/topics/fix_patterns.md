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

### Preserve contract-locked terminology during architecture rewrites
- **Date**: 2026-08-28
- **Symptom**: The SenseNova documentation regression test failed after the architecture table retained the correct behavior but dropped the contract phrase `ordered variable-count references`.
- **Root Cause**: A broad documentation rewrite paraphrased a tested semantic distinction without checking the existing documentation contract.
- **Fix**: `.agents/knowledge/architecture.md` now restores the exact phrase while retaining the grouped `images` and non-NaViT execution details; the documentation suite verifies both distinctions.
- **Lesson**: Search documentation tests before rewriting architecture terminology, especially where wording distinguishes adapters with superficially similar multi-reference inputs.
- **Related Constraint**: N/A

### Distributed exact-state phases need synchronized failure and one publisher
- **Date**: 2026-08-28
- **Symptom**: One rank could fail exact-resume RNG/hash preflight while a peer entered `Accelerator.load_state()` and hung; concurrent saves could also pass destination preflight together and write the same staging directory.
- **Root Cause**: Rank-local filesystem work was followed by raw barriers or core mutation without first gathering errors, and publication ownership was checked non-atomically before artifact creation.
- **Fix**: `trainers/abc.py` now gathers all-rank errors after barrier-free path resolution, runtime preflight, and core load; commits runtime progress only after every core load succeeds; atomically elects the global publisher before core save; and keeps every filesystem claim until all publishers install their final directory. `trainers/common/runtime_identity.py` hashes FSDP wrap/state-dict topology and the full DeepSpeed batch/accumulation plan, while `runtime_state.py` validates per-device RNG topology and state installability. Runtime manifests are written only after Accelerator artifacts finish. Targeted multi-rank failure simulations cover preflight, load, and publisher races.
- **Lesson**: A raw barrier is unsafe after rank-local I/O that can raise. Structure distributed checkpoints as preflight → synchronized error gather → backend mutation → synchronized error gather → manifest → atomic publication, acquire the publication claim before any process writes shared staging, and retain it until every filesystem reports success. Exact resume identity must cover backend topology, not only parameter names and shapes.
- **Related Constraint**: #18

### Validate nested media cardinality before flattening
- **Date**: 2026-08-28
- **Symptom**: Valid offline Flux1-Kontext batches shaped as `List[List[PIL]]` raised `TypeError` while checking whether a sample carried multiple condition images.
- **Root Cause**: `_standardize_image_input()` flattened the nested batch before running its per-sample cardinality check, so the check called `len()` on each PIL image.
- **Fix**: Flux1-Kontext now checks and warns on the original nested batch before selecting the first image, with a regression covering two offline single-image rows.
- **Lesson**: Perform shape and cardinality validation at the boundary where that structure still exists; flattening destroys the evidence needed to validate it safely.
- **Related Constraint**: N/A

### Optional conditions require row-preserving empty sentinels
- **Date**: 2026-08-28
- **Symptom**: Legal batches mixing omitted and present negative prompts or condition images could reach tokenizers as `None`, lose their outer batch interpretation, or fail in an image encoder on an empty list.
- **Root Cause**: Optionality was validated per record, but preprocessing and collation lacked homogeneous representations for a missing value inside a mixed batch.
- **Fix**: Mixed optional negative prompts project missing values as empty strings; `is_multi_image_batch()` recognizes empty per-sample lists; Flux2/Klein emit aligned `None` latent/ID slots; and Bagel preserves empty condition-image slots. Arrow and offline-collator regressions cover the complete path.
- **Lesson**: A batch-level optional field still needs one explicit slot per row. Normalize at the preprocessing boundary while retaining the original record identity for provenance.
- **Related Constraint**: N/A

### Preprocessing cache identity includes precision policy
- **Date**: 2026-08-28
- **Symptom**: Changing `component_load_dtypes` or `frozen_parameters_dtype` could reuse condition embeddings computed under a different component precision policy.
- **Root Cause**: The offline cache fingerprint named the model but omitted the precision configuration that controls preprocessing component loading and storage.
- **Fix**: Offline condition-cache extras now include a sorted canonical JSON representation of both dtype policies, with regressions for load-policy changes, frozen-policy changes, and mapping-order stability.
- **Lesson**: Cache identity must include every configuration value that can change preprocessing numerics, even when that value is enforced during model loading rather than passed to the preprocessing function.
- **Related Constraint**: #20

### Batched Arrow schemas must cover later optional values
- **Date**: 2026-08-28
- **Symptom**: A valid optional-image dataset ordered as two prompt-only rows followed by two image-conditioned rows failed in the second `Dataset.map` chunk while casting image bytes or latent tensors to the first chunk's empty/string schema.
- **Root Cause**: HuggingFace fixed the writer schema from the all-empty first output chunk, while Flux2 also omitted its image output keys whenever that individual chunk contained no images.
- **Fix**: `data_utils/dataset.py` now scans real map-sized slices, dropping resolved columns from the bounded scan until it finds each later typed representative chunk; it probes those chunks only when a source column transitions from empty in the first chunk to typed later, restores Python/NumPy/torch/MPS and explicit-generator RNG state after that schema-only probe, and passes the resulting explicit `Features` through the `datasets==3.3.2`-compatible map surface. Flux2 now emits aligned image columns whenever the source image field exists; Flux2 and Flux2-Klein regressions verify identical empty-chunk output structure, a spy rejects whole-column reads, and a four-row offline cache regression covers the real Arrow path.
- **Lesson**: A batched map's output key set and feature types are dataset-level contracts, not properties of whichever values happen to appear in the current chunk. Optional adapters must emit empty row slots consistently, and data writers must derive schemas from representative typed values before committing the first Arrow batch. A schema probe is intentionally narrow and restores RNG, but preprocessors still own their normal deterministic/cache-safe behavior; arbitrary adapter-owned mutable state is not rollback-safe.
- **Related Constraint**: N/A

### Exact resume must lock future execution and preserve acquisition boundaries
- **Date**: 2026-08-28
- **Symptom**: An exact checkpoint could pass preflight after objective, seed, scheduler, ordered training-data, or replayed evaluation semantics changed; a remote-rank runtime-child failure could let peers return from resume; an online resume retried its immutable source `checkpoint-N`; offline save-before-eval captured RNG that did not match the next uninterrupted epoch; and MPS could publish an exact checkpoint that its loader would always reject.
- **Root Cause**: Resume identity stopped at physical model/optimizer/backend structure and treated RNG-consuming evaluation as operational, runtime-child commit was outside the synchronized distributed phase, one generic checkpoint/evaluation order ignored the different replay boundaries of generated versus finite-dataset acquisition, and save preflight did not reject a device RNG unsupported by Accelerate.
- **Fix**: Runtime identity now includes trainer-extensible execution and rank-free data-contract digests derived from resolved objective/forward settings, realized training loader provenance/order/geometry, and the ordered realized evaluation path (cadence, arguments, dataset overrides, rewards, and prepared loaders). Runtime-child commit and attachment use a synchronized all-rank error phase. The first online boundary skips a save only when its resolved real path equals the exact-resume source, while still evaluating; offline boundaries evaluate before saving so exact checkpoints capture post-evaluation RNG, model-only saves retain the same observable order without claiming RNG restoration, and MPS exact save fails before adapter or filesystem mutation.
- **Lesson**: Exact resume compatibility covers every computation and ordered data stream that can affect future state, including evaluation replay, not only the state container. Every rank-local resume mutation needs a synchronized failure boundary, checkpoint placement must match whether acquisition resumes before a rollout or after a completed data epoch, and save must not publish a state the matching load path cannot restore.
- **Related Constraint**: #18

### Multi-source schedule seeds must be process-independent
- **Date**: 2026-08-28
- **Symptom**: The same multi-source counts, configured seed, and epoch could produce a different source order on separate ranks or after restart.
- **Root Cause**: `WeightedSourceBatchScheduler` seeded its generator with Python's salted `hash()` over a tuple containing a string, so the result depended on each process's `PYTHONHASHSEED`.
- **Fix**: `data_utils/multi_source.py` now derives a domain-separated unsigned 64-bit seed from SHA-256; a subprocess regression compares schedules under distinct `PYTHONHASHSEED` values.
- **Lesson**: Never use Python object hashes as distributed or persistent RNG seeds. Derive seeds from an explicitly versioned, stable byte representation.
- **Related Constraint**: #9

### Exact resume must include non-param-group optimizer semantics
- **Date**: 2026-08-28
- **Symptom**: An exact checkpoint could pass compatibility preflight after a role's gradient clipping threshold or update frequency changed.
- **Root Cause**: Runtime identity covered realized optimizer parameter groups, but `max_grad_norm` and `update_frequency` are consumed by role optimization outside those groups.
- **Fix**: `trainers/common/runtime_identity.py` now hashes resolved per-role optimizer arguments, including algorithm-provided defaults; runtime-identity regressions verify clipping and cadence drift change the execution digest while operational controls remain mutable.
- **Lesson**: Exact-resume identity must cover every value that controls whether and how an optimizer update occurs, not only values serialized in optimizer parameter groups.
- **Related Constraint**: #18

### Distillation rollout cursors derive from persisted progress
- **Date**: 2026-08-28
- **Symptom**: DMD2, TDM, and TDM-R1 exact resumes restarted prompt acquisition from dataloader epoch zero even though the checkpoint recorded completed rollout iterations.
- **Root Cause**: The trainers retained a live Python iterator and local dataloader epoch counter, while exact runtime state persisted only `TrainingProgress`; infinite grouped samplers never raised `StopIteration`, so the local epoch counter did not describe their real position either.
- **Fix**: `trainers/distillation/distillation_runtime.py` now reconstructs the global consumed-batch cursor as `rollout_iteration * gradient_accumulation_steps`, uses the realized finite loader length before sampler/config fallbacks, maps the cursor to sampler epoch and intra-epoch offset, and restores Python/NumPy/torch CPU/CUDA/MPS plus explicit loader-generator RNG around iterator reconstruction and skips. Regressions compare uninterrupted and resumed real `DataLoader`, infinite grouped-sampler, and finite multi-source sequences.
- **Lesson**: Do not serialize Python iterators or maintain a second checkpoint authority. At legal checkpoint boundaries, derive replayable loader position from persisted progress plus identity-locked realized geometry, and treat iterator construction and replayed skips as RNG-consuming side effects that must be neutralized.
- **Related Constraint**: #18

### Offline media bytes belong to the exact data identity
- **Date**: 2026-08-28
- **Symptom**: Replacing target, chosen, or rejected media in place left the same path-based record digest, so exact-resume preflight accepted a run whose next on-the-fly VAE inputs had changed. Input media replacement could likewise reuse stale condition embeddings.
- **Root Cause**: Offline identities included normalized media type, path, and rate metadata but not file content.
- **Fix**: `data_utils/offline_dataset.py` streams each unique normalized media path through SHA-256 once per source build. Input digests participate in condition IDs and cache fingerprints; supervision digests participate in full record IDs. No media, decoded pixels, or VAE latents are copied or cached.
- **Lesson**: A path is provenance, not immutable content. Exact future-data contracts and preprocessing caches must identify external file bytes when those bytes are decoded on demand.
- **Related Constraint**: #18

### Runtime identity excludes transport-only global source IDs
- **Date**: 2026-08-28
- **Symptom**: Inserting or reordering an eval-only dataset renumbered training sources and caused exact-resume rejection even though the ordered training data and reward mathematics were unchanged.
- **Root Cause**: Data identity hashed the full global name-to-ID registry and offline numeric `source_id`, which are transport metadata assigned across both train and eval entries.
- **Fix**: `trainers/common/runtime_identity.py` now locks ordered realized training source names and loader schemas while excluding global numeric IDs. Structural and real-`Arguments` regressions verify eval-only changes preserve both execution and data digests, while training-source order/count changes remain incompatible.
- **Lesson**: Compatibility identities should include semantic names and order, not remappable integer handles whose only purpose is runtime transport.
- **Related Constraint**: #18

### Offline epoch semantics follow the realized official sampler
- **Date**: 2026-08-28
- **Symptom**: Loader documentation claimed that every global sample appears exactly once and that an offline epoch is never padded, while the intentionally selected official `DistributedSampler(drop_last=False)` repeats tail indices when dataset size is not divisible by world size.
- **Root Cause**: The design correctly defined an epoch as a complete rank-local dataloader traversal, but documentation conflated that with global sample uniqueness and with the separate prohibition on inventing batches to close a gradient-accumulation window.
- **Fix**: Loader, workflow, dataset, sampler, and constraint documentation now preserve PyTorch's standard tail-equalization semantics and state that the framework adds no batches merely for accumulation. Existing uneven-geometry tests continue to lock official sampler behavior.
- **Lesson**: When delegating sharding to an official sampler, define epoch semantics over its realized finite loader. Distinguish sampler-level repeated indices from optimizer-level synthetic padding.
- **Related Constraint**: #9

## Cross-refs

- `constraints.md` (archival target for constraint violations)
- `architecture.md` (archival target for data-flow misunderstandings)
- `ff-debug/SKILL.md` Phase 5 (knowledge capture workflow)
