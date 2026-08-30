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

### Distributed reward gathering must preserve reconstruction invariants
- **Date**: 2026-08-30
- **Symptom**: Two-rank H3 Ref2VA GRPO failed before reward execution because `MiniMaxH3Ref2VASample` was reconstructed with `reference_manifest=None`.
- **Root Cause**: The distributed group-reward path gathered only reward-consumed fields, although `gather_samples` reconstructs the concrete sample class and that class can require additional state.
- **Fix**: `BaseSample` now declares an empty `reconstruction_required_fields` contract, `OrderedReferenceConditionSample` adds `reference_manifest`, and `RewardProcessor` unions that contract into its distributed gather fields without forwarding it to the reward call.
- **Lesson**: Communication payload requirements and reward-call requirements are distinct contracts; partial gathers must preserve constructor invariants even for fields that downstream computation does not consume.
- **Related Constraint**: N/A

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

### Recipe migrations must move tests and documentation with the config
- **Date**: 2026-08-28
- **Symptom**: Rebasing onto the precision-aware loading branch changed the MiniMax H3 T2VA default to the shared `vid_prompt` source, added ImageBind routing, and removed its old unvalidated warning, while the executable example test and user guides still required the prior dataset and wording.
- **Root Cause**: The recipe-only commit updated YAML semantics without treating example assertions, dataset links, dependency notes, and validation-status language as one public workflow contract.
- **Fix**: The T2VA default test now locks the shared TXT manifests and CLAP/ImageBind routing while retaining the dedicated JSONL fixture check for the validated debug recipe. The example and dataset guides now describe the shared source, ImageBind dependency, and the exact evidence boundary without claiming a completed long run.
- **Lesson**: An example configuration is executable documentation. Any recipe migration must update its production parse test, linked data provenance, optional dependency instructions, and validation claims in the same integration change.
- **Related Constraint**: #15

### Optional-kernel adapter tests must lazy-load behind the dependency seam
- **Date**: 2026-08-28
- **Symptom**: Collecting the Bagel TDM contract test on macOS failed before any test ran because `flash-attn>=2.5.8` was unavailable.
- **Root Cause**: The test imported the Bagel adapter at module scope instead of installing the existing fake optional-kernel modules before the adapter import.
- **Fix**: `tests/models/test_bagel_tdm_contracts.py` now lazily imports Bagel after stubbing `flash_attn`, OpenCV, and the availability probes; new Python files also receive the required Apache 2.0 headers.
- **Lesson**: Contract tests for CUDA-only optional adapters must exercise the adapter through its dependency boundary so CPU and macOS collection remains valid; importing such adapters at module scope turns an optional dependency into a repository-wide test dependency.
- **Related Constraint**: N/A

### Distillation exact cursors count rollout batches, not backend work items
- **Date**: 2026-08-29
- **Symptom**: After timestep-aligned TDM accumulation made each trajectory boundary one backend work item, exact resume skipped `num_inference_steps` times too many prompt batches.
- **Root Cause**: Cursor reconstruction still multiplied completed rollout iterations by backend `gradient_accumulation_steps`, even though one rollout now contributes multiple boundary losses to that accumulation window.
- **Fix**: `trainers/distillation/distillation_runtime.py` now derives consumed prompt batches through `resolve_rollout_accumulation_steps()`, and the cursor regression locks `gradient_accumulation_steps=8`, four losses per rollout, and two completed iterations to four consumed batches.
- **Lesson**: Persisted acquisition progress must be projected through the current acquisition-to-backend work-item ratio. Backend GAS is not a valid dataloader cursor when one acquired batch expands into multiple backward graphs.
- **Related Constraint**: #18a

### Sparse media arguments require semantic input slots
- **Date**: 2026-08-29
- **Symptom**: A last-frame-only MiniMax H3 record could not be represented without pretending its
  image was the first frame, and heterogeneous Ref2VA cardinality rules could not be expressed by
  independent per-type counts.
- **Root Cause**: The public offline schema and pipeline contract treated media position and
  per-modality cardinality as the complete binding model.
- **Fix**: V2 input media now accepts an input-only semantic `slot`; contracts declare ordered and
  required slots plus aggregate count/required-any-type rules; projection resolves explicit slots
  first and fills remaining slots positionally. Outputs reject slots. Construction rejects
  multi-slot rules that claim order-insensitivity and aggregate bounds that cannot satisfy their
  per-type rules.
- **Lesson**: Use generic argument-binding metadata for sparse conditions and keep model-specific
  argument names in adapter contracts, not algorithm code or ad-hoc dataset columns.
- **Related Constraint**: #5

### Offline velocity objectives must bypass unused scheduler transitions
- **Date**: 2026-08-29
- **Symptom**: LTX2 near-clean offline targets lost velocity precision after a velocity-to-x0-to-
  velocity round trip, while exact velocity-only Wan/LTX forwards still invoked scheduler steps that
  their loss never consumed. LTX2 I2AV also dropped cached negative prompts during preprocessing.
- **Root Cause**: Generation-oriented forward paths performed transition reconstruction before
  checking the requested offline component, and the I2AV preprocessing override failed to forward
  one base prompt argument.
- **Fix**: LTX2 retains official online reconstruction by default but opts offline forwards into raw
  model velocity; Wan and LTX return exact velocity requests before scheduler stepping; I2AV now
  forwards `negative_prompt` explicitly. Component, parity, and initialization regressions cover
  the split behavior.
- **Lesson**: Offline objectives may share a model forward with generation but must not inherit
  numerically lossy or RNG-consuming transition work that is outside their requested output.
- **Related Constraint**: #7

### Documented dataset tools must use their package invocation mode
- **Date**: 2026-08-30
- **Symptom**: Running the documented `python dataset/offline_smoke/prepare.py ...` command failed before argument parsing with `attempted relative import with no known parent package`.
- **Root Cause**: The package uses relative imports, while the documentation incorrectly advertised direct file execution instead of Python's module mode.
- **Fix**: All offline-smoke commands now use `python -m dataset.offline_smoke.<tool>` with unconditional package imports, and a subprocess regression executes the documented form.
- **Lesson**: A checked-in CLI example is part of the public interface; standardize on one package-aware invocation and test that exact process rather than adding conditional import fallbacks.
- **Related Constraint**: N/A

### Exact resume must lock the checkpoint-realized pipeline contract
- **Date**: 2026-08-29
- **Symptom**: Exact resume could accept a checkpoint after an in-place model configuration change
  switched a Wan adapter between first-only and first/last-frame semantics.
- **Root Cause**: Runtime identity hashed model arguments and trainer execution but omitted the
  adapter's resolved `effective_pipeline_io_contract`.
- **Fix**: The default execution identity now canonicalizes and hashes the realized pipeline I/O
  contract after adapter initialization; a regression changes only that contract and observes only
  the execution digest change.
- **Lesson**: Any checkpoint-dependent specialization that changes legal inputs or forward binding
  is future-execution state and belongs in exact-resume identity.
- **Related Constraint**: #18

### Offline condition caches need contract-stable schemas across sources
- **Date**: 2026-08-29
- **Symptom**: Changing only semantic slot order could reuse an Arrow cache with the old media
  projection, while a multi-source batch could fail because an all-empty optional source omitted
  columns that a populated source emitted.
- **Root Cause**: The source hash covered record identities but not the effective input projection
  contract, and projection decided column existence from each source's observed values.
- **Fix**: Condition source identity now includes the canonical effective input contract. With a
  contract, negative-prompt, declared media, and semantic-slot columns are projected consistently
  even when every row in one source is empty. A real two-source `DistributedSampler` loader
  regression mixes empty and populated optional conditions in one batch.
- **Lesson**: A concatenated cache schema is defined by the model contract, not by local source
  sparsity; cache identity must cover every declaration that can reorder or reshape projection.
- **Related Constraint**: #9

### Distributed Arrow schemas must be inferred before rank sharding
- **Date**: 2026-08-29
- **Symptom**: A distributed condition-cache build could write `List(null)` on an all-empty rank
  and `List(Image)` on a populated rank, then fail when the per-rank Arrow files were consolidated.
- **Root Cause**: Cross-chunk schema discovery ran after rank sharding, so each process inferred
  features from only its local value distribution. The standalone cache entry point also ignored a
  pipeline's single-sample preprocessing capability.
- **Fix**: Distributed preprocessing now derives one explicit feature schema from the full source
  before selecting rank-local rows, and both cache entry points force batch size one for ordered or
  `SINGLE_SAMPLE` contracts. A real two-part Arrow regression separates empty and populated rows
  across ranks, consolidates the files, and loads the merged dataset.
- **Lesson**: Distributed writers need a global serialization contract even when their data is
  disjoint. Batch capability is likewise part of the preprocessing boundary, not only trainer
  orchestration.
- **Related Constraint**: #9

### Validate output candidates before stochastic condition preparation
- **Date**: 2026-08-29
- **Symptom**: An invalid offline target correctly raised an exception but first consumed condition
  preparation RNG, so retrying with corrected media no longer reproduced the original encoding.
- **Root Cause**: `BaseAdapter.encode_output_state()` prepared raw conditions before validating the
  generator and exact output-media sequence.
- **Fix**: The lifecycle wrapper now validates generator type and candidate media before invoking
  any condition preparer or codec. A stochastic-preparer regression proves invalid media leaves the
  explicit generator unchanged and performs no preparation work.
- **Lesson**: Pure boundary validation must precede expensive or random transformations. Failed
  inputs should not mutate the state that determines a later valid retry.
- **Related Constraint**: #7

### Aggregate media guarantees must be canonical contract state
- **Date**: 2026-08-29
- **Symptom**: `INPUT_MEDIA` geometry rejected a valid contract whose aggregate `min_total_count` or
  `required_any_types` guaranteed a condition, while semantically identical required-type tuples
  in different orders produced different cache and resume identities.
- **Root Cause**: Geometry validation recognized only per-type minima, and the set-like aggregate
  field had no canonical ordering rule.
- **Fix**: Input-derived geometry now accepts every nonempty guarantee enforced by runtime
  validation, `required_any_types` must follow canonical media-type order, and required slots must
  follow their declaration order.
- **Lesson**: Declarative invariants should be interpreted consistently at construction and runtime,
  and set-like identity fields require one canonical representation.
- **Related Constraint**: #5

### ZeRO optimizer identity must use logical model groups
- **Date**: 2026-08-30
- **Symptom**: Every ZeRO-2 trainer failed during initialization because its optimizer schema
  contained a parameter not owned by the rebound component-variant registry.
- **Root Cause**: DeepSpeed ZeRO-1/2 replaces each public optimizer group with a rank-local flat
  FP32 master partition, while runtime identity incorrectly treated those partitions as the live
  model parameters owned by the registry.
- **Fix**: Runtime identity now maps stable parameter ownership through DeepSpeed's retained
  `bit16_groups` and continues to serialize settings from the public optimizer groups. It fails
  closed if logical groups are absent or do not match the partitioned group count.
- **Lesson**: A distributed optimizer's public parameter groups may be physical state partitions;
  exact-resume identity must separate logical model ownership from physical group settings.
- **Related Constraint**: #18a

### Custom Transformers models must implement their declared checkpointing seam
- **Date**: 2026-08-30
- **Symptom**: Every Bagel trainer failed during initialization when full gradient checkpointing
  called `Qwen2ForCausalLM.gradient_checkpointing_enable()` and Transformers reported that the
  architecture was incompatible.
- **Root Cause**: Bagel's custom Qwen2-NaViT model inherited
  `supports_gradient_checkpointing = True` but did not expose a `gradient_checkpointing` state or
  invoke the installed checkpoint function in its active decoder loop.
- **Fix**: The custom Qwen2 model now owns the standard checkpointing flag and routes each pure
  decoder layer through Transformers' installed non-reentrant checkpoint function while training.
  Cache-updating and TaylorSeer paths remain direct to avoid replaying mutations. A backward
  regression proves the layer is recomputed rather than merely accepting the API call.
- **Lesson**: A custom `PreTrainedModel` must pair its capability declaration with both the state
  seam expected by Transformers and an execution-time checkpoint boundary in the forward path.
- **Related Constraint**: #7

### FSDP wrap metadata must follow the instantiated architecture variant
- **Date**: 2026-08-30
- **Symptom**: Every Bagel FSDP2 trainer failed during `accelerator.prepare()` because Accelerate
  could not find the declared `Qwen2DecoderLayer` in the loaded model.
- **Root Cause**: Bagel's custom Qwen2 classes inherited fixed `_no_split_modules` metadata from the
  standard decoder even though `config.layer_module` instantiated a MoE or MoT decoder variant.
- **Fix**: Both the inner Qwen2 model and outer causal LM now derive their no-split class from the
  realized `layer_module`. CPU FSDP2 auto-wrap regressions verify all decoder variants resolve
  through the same PEFT wrapper used by LoRA training.
- **Lesson**: Distributed wrap metadata is realized model state. Config-selectable architectures
  must not inherit a fixed block class that may be absent from the instantiated module tree.
- **Related Constraint**: #9

### Parameter-sharded submodule work must remain inside the prepared root forward
- **Date**: 2026-08-30
- **Symptom**: Every Bagel FSDP2 trainer reached sampling or offline replay but failed at token
  embedding with a mixed `torch.Tensor` and `DTensor` operator error.
- **Root Cause**: Bagel's cache helpers and denoising path reached through the physical pipeline to
  `language_model.model.embed_tokens` before calling the routed transformer. Decoder layers owned
  nested FSDP groups, but embedding and final normalization belonged to the prepared `ModelBundle`
  root, whose unshard hook was bypassed by those direct calls.
- **Fix**: The outer Qwen forward now accepts raw packed token IDs and inserts their embeddings into
  an optional query-local auxiliary sequence. Bagel cache helpers accept an injected language-model
  forward, and the adapter supplies its routed transformer for text, VAE, ViT, denoising, and CFG
  passes. Each logical language-model pass therefore performs embedding, decoder execution, and
  final normalization inside one prepared-root call.
- **Lesson**: Under compositional FSDP, wrapping transformer blocks does not make arbitrary child
  access safe. Any computation using parameters owned by the prepared root must execute beneath
  that root's forward hooks; converting ordinary inputs to `DTensor` or manually unsharding only
  masks the first failure and breaks distributed lifecycle semantics.
- **Related Constraint**: #9

### FSDP2 activation checkpoints must replay inside the mixed-precision boundary
- **Date**: 2026-08-30
- **Symptom**: All four Wan FSDP2 trainers failed on their first backward because checkpointed
  tensors were saved as BF16 but recomputed as FP32.
- **Root Cause**: Model-level checkpointing captured FP32 block inputs before FSDP2's forward-input
  cast, while backward replay re-entered a block in `PRE_BACKWARD` state where PyTorch deliberately
  skips that cast.
- **Fix**: When full model checkpointing and FSDP2 activation checkpointing are both requested, the
  trainer now disables model-level boundaries and keeps Accelerate's backend checkpoint wrappers,
  which replay inside the fully-sharded mixed-precision boundary. Selective policies fail closed
  because backend checkpointing cannot preserve their exact selection, while FSDP1 retains its
  existing owner. Wan FSDP2 GRPO, TDM, SFT, and offline DPO plus SD3.5 and Bagel regressions verify
  the shared path.
- **Lesson**: Checkpoint placement is part of distributed precision semantics. A recompute boundary
  outside a sharded module may not replay its forward hooks, so backend-aligned checkpoint wrappers
  must own FSDP2 full checkpointing instead of nesting model-level boundaries around sharded blocks.
- **Related Constraint**: #9, #20

### Adapter dtype manifests may span optional checkpoint components
- **Date**: 2026-08-30
- **Symptom**: Every Wan2.2 TI2V trainer rejected the Wan I2V adapter's `image_encoder` load-dtype
  default even though the checkpoint legitimately omits that optional component.
- **Root Cause**: The eager loader validated an adapter-wide dtype manifest only against components
  present in one checkpoint, conflating the checkpoint instance with the pipeline class's wider
  optional-component contract.
- **Fix**: Eager pipeline loading now validates adapter manifest selectors against the union of the
  checkpoint components and the pipeline class's declared optional components, while resolving
  dtype arguments only for components actually present. User overrides remain strict against the
  selected checkpoint. Regressions cover absent and present optional components, invalid manifest
  selectors, and explicit user selection of an absent component.
- **Lesson**: Adapter defaults may intentionally cover several checkpoint variants of one pipeline
  class. Optional class-level declarations belong to manifest validation, but they must not create
  components or weaken the fail-fast contract for checkpoint-specific user overrides.
- **Related Constraint**: #20

### Absent optional components need distinct physical roots
- **Date**: 2026-08-30
- **Symptom**: Every Wan2.2 TI2V trainer rejected its load plan because the physical
  `image_encoder` root appeared to combine incompatible auxiliary and host roles.
- **Root Cause**: The eager runtime used object identity to collapse logical aliases, so multiple
  optional components whose value was the singleton `None` were mistaken for one shared physical
  object.
- **Fix**: Classic pipelines now preserve a declared optional `None` under its own logical root
  while retaining identity aliasing for real objects. The load coordinator finalizes only
  replicated roots that actually materialized, preventing FSDP replica checks from resolving an
  allowed absent component.
- **Lesson**: Object identity establishes physical aliasing only for materialized objects. Optional
  declarations retain distinct lifecycle identities, and backend finalization must follow observed
  materialization rather than the requested name set.
- **Related Constraint**: #9

### Supported adapter paths require their upstream optional dependencies
- **Date**: 2026-08-30
- **Symptom**: Every Wan I2V trainer reached prompt preprocessing and then failed with
  `NameError: name 'ftfy' is not defined` inside Diffusers prompt normalization.
- **Root Cause**: Diffusers imports `ftfy` conditionally and declares it only in development/test
  extras, while its Wan I2V prompt helper calls the package unconditionally. Flow-Factory exposes
  Wan I2V as a core adapter but did not close that runtime dependency gap.
- **Fix**: `ftfy` is now a core project dependency, with a metadata regression that keeps exactly
  one install requirement for Wan prompt normalization.
- **Lesson**: A framework that promotes an upstream optional code path to a supported core feature
  also owns the path's transitive runtime dependencies; successful import of the upstream module
  does not prove its conditionally imported helpers are callable.
- **Related Constraint**: N/A

## Cross-refs

- `constraints.md` (archival target for constraint violations)
- `architecture.md` (archival target for data-flow misunderstandings)
- `ff-debug/SKILL.md` Phase 5 (knowledge capture workflow)
