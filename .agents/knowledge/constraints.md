# Hard Constraints

Quick index: **#1-5** Registry | **#6-10** Training Pipeline | **#11-14** Base Classes | **#15-17** Config | **#18-20** Distributed | **#21-27** Code Quality | **#28-29** Agent Workflow

These constraints MUST NOT be violated. Consult this file before making any code changes.

---

## Registry & Loading (1–5)

### 1. Registry Path Accuracy
The four registries (`_TRAINER_REGISTRY`, `_MODEL_ADAPTER_REGISTRY`, `_REWARD_MODEL_REGISTRY`, `_ACCELERATOR_REGISTRY`) map string identifiers to **fully qualified Python class paths** for lazy import. If you move, rename, or restructure a class, the corresponding registry entry MUST be updated, or `ImportError` will occur at runtime.

### 2. Registry Identifier Convention
Registry keys are **case-insensitive** (lowered at lookup). Model adapter keys use lowercase with hyphens (e.g., `flux1-kontext`). Trainer keys use lowercase (e.g., `grpo-guard`). Reward keys use lowercase (e.g., `pickscore`). New entries must follow the same convention.

### 3. Dynamic Import Fallback
All four registries support a **direct Python path** fallback (e.g., `my_package.models.CustomAdapter`). If an identifier is not found in the registry, it is treated as a fully qualified import path. Do not break this two-mode resolution logic.

### 4. Decorator Registration
`@register_trainer` and `@register_reward_model` decorators exist for convenience but the canonical entries are the static dicts. If you use the decorator, ensure the static dict is also updated if the class should be discoverable by default.

### 5. Adapter Component Runtime Contract
Existing `BaseAdapter` subclasses keep implementing `load_pipeline()` and use the default
`ClassicPipelineRuntime`. Adapters backed by lazy modular pipelines or explicit pseudo-pipeline
containers override the concrete `build_component_runtime()` hook instead; they must retain
`adapter.pipeline` as the backend compatibility alias and ensure the canonical scheduler is
materialized before scheduler construction. Trainer stage lifecycle must call the adapter's public
component methods, preserving model-specific override points. Runtime-wide device and dtype
enumeration includes only materialized canonical `torch.nn.Module` entries; declared lazy specs,
optional `None` entries, and pseudo-pipeline aliases are not implicitly loaded or moved.
`materialize_components(None)` means already-materialized modules, never all declared specs.
Text-encoder/transformer role groups include non-`None` declarations/specs only.
The canonical scheduler remains `adapter.scheduler`; `adapter.scheduler_group` owns ordered mode
and seed dispatch, and its immutable names must equal `trajectory_component_order`.

---

## Training Pipeline (6–10)

### 6. Execution Contract and Stage Order
Every trainer declares an immutable `ExecutionContract`; acquisition, outer-cycle unit, training
feedback, and loader distribution are independent of the algorithm's `paradigm`. Online
reward-based execution preserves the six-stage order: Data Preprocessing → K-Repeat Sampling →
Trajectory Generation → Reward Computation → Advantage Computation → Policy Optimization.
Reward-free online distillation omits only the declared feedback stages. Offline execution is a
finite `DistributedSampler` dataloader traversal with per-batch optimization; it MUST NOT emulate
this by making `sample()` a no-op or by materializing the entire epoch. One complete offline
dataloader traversal is one `data_epoch`, and an interrupted traversal does not advance it.
Periodic boundaries are acquisition-specific: online execution preserves its historical
pre-rollout save/eval boundary, while offline execution runs save/eval only after the dataloader
has been exhausted, `_after_training_cycle()` has returned, and `data_epoch` has advanced. A
failed or interrupted offline traversal therefore reaches neither epoch advancement nor its
periodic boundary.
Runtime reward feedback is not yet implemented for dataset acquisition and MUST fail at contract
construction instead of being silently ignored; SFT and offline preference training consume their
supervision directly from each batch. Any `FeedbackMode.NONE` algorithm rejects training
`rewards` while still permitting explicit `eval_rewards`. Offline configuration requires an
explicit positive integer `gradient_accumulation_steps`, unit dataset-source weights, and leaves
the online `sampler_type` at `auto`; it never resolves or aligns K-repeat/group geometry and never
multiplies accumulation by the number of training timesteps. Offline flow-matching algorithms
average all configured timestep loss terms within one dataloader-batch microstep.

### 7. Coupled vs Decoupled Paradigm
- **Coupled** (GRPO, GRPO-Guard, DPPO): Training timesteps are coupled with SDE-based sampling. Requires log-probability computation. Must use SDE dynamics (`Flow-SDE`, `Dance-SDE`, `CPS`).
- **Decoupled** (SFT, online/offline DPO, NFT, AWM, DGPO, CRD): Training timesteps are decoupled from sampling or dataset acquisition. Can use any dynamics including `ODE`; SFT has no policy-ratio or reference-model requirement.
- **Distillation** (`diffusion-opd`): On-policy multi-teacher distillation; dynamics-agnostic (ODE or SDE) and has no reward/advantage stage.

Mixing paradigms (e.g., using `ODE` dynamics with `GRPO`) will produce incorrect gradients silently.

### 8. Component Offloading Lifecycle
Input encoders are loaded for Stage 1 and may be offloaded after cached conditions are built. Online
execution reloads the components required for inference. Offline target pixels/latents MUST NOT be
preprocessed or cached: output-codec components such as the VAE remain runtime-available and encode
each fetched target batch under `torch.no_grad()`. Do not assume identical component lifecycles
across acquisition modes. Before offline input preprocessing or cache lookup, every normalized
input and the preprocessor's grouped/ordered binding MUST be validated against the adapter's
`PipelineIOContract`; unsupported media or negative prompts must never be silently dropped.
Condition `encode_*` methods and output-state codecs MUST remain separate lifecycle boundaries:
the former owns reusable input caching and condition orchestration, while the latter owns
on-the-fly target geometry, packing, masks, and forward/decode context. When both roles use the
same checkpoint VAE transform, they MUST call one role-neutral numerical encoding primitive rather
than maintain duplicate transforms.
An offline-capable adapter MUST declare a static `PipelineIOContract` and a real
`build_output_state_codec()` implementation. A known blocker MUST instead be declared through a
non-empty actionable `output_state_codec_unavailable_reason`; it MUST NOT be hidden behind a fake
codec builder or by omitting one output modality. Offline trainer loading validates this boundary
before Accelerator construction or model loading, while online execution remains unaffected.

### 9. Accelerator `prepare()` Scope
All target components (trainable **and** frozen-but-shardable) are bundled into a single `ModelBundle` (`models/model_bundle.py`) and prepared with the **optimizer** as one root via `accelerator.prepare()` — DeepSpeed (one engine) and FSDP2 (one root) cannot prepare multiple models separately. After prepare, each component is exposed as a `RoutedComponentProxy` that routes forwards through the bundle root; the optimizer/EMA/reference params still target only the `requires_grad` subset (frozen members are sharded for memory but never trained). Online train dataloaders use the framework's rank-aware grouped samplers; offline train dataloaders use PyTorch's official `DistributedSampler`, including for one-process execution. Neither train loader is passed to `accelerator.prepare()` or `prepare_data_loader()`, because both are already distributed and a second shard would duplicate or drop data. Eval dataloaders remain prepared with the model bundle and optimizer in the single root call.

### 9a. Sampler Geometric Constraints
`DistributedKRepeatSampler` and `GroupContiguousSampler` require `M * K ≡ 0 (mod W * B * G)` where M=unique_sample_num, K=group_size, W=world_size, B=per_device_batch_size, G=gradient_step_per_epoch — **unless** `gradient_accumulation_steps` is set manually, in which case the constraint reduces to `M * K ≡ 0 (mod W * B)`. **GroupContiguousSampler** adds: `M ≡ 0 (mod W)`. **GroupDistributedSampler** (DGPO) requires: `K % W == 0` and `(W * B) % K == 0`; auto-aligned by `_align_for_group_distributed`. See `topics/samplers.md` for full details.

### 9b. Checkpoint Save/Load Symmetry Under the Bundle
Checkpoints are written and read for **trainable members only** — components whose `target_module_map[name]` is non-empty (`adapter.trainable_component_names`). Frozen-but-shardable bundle members (e.g. Wan2.2's `transformer_2`, kept in `target_components` only to be FSDP-sharded for memory; see #9) map to `None` and are skipped by both `save_checkpoint` and `_load_lora`/`_load_full_model`. Loaders MUST iterate `trainable_component_names`, not `target_components`, or resume logs a spurious error for a per-component subdir that was never written. `resume_type='state'` restores via `accelerator.load_state` into the prepared bundle root and is therefore keyed to bundle membership — resuming into a different `target_components` / bundle composition will mismatch.

Offline exact-state runtime v1 MUST NOT register an Accelerate custom checkpoint object because
Accelerate loads those files through pickle. It uses an immutable staged JSON+safetensors sidecar
and validates algorithm/model identity, canonical parameter ordering, optimizer groups, core
artifact SHA-256 digests, RNG, and EMA/reference payloads before policy mutation. This exact-state
path is single-process only. Distributed offline training MUST use model-only checkpoints until a
collective atomic runtime format exists; state save/resume must fail fast rather than restore only a
subset.

### 10. DeepSpeed ZeRO-3 Is Unsupported
Supported distributed plans are DDP, FSDP, and DeepSpeed ZeRO-1/2. Reward model sharding under ZeRO-3 is broken even with DeepSpeed's own `zero.GatheredParameters` context manager, and parameter sharding also breaks frozen-component synchronization. `validate_supported_distributed_plan` (`trainers/abc.py`) rejects it at `BaseTrainer.__init__`, before any weights load, and `config/deepspeed/` ships no ZeRO-3 profile. Multi-role training narrows this further: `_validate_multirole_backend` requires ZeRO-1/2 and, under FSDP2, `use_orig_params=True`.

---

## Base Class Interfaces (11–14)

### 11. BaseTrainer Execution Contract
`BaseTrainer.__init__` expects `(accelerator, config, adapter)`. `start()` owns common save/eval
boundaries and delegates each outer cycle to the driver selected by `execution_contract`.
`TrainingProgress` tracks `optimizer_step`, `rollout_iteration`, and `data_epoch` independently;
the legacy `step` alias maps to the primary optimizer step, while `epoch` maps to the contract's
cycle unit. Online trainers override `optimize(samples)`. Offline trainers override
`optimize_batch(batch)`. Both hooks are concrete fail-fast methods, and initialization verifies the
matching hook was overridden; subclasses must not add a fake implementation for the other mode.
Prefer `sampling_context`, `_run_training_step`, `_after_gradient_step`, and
`_after_training_cycle` over restating the loop. `_initialization()` continues to own dataloaders,
optimizer/preparation, feedback runtime, and model lifecycle.

**Online rollout order**: seed → periodic save/eval boundaries → `sample()` (Stages 2–3) →
`prepare_feedback()` when declared (Stages 4–5) → `optimize()` (Stage 6) → shared EMA →
`_after_training_cycle()` → increment `rollout_iteration`. The online boundary therefore uses the
pre-cycle completed-rollout index, preserving the historical index-zero boundary. Online
`DPOTrainer` forms chosen/rejected pairs at the start of `optimize()`. **Offline epoch order**:
official `DistributedSampler.set_epoch(data_epoch)` → exhaust the finite dataloader through
`optimize_batch()` → `_after_training_cycle()` → increment `data_epoch` → periodic save/eval
boundaries. Here one `data_epoch` is exactly one complete dataloader traversal, so the offline
boundary uses the newly completed-epoch index and never represents a partial traversal. Offline
shared EMA and `_after_gradient_step()` use optimizer-step cadence; online shared EMA uses
rollout-iteration cadence.

**Trainer hierarchy**: New trainers MUST inherit directly from `BaseTrainer`. The only sanctioned exceptions are strict behavioral variants of GRPO that change only the per-step loss while reusing GRPO's sampling/advantage/eval machinery: `GRPOGuardTrainer → GRPOTrainer` (adds ratio-normalization) and `DPPOTrainer → GRPOTrainer` (replaces the PPO ratio-clip with a KL trust-region mask). Trainer-to-trainer inheritance creates fragile coupling; when in doubt, inherit from `BaseTrainer` and extract shared logic into helper methods. All reward-based trainers delegate advantage computation to `self.advantage_processor.compute_advantages()`. Reward-free online trainers MUST declare `ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT`; the shared feedback gate omits reward and advantage dispatch.

### 12. BaseAdapter Abstract Methods
Subclasses of `BaseAdapter` MUST implement these **4 abstract methods**:
- `load_pipeline()` → returns the adapter backend pipeline/container (the default runtime expects
  a DiffusionPipeline-compatible eager object)
- `decode_latents()` → latents → pixels
- `inference()` → full multi-step denoising (corresponds to pipeline `__call__`)
- `forward()` → single-step denoising for training loss computation

**Optional encoder overrides (no-op default)**: All four per-modality encoders are non-abstract on `BaseAdapter`. Their default body is `pass` (returns `None`). Override only the modalities your model actually consumes — text/image/video-only adapters do **not** need stub `pass` overrides for unused modalities.
- `encode_prompt()` → text → embeddings
- `encode_image()` → image → latents
- `encode_video()` → video frames → latents
- `encode_audio()` → audio waveforms → embeddings/features

Note: `preprocess_func()` is a **concrete method** on `BaseAdapter` that dispatches to all four encoders (`prompt`, `images`, `videos`, `audios`) and skips integration when the called encoder returns `None`. It does NOT need to be overridden unless the model requires cross-modal preprocessing (e.g. prompt rewriting from images).

Breaking the signature of any of the four abstract methods (or changing the encoder return contract from "dict-or-`None`") breaks the entire training pipeline.

**Adapter hierarchy**: All model adapters MUST inherit directly from `BaseAdapter` — never from another adapter. Shared logic between adapters for the same model family should use private helper functions, code duplication, or mixins — not adapter-to-adapter inheritance. Adapter subclassing creates fragile coupling where changes to a parent adapter silently break child adapters, and makes the 4-abstract-method contract harder to verify (the 4 per-modality encoders have no-op defaults, so a fresh subclass of `BaseAdapter` is always valid; chained inheritance hides which encoder a model actually overrides).

### 13. BaseRewardModel Paradigm Split
- `PointwiseRewardModel.__call__` receives batches of size `batch_size`, returns rewards of shape `(batch_size,)`
- `GroupwiseRewardModel.__call__` receives all samples in a group (size `group_size`), returns rewards of shape `(group_size,)`

The `RewardProcessor` dispatches differently based on the model type. Do not change the calling convention.

### 14. Sample Dataclass Hierarchy
`BaseSample` → `T2ISample`, `ImageConditionSample`, `T2VSample`, `T2AVSample`, etc. The `_shared_fields` class variable determines which fields are NOT stacked across a batch. Incorrect `_shared_fields` causes silent data corruption during collation.

**Two-layer hierarchy**: Task-level samples (`T2ISample`, `I2VSample`, `I2AVSample`, ...) are defined in `samples/samples.py` and inherit from `BaseSample` or its condition mixins (`ImageConditionSample`, `VideoConditionSample`). Model-specific samples (`LTX2Sample`, `LTX2I2AVSample`, ...) MUST inherit from the appropriate task-level sample — never from another model-specific sample across files. This mirrors the flat adapter hierarchy: `LTX2I2AVSample(I2AVSample)`, NOT `LTX2I2AVSample(LTX2Sample)`.

Legacy trajectory fields remain authoritative when `BaseSample.trajectory is None`. Structured
trajectory collation requires identical ordered component keys and shared state/log-prob index maps;
component mapping iteration never defines scheduler RNG order.

---

## Configuration System (15–17)

### 15. Pydantic Hparams Synchronization
All config dataclasses live in `hparams/`. The top-level `Arguments` aggregates `DataArguments`, `ModelArguments`, `TrainingArguments`, `RewardArguments`, `LogArguments`, etc. Field changes MUST be reflected in:
1. The dataclass definition
2. ALL YAML configs under `examples/` (renames/removals: search-replace; new user-facing fields: add with defaults and `# Options:` comments)
3. Any code that accesses `config.<field_name>`

### 16. Algorithm-Specific Training Args
`TrainingArguments` has algorithm-specific subclasses, including separate `DPOTrainingArguments`
for online DPO and `OfflineDPOTrainingArguments` for finite preference data, plus
`SFTTrainingArguments` for demonstrations. The correct subclass is resolved by
`get_training_args_class()` (registry in `hparams/training_args/_registry.py`). Adding a new
algorithm requires a corresponding subclass and resolver entry.

### 17. YAML Config Structure
Config keys must exactly match Pydantic field names. Typos fail silently with default values. See
`examples/` for canonical templates; structure is defined in `hparams/args.py`. Offline split JSONL
uses strict schema version 2 and the public `type` discriminator; algorithm names never appear in
the record schema.

---

## Distributed Training (18–20)

### 18. All-Rank Synchronization Points
`accelerator.wait_for_everyone()` must be called at critical synchronization points (after preprocessing, before/after evaluation, checkpoint saving). Missing barriers cause deadlocks or race conditions.

### 19. FSDP CPU Efficient Loading
When using FSDP with CPU offloading, frozen components (text encoder, VAE) may be uninitialized on Rank > 0. The `_synchronize_frozen_components()` method handles this. Do not remove or bypass it. Lazy components must be materialized before synchronization: both preprocessing and inference stages load and synchronize their explicit component sets before first use. Synchronize module parameters and buffers; skip non-module declarations such as tokenizers and processors.

### 20. Mixed Precision Consistency
The adapter sets inference dtype for frozen components and training dtype for trainable parameters in `_mix_precision()`. Components materialized later must receive the same policy through `on_load_components()`; laziness must not bypass an explicit `frozen_parameters_dtype`. Autocast context is configured in `BaseTrainer.__init__`. Do not manually cast tensors unless you understand the precision boundary. Details: `topics/dtype_precision.md`.

### 20a. Autocast Weight Cache Must Not Span a Forward
`torch.autocast`'s weight cache (keyed by tensor `data_ptr`) serves **stale** casts after any in-place weight change — `optimizer.step()` or a `use_ref/ema/named_parameters` swap (`param.data.copy_`). So wrap **each** forward (and its KL) in its own `with self.autocast():`; never one autocast around the optimize loop. Active for fp32 trainable weights (`trainable_parameters_dtype: fp32`), dormant for the bf16 default, LoRA `disable_adapter()` safe. Details + DDP/DeepSpeed caveat: `topics/autocast_param_swap.md`.

---

## Code Quality (21–27)

### 21. Formatting Standards
- **Black** with `line-length=100`, targeting Python 3.10–3.12
- **isort** with `profile="black"`, `line_length=100`
- Comments and docstrings in **English**

### 22. Import Style
- Use relative imports within `flow_factory` package (e.g., `from ..hparams import *`)
- Use absolute imports for external packages
- Follow existing wildcard import patterns for `hparams`
- **Top-level imports only**: All `import` / `from ... import ...` statements MUST live at the top of the module, never inside function bodies, methods, `__init__`, or conditional branches. Sanctioned exceptions: (a) optional dependencies wrapped in `try/except ImportError` (e.g., `deepspeed`, `xformers`); (b) backend-gated imports where the target symbol is only resolvable under a specific runtime backend already selected by a preceding feature check (e.g., DeepSpeed/FSDP submodules guarded by `is_deepspeed()` / `is_fsdp2()` in `models/abc.py`); (c) genuine unresolvable circular imports documented inline. Lazy imports added merely for "import speed" or "to keep the module light" are NOT acceptable — every hard dependency already runs through Python's import machinery on a typical import path. Inline imports hide the dependency surface from readers, `isort`, and static-analysis tools, and re-execute on every call in hot loops.

### 23. Type Annotations
All public methods must have type annotations. Use `typing` module types (`List`, `Dict`, `Optional`, `Tuple`, `Union`) for Python 3.10 compatibility.

### 24. License Header
All source files must include the Apache 2.0 license header with `Copyright 2026 Jayce-Ping`.

### 25. Logger Message Style
Logger messages referencing config parameters MUST use user-facing field names (not shorthand like `M`, `K`, `W`), show concrete values in parentheses (e.g., `unique_sample_num_per_epoch(32)`), and structure multi-constraint messages with numbered lines.

### 26. Fail-Fast Error Handling
Raise exceptions with detailed debug information over silent auto-fallback. Do not add defensive fallback code that silently recovers from invalid inputs. Auto-fallback is only acceptable when documented as intentional design. Details: `.cursor/rules/no-defensive-except.mdc`.

### 27. Docstring Style
All public functions and methods must have Google-style docstrings in English: imperative one-liner summary, `Args:`, `Returns:`, optional `Note:`. Private helpers (`_func`) may use a one-liner docstring if the behavior is obvious.

### 28. Agent Scratch Files
When an agent (sub-agent, background agent, or any automated tool) needs to write temporary files — investigation reports, analysis documents, checklists, diagrams, or any intermediate artifact that is NOT part of the final deliverable — it MUST write them under the `.scratch/` directory at the repository root. **Never** write temporary files to the project root or any tracked directory (`src/`, `guidance/`, `.agents/`, `.docs/`, `examples/`). `.scratch/` is git-ignored, so files there will not pollute the working tree or accidentally get staged.

### 29. Examples Directory Convention
Example configs follow the path convention `examples/{algorithm}/{finetune_type}/{model_type}/{variant}.yaml`. Model directory names use underscores matching the config `model_type` field (e.g., `sd3_5`, `flux1_kontext`). The baseline config for a model is `default.yaml`. When adding, renaming, or removing examples, update all path references in `README.md`, `guidance/*.md`, and `examples/README.md`.
