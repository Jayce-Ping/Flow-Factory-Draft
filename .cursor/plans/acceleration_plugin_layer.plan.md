# Flow-Factory Acceleration Roadmap & Phase-1 Plan

## Context

Flow-Factory is an **online RL fine-tuning framework** for diffusion/flow-matching models. Its
dominant runtime cost is **Stage 3 — Trajectory Generation (rollout)**: for every epoch it runs
`num_inference_steps` denoising steps × `group_size (K)` samples per prompt × `num_batches_per_epoch`,
all under `torch.no_grad()`. This is *exactly* the inference workload that cache-dit / lightx2v / SGLang
accelerate. Stage 6 (policy optimization, forward+backward) is the secondary cost.

The framework already has a solid **lossless** optimization base (FSDP/FSDP2/DeepSpeed, grad-checkpoint,
LoRA, bf16/fp16, CPU-offload + async H2D prefetch, Qwen CFG-merge, comm fusion, `StackedSampleBatch`,
single-root `ModelBundle`). What is missing is the **inference-acceleration layer** that the referenced
projects specialize in: feature caching (cache-dit DBCache/TaylorSeer/FBCache), whole-/region
`torch.compile`, long-sequence context parallelism (lightx2v / xDiT / SGLang style), and
quantization of frozen modules.

This plan proposes a **model-agnostic, registry-based acceleration plugin layer** that respects the
framework's hard algorithm/model decoupling, plus a phased roadmap. Goal of Phase 1: cut rollout wall-time
on image models (FLUX/Qwen-Image) with **zero changes to trainers or model math**, gated for correctness.

---

## The Decoupling-Critical Insight (read first — shapes the whole design)

Acceleration techniques split into two correctness classes, and the split maps **exactly** onto
Constraint #7 (coupled vs decoupled) + Philosophy #1 / Constraint #20a (train-inference consistency):

| Class | Examples | Where safe | Why |
|---|---|---|---|
| **Lossy** (changes `noise_pred`) | feature caching, step-reduction, int8/Sage attention | **rollout-only, decoupled/distillation-only** | Caching skips block compute; it cannot be replicated in the Stage-6 training forward (that needs full gradient through every block, every step). For **coupled** algos (GRPO/GRPO-Guard/DPPO) the rollout's per-step `log_prob` becomes the PPO "old log-prob"; if rollout is approximated but the training forward is exact, the importance ratio is biased → **silently wrong gradients**. For **decoupled** algos (NFT/AWM/DGPO/DPO/CRD) rollout runs with `compute_log_prob=False`; only the final image enters the reward → caching merely shifts the sample distribution, which is acceptable/tunable. |
| **Lossless** (numerically ~identical, applied to the *shared* module) | `torch.compile`, exact attention backends (FA2/FA3/xformers), context/sequence parallelism, quantizing **frozen-only** modules, comm/overlap/offload | **everywhere** | Because both `inference()` and `forward()` call the *same* `adapter.transformer`, a transform applied to that module is consistent across rollout and training by construction. |

**Design rule enforced by the plan:** every accelerator declares `safety ∈ {lossless, lossy}` and
`stage ∈ {rollout, train, both}`. A fail-fast validator (Constraint #26) rejects `lossy` accel on a
coupled trainer, and rejects `lossy` accel on the `train` stage for any trainer. This is the mechanism
that lets us add aggressive rollout speedups *without* touching the train-inference consistency contract.

---

## Current State (verified)

**Already present / reusable seams:**
- `attn_backend` config (`hparams/model_args.py:110`) → `BaseAdapter._set_attention_backend()`
  (`models/abc.py:852-866`) routes to diffusers' attention dispatcher: `native / flash / flash_hub /
  _flash_3 / _flash_3_hub / sage / xformers`. **Attention backends are already wired framework-wide** —
  they need benchmarking/per-stage exposure, not invention.
- Transformers are diffusers `CacheMixin` models: adapters already open `transformer.cache_context(...)`
  (Qwen, Wan, LTX2, FLUX2-Klein). This is the exact hook diffusers-native & cache-dit caching plug into.
- Pluggable **scheduler registry** (`scheduler/registry.py`): FlowMatchEuler + UniPC multistep — solver
  swaps / step schedules are already extensible.
- `RoutedComponentProxy` (`models/model_bundle.py:93-121`) delegates attribute access (incl.
  `enable_cache`, `cache_context`, `compile`) to the inner module → an accelerator can call
  `adapter.transformer.enable_cache(...)` model-agnostically, post-`prepare`.
- Rollout entry points: `BaseTrainer.generate_samples()` / `sample_batch()` (`trainers/abc.py:550-798`).

**Absent (opportunities):** feature caching of any kind; whole-/region `torch.compile` (only
`bagel/.../qwen2_navit.py:43` compiles `flex_attention`); context/sequence parallelism for video;
quantization integration (`bitsandbytes` is an optional dep but unused); rollout throughput/VRAM/quality
benchmark harness.

---

## Roadmap (phased, by ROI × safety × effort)

| Phase | Track | What | Safety | Primary win | Effort |
|---|---|---|---|---|---|
| **0** | Foundation | Benchmark harness (rollout samples/s, step-time, peak VRAM, **reward-distribution regression**) + the `acceleration/` plugin layer + paradigm-gated validator | infra | enables safe tuning | S |
| **1** | Rollout (image-first) | **Feature caching** (cache-dit / diffusers-native) gated decoupled-rollout-only; **`torch.compile`** of shared transformer (lossless, everywhere); finish **attn_backend** benchmark + per-stage selection | mixed | **2–4× rollout on NFT/AWM/DGPO/CRD; ~1.2–1.8× compile everywhere** | M |
| **2** | Video long-seq | **Context/sequence parallelism** (Ulysses/Ring attention) for the DiT during rollout *and* training (Wan2, LTX2) — lossless | lossless | enables/scales long video; near-linear on seq dim | L |
| **3** | Memory→throughput | Quantize **frozen** modules only (text encoders, VAE, reference/EMA, reward models) via torchao fp8 / bnb nf4; FSDP2 selective activation checkpointing; smarter offload scheduling rollout vs train | lossless | bigger batch / larger models | M |
| **4** | Pipeline overlap | Async reward (Stage 4) + advantage (Stage 5) overlapped with next rollout; extend existing comm fusion | lossless | hide reward/comm latency | M |

Phases 1 and 2 are the user-prioritized fronts (rollout throughput, then video parallelism). 3–4 are
follow-ons. Everything hangs off the Phase-0 plugin layer so each later track is a registry entry, not a
trainer/adapter edit.

**Execution order requested by user:** implement all **lossless** accelerators first (torch.compile +
per-stage attention backend within the plugin layer), then the **lossy** feature-caching accelerators.

---

## Proposed Architecture: `src/flow_factory/acceleration/` (model-agnostic plugin layer)

Mirror the existing `rewards/` module shape (registry + abc + concrete impls + hparams), so it inherits
the framework's decoupling guarantees and lazy-import conventions (Constraints #1–#4).

```
src/flow_factory/acceleration/
  __init__.py
  abc.py          # BaseAccelerator: declares safety + stage; setup()/rollout_context()
  registry.py     # _ACCELERATOR_REGISTRY {id -> path} + get_accelerator_class() w/ direct-path fallback
  validator.py    # paradigm-gated safety check (fail-fast, Constraint #26)
  torch_compile.py    # CompileAccelerator (lossless; applied to shared transformer at init)
  attention_backend.py  # AttentionBackendAccelerator (lossless exact / lossy sage, per-stage)
  cache_dit.py    # CacheDitAccelerator  (lossy; wraps cache_dit.enable_cache / disable_cache)
  diffusers_cache.py  # DiffusersCacheAccelerator (lossy; FirstBlockCache/FasterCache/PAB)
```

**`BaseAccelerator` (abc.py)** — minimal contract:
```python
class BaseAccelerator(ABC):
    safety: ClassVar[Literal["lossless", "lossy"]]
    stage:  ClassVar[Literal["rollout", "train", "both"]]

    @abstractmethod
    def setup(self, adapter: "BaseAdapter") -> None: ...      # one-time (e.g. torch.compile)
    @contextmanager
    def rollout_context(self, adapter: "BaseAdapter"): ...    # enable cache → yield → disable+reset
```

**Config (`hparams/acceleration_args.py`)** — new `AccelerationArguments`, aggregated into top-level
`Arguments` (Constraint #15). Per-stage so lossy stays rollout-only:
```yaml
acceleration:
  rollout: { name: cache_dit, params: { policy: DBCache, rdt: 0.08 } }  # lossy, rollout-only
  shared:  { name: torch_compile, params: { mode: default, dynamic: true } }  # lossless, both stages
```

**Trainer paradigm tag.** Add `paradigm: ClassVar[Literal["coupled","decoupled","distillation"]]` to each
trainer (values already fixed by Constraint #7). `validator.py` reads it and the chosen accelerators'
`safety/stage` to enforce the table above before training starts.

**Integration points (only two, both model-agnostic, no per-adapter math change):**
1. `BaseTrainer._initialization()` builds `self.rollout_accelerator` / `self.shared_accelerator` from
   config via the registry, runs `validator.validate(paradigm, accelerators)`, and calls
   `shared_accelerator.setup(adapter)` (e.g. compile) after `accelerator.prepare()`.
2. `BaseTrainer.generate_samples()` (`trainers/abc.py:701`) wraps the batch loop:
   `with self.rollout_accelerator.rollout_context(self.adapter): ...`. The context calls
   `adapter.transformer.enable_cache(...)` (reachable through `RoutedComponentProxy`) and tears it down
   after, so cache state never leaks into the Stage-6 forward.

Trainers and adapters stay unchanged except: (a) the one-line `paradigm` tag, (b) the two
`BaseTrainer` hooks above. No `optimize()` / `forward()` / `inference()` signature changes
(Constraint #12), no scheduler-to-trainer coupling.

---

## Phase 1 — Detailed Implementation Spec

**Objective:** lossless `torch.compile` + per-stage attention path usable by all algos, then 2–4×
rollout speedup on decoupled image trainers via feature caching, with a benchmark that proves reward
isn't degraded.

### 1. Phase-0 prerequisites bundled in
- **Benchmark harness** under `.scratch/bench/` (scratch per Constraint #28) + a small reusable profiler
  in `utils/`: measure rollout samples/s, mean step time, peak VRAM, and **mean/std of the reward
  buffer** before vs after accel. The reward-regression check is mandatory for the lossy path.
- Build `acceleration/{abc,registry,validator}.py` and `hparams/acceleration_args.py`; wire
  `AccelerationArguments` into `hparams/args.py` `Arguments`.

### 2. Lossless first — `torch.compile` (both stages)
- **`CompileAccelerator`** (`acceleration/torch_compile.py`): in `setup(adapter)` apply
  `torch.compile` to the shared transformer — prefer diffusers `transformer.compile_repeated_blocks()`
  (region compile → fast warmup, robust to the variable image/seq-len shapes set by
  `calculate_shift`/`set_scheduler_timesteps`). Because the same compiled module backs both
  `inference()` and `forward()`, it is consistent for coupled algos too. Handle LoRA (compile after PEFT
  wrap) and `dynamic=True` for variable resolution.

### 3. Lossless — attention backend (per-stage)
- No new mechanism — extend `attn_backend` to be **per-stage** through the same accel config, and add a
  benchmark sweep (FA2/FA3/xformers exact = lossless/both; Sage int8 = lossy/rollout-only). Document
  results in `guidance/`.

### 4. Lossy — feature caching (rollout-only, decoupled-only)
- **`DiffusersCacheAccelerator`** (`acceleration/diffusers_cache.py`): zero-extra-dep, uses diffusers'
  built-in `transformer.enable_cache(FirstBlockCacheConfig | FasterCacheConfig | PyramidAttentionBroadcastConfig)` / `disable_cache()`. **Default lossy backend.**
- **`CacheDitAccelerator`** (`acceleration/cache_dit.py`): optional dep (`cache-dit`), imported under
  `try/except ImportError` per Constraint #22(a). `rollout_context` calls
  `cache_dit.enable_cache(adapter.transformer, ...)` on enter, `cache_dit.disable_cache(...)` on exit.
- **Adapter cache-readiness:** caching hooks fire only inside a `cache_context`. Adapters that already
  wrap their transformer call in `transformer.cache_context(...)` (Qwen-Image `qwen_image.py:598`, Wan,
  LTX2, FLUX2-Klein) are ready. Adapters that call the transformer bare (e.g. FLUX.1 `flux1.py:323`)
  need the existing call wrapped in a `cache_context` — a localized, behavior-preserving change, not a
  decoupling violation. **Validate first on Qwen-Image (already ready) + an NFT/AWM/DGPO config.**
- New optional dep: add `cache-dit` to `[project.optional-dependencies]` as `acceleration = [...]`.

### 5. Files touched (Phase 1)
- **New:** `src/flow_factory/acceleration/*` (7 files), `src/flow_factory/hparams/acceleration_args.py`.
- **Edited (small):** `hparams/args.py` (aggregate args); `trainers/abc.py` (`_initialization` build+validate+setup, `generate_samples` context wrap); each trainer class (one `paradigm` ClassVar); `flux1.py`-style adapters lacking `cache_context` (wrap existing transformer call); `pyproject.toml` (optional dep); `examples/` configs gain an optional `acceleration:` block (Constraint #15/#17); `guidance/` + `.agents/knowledge/architecture.md` doc updates (registry table + new module).

### 6. Reuse (do not reinvent)
- Registry pattern: copy `rewards/registry.py` resolution + direct-path fallback.
- Hook seam: `RoutedComponentProxy.__getattr__` already exposes `enable_cache`/`cache_context`/`compile`.
- Rollout loop already isolated in `generate_samples()`/`sample_batch()` — single wrap point.
- diffusers native caches need no new dependency.

---

## Verification

1. **Correctness gating (unit):** assert `validator` raises (Constraint #26) for `lossy` accel on a
   coupled trainer and for `lossy` on the `train` stage; passes for decoupled-rollout.
2. **Numerical (lossless):** with `torch.compile` only, a short GRPO run's per-step `log_prob` and loss
   match the eager baseline within tolerance (confirms coupled-safety of the shared-module path).
3. **Rollout speed + quality (lossy):** run an NFT (or AWM/DGPO) image config on Qwen-Image with
   caching vs baseline for N epochs; report rollout samples/s, peak VRAM, and **reward mean/std drift**
   from the harness. Accept only if speedup ≥ target and reward drift within budget.
4. **End-to-end:** `ff-train examples/<algo>/lora/qwen_image/default.yaml` with and without the
   `acceleration:` block trains and checkpoints identically in shape; `black --check src/ && isort
   --check src/` clean (Constraint #21); run `/ff-review` before commit.
5. **Video smoke (Phase 2 entry):** confirm the same plugin enables on Wan2/LTX2 (already use
   `cache_context`) before building context parallelism.

---

## Open Risks / Decisions
- **cache-dit vs diffusers-native** as the default lossy backend: cache-dit has richer policies
  (DBCache/TaylorSeer) but adds a dep; diffusers-native (FBCache/FasterCache/PAB) is dep-free. Plan
  ships **both** behind the registry; default = diffusers-native, cache-dit opt-in.
- **FLUX.1 cache_context wrap** is the only per-adapter code change in Phase 1 — keep it behavior-identical
  when caching is off (empty context).
- **Reward-drift budget** for the lossy path must be set with the user (per reward model) — the harness
  measures it but the accept/reject threshold is a product decision.
- Phase 2 context parallelism interacts with FSDP/`ModelBundle` sharding and the SDE scheduler's
  per-token noise (`scheduler/flow_match_euler_discrete.py`) — needs its own design pass (separate plan).
