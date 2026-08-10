# Flow-Factory Architecture Overview

## Module Dependency Graph

```
                         ┌──────────┐
                         │ cli.py   │
                         │ train.py │
                         └────┬─────┘
                              │
                    ┌─────────▼─────────┐
                    │     Arguments     │  (hparams/)
                    │  Top-level config │
                    └──┬────┬────┬──────┘
                       │    │    │
          ┌────────────┘    │    └────────────┐
          ▼                 ▼                  ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  BaseTrainer  │  │ BaseAdapter  │  │BaseRewardModel│
   │  (trainers/)  │  │  (models/)   │  │  (rewards/)  │
   └──┬───┬───┬───┘  └──┬───┬───┬──┘  └──┬───┬───┬───┘
      │   │   │         │   │   │         │   │   │
      ▼   ▼   ▼         ▼   ▼   ▼         ▼   ▼   ▼
    GRPO NFT AWM     Flux SD3 Wan     PickScore CLIP OCR
```

### Key Dependency Rules

| Module | Depends On | Depended By |
|--------|-----------|-------------|
| `hparams/` | (standalone) | Everything |
| `models/abc.py` | `hparams`, `samples`, `ema`, `scheduler`, `utils` | All model adapters, `trainers/abc.py` |
| `trainers/abc.py` | `hparams`, `models/abc.py`, `rewards/`, `advantage/`, `data_utils/`, `logger/` | All trainer subclasses |
| `advantage/` | `hparams`, `rewards/`, `samples/` | `trainers/abc.py` |
| `rewards/abc.py` | `hparams` | All reward models, `trainers/abc.py` |
| `data_utils/` | `hparams` | `trainers/abc.py` |
| `scheduler/` | (standalone) | `models/abc.py` |
| `samples/` | `utils/` | `models/`, `rewards/`, `advantage/`, `trainers/` |
| `ema/` | `utils/` | `models/abc.py` |
| `logger/` | `hparams` | `trainers/abc.py` |
| `utils/` | (standalone) | Most modules |

---

## Six-Stage Training Pipeline

> Authoritative reference: `guidance/workflow.md`

```
Stage 1: Data Preprocessing (offline, cached)
  │  GeneralDataset + adapter.preprocess_func()
  │  Text/image/video/audio → encoded tensors (prompt_embeds, image_latents, audio_features, ...)
  │  Result cached with hash fingerprint
  ▼
Stage 2: K-Repeat Sampling
  │  Three sampler strategies (see `topics/samplers.md`):
  │  - GroupContiguousSampler (preferred, auto-selected): keeps K copies on same rank
  │  - DistributedKRepeatSampler (fallback): shuffles K copies across ranks
  │  - GroupDistributedSampler (DGPO): rank-identical prompt sequence, K/W copies per rank
  │  K = training_args.group_size
  ▼
Stage 3: Trajectory Generation
  │  adapter.inference() — full multi-step SDE/ODE denoising
  │  Produces: generated images/videos + trajectory data (noises, log-probs)
  ▼
Stage 4: Reward Computation
  │  RewardProcessor dispatches to Pointwise or Groupwise models
  │  Multi-reward aggregation with configurable weights
  ▼
Stage 5: Advantage Computation
  │  AdvantageProcessor (advantage/advantage_processor.py)
  │  Communication-aware: auto-selects gather vs local path
  │  Strategies: "sum" (weighted-sum, GRPO) or "gdpo"
  ▼
Stage 6: Policy Optimization
  │  adapter.forward() — single-step denoising for loss computation
  │  Policy gradient (GRPO) or weighted matching (NFT/AWM) or DPO preference loss
  │  Gradient update via accelerator
  ▼
  (Repeat Stages 2–6 for next epoch)
```

**Trainer methods vs stages** (each epoch, after Stage 1):

| Method | Stages |
|--------|--------|
| `sample()` | 2–3 (K-repeat batches + `adapter.inference` trajectories) |
| `prepare_feedback()` | 4–5: reward buffer finalize, `AdvantageProcessor` |
| `optimize()` | 6: `adapter.forward` and optimizer step (DPO: form chosen/rejected pairs at entry, then loss) |

---

## Registry System

All four registries map string keys → lazy import paths. Resolution: registry lookup → fallback to direct Python path → dynamic import. See `trainers/registry.py`, `models/registry.py`, `rewards/registry.py`, `acceleration/registry.py` for implementation.

### Registered Components

**Trainers** (`trainers/registry.py`):

| Key | Class | Paradigm | Base Class |
|-----|-------|----------|------------|
| `grpo` | `GRPOTrainer` | Coupled | `BaseTrainer` |
| `grpo-guard` | `GRPOGuardTrainer` | Coupled | `GRPOTrainer` |
| `dppo` | `DPPOTrainer` | Coupled | `GRPOTrainer` |
| `dpo` | `DPOTrainer` | Decoupled | `BaseTrainer` |
| `dgpo` | `DGPOTrainer` | Decoupled | `BaseTrainer` |
| `nft` | `DiffusionNFTTrainer` | Decoupled | `BaseTrainer` |
| `awm` | `AWMTrainer` | Decoupled | `BaseTrainer` |
| `crd` | `CRDTrainer` | Decoupled | `BaseTrainer` |
| `diffusion-opd` | `DiffusionOPDTrainer` | Distillation (on-policy) | `BaseTrainer` |

**Flat hierarchy**: New trainers inherit from `BaseTrainer` directly. The sanctioned exceptions are `GRPOGuardTrainer → GRPOTrainer` and `DPPOTrainer → GRPOTrainer` (strict GRPO loss variants; see constraint #11).

**Model Adapters** (`models/registry.py`):
| Key | Class | Task |
|-----|-------|------|
| `sd3-5` | `SD3_5Adapter` | Text-to-Image |
| `flux1` | `Flux1Adapter` | Text-to-Image |
| `flux1-kontext` | `Flux1KontextAdapter` | Image-to-Image |
| `flux2` | `Flux2Adapter` | Text-to-Image & Image(s)-to-Image |
| `flux2-klein` | `Flux2KleinAdapter` | Text-to-Image & Image(s)-to-Image |
| `qwen-image` | `QwenImageAdapter` | Text-to-Image |
| `qwen-image-edit-plus` | `QwenImageEditPlusAdapter` | Image(s)-to-Image |
| `z-image` | `ZImageAdapter` | Text-to-Image |
| `wan2_t2v` | `Wan2_T2V_Adapter` | Text-to-Video |
| `wan2_i2v` | `Wan2_I2V_Adapter` | Image-to-Video |
| `wan2_v2v` | `Wan2_V2V_Adapter` | Video-to-Video |
| `ltx2_t2av` | `LTX2_T2AV_Adapter` | Text-to-Audio-Video |
| `ltx2_i2av` | `LTX2_I2AV_Adapter` | Image-to-Audio-Video |
| `bagel` | `BagelAdapter` | Text-to-Image & Image(s)-to-Image (T2I & I2I both batched via NaViT packing; subset-round packing handles variable I2I reference-image count, no per-sample fallback — see `topics/adapter_conventions.md`) |

**Reward Models** (`rewards/registry.py`):
| Key | Class | Type |
|-----|-------|------|
| `pickscore` | `PickScoreRewardModel` | Pointwise |
| `pickscore_rank` | `PickScoreRankRewardModel` | Groupwise |
| `clip` | `CLIPRewardModel` | Pointwise |
| `clap` | `CLAPRewardModel` | Pointwise |
| `imagebind` | `ImageBindRewardModel` | Pointwise |
| `ocr` | `OCRRewardModel` | Pointwise |
| `vllm_evaluate` | `VLMEvaluateRewardModel` | Pointwise |
| `rational_rewards_t2i` | `RationalRewardsT2IRewardModel` | Pointwise |
| `rational_rewards_edit` | `RationalRewardsEditRewardModel` | Pointwise |
| `geneval` | `GenEvalRewardModel` | Pointwise |
| `geneval2_soft_tifa` | `GenEval2SoftTIFARewardModel` | Pointwise |
| `hpsv2` | `HPSv2RewardModel` | Pointwise |
| `qwen_image_bench` | `QwenImageBenchRewardModel` | Pointwise |

**Accelerators** (`acceleration/registry.py`):
| Key | Class | Safety | Stage | Notes |
|-----|-------|--------|-------|-------|
| `attention_backend` | `AttentionBackendAccelerator` | lossless | both | Sets the diffusers attention backend on every transformer (requires a `backend` param). Listed as a `shared` entry (before `torch_compile`); this is the single code path for backend selection (the old `BaseAdapter._set_attention_backend` and the `model.attn_backend` knob were both removed — a config still setting `model.attn_backend` fails fast). Bagel forces flash_attention_2 at load and does not use it. |
| `torch_compile` | `CompileAccelerator` | lossy | both | `torch.compile` of the shared transformer (`auto` default: regional when `_repeated_blocks` is available, otherwise full; explicit regional/full overrides); applied in-place after `post_init` so checkpoint keys / param identity stay stable. Marked `lossy` because it is applied symmetrically but is **not bit-exact across rollout vs training** (Inductor's grad/no-grad graph split → intermittent ~1e-5 on-policy residual, within `clip_range`); still allowed on coupled algos, validator warns. |
| `diffusers_cache` | `DiffusersCacheAccelerator` | lossy | rollout | Diffusers `CacheMixin` feature caching (first_block/faster/pyramid/taylorseer/magcache), gated by the adapter's explicit `supports_diffusers_cache` capability before any component is mutated. |

Configured via the `acceleration:` block (`hparams/acceleration_args.py`): two ordered lists of `{name, params}` entries — `shared` (persistent `stage='both'` accelerators applied to rollout and training) and `rollout` (Stage-3 only). **List order is application order**: `shared` entries run their `setup()` in order (so `attention_backend` must precede `torch_compile`), and `rollout` entries nest their `rollout_context()` in order. The `acceleration/validator.py` enforces that a **lossy `rollout`** accelerator runs only on `decoupled`/`distillation` trainers (each trainer declares a `paradigm`), preserving train-inference consistency (constraint #7). A **lossy `stage='both'` accelerator in the `shared` slot** (e.g. `torch_compile`, applied symmetrically but not bit-exact across stages) is allowed on any paradigm but the validator **warns** on coupled trainers that the on-policy ratio will be ≈1, not exactly 1. Off by default.

---

## Extension Points

- **New model adapter**: `guidance/new_model.md`, skill `/ff-new-model`, conventions `topics/adapter_conventions.md`
- **New reward model**: `guidance/rewards.md`, skill `/ff-new-reward`
- **New algorithm**: `guidance/algorithms.md`, skill `/ff-new-algorithm`
- **New accelerator**: subclass `acceleration/abc.py::BaseAccelerator` (declare `safety`/`stage`), register in `acceleration/registry.py`

---

## Key Design Patterns

### Timestep & Sigma Convention

Timesteps are `[0, 1000]` (scheduler scale); sigmas are `[0, 1]` (flow-matching noise level). Details: `topics/timestep_sigma.md`.

### Adapter Pattern (Models)
Each model adapter wraps a diffusers pipeline into the `BaseAdapter` interface:
- `preprocess_func()` — offline encoding (Stage 1)
- `inference()` — full denoising loop (Stage 3)
- `forward()` — single-step denoising (Stage 6)

**Per-modality encoders** (`encode_prompt`, `encode_image`, `encode_video`, `encode_audio`) are no-op by default on `BaseAdapter` — override only the modalities your model consumes. `preprocess_func` dispatches to all four and skips any that return `None`, so text/image/video-only adapters need no stub overrides for unused modalities.

**Flat hierarchy**: All adapters inherit directly from `BaseAdapter` — never from another adapter (see constraint #12). Shared logic within a model family uses helper functions, code duplication, or mixins — not adapter subclassing.

Details: `topics/adapter_conventions.md`

### Sample Dataclass Hierarchy
Two-layer structure (constraint #14): task-level samples (`T2ISample`, `I2VSample`, `I2AVSample`, ...) live in `samples/samples.py` and inherit from `BaseSample` or condition mixins. Model-specific samples (`LTX2Sample`, `LTX2I2AVSample`, ...) inherit from the matching task-level sample — never from another model-specific sample.

### Component Management
`BaseAdapter` delegates component discovery, canonical access, runtime overrides, lazy
materialization, and stage-device lifecycle to `models/runtime/`. Runtime overrides include
prepared/proxied modules and LoRA/checkpoint replacements; all are excluded from manual device
management. Declared lazy specs are separate from materialized module enumeration, so stage-wide
operations and `materialize_components(None)` never materialize tokenizers, schedulers, processors,
or configs implicitly. Explicit names are required to materialize a lazy spec. Role groups retain
non-`None` modular specs but exclude absent classic optional components. The default
`ClassicPipelineRuntime` preserves eager DiffusionPipeline behavior;
`ModularPipelineRuntime` materializes selected lazy component specs; and
`PseudoPipelineRuntime` manages explicit containers and non-enumerated aliases such as Bagel's
`transformer -> bagel.language_model`. `adapter.pipeline` remains the backend compatibility alias,
while `ModelBundle` and `RoutedComponentProxy` remain the sole distributed preparation runtime.

#### Component runtime enumeration boundaries
- **Date**: 2026-08-10
- **Symptom**: Lazy stage-wide operations could materialize non-module specs, Bagel's nested
  transformer could be moved twice, and trainers bypassed adapter lifecycle overrides.
- **Root Cause**: The first runtime abstraction conflated declared specs, materialized modules,
  aliases, and prepared/replacement overrides under one component-name path.
- **Fix**: Split declared and materialized discovery, added non-enumerated pseudo aliases and
  generic device-excluded overrides, and restored trainer routing through adapter lifecycle APIs.
- **Lesson**: Discovery for explicit lookup and enumeration for lifecycle operations require
  separate contracts; aliases and overrides must remain addressable without becoming lifecycle
  roots.
- **Related Constraint**: #5.

#### Optional role discovery and lazy default materialization
- **Date**: 2026-08-10
- **Symptom**: A declared classic `transformer_2=None` entered the transformer role group and
  adapter freezing called `requires_grad_` on `None`; separately, `materialize_components(None)`
  eagerly loaded every modular spec.
- **Root Cause**: Role discovery filtered names rather than non-`None` values, and the default
  materialization request expanded declared names instead of materialized modules.
- **Fix**: Role discovery now excludes `None` values while retaining non-`None` modular specs;
  default materialization uses already-materialized module names, and normal canonical lookup
  returns direct materialized attributes before consulting the expensive declared component map.
- **Lesson**: Optional declarations are valid for explicit compatibility lookup but cannot imply
  role membership, and an omitted lazy-materialization selection must never mean "load all."
- **Related Constraint**: #5.

### Reward Processing
`RewardProcessor` dispatches by model type:
- **Pointwise**: batch by `batch_size`
- **Groupwise**: group by `unique_id` (local or distributed path)
- **Multi-reward**: weighted aggregation
- **Async**: optional non-blocking computation

### Advantage Computation
`AdvantageProcessor` (`advantage/advantage_processor.py`): communication-aware, auto-selects gather vs local path. Strategies: `"sum"` (GRPO) and `"gdpo"`. All reward-based trainers delegate to `self.advantage_processor.compute_advantages()`; the distillation trainer `diffusion-opd` is the exception (its `prepare_feedback()` is a no-op — no reward/advantage stage).

### Configuration Hierarchy
```
Arguments (top-level)
├── ModelArguments        # model_type, model_path, finetune_type, LoRA config
├── TrainingArguments     # Algorithm-specific (GRPO/DPO/NFT/AWM subclass)
├── SchedulerArguments    # dynamics_type, timestep_range, num_inference_steps
├── DataArguments         # dataset, preprocessing, resolution, sampler_type
├── MultiRewardArguments  # reward_model configs (list of RewardArguments)
├── LogArguments          # logger type, verbose, project name
└── EvaluationArguments   # evaluation settings
```
