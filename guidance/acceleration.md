# Acceleration

Flow-Factory's acceleration layer is a **model-agnostic, registry-based plugin system**
that speeds up training without touching trainer or model math. It lives in
`src/flow_factory/acceleration/` and mirrors the reward/model/trainer registries.

The dominant cost of online RL fine-tuning is **Stage 3 — rollout** (multi-step denoising
under `torch.no_grad()`), so that is where most accelerators apply.

## Safety model (read this first)

The correctness axis is **symmetric application** (`stage`), not numerical bit-exactness.
Every accelerator declares two markers, and a validator (`acceleration/validator.py`)
enforces them against the trainer's `paradigm` before training starts (fail-fast):

| marker | values | meaning |
|--------|--------|---------|
| `stage` | `both` / `rollout` | `both`: persistent transform applied to the transformer shared by rollout `inference()` and training `forward()` → **consistent by construction, safe for any algorithm** (the `shared` slot). `rollout`: per-epoch context, torn down before training (the `rollout` slot). |
| `safety` | `lossless` / `lossy` | Only consulted for `stage='rollout'`. `lossy` = changes rollout outputs → diverges from training. `lossless` = bit-identical. |

The single restriction: a **`lossy` rollout** accelerator is allowed **only on
`decoupled` / `distillation`** trainers. Why: for **coupled** algorithms (GRPO,
GRPO-Guard, DPPO) the rollout's per-step log-prob becomes the PPO "old log-prob";
changing the rollout while the training forward stays exact biases the importance ratio
and silently corrupts gradients (`.agents/knowledge/constraints.md` #7, #20a). For
**decoupled** (NFT, AWM, DGPO, DPO, CRD) and **distillation** (diffusion-opd), the rollout
log-prob does not enter the loss, so a lossy rollout only shifts the generated-sample
distribution — acceptable and tunable. Monitor the reward mean/std when enabling it.

A `stage='both'` accelerator is **always safe** — even a numerically-approximate one. For
example, Sage int8 attention used as the attention backend runs in *both* rollout and
training, so the two stay consistent; there is no need to reject it or give it a special
"lossy" status. Numerical exactness only matters when a transform is applied to one stage
but not the other, which is exactly what `stage='rollout'` + `safety='lossy'` captures.

## Configuration

Add an optional `acceleration:` block to any config. Two independent slots, both off by
default:

```yaml
acceleration:
  # Lossless, applied to BOTH rollout and the training forward.
  shared_accelerator: "torch_compile"        # torch_compile
  shared_params: { mode: "regional" }        # regional (compile_repeated_blocks) | full

  # Rollout-only (Stage 3). May be lossy (paradigm-gated).
  rollout_accelerator: "diffusers_cache"     # diffusers_cache | cache_dit | <lossless id>
  rollout_params: { policy: "first_block", threshold: 0.08 }
```

Either slot may be omitted. A direct python path (e.g. `my_pkg.accel.MyAccelerator`) is
accepted in place of a registered id.

## Available accelerators

| id | safety | stage | Notes |
|----|--------|-------|-------|
| `attention_backend` | lossless | both | Sets the diffusers attention backend on every transformer. Configured via `model.attn_backend` (see below); `backend` param can override. Forwards any backend (`native` / `flash` / `_flash_3` / `_flash_3_hub` / `sage` / `xformers`) to `set_attention_backend`. |
| `torch_compile` | lossless | both | `torch.compile` of the shared transformer. `mode: regional` uses diffusers' `compile_repeated_blocks` (fast warmup, robust to variable resolution); `mode: full` compiles the whole module. Extra `compile_kwargs` forwarded to the compile call. Compiles in place (checkpoint- and EMA/ref-safe), applied after `post_init`. |
| `diffusers_cache` | lossy | rollout | Diffusers-native feature caching (no extra dependency). `policy`: `first_block` (default) / `faster` / `pyramid` / `taylorseer` / `magcache`; remaining params forwarded to the policy's diffusers config (e.g. `threshold`). |
| `cache_dit` | lossy | rollout | [cache-dit](https://github.com/vipshop/cache-dit) backend (DBCache/TaylorSeer). Requires `pip install flow-factory[acceleration]`. All params forwarded to `cache_dit.enable_cache`. |

### Attention backend

Attention-backend selection keeps its dedicated config knob, `model.attn_backend`:

```yaml
model:
  attn_backend: "_flash_3_hub"   # native | flash | flash_hub | _flash_3 | _flash_3_hub | sage | xformers
```

It is applied through `AttentionBackendAccelerator` by the trainer
(`BaseTrainer._apply_shared_acceleration`) — after `accelerator.prepare` / `post_init`
and **before** compile, so the compiled graph captures the chosen backend. This is the
single code path for backend selection (the old `BaseAdapter._set_attention_backend` was
removed); it applies whether or not an `acceleration:` block is present.

## Model cache-readiness (lossy caching)

Feature caching reuses block outputs across denoising steps via the transformer's
`cache_context(...)`. Adapters that already wrap their transformer call in
`transformer.cache_context(...)` — Qwen-Image, Qwen-Image-Edit-Plus, Wan2, LTX2,
FLUX.2-Klein — are cache-ready. Adapters that call the transformer bare (e.g. FLUX.1)
need their forward wrapped in a `cache_context` first. Validate the reward distribution
before/after enabling on a new model.

`torch_compile` is model-agnostic and applies to every adapter.

## Adding a new accelerator

1. Subclass `acceleration/abc.py::BaseAccelerator`; set the `safety` and `stage` class
   attributes. Implement `setup()` (one-time mutation, e.g. compile) and/or
   `rollout_context()` (per-epoch context, e.g. caching).
2. Register the id → class path in `acceleration/registry.py`.
3. Optional dependencies must be imported defensively (`try/except ImportError`) per
   constraint #22(a).
