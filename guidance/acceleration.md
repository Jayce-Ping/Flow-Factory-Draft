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

Add an optional `acceleration:` block to any config. Two independent slots, each an
**ordered list** of `{name, params}` entries (both empty by default). **List order is the
application order**:

```yaml
acceleration:
  # Lossless, applied to BOTH rollout and the training forward, in list order.
  shared:
    - name: attention_backend                # set the diffusers backend first...
      params: { backend: _flash_3_hub }
    - name: torch_compile                    # ...so the compiled graph captures it
      params: { mode: regional }             # regional (compile_repeated_blocks) | full

  # Rollout-only (Stage 3), nested in list order. May be lossy (paradigm-gated).
  rollout:
    - name: diffusers_cache                   # diffusers_cache | cache_dit | <lossless id>
      params: { policy: first_block, threshold: 0.08 }
```

Either slot may be omitted or left empty. A single entry dict (without the list dashes) is
accepted as shorthand for a one-element list. A direct python path (e.g.
`my_pkg.accel.MyAccelerator`) is accepted in place of a registered id.

## Available accelerators

| id | safety | stage | Notes |
|----|--------|-------|-------|
| `attention_backend` | lossless | both | Sets the diffusers attention backend on every transformer. Requires a `backend` param. Forwards any backend (`native` / `flash` / `_flash_3` / `_flash_3_hub` / `sage` / `xformers`) to `set_attention_backend`. List it in `shared` **before** `torch_compile` so the compiled graph captures the backend. |
| `torch_compile` | lossless | both | `torch.compile` of the shared transformer. `mode: regional` uses diffusers' `compile_repeated_blocks` (fast warmup, robust to variable resolution); `mode: full` compiles the whole module. Extra `compile_kwargs` forwarded to the compile call. Compiles in place (checkpoint- and EMA/ref-safe), applied after `post_init`. |
| `diffusers_cache` | lossy | rollout | Diffusers-native feature caching (no extra dependency). `policy`: `first_block` (default) / `faster` / `pyramid` / `taylorseer` / `magcache`; remaining params forwarded to the policy's diffusers config (e.g. `threshold`). |
| `cache_dit` | lossy | rollout | [cache-dit](https://github.com/vipshop/cache-dit) backend (DBCache/TaylorSeer). Requires `pip install flow-factory[acceleration]`. All params forwarded to `cache_dit.enable_cache`. |

### Attention backend

Attention-backend selection is a `shared` accelerator (it transforms the module shared by
rollout and training, so it is consistent for any algorithm — even an approximate kernel
like Sage int8). It used to live under the dedicated `model.attn_backend` knob; that knob
was **removed** and folded into the acceleration layer:

```yaml
acceleration:
  shared:
    - name: attention_backend
      params: { backend: _flash_3_hub }   # native | flash | flash_hub | _flash_3 | _flash_3_hub | sage | xformers
    - name: torch_compile                 # optional; if present, list it AFTER attention_backend
      params: { mode: regional }
```

It is applied through `AttentionBackendAccelerator` by the trainer
(`BaseTrainer._apply_shared_acceleration`) — after `accelerator.prepare` / `post_init`, in
list order. Place it **before** `torch_compile` so the compiled graph captures the chosen
backend. This is the single code path for backend selection (the old
`BaseAdapter._set_attention_backend` and `model.attn_backend` were both removed). A config
that still sets `model.attn_backend` fails fast with a migration error.

> **Bagel** forces `flash_attention_2` at model load (requires `pip install -e ".[bagel]"`)
> and its custom transformer has no `set_attention_backend`, so it does **not** take an
> `attention_backend` entry — omit it (the accelerator raises if applied to bagel).

### torch.compile train-inference consistency (coupled algorithms)

For coupled algorithms (GRPO / GRPO-Guard / DPPO) the on-policy PPO ratio on the first inner
step must be **1.0**. `torch.compile` (Inductor) threatens this because it compiles a
**separate, numerically non-identical graph for grad vs no-grad mode** (Dynamo guards on
`grad_mode`): rollout normally runs the transformer under `torch.no_grad()` and the training
forward under grad, so a naive compiled rollout would diverge from training (~1e-5) and bias
the ratio.

`CompileAccelerator` handles this automatically (no user action needed): it declares
`requires_grad_rollout = True`, so the trainer runs the rollout transformer under
`torch.enable_grad()` (overriding the `@torch.no_grad()` on `inference()`) and detaches only
the per-step latent feedback (in `cast_latents`) to keep memory bounded. It also compiles the
**base** transformer under any PEFT/LoRA wrapper (not the wrapper itself). With this, the
on-policy ratio is **bit-exact** (`max|ratio-1| = 0.000e+00`, verified on SD3.5) for coupled
training — **both with and without CFG** (CFG concatenates `[uncond, cond]` identically in
rollout and training, so it aligns exactly).

Two implementation notes that matter for correctness:
- The grad-force wrapper must **not** detach its own output — an inner detach lets Inductor
  pick a divergent inference-optimized kernel and re-introduces ~1e-5 drift. The detach lives
  at the latent-feedback chokepoint (`cast_latents`) instead.
- Determinism knobs (`cudnn.deterministic`, `fallback_random`) are irrelevant here — the cause
  was the grad-vs-no-grad graph split, not RNG (a `train-vs-train` recompute is exactly 0.0).

See `.scratch/torch_compile_consistency_report.md` for the full analysis.


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
