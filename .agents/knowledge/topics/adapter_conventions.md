# Adapter Conventions

**Read when**: Adding or modifying a model adapter.

---

## Classifier-Free Guidance (CFG) Convention

All adapters that support CFG must follow a consistent two-stage pattern. Guidance-distilled models (FLUX.1, FLUX.1-Kontext, FLUX.2) do not use CFG — they pass `guidance_scale` as a guidance embedding directly to the transformer.

### Stage 1: `encode_prompt()` / data preprocessing

- **CFG condition**: `do_classifier_free_guidance = guidance_scale > 1.0` (exception: Z-Image uses `> 0.0`).
- `encode_prompt()` **must** accept `guidance_scale` and compute the CFG flag internally — callers should not need to decide.
- If `do_classifier_free_guidance` is true and `negative_prompt is None`, default to `""`.
- When CFG is active, encode the negative prompt and include `negative_prompt_embeds` (plus `negative_prompt_embeds_mask` or `negative_pooled_prompt_embeds` where applicable) in the returned dict.

### Stage 2: `forward()` / denoising step

- `forward()` receives `negative_prompt_embeds` (may be `None`).
- **CFG condition**: `do_classifier_free_guidance = guidance_scale > 1.0 and negative_prompt_embeds is not None`.
- If `guidance_scale > 1.0` but `negative_prompt_embeds is None`, emit `logger.warning(...)` and **fall back to the no-CFG path** (no error). The warning message must mention both the passed scale and the missing embeddings.
- CFG formula: `noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)`.

### Reference implementation

`flux/flux2_klein.py` — `encode_prompt()` (line ~165) and `_forward()` (line ~769).

### Models with model-specific CFG extensions

| Model | Extension | Notes |
|---|---|---|
| Z-Image | `cfg_truncation`, `cfg_normalization` | Applied after standard CFG formula |
| Qwen-Image / Qwen-Image-Edit-Plus | Norm rescale after CFG | `comb_pred * (cond_norm / noise_norm)` |
| LTX2 | x0-space multi-guidance (CFG + STG + Modality Isolation) | CFG delta computed in x0-space, not velocity-space |
| SD3.5 | Requires `negative_pooled_prompt_embeds` in addition to `negative_prompt_embeds` | Two embedding checks in forward |

## `forward()` as the Consistency Boundary

`adapter.forward()` is the atomic unit for train-inference consistency (-> `train_inference_consistency.md`).

1. **Inference/forward identity**: `inference()` loop must call `forward()` — not duplicate its logic. Any code that affects model output belongs inside `forward()`.
2. **Argument preservation**: All arguments affecting `forward()` output must be stored on the Sample dataclass during rollout and replayed identically by `optimize()`. This includes `guidance_scale`, `stg_scale`, `connector_prompt_embeds`, `noise_level`, etc.

## Upstream Pipeline Alignment

- **Structural vs behavioral separation**: First commit matches the reference diffusers pipeline's numerical output; second commit cleans up style. Never combine both in a single change.
- **`inference()` must reproduce `Pipeline.__call__()` output** given the same seed, dtype, and parameters. Verify via parity testing (-> `parity_testing.md`).
- **Timestep convention**: Adapter receives `t` in `[0, 1000]`; converts internally per model needs. Detail: `topics/timestep_sigma.md`.

## Component Lifecycle

| Category | Property | Frozen | Offloadable | Examples |
|---|---|---|---|---|
| Preprocessing | `preprocessing_modules` | yes | yes | `text_encoders`, `vae` |
| Inference/Training | `inference_modules` | transformer: trainable; VAE: frozen | VAE: yes | `transformer`, `vae` |

Defined in `models/abc.py` L380-387. Override in subclasses to add model-specific components (e.g., `connectors`, `image_encoder`).

## Batch Dimension Convention

- All adapter methods (`preprocess_func`, `encode_*`, `inference`, `forward`) receive tensors with batch dim `(B, ...)`.
- `BaseSample` fields are **per-sample** (no batch dim) — the sample collator handles stacking.
- `condition_images` is model-dependent: `Tensor(B,C,H,W)` for uniform shape, `List[List[Tensor]]` for variable shape.
- `inference()` condition parameters (`images`, `videos`, `audios`) arrive as `MultiImageBatch` / `MultiVideoBatch` / `MultiAudioBatch` (nested batch, e.g. `List[List[Image.Image]]`, `List[List[Tensor]]`) from the training pipeline collator (`data_utils/dataset.py` `collate_fn`). Type annotations on `inference()` must use the multi-form, not the bare `ImageBatch` / `VideoBatch` / `AudioBatch`.
- **Multi-media batch homogeneity**: `_preprocess_batch` (`data_utils/dataset.py`) guarantees `List[List[Media]]` for every modality column — empty samples contribute `[]`, single-item samples contribute `[item]`, multi-item samples contribute `[item1, ..., itemN]`. This keeps HF Arrow columns homogeneous and lets every `encode_*` consume a single shape.
- **Image-column persistence (HF Image feature)**: the raw `images` column and any `encode_image` output listed in `python_format_columns` (ClassVar on `BaseAdapter`, empty by default) are stored via the HuggingFace `Image` feature (PNG bytes) instead of raw tensors, and **read back as PIL** (`List[List[PIL.Image]]`). This is what lets ragged multi-reference batches (variable size/count) serialize — raw tensors are only Arrow-serializable when uniform. Opt in per adapter only for genuine RGB images (e.g. Bagel `condition_images`); never declare preprocessed/non-RGB tensors (VAE-ready video tensors, latents) — PIL conversion is lossy and breaks tensor consumers (e.g. LTX2-I2AV `condition_images` stays a tensor). Consumers must normalize via `_standardize_image_input` / `standardize_image_batch` before any tensor op. To keep PIL on the **sample** too (not just the dataset cache), the adapter's `ImageConditionSample` subclass must set `condition_images_as_pil = True` (else `__post_init__` re-canonicalizes to `List[Tensor(C,H,W)]` [0,1]); e.g. `BagelI2ISample`.
- Single-condition adapters must flatten internally via `_standardize_image_input` / `_standardize_video_input` using `is_multi_image_batch` / `is_multi_video_batch` to extract the first element per sample (e.g. `Wan2_I2V._standardize_image_input`, `Wan2_V2V._standardize_video_input`, `LTX2_I2AV._standardize_image_input`). Multi-condition adapters (e.g. `Flux2`) consume the nested structure directly.

## Numbered Gotchas (append-only)

1. Never call `pipeline.__call__()` from `inference()` — decompose it into individual pipeline steps.
2. `encode_prompt()` must match the pipeline's tokenizer settings exactly (padding, truncation, max_length).
3. `_shared_fields` on Sample determines which fields are shared across batch in sampling. Missing fields cause silent data duplication.
4. `default_target_modules` must list all Linear layers to be LoRA'd; verify with `named_modules()`. Default is `['to_q', 'to_k', 'to_v', 'to_out.0']`.
5. `inference()` `images`/`videos` params are always `MultiImageBatch`/`MultiVideoBatch`. Single-condition adapters must flatten via `_standardize_*_input` with `is_multi_image_batch`/`is_multi_video_batch` (e.g. `Wan2_I2V._standardize_image_input`); annotate as `MultiImageBatch`/`MultiVideoBatch`, never `ImageBatch`/`VideoBatch`.
6. **Multi-media batch homogeneity** — `_preprocess_batch` always emits `List[List[Media]]` per modality. Do NOT unwrap single-element lists in `encode_*` and do NOT return a bare `Tensor` or `None` for empty samples — return `[]`. Returning a bare `Tensor` for single-audio samples (or `None` for empty image samples) breaks Arrow column homogeneity and forces downstream consumers to handle three input shapes. Applies symmetrically to `images`, `videos`, and `audios`.
7. **CFG two-stage consistency** — `encode_prompt()` and `forward()` must use the same threshold for CFG activation (`guidance_scale > 1.0`, or `> 0.0` for Z-Image). `forward()` must gracefully handle the case where `guidance_scale > threshold` but negative embeds are `None` (warn + fallback, never error). See "Classifier-Free Guidance (CFG) Convention" section above.
8. **Bagel batch handling (NaViT subset-round packing)** — Bagel uses sequence packing, not a leading batch dim. **Both T2I and I2I** pack all B samples into one block-diagonal forward (`_build_gen_context` + `_forward_packed`; the framework's `(B, num_tokens, dim)` latents reshape to packed `(B*num_tokens, dim)` and back). For I2I, reference images are added in per-image rounds (`num_rounds = max per-sample count`); a sample without an r-th image is passed as `None` to `prepare_vae_images` / `prepare_vit_images`, which keep its cached KV and add a **zero-length query segment**. So a **variable per-sample reference-image count** (and varying sizes) is handled by packing directly — there is no per-sample (`batch_size=1`) fallback. The cache merge requires every sample to remain on the key/value side, so only the query may be a subset. The prefill is `@torch.no_grad` and every round has >=1 active image (`max_seqlen_q > 1`), avoiding flash-attn zero-length pitfalls (no backward, no `max_seqlen_q==1`). `_is_i2i(condition_images)` depends only on condition-image presence (distributed-safe). CFG global renorm is computed **per sample** over `packed_seqlens - 2`, and `forward()` returns per-sample `(B,)` log-prob (not per-token). **Distributed**: the prefill makes a data-dependent number of `language_model` forward calls (`2*num_rounds + 2`); `language_model` is the only FSDP-sharded module (frozen ViT/VAE are unsharded, so they don't count). Under FSDP FULL_SHARD/HYBRID (and ZeRO-3) each call AllGathers `language_model`'s shard, so per-rank counts mismatch and deadlock — `_assert_variable_count_supported` fails fast there (`@torch.no_grad` does not help; FSDP still all-gathers to compute). DDP / DeepSpeed ZeRO-1/2 (the Bagel I2I backends) replicate params (local forward, fixed grad sync at backward), so variable counts are safe. The FSDP-safe alternative is to gather `language_model` once for the generation (`summon_full_params` / `reshard_after_forward=False`).
9. **Image columns persist via HF Image feature (variable-size/count I2I)** — preprocessing stores image data as PIL via the HF `Image` feature, not raw tensors; ragged tensor columns (multi-reference images of varying size/count) are NOT Arrow-serializable and otherwise crash in `Dataset.map` with `TypeError: a bytes-like object is required, not 'Tensor'` / `OverflowError`. The raw `images` column is always stored this way; an `encode_image` output is stored this way only when its name is listed in the adapter's `python_format_columns` ClassVar (default empty — opt in for RGB images only, e.g. Bagel `condition_images`). These columns **read back as PIL** (`List[List[PIL.Image]]`); the `torch` format excludes them (`_apply_torch_format` in `dataset.py`), and `collate_fn` keeps them as a `MultiImageBatch`. To keep PIL end-to-end on the **sample** (not just the cache), the adapter's `ImageConditionSample` subclass must also set `condition_images_as_pil=True` (else `ImageConditionSample.__post_init__` re-canonicalizes to `List[Tensor(C,H,W)]` [0,1]); e.g. `BagelI2ISample`. Bump `_PREPROCESS_FORMAT_VERSION` if the on-disk image format changes again.

## Cross-refs

- UP: `architecture.md` "Adapter Pattern", `constraints.md` #5 #11-12
- PEER: `train_inference_consistency.md`, `parity_testing.md`, `ff-new-model` Pitfall #6
