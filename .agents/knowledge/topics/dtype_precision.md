# Dtype & Precision

**Read when**: Touching dtype/precision, mixed precision config, or debugging NaN/overflow.

---

## Precision Boundaries

| Component | Runtime dtype | Why |
|-----------|--------------|-----|
| Frozen params/buffers (frozen transformer base, VAE, text encoders) | `frozen_parameters_dtype` — default `None` **preserves each component's `from_pretrained` dtype** (no downcast) | Released checkpoints ship components in different dtypes (e.g. Z-Image: transformer fp32, text encoder bf16); set an explicit dtype to force one / save memory |
| Transformer (trainable) | `trainable_parameters_dtype` (fp32/bf16) | Gradient precision |
| Scheduler math | `float32` always | `1/sigma` amplification (see below) |
| Latent storage (trajectory) | `latent_storage_dtype` (configurable) | Memory vs. precision tradeoff |
| Forward-process adapter boundary | Exact dtype/device of each clean latent component | Structured adapters reject noise with a different storage representation |
| Advantage computation | `float64` (numpy) | Normalization stability |

Boundaries are set in `BaseAdapter._mix_precision()` (`models/abc.py`) and `BaseTrainer.__init__` (autocast context). Autocast weight-cache invariant + in-place ref/EMA/named swaps: `topics/autocast_param_swap.md` (#20a).

`frozen_parameters_dtype` accepts either one dtype or a selector mapping. Resolution order is
concrete component, component group (`transformers` / `text_encoders`), then `default`. A null
result preserves the checkpoint dtype. Under FSDP2 mixed precision, trained components keep
uniform fp32 original parameters while the resolved policy still applies to fully frozen
components.

## `cast_latents()` Contract

`BaseAdapter.cast_latents()` (`models/abc.py`) casts latents to `latent_storage_dtype` for trajectory storage.

- **float16 overflow protection**: clamps values exceeding 65504.0 with a warning.
- **Identity when no target**: returns latents unchanged if `latent_storage_dtype` is unset and no default provided.
- **Must be applied identically** in both rollout and training paths — inconsistency breaks train-inference consistency (-> `train_inference_consistency.md` item #3).

```python
def cast_latents(self, latents, default_dtype=None):
    target = self.latent_storage_dtype or default_dtype
    if target is None or latents.dtype == target:
        return latents
    if target == torch.float16:
        abs_max = latents.abs().max().item()
        if abs_max > 65504.0:
            latents = latents.clamp(-65504.0, 65504.0)
    return latents.to(target)
```

## 1/sigma Error Amplification

Scheduler math uses `1/sigma` to scale velocity predictions. Near the end of the denoising schedule, sigma approaches zero and errors are amplified:

```
Example: sigma = 0.01, epsilon_error = 1.5e-4
Amplified error: epsilon_error / sigma = 1.5e-4 / 0.01 = 1.5e-2

Over 30 steps with accumulated error: total_drift ≈ 6.0
```

This is why scheduler math is always `float32` and why the dtype round-trip guard exists in schedulers:

```python
next_latents = next_latents.to(_input_dtype).float()
```

The round-trip ensures that the precision of stored latents matches what training will see — without it, float32 scheduler output stored as bf16 loses precision, and the training replay produces different `log_prob`.

## Diagnosis Checklist

| Symptom | Check |
|---------|-------|
| NaN in loss after few steps | `latent_storage_dtype=float16` with large latent values? Check `cast_latents()` clamp warnings. |
| `ratio` drifts from 1.0 at epoch start | Compare `forward()` output dtype between rollout and training. Verify `cast_latents()` is called in both paths. |
| Gradients explode near end of schedule | Scheduler using lower-than-float32 precision? Check `_input_dtype` round-trip in scheduler. |
| Reward NaN but generation looks normal | Advantage normalization overflow — verify `float64` in `advantage_processor.py`. |
| Forward noising rejects float32 noise for fp16/bf16 latents | Keep stochastic/likelihood math in float32, then cast the adapter-bound noise component with `.to(clean_component)` before calling `apply_forward_process_noise()`. |

## Fix records

### TDM conditional noise crossed the latent boundary in float32
- **Date**: 2026-08-27
- **Symptom**: MiniMax H3 rejected TDM fake-stage noise because the clean video/audio latents were float16 while conditionally mixed noise was float32.
- **Root Cause**: TDM correctly promoted conditional re-noising and likelihood math to float32 but passed that compute representation directly into the adapter's strict storage boundary.
- **Fix**: TDM now retains float32 mixed/fresh noise for importance weighting and creates a separate adapter-bound noise state cast component-wise to each clean latent's actual dtype and device.
- **Lesson**: Compute precision and boundary representation are separate contracts. Restore representation at the producer boundary instead of weakening adapter validation or casting from global configuration.
- **Related Constraint**: N/A

## Cross-refs

- `constraints.md` #18 (all-rank synchronization — precision errors may manifest differently per rank)
- `constraints.md` #20 (mixed precision consistency)
- `topics/autocast_param_swap.md` (#20a)
- `train_inference_consistency.md` (log_prob mismatch from precision)
- `topics/timestep_sigma.md` (scheduler math always float32)
