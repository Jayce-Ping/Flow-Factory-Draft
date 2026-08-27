# Timestep & Sigma Convention

**Read when**: Touching `TimeSampler`, `adapter.forward(t=...)`, `scheduler.step(timestep=...)`, `flow_match_sigma()`, or `timestep_range` config fields.

---

Throughout the codebase, two related but distinct scales are used for time:

| Name | Variable | Scale | Meaning |
|------|----------|-------|---------|
| **Timestep** | `t`, `timestep` | `[0, 1000]` | Scheduler-scale time. All public interfaces (`TimeSampler` outputs, `adapter.forward(t=...)`, `scheduler.step(timestep=...)`) use this scale. |
| **Sigma** | `σ`, `sigma` | `[0, 1]` | Flow-matching noise level. Used for latent interpolation `x_t = (1-σ) x_0 + σ ε` and loss weighting. Obtained via `flow_match_sigma(t) = t / 1000`. |

## Rules

1. `TimeSampler` always returns `t` in `[0, 1000]`. Trainers pass it directly to `adapter.forward(t=...)` without scaling.
2. When interpolating latents or computing noise-level-dependent weights, convert explicitly: `sigma = flow_match_sigma(t)`.
3. Each model adapter internally converts `t` to whatever its underlying transformer expects (e.g., Flux divides by 1000, SD3.5 passes as-is). This conversion is encapsulated inside the adapter's `forward()` method.
4. `timestep_range=(frac_lo, frac_hi)` is a fraction along the denoising axis from 1000 (noisy) toward 0 (clean), mapped via `t = 1000 * (1 - frac)`. So `(0, 0.99)` corresponds to `t ∈ [10, 1000]`.
5. When validating a stored `(timestep, sigma)` pair, compare in sigma space with `flow_match_coordinates_close()`. The coordinates may differ by one native ULP after independent materialization.

## Gotchas

1. **Don't divide `t` by 1000 before passing to `adapter.forward()`** — the adapter handles internal conversion.
2. **`timestep_range` uses denoising-axis fractions, not raw timesteps** — `(0, 0.5)` means the noisier half `t ∈ [500, 1000]`, not the cleaner half.
3. **`flow_match_sigma()` is the only sanctioned conversion** — do not use `t / 1000` directly; use the function for traceability.
4. **Do not validate redundant coordinates with a fixed scaled tolerance** — normalize with `flow_match_sigma()` and permit one ULP of the stored dtypes. For a discrete multi-component endpoint, reuse each scheduler's authoritative stored coordinate instead of reconstructing it through inverse/forward shifts.

## Fix records

### Independently rounded flow-matching coordinates
- **Date**: 2026-08-27
- **Symptom**: MiniMax H3 rejected valid TDM forward noising because `990.4219970703125` and `0.9904220700263977 * 1000` differed by roughly `6e-5`; independently shifted audio endpoints could also miss exact trajectory topology checks.
- **Root Cause**: H3 compared redundant float32 coordinates after multiplying sigma by 1000 with a fixed `1e-5` tolerance, while continuous mapping recreated discrete dual-scheduler endpoints from rounded video values.
- **Fix**: Added canonical one-ULP comparison in sigma space, routed explicit scheduler conversions through `flow_match_sigma()`, and made H3 reuse the exact video/audio scheduler coordinates whenever the sampled primary time is an existing grid point.
- **Lesson**: Validate redundant coordinates in their normalized domain at representation precision; preserve authoritative discrete grid values at interval boundaries and use analytic mapping only for interior samples.
- **Related Constraint**: N/A

## Cross-refs

- `constraints.md` #7 (coupled/decoupled paradigm — affects which timestep sampling is valid)
- `topics/train_inference_consistency.md` (same `t` must produce same output in rollout vs training)
- `topics/adapter_conventions.md` (adapter encapsulates timestep-to-model conversion)
