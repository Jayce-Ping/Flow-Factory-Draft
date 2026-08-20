# Examples

Training configs for all supported algorithm–model combinations.

## Directory Structure

```
examples/{algorithm}/{finetune_type}/{model_type}/{variant}.yaml
```

| Level | Description | Examples |
|-------|-------------|---------|
| `algorithm` | Training algorithm | `grpo`, `dppo`, `nft`, `awm`, `dgpo`, `dpo`, `crd`, `opd`, `dmd2`, `tdm` |
| `finetune_type` | Parameter-efficient or full | `lora`, `full` |
| `model_type` | Model family (underscore-separated) | `flux1`, `sd3_5`, `wan21`, `ltx2` |
| `variant` | Config variant | `default.yaml`, `nocfg.yaml`, `t2v.yaml` |

**Naming rules**:
- Model directory names use underscores matching the config's `model_type` field (e.g., `sd3-5` → `sd3_5`, `flux1-kontext` → `flux1_kontext`).
- `default.yaml` is the baseline config for a model. Use descriptive names for variants (`nocfg.yaml`, `rational_rewards_t2i.yaml`, `t2v.yaml`, `i2v.yaml`).

**Quick start**:
```bash
ff-train examples/grpo/lora/flux1/default.yaml
```

## DMD2 and TDM

- [`dmd2` SD3.5 OCR recipe](dmd2/lora/sd3_5/ocr.yaml) — validated by
  [W&B run uf4dgbgv](https://wandb.ai/315229706-xi-an-jiaotong-university-/Flow-Factory-DMD2/runs/uf4dgbgv).
- [`tdm` SD3.5 OCR recipe](tdm/lora/sd3_5/ocr.yaml) — official conditional-noise objective.

## MiniMax H3 examples

- [`minimax-h3-t2va`](../examples/grpo/lora/minimax_h3_t2va/default.yaml)
- [`minimax-h3-t2va` real-weight debug recipe](../examples/grpo/lora/minimax_h3_t2va/debug.yaml)
- [`minimax-h3-fl2va`](../examples/grpo/lora/minimax_h3_fl2va/default.yaml)
- [`minimax-h3-ref2va`](../examples/grpo/lora/minimax_h3_ref2va/default.yaml)

The T2VA `debug.yaml` recipe is real-weight validated with the 61 GB checkpoint
(61.74 GiB transformer):
one H20 and 16 H20s across two nodes both completed CPS rollout, video/audio decode,
CLAP reward, GRPO replay/backward/optimizer step, and LoRA checkpoint save/resume.
Its 64x96 canvas is intentionally a correctness geometry. The quality-oriented T2VA
default remains an unverified quality starting point. FL2VA and Ref2VA are
**Schema/API validated only** rather than claims of training stability or reward
improvement.

## Contributing

We welcome community contributions! Here's what you can contribute and how:

### Verified Training Configs

If you've tested a model–algorithm combination and confirmed reward improvement, submit a PR with:
- The config YAML following the directory structure above
- A brief note in the PR description about hardware used and observed reward trend

> **Example**: [#145 — LTX-2.3 + PickScore](https://github.com/X-GenGroup/Flow-Factory/pull/145) added a GRPO + LoRA config for text-to-audio-video, with a training curve (8×H200, 18h) confirming reward improvement.

### Custom Reward Models

New reward models are welcome — add the implementation under `src/flow_factory/rewards/` and include an example config that uses it. Please ensure your reward model's dependencies are compatible with the existing environment (check `pyproject.toml`).

### New Model Adapters

See the [New Model Guide](../guidance/new_model.md) for how to add a new diffusion/flow-matching model. Include at least one example config with your PR.

### Guidelines

- Configs should be self-contained and runnable with `ff-train`
- Include comments for non-obvious parameter choices
- If your config requires a specific dataset, document how to obtain it
- Test on at least one hardware configuration before submitting
