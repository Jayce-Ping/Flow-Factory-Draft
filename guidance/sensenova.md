# SenseNova-U1 adapter

Flow-Factory registers both SenseNova-U1 1.0 and SenseNova-U1 1.5 under
`model_type: "sensenova"`:

```yaml
model:
  model_type: "sensenova"
  model_name_or_path: "sensenova/SenseNova-U1.5-8B-MoT"
```

The U1.0 checkpoint is selected by changing the path to
`sensenova/SenseNova-U1-8B-MoT`. Both checkpoints use the same NEO-Unify
architecture. The checkpoint config selects the flow head (`use_pixel_head:
false` for U1.0 and `true` for U1.5), so the adapter does not duplicate model
logic.

SenseNova-U1 is not loaded through a diffusers pipeline. The adapter vendors the
official NEO-Unify Transformers implementation under
`src/flow_factory/models/sensenova/modeling/`, wraps it in an explicit
`PseudoPipelineRuntime`, and routes the differentiable image-generation pass
through `SenseNovaDenoiser`. Prefix KV-cache construction is deterministic and
no-gradient, while the denoising prediction and SDE transition are shared by
rollout and replay.

The adapter supports T2I and I2I training/evaluation. I2I accepts an ordered
`List[List[PIL.Image]]` condition batch, so each sample may carry multiple reference
images. The official image-prefill path expands each `<image>` marker into its
visual-token block and prepares separate text+image, image-only, and optional
unconditional KV caches. `cfg_scale` controls text guidance and `img_cfg_scale`
controls image guidance; the default `img_cfg_scale: 1.0` follows the official
edit configuration.

Example I2I data shape:

```python
condition_images = [
    [reference_a, reference_b],  # sample 0: two ordered references
    [reference_c],                # sample 1: one reference
]
```

Reference images are persisted through the HF Image feature as PIL images, allowing
variable reference-image counts and spatial sizes to survive dataset caching and
sample replay. This integration covers the official local image-editing contract;
other multimodal/interleaved tasks remain outside the adapter scope.

Recommended starting points are:

- [GRPO + LoRA example](../examples/grpo/lora/sensenova/default.yaml)
- [SenseNova-U1 official repository](https://github.com/OpenSenseNova/SenseNova-U1)
- [SenseNova-U1.5 preview notes](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/u1.5_preview.md)
