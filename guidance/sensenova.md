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

The initial integration supports text-to-image training and evaluation. Official
image-editing/multimodal generation requires a different image-prefill and cache
contract, so it is intentionally not exposed by this adapter yet.

Recommended starting points are:

- [GRPO + LoRA example](../examples/grpo/lora/sensenova/default.yaml)
- [SenseNova-U1 official repository](https://github.com/OpenSenseNova/SenseNova-U1)
- [SenseNova-U1.5 preview notes](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/u1.5_preview.md)
