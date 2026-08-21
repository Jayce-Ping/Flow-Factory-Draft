# Dataset Guide

Flow-Factory datasets describe generation conditions, prompts, and optional reward metadata. The
model adapter determines which columns it consumes; a column supported by the generic loader is
not necessarily meaningful to every model or workflow.

## Directory and split layout

Each dataset is a directory with a required training split and an optional evaluation split:

```text
dataset/example/
├── train.txt or train.jsonl
├── test.txt or test.jsonl
├── images/                 # optional
├── videos/                 # optional
└── audios/                 # optional
```

When both formats are absent, loading fails. JSONL is required when a row contains media paths,
metadata, negative prompts, or heterogeneous references. Plain text files contain one prompt per
line.

Configure a dataset under `data.datasets`:

```yaml
data:
  datasets:
    - name: example
      dataset_dir: "dataset/example"
      train: {weight: 1, max_dataset_size: 1024}
      eval: {}
  preprocessing_batch_size: 8
  cache_dir: "~/.cache/flow_factory/datasets"
```

For ordinary media columns, `image_dir`, `video_dir`, and `audio_dir` default to
`{dataset_dir}/images`, `{dataset_dir}/videos`, and `{dataset_dir}/audios`. Override them when JSONL
paths use a different root:

```yaml
data:
  datasets:
    - name: example
      dataset_dir: "dataset/example"
      image_dir: "dataset/example"
      video_dir: "/shared/videos"
      audio_dir: "/shared/audio"
```

Relative paths in the ordinary `image`/`images`, `video`/`videos`, and `audio`/`audios` columns are
resolved against their corresponding media directory. MiniMax H3 Ref2VA is different:
relative `references[*].path` and `references[*].audio_path` values are resolved against
`dataset_dir`; absolute paths are accepted unchanged.

## Common task formats

### Text-conditioned generation

Text-to-image, text-to-video, and text-to-audio-video datasets may use plain text:

```text
A hill at sunset.
An astronaut riding a horse on Mars.
```

The equivalent JSONL form is:

```jsonl
{"prompt":"A hill at sunset."}
{"prompt":"An astronaut riding a horse on Mars."}
```

Adapters that implement classifier-free guidance may also consume `"negative_prompt"`. Do not add
it to workflows that explicitly reject CFG, including MiniMax H3.

Examples:

- [T2I prompts](../dataset/t2is/train.jsonl)
- [T2I prompts with negative prompts](../dataset/t2is_neg/train.jsonl)
- [MiniMax H3 T2VA prompts](../dataset/minimax_h3_t2va/train.jsonl)

### Image-conditioned generation

Use `"image"` for one condition image or `"images"` for an ordered list:

```jsonl
{"prompt":"Restyle this scene as a watercolor.","image":"input.png"}
{"prompt":"Combine these visual references.","images":["first.png","second.png"]}
```

The loader normalizes the singular form to the batched image path and decodes images as RGB. The
adapter decides whether multiple images are valid and what their order means.

### Video-conditioned generation

Use `"video"` for one condition video or `"videos"` for a list:

```jsonl
{"prompt":"Restyle this motion.","video":"input.mp4"}
{"prompt":"Use both motion references.","videos":["motion-a.mp4","motion-b.mp4"]}
```

Videos are decoded before adapter preprocessing. As with images, list cardinality and semantics
belong to the selected adapter.

### Audio columns

The generic loader accepts `"audio"` or `"audios"` and resolves them against `audio_dir`:

```jsonl
{"prompt":"Generate visuals matching this sound.","audio":"sound.wav"}
{"prompt":"Use these sound references.","audios":["voice.wav","ambience.wav"]}
```

Only use these columns with an adapter that declares an audio input. MiniMax H3 Ref2VA instead
places audio entries inside the ordered `"references"` array.

## MiniMax H3 datasets

All three MiniMax H3 workflows require preprocessing and training batch size `B=1`. They do not use
CFG, so both training and evaluation must keep:

```yaml
data:
  preprocessing_batch_size: 1
train:
  per_device_batch_size: 1
  guidance_scale: 1.0
eval:
  per_device_batch_size: 1
  guidance_scale: 1.0
```

MiniMax H3 accepts 5–15 seconds at 24 fps. `num_frames` is rounded up to
`17*n+5`, so 124 is the smallest explicit frame count that satisfies both the
duration and VAE chunking contracts.

T2VA `debug.yaml` is real-weight validated on 1 and 16 GPUs, including LoRA
checkpoint save/resume. Its 64x96 canvas validates
correctness and memory fit, not visual quality or reward improvement.
`quality_720p_fsdp2.yaml` has real-weight initialization, checkpoint, native-resolution
decode, and evaluation coverage; no long-run reward trend is claimed. The default,
FL2VA, and Ref2VA configs remain schema/API-validated starting points.

### T2VA: `minimax-h3-t2va`

T2VA is prompt-only:

```jsonl
{"prompt":"A small paper windmill turns beside a quiet stream with synchronized birdsong."}
```

Do not include negative prompts, images, or references. Use the
[T2VA dataset fixture](../dataset/minimax_h3_t2va/train.jsonl) with the
[T2VA GRPO configuration](../examples/grpo/lora/minimax_h3_t2va/default.yaml).

### FL2VA: `minimax-h3-fl2va`

FL2VA uses an ordered `"images"` list:

```jsonl
{"prompt":"Animate this scene with matching sound.","images":["images/first.png"]}
{"prompt":"Interpolate between these scenes.","images":["images/first.png","images/last.png"]}
```

- One image is the first frame.
- Two images are the first frame followed by the last frame.
- Any other cardinality is invalid.
- Order must not be sorted, deduplicated, or inferred from filenames.

The example stores paths relative to the dataset root, so its YAML sets `image_dir` to the dataset
directory. See the [FL2VA dataset fixture](../dataset/minimax_h3_fl2va/train.jsonl) and
[FL2VA GRPO configuration](../examples/grpo/lora/minimax_h3_fl2va/default.yaml).

### Ref2VA: `minimax-h3-ref2va`

Ref2VA uses a non-empty ordered `"references"` array containing image, video, and audio entries:

```jsonl
{"prompt":"Create a coherent scene using the references in order.","references":[{"kind":"image","path":"references/style.png"},{"kind":"video","path":"references/motion.mp4","fps":12.0},{"kind":"audio","path":"references/ambience.wav","sample_rate":16000}]}
```

Array order is semantically significant. It is preserved during validation, encoding, caching, and
sample identity hashing. At least one image or video reference is required; an audio-only array is
invalid.

Supported entries:

| `kind` | Required keys | Optional keys | Decoded value |
|---|---|---|---|
| `image` | `kind`, `path` | none | RGB image |
| `video` | `kind`, `path` | `fps`, `audio_path`, `sample_rate` | frames and optional soundtrack |
| `audio` | `kind`, `path` | `sample_rate` | waveform |

`fps` and `sample_rate` overrides must be finite positive numbers. A video may use its embedded
soundtrack or a separate dataset-relative `audio_path`; a video `sample_rate` override requires
`audio_path`. Unknown keys and unsupported `kind` values fail before preprocessing.

See the [Ref2VA dataset fixture](../dataset/minimax_h3_ref2va/train.jsonl), its
[local fixture notes](../dataset/minimax_h3_ref2va/README.md), and the
[Ref2VA GRPO configuration](../examples/grpo/lora/minimax_h3_ref2va/default.yaml).

## Preprocessing and cache flow

Normal datasets follow this path:

```text
TXT/JSONL row
  -> resolve paths and decode media
  -> adapter.preprocess_func under no-grad
  -> move encoded tensors to CPU
  -> write Arrow cache
  -> collate cached fields for rollout
```

Prompt encoders, VAEs, and processors are preprocessing components. The trainer can offload them
after cache creation because optimization consumes their cached outputs rather than decoding and
encoding every source row again.

Ref2VA adds an ordered-reference path:

```text
references array
  -> validate and canonicalize as reference_manifest
  -> decode images/video/audio with PIL and PyAV
  -> construct transient pinned Diffusers reference objects
  -> run Ref2VA setup, text, reference-encoder, and layout blocks under no-grad
  -> cache Arrow-safe prompt embeddings, condition latents, layout, geometry, and manifest
```

Reference media is not stored in the Arrow cache. The decoded PIL/PyAV values and upstream
reference objects exist only during preprocessing. Cache identity includes the configured dataset
root, TXT/JSONL source bytes, semantic preprocessing fields, adapter preprocessing version, and
model identity.
`reference_manifest` is the canonical JSON string retained for reproducibility and sample
identity. Replacing a media file in place without changing its path or manifest does not change
the source hash; set `force_reprocess: true` after such a replacement.

## How Ref2VA references affect training

At rollout time, the cached reference encodings become immutable video/audio
`condition_prefixes`. Fresh random video and audio target states are created separately. The
packed layout presented to `transformer_ref` is conceptually:

```text
[reference condition rows | generated target rows]
```

Joint attention lets the references change the predicted target video/audio velocity and therefore
the rollout distribution, generated media, reward, and replay log-probability. After each forward
pass, condition-row predictions are discarded and only target rows are stepped by the video and
audio schedulers.

Reference condition rows are not trajectory states. Reference condition rows do not contribute
policy log-probability degrees of freedom. They are fixed observations, not sampled actions.
Gradients flow through `transformer_ref` (or its LoRA parameters) because its target prediction
depends on those observations, but gradients do not update the source media, text encoder,
reference encoder, or VAE.

The resulting sample retains:

- the target-only video/audio structured trajectory;
- cached prompt embeddings;
- `condition_prefixes`, packed layout, and geometry for identical replay conditions;
- `reference_manifest` for identity and auditability.

This separation keeps rollout and optimization mathematically aligned without treating reference
tokens as generated state.
