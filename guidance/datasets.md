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

### Strict V2 records for offline supervision

SFT and offline DPO require `train.jsonl` in the strict V2 format. Existing prompt-only online
datasets remain valid and do not need migration. V2 uses the discriminator key `"type"` at every
public media or supervision boundary; legacy `"kind"` is rejected. Unknown keys are rejected
rather than silently ignored.

An SFT demonstration stores one input and one ordered output candidate:

```jsonl
{"schema_version":2,"input":{"prompt":"A red fox in fresh snow.","media":[]},"supervision":{"type":"demonstration","target":{"media":[{"type":"image","path":"targets/fox.png"}]}},"metadata":{"license":"example"}}
```

An offline-DPO record stores two candidates under the same input:

```jsonl
{"schema_version":2,"input":{"prompt":"A red fox in fresh snow.","media":[]},"supervision":{"type":"preference","chosen":{"media":[{"type":"image","path":"chosen/fox.png"}]},"rejected":{"media":[{"type":"image","path":"rejected/fox.png"}]}},"metadata":{"annotator":"example"}}
```

The schema is algorithm-neutral: `"demonstration"` and `"preference"` describe supervision, not
trainer class names. Each enabled source must be homogeneous, and every candidate's ordered media
sequence must match the selected model adapter's pipeline I/O contract. For example, SD3.5 accepts
no input media and exactly one image output. A future audio-video pipeline may instead declare a
fixed video/audio output sequence without changing SFT or offline-DPO code.

V2 media paths are absolute or relative to that source's `dataset_dir`. Video entries may declare
a positive `fps`; audio entries may declare a positive integer `sample_rate`. Image entries accept
only `type` and `path`.

Offline source configuration uses the existing unified dataset list:

```yaml
data:
  datasets:
    - name: demonstrations
      dataset_dir: dataset/my_demonstrations
      train: {split: train, weight: 1}
  enable_preprocess: true
  force_reprocess: false
  sampler_type: auto
```

All offline sources use unit weight. The train loader concatenates them and uses PyTorch's official
`DistributedSampler`; one complete traversal is one data epoch. Before any condition-cache lookup,
each normalized input is validated against the adapter's pipeline contract, including accepted
media types/counts/rates, negative-prompt policy, and grouped-versus-ordered preprocessing binding.
Unsupported conditions fail instead of being silently ignored by a model preprocessor.

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

## SenseNova-U1 datasets

SenseNova-U1 1.0/1.5 supports prompt-only T2I and ordered multi-reference I2I.
T2I can use the ordinary text or JSONL formats above; the
[default GRPO recipe](../examples/grpo/lora/sensenova/default.yaml) uses OCR prompts.

For I2I, each row uses the ordinary `"images"` list. List order is preserved and
each sample may contain a different number and size of reference images:

```jsonl
{"prompt":"Combine these images together.","images":["first.png","second.png"]}
{"prompt":"Combine these images together.","images":["style.png","subject.png","layout.png"]}
```

Prepare the 2–3-reference example dataset with:

```bash
python dataset/multi_ref_image/prepare.py
```

Then launch the
[multi-reference GRPO recipe](../examples/grpo/lora/sensenova/multi_reference_image.yaml).
The dataset cache stores ragged reference lists as PIL images. SenseNova builds a
separate variable-length NEO-Unify prefix for each generated sample; it does not
NaViT-pack independent samples like Bagel.

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

Offline supervision deliberately has two lifecycles:

```text
input prompt/media
  -> adapter input preprocessing
  -> input-only Arrow condition cache

target/chosen/rejected path
  -> CPU decode on each dataset access
  -> adapter output codec on the training device
  -> on-the-fly VAE encode for the current microbatch
  -> discard pixels and clean target latents after optimization
```

Target pixels, target videos, and target latents are never written to the preprocessing cache.
Changing only target paths or metadata therefore reuses the input-condition cache when
`force_reprocess` is false, while changing the prompt or input-media path/spec produces a different
condition identity. Replacing input-media bytes at the same path requires `force_reprocess: true`.
This avoids a second large media/latent copy on disk and keeps output geometry and VAE normalization
under the model adapter that consumes them. The VAE stays available during offline training and
evaluation; target encoding runs under `torch.no_grad()` but is performed again when a record is
revisited in a later epoch.

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
