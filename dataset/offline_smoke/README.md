# Offline smoke datasets

This directory contains the reproducible build, preparation, validation, and
publication tooling for two independent public fixtures:

- [Jayce-Ping/Flow-Factory-SFT-Smoke](https://huggingface.co/datasets/Jayce-Ping/Flow-Factory-SFT-Smoke)
- [Jayce-Ping/Flow-Factory-Offline-DPO-Smoke](https://huggingface.co/datasets/Jayce-Ping/Flow-Factory-Offline-DPO-Smoke)

The repositories are intentionally small correctness fixtures. They are not
quality-training corpora. All media are deterministic procedural assets released
under CC0-1.0; the builder does not download third-party data or create VAE
latent caches.

See [SOURCES.md](SOURCES.md) for the DiffSynth, DyRef, image, video, and audio-video
datasets reviewed for schema compatibility and the reasons they are not silently
mirrored into these public fixtures.

## Build staging repositories

Install the project dependencies, including Pillow, NumPy, and PyAV, then run:

```bash
python -m dataset.offline_smoke.build_mini \
  --staging-root dataset/offline_smoke/_staging
```

The command creates two self-contained trees. It refuses to overwrite an
existing staging tree unless `--replace` is supplied. A fixed seed and fixed
dependency versions produce the same logical records and media.

Each tree contains:

```text
README.md
LICENSE
dataset_manifest.json
provenance.jsonl
media/
profiles/<runtime-alias>/train.jsonl
```

Every runtime alias has 32 strict V2 records, enough for two local batches on up
to 16 ranks with per-device batch size one. Media paths in each JSONL are
relative to its profile directory (`../../media/...`). The aliases cover the ten
main GPU modes plus the supplemental `image-i2i` contract gate.

Video DPO corruptions preserve the decoded first and last frames. Audio-video
candidates follow the declared exact output order `[video, audio]`. The generic
candidate projection follows `profiles.py` rather than assuming that every
multi-component output is AV, so a future pure-audio or other output contract can
be added without changing the public V2 record model.

Preparation performs V2, task-profile contract, path, and media validation; see
`python -m dataset.offline_smoke.prepare --help`. Publication is a separate
external-state operation and requires the explicit confirmation flag shown by
`python -m dataset.offline_smoke.publish --help`.
