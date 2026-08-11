<p align="center">
  <img src="./assets/logo-no-bg.png" alt="Flow-Factory logo" height="200">
</p>
<h1 align="center">Flow-Factory</h1>

<p align="center">
  <b>Easy Reinforcement Learning for Diffusion and Flow-Matching Models</b>
</p>

# 🔥 News

* **[2026-08-11]** **MiniMax H3 Audio-Video** support is now available for
  [`minimax-h3-t2va`](examples/grpo/lora/minimax_h3_t2va/default.yaml),
  [`minimax-h3-fl2va`](examples/grpo/lora/minimax_h3_fl2va/default.yaml), and
  [`minimax-h3-ref2va`](examples/grpo/lora/minimax_h3_ref2va/default.yaml). The integration adds
  modular workflow loading, separate video/audio trajectories and schedulers, and ordered
  heterogeneous reference conditioning. See the [Dataset Guide](guidance/datasets.md) for all
  three input schemas. These examples are schema/API validated against the pinned `diffusers`
  revision; real-checkpoint GPU generation, memory fit, training quality, and numerical parity
  remain to be established.

# 📕 Table of Contents

- [Supported Models](#-supported-models)
- [Supported Algorithms](#-supported-algorithms)
- [Get Started](#-get-started)
  - [Installation](#installation)
  - [Experiment Trackers](#experiment-trackers)
  - [Quick Start Example](#quick-start-example)
- [Guidance](#-guidance)
- [Dataset](#-dataset)
- [Reward Model](#-reward-model)
- [Acknowledgements](#-acknowledgements)

# 🤗 Supported Models

<table>
  <tr><th>Task</th><th>Model</th><th>Model Size</th><th>Model Type</th></tr>
  <tr><td rowspan="6">Text-to-Image</td><td><a href="https://huggingface.co/collections/stabilityai/stable-diffusion-35">stable-diffusion-3.5-medium/large</a></td><td>2.5B/8.1B</td><td>sd3-5</td></tr>
  <tr><td><a href="https://huggingface.co/black-forest-labs/FLUX.1-dev">FLUX.1-dev</a></td><td>13B</td><td>flux1</td></tr>
  <tr><td><a href="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo">Z-Image-Turbo</a></td><td>6B</td><td>z-image</td></tr>
  <tr><td><a href="https://huggingface.co/Tongyi-MAI/Z-Image">Z-Image</a></td><td>6B</td><td>z-image</td></tr>
  <tr><td><a href="https://huggingface.co/Qwen/Qwen-Image">Qwen-Image</a></td><td>20B</td><td>qwen-image</td></tr>
  <tr><td><a href="https://huggingface.co/Qwen/Qwen-Image-2512">Qwen-Image-2512</a></td><td>20B</td><td>qwen-image</td></tr>

  <tr><td>Image-to-Image</td><td><a href="https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev">FLUX.1-Kontext-dev</a></td><td>13B</td><td>flux1-kontext</td></tr>
  
  <tr><td rowspan="2">Image(s)-to-Image</td><td><a href="https://huggingface.co/Qwen/Qwen-Image-Edit-2509">Qwen-Image-Edit-2509</a></td><td>20B</td><td>qwen-image-edit-plus</td></tr>
  <tr><td><a href="https://huggingface.co/Qwen/Qwen-Image-Edit-2511">Qwen-Image-Edit-2511</a></td><td>20B</td><td>qwen-image-edit-plus</td></tr>

  <tr><td rowspan="6">Text-to-Image & Image(s)-to-Image</td><td><a href="https://huggingface.co/black-forest-labs/FLUX.2-dev">FLUX.2-dev</a></td><td>32B</td><td>flux2</td></tr>
  <tr><td><a href="https://huggingface.co/black-forest-labs/FLUX.2-klein-4B">FLUX.2-klein-4B</a></td><td>4B</td><td>flux2-klein</td></tr>
  <tr><td><a href="https://huggingface.co/black-forest-labs/FLUX.2-klein-9B">FLUX.2-klein-9B</a></td><td>9B</td><td>flux2-klein</td></tr>
  <tr><td><a href="https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B">FLUX.2-klein-base-4B</a></td><td>4B</td><td>flux2-klein</td></tr>
  <tr><td><a href="https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B">FLUX.2-klein-base-9B</a></td><td>9B</td><td>flux2-klein</td></tr>
  <tr><td><a href="https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT">BAGEL-7B-MoT</a></td><td>14B</td><td>bagel</td></tr>

  <tr><td rowspan="4">Text-to-Video</td><td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers">Wan2.1-T2V-1.3B</a></td><td>1.3B</td><td>wan2_t2v</td></tr>
  <tr><td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers">Wan2.1-T2V-14B</a></td><td>14B</td><td>wan2_t2v</td></tr>
  <tr><td><a href="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers">Wan2.2-TI2V-5B</a></td><td>5B</td><td>wan2_t2v</td></tr>
  <tr><td><a href="https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers">Wan2.2-T2V-A14B</a></td><td>A14B</td><td>wan2_t2v</td></tr>

  <tr><td rowspan="5">Image-to-Video</td><td><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers">Wan2.1-I2V-14B-480P</a></td><td>14B</td><td>wan2_i2v</td></tr>
  <tr><td><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers">Wan2.1-I2V-14B-480P</a></td><td>14B</td><td>wan2_i2v</td></tr>
  <tr><td><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers">Wan2.1-I2V-14B-720P</a></td><td>14B</td><td>wan2_i2v</td></tr>
  <tr><td><a href="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers">Wan2.2-TI2V-5B</a></td><td>5B</td><td>wan2_i2v</td></tr>
  <tr><td><a href="https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers">Wan2.2-I2V-A14B</a></td><td>A14B</td><td>wan2_i2v</td></tr>

  <tr><td rowspan="2">Text-to-Audio-Video</td><td><a href="https://huggingface.co/Lightricks/LTX-2">LTX-2</a></td><td>19B</td><td>ltx2_t2av</td></tr>
  <tr><td><a href="https://huggingface.co/Lightricks/LTX-2.3">LTX-2.3</a></td><td>22B</td><td>ltx2_t2av</td></tr>
  <tr><td rowspan="2">Image-to-Audio-Video</td><td><a href="https://huggingface.co/Lightricks/LTX-2">LTX-2</a></td><td>19B</td><td>ltx2_i2av</td></tr>
  <tr><td><a href="https://huggingface.co/Lightricks/LTX-2.3">LTX-2.3</a></td><td>22B</td><td>ltx2_i2av</td></tr>
  <tr><td>Text-to-Audio-Video</td><td><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3">MiniMax H3 T2VA</a> (<a href="examples/grpo/lora/minimax_h3_t2va/default.yaml">schema/API-validated example</a>)</td><td>61 GB checkpoint</td><td>minimax-h3-t2va</td></tr>
  <tr><td>First/Last-Frame-to-Audio-Video</td><td><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3">MiniMax H3 FL2VA</a> (<a href="examples/grpo/lora/minimax_h3_fl2va/default.yaml">schema/API-validated example</a>)</td><td>61 GB checkpoint</td><td>minimax-h3-fl2va</td></tr>
  <tr><td>Ordered-Reference-to-Audio-Video</td><td><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3">MiniMax H3 Ref2VA</a> (<a href="examples/grpo/lora/minimax_h3_ref2va/default.yaml">schema/API-validated example</a>)</td><td>61 GB checkpoint</td><td>minimax-h3-ref2va</td></tr>
</table>

> To support new models, see [Guidance/New Model](guidance/new_model.md).

> **MiniMax H3 status and limits:** The three examples are schema/API validated, including a real
> pinned no-weight component-spec/workflow-build probe; the 61 GB checkpoint was not downloaded.
> No GPU, real-weight generation/training, memory fit, reward improvement, or numerical parity is
> claimed. H3 requires B=1 (including preprocessing), has no CFG (neutral guidance `1.0` only),
> and keeps video/audio states separate. Video uses shift 12, audio uses shift 3, and H3's
> data-ward velocity is converted only at the scheduler's standard-flow boundary.
> `num_inference_steps=N` means N transitions and N + 1 states.

# 💻 Supported Algorithms

| Algorithm      | `trainer_type` | Paper |
|----------------|----------------|-------|
| DPO            | dpo            | [Diffusion-DPO](https://arxiv.org/abs/2311.12908) |
| GRPO           | grpo           | [Flow-GRPO](https://arxiv.org/abs/2505.05470) / [Dance-GRPO](https://arxiv.org/abs/2505.07818) |
| DiffusionNFT   | nft            | [DiffusionNFT](https://arxiv.org/abs/2509.16117) |
| AWM            | awm            | [Advantage Weighted Matching](https://arxiv.org/abs/2509.25050) |
| DGPO           | dgpo           | [DGPO](https://arxiv.org/abs/2510.08425) |
| GRPO-Guard     | grpo-guard     | [GRPO-Guard](https://arxiv.org/abs/2510.22319) |
| DPPO           | dppo           | [Flow-DPPO](https://arxiv.org/abs/2606.11025) |
| CRD            | crd            | [Centered Reward Distillation](https://arxiv.org/abs/2603.14128) ([Blog (Chinese)](https://mp.weixin.qq.com/s/fpTi7PPi3APSNJQ2kXN3Dw))|
| DiffusionOPD   | diffusion-opd  | [DiffusionOPD](https://arxiv.org/abs/2605.15055) |

See [`Algorithm Guidance`](guidance/algorithms.md) for more information.

> Model and algorithm adapters are decoupled at the framework interface, but this does not imply
> every combination has completed training validation. Validation status varies by example:
> training-verified configurations include documented hardware and reward-trend evidence, while
> MiniMax H3 examples are schema/API validated only. Treat unlisted combinations as starting
> points that require their own compatibility and training evidence.

# 💾 Hardware Requirements

# 🚀 Get Started

## Installation

```bash
git clone https://github.com/Jayce-Ping/Flow-Factory.git
cd Flow-Factory
pip install -e .
```

Optional dependencies, such as `deepspeed`, are also available. Install them with:

```bash
pip install -e .[deepspeed]
```

> **Note**: The Bagel adapter requires `flash-attn` (>= 2.5.8) and `opencv-python`. Install them with `pip install -e .[bagel]` (the `[bagel]` extra is intentionally not part of `[all]` because flash-attn is heavy to build).

> **Dependency pin**: Project metadata installs `diffusers` from exact Git commit
> `f53d552036a0d1bd5570782a39cd40cfabf112bc`; MiniMax H3 depends on unreleased modular APIs at
> this revision. PyAV >=18.0.0 is a core dependency for reliable ordered video/audio reference
> decoding. A future stable-release upgrade must rerun the H3 feature probe, real no-weight
> component-spec/workflow checks, focused tests, and a separately documented real-weight smoke
> before this pin changes.

A CUDA training image (Python 3.12, **uv**-based install, PyTorch 2.8 + `cu129`, `deepspeed`, `wandb`, bundled `diffusers`) is defined under [`docker/docker-cuda/`](docker/docker-cuda/Dockerfile). See [`docker/README.md`](docker/README.md) for build and run instructions (including `linux/amd64` on Apple Silicon).

## Experiment Trackers

To use [Weights & Biases](https://wandb.ai/site/) or [SwanLab](https://github.com/SwanHubX/SwanLab) to log experimental results, install extra dependencies via `pip install -e .[wandb]` or `pip install -e .[swanlab]`.

After installation, set corresponding arguments in the config file:

```yaml
run_name: null  # Run name (auto: {model_type}_{finetune_type}_{trainer_type}_{timestamp})
project: "Flow-Factory"  # Project name for logging
logging_backend: "wandb"  # Options: wandb, swanlab, tensorboard, none
```

These trackers allow you to visualize both **training samples** and **metric curves** online:

![Online Image Samples](assets/wandb_images.png)

![Online Metric Examples](assets/wandb_metrics.png)

## Quick Start Example

Start training with the following simple command:

```bash
ff-train examples/grpo/lora/flux1/default.yaml
```

# 📖 Guidance

We provide a set of guidance documents to help you understand the framework and extend it. For a comprehensive understanding of the framework's design and motivation, refer to our [technique report](https://arxiv.org/abs/2602.12529).

| Document | Description |
|---|---|
| [Workflow](guidance/workflow.md) | End-to-end training pipeline: the overall stages from data preprocessing to policy optimization |
| [Algorithms](guidance/algorithms.md) | Supported RL algorithms (GRPO, GRPO-Guard, DPPO, DiffusionNFT, AWM, DPO, DGPO, CRD, DiffusionOPD) and their configurations |
| [Rewards](guidance/rewards.md) | Reward model system: built-in models, custom rewards, and remote reward servers |
| [Datasets](guidance/datasets.md) | Dataset layouts, task-specific JSONL schemas, media preprocessing, caching, and MiniMax H3 inputs |
| [New Model](guidance/new_model.md) | How to add support for a new Diffusion/Flow-Matching model |

# 📊 Dataset

Each dataset contains `train.txt` or `train.jsonl` and may provide an optional test split plus
image, video, or audio assets:

```plaintext
dataset/example/
├── train.txt or train.jsonl
├── test.txt or test.jsonl (optional)
├── images/ (optional)
├── videos/ (optional)
└── audios/ (optional)
```

See the [Dataset Guide](guidance/datasets.md) for text, image, video, and audio conventions;
single- and multi-condition schemas; media-root configuration; preprocessing and Arrow caching;
and the MiniMax H3 T2VA, FL2VA, and ordered Ref2VA formats.

# 💯 Reward Model

Flow-Factory provides a flexible reward model system that supports both built-in and custom reward models for reinforcement learning.

## Reward Model Types

Flow-Factory supports two types of reward models:

- **Pointwise Reward**: Computes independent scores for each sample (e.g., aesthetic quality, text-image alignment).
- **Pairwise Reward**: Computes rewards based on the pairwise comparison within the group. This is a special case of the following **Groupwise Reward**.
- **Groupwise Reward**: Computes rewards that requires the all samples in a group (e.g., ranking-based score or pairwise comparison).

## Built-in Reward Models

The following reward models are pre-registered and ready to use:

| Name | Type | Description | Reference |
|------|------|-------------|-----------|
| `PickScore` | Pointwise | CLIP-based aesthetic scoring model | [PickScore](https://huggingface.co/yuvalkirstain/PickScore_v1) |
| `PickScore_Rank` | Groupwise | Ranking-based reward using PickScore | [PickScore](https://huggingface.co/yuvalkirstain/PickScore_v1) |
| `CLIP` | Pointwise | Image-text cosine similarity | [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) |
| `OCR` | Pointwise | Text rendering in images | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| `GenEval` | Pointwise | Compositional T2I evaluation (object count, color, position) | [GenEval](https://github.com/djghosh13/geneval) |
| `vllm_evaluate` | Pointwise | VLM Yes/No judge + logprobs over an OpenAI-compatible API | [Rewards: VLM-as-Judge](guidance/rewards.md#vlm-as-judge) |
| `rational_rewards_t2i` | Pointwise | A reasoning reward model that provides multi-aspect reward for text-to-image; parsed aspects → scalar in [0, 1] | [RationalRewards-8B-T2I](https://huggingface.co/TIGER-Lab/RationalRewards-8B-T2I) |
| `rational_rewards_edit` | Pointwise | A reasoning reward model that provides multi-aspect reward for image edit; four aspects → scalar in [0, 1] | [RationalRewards-8B-Edit](https://huggingface.co/TIGER-Lab/RationalRewards-8B-Edit) |
| `qwen_image_bench` | Pointwise | Qwen-Image-Bench "Q-Judger"; hierarchical 5-dim / 56-facet scoring with per-prompt `dims_en` → scalar in [0, 1] | [Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench) |

> **GenEval** requires extra dependencies (mmcv, mmdet, open_clip). Install with: `bash scripts/install_geneval_deps.sh` (Python 3.10 recommended). See [guidance/rewards.md](guidance/rewards.md#dataset-metadata-convention) for dataset format.

> **VLM-as-Judge** (remote vLLM / OpenAI-style HTTP) is covered in [guidance/rewards.md#vlm-as-judge](guidance/rewards.md#vlm-as-judge) (`vllm_evaluate`, Rational Rewards, `qwen_image_bench`, async tips). For [RationalRewards](https://github.com/TIGER-AI-Lab/RationalRewards) specifically, serve the judge with [`scripts/start_vllm_rational_reward.sh`](scripts/start_vllm_rational_reward.sh) and set YAML `api_base_url` / `vlm_model` to match `--served-model-name` (defaults: `RationalRewards-8B-T2I` / `RationalRewards-8B-Edit`). For [Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench), use [`scripts/start_vllm_qwen_image_bench.sh`](scripts/start_vllm_qwen_image_bench.sh) and build the dataset with `python dataset/qwen_image_bench/prepare.py`.

## Using Built-in Reward Models

Simply specify the reward model name in your config file:
```yaml
rewards:
  name: "aesthetic" # Alias for this reward model
  reward_model: "PickScore" # Reward model type or a path like 'my_package.rewards.CustomReward'
  batch_size: 16
  device: "cuda"
  dtype: bfloat16
```

Refer to [Rewards Guidance](guidance/rewards.md) for more information about advanced usage, such as creating a custom reward model.


# 🤗 Acknowledgements

This repository is based on [diffusers](https://github.com/huggingface/diffusers/), [accelerate](https://github.com/huggingface/accelerate) and [peft](https://github.com/huggingface/peft).
We thank them for their contributions to the community!!!

# 📝 Citation

If you find Flow-Factory useful in your research, please consider citing our paper:

```bibtex
@article{ping2026flowfactory,
  title={Flow-Factory: A Unified Framework for Reinforcement Learning in Flow-Matching Models}, 
  author={Bowen Ping and Chengyou Jia and Minnan Luo and Hangwei Qian and Ivor Tsang},
  journal={arXiv preprint arXiv:2602.12529},
  year={2026},
  url={https://arxiv.org/abs/2602.12529}, 
}
```