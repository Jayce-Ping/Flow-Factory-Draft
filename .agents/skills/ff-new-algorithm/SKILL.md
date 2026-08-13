---
name: ff-new-algorithm
description: "Complete workflow for adding a new RL training algorithm. Covers paradigm selection, TrainingArguments subclass, trainer implementation, registry, example config, and verification. Trigger: 'add algorithm', 'new trainer', 'new training method', 'implement algorithm'."
---

# New RL Algorithm Integration

> **Authoritative reference**: `guidance/algorithms.md`

## Prerequisites

Determine your algorithm's characteristics:
- **Paradigm**: Coupled or Decoupled? (`constraints.md` #7)
- **Dynamics**: Which SDE/ODE formulation? (`Flow-SDE`, `Dance-SDE`, `CPS`, `ODE`)
- **Advantage**: How are advantages computed from rewards? (Most algorithms can delegate to `AdvantageProcessor`)
- **Loss**: What is the policy optimization objective?

## Phase 1: Design

1. **Study existing implementations**:
   - Coupled example: `trainers/rl/grpo.py` (GRPO)
   - Decoupled example: `trainers/rl/nft.py` (DiffusionNFT) or `trainers/rl/awm.py` (AWM)
2. **Identify what's shared vs unique** (`constraints.md` #11):
   - Shared: the epoch loop (`BaseTrainer.start`), data loading, reward computation,
     `AdvantageProcessor`, `prepare_feedback`, `compute_advantages`, adapter interface, checkpoint logic
   - Unique: the loss function and the algorithm-specific hyperparameters. Never restate the loop
   - Per-epoch hook order: `sample()` → `prepare_feedback()` → `optimize()` (see `guidance/workflow.md`)

## Phase 2: Configuration

### Step 1 — Define Algorithm-Specific Arguments

Create a new file `src/flow_factory/hparams/training_args/my_algo.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field

from ._base import TrainingArguments


@dataclass
class MyAlgoTrainingArguments(TrainingArguments):
    """Training arguments specific to MyAlgo."""
    my_specific_param: float = field(
        default=0.1,
        metadata={"help": "Description of param."},
    )
    another_param: int = field(
        default=10,
        metadata={"help": "Description of param."},
    )
```

If the algorithm uses a different CFG `guidance_scale` at optimize time than at sampling/rollout time (e.g., `kl_cfg` for a reference-model branch), override `get_preprocess_guidance_scale()` so the data preprocessing stage encodes negative prompts:

```python
def get_preprocess_guidance_scale(self) -> float:
    """Ensure negative prompts are encoded when optimize-time CFG needs them."""
    return max(self.guidance_scale, self.my_optimize_cfg)
```

See `topics/adapter_conventions.md` "Classifier-Free Guidance (CFG) Convention" for the full two-stage CFG contract.

### Step 2 — Register in Argument Resolver

Update three files in `src/flow_factory/hparams/training_args/`:

**a)** Add import + registry entry in `_registry.py`:

```python
from .my_algo import MyAlgoTrainingArguments

_TRAINING_ARGS_REGISTRY: Dict[str, Type[TrainingArguments]] = {
    ...
    'my_algo': MyAlgoTrainingArguments,  # Add this
}
```

**b)** Add re-export in `__init__.py`:

```python
from .my_algo import MyAlgoTrainingArguments
# Also add to __all__
```

**c)** Add re-export in `src/flow_factory/hparams/__init__.py`:

```python
from .training_args import MyAlgoTrainingArguments
# Also add to __all__
```

## Phase 3: Trainer Implementation

### Step 3 — Create Trainer Class

```python
# src/flow_factory/trainers/my_algo.py
from .abc import BaseTrainer
from .registry import register_trainer

@register_trainer('my_algo')
class MyAlgoTrainer(BaseTrainer):
    """My custom RL algorithm trainer."""

    # Do NOT define start(). BaseTrainer.start() owns the epoch loop: reseed, checkpoint on
    # save_freq, evaluate on eval_freq, _run_training_step(), ema_step, _after_optimizer_step.
    # evaluate(), prepare_feedback() and compute_advantages() are likewise CONCRETE base
    # methods. optimize() is the only abstract one.
    #
    # Vary behavior through hooks instead of restating the loop:
    #   sampling_context()       - wrap the rollout (e.g. install a snapshot's weights)
    #   _run_training_step()     - replace the sample -> feedback -> optimize middle
    #   _after_gradient_step()   - run right after each optimizer step
    #   _after_optimizer_step()  - run once per epoch, after the EMA step
    #   _declare_model_variants() - declare several trainable copies (see component_variants.md)

    def sample(self):
        """Stages 2-3: K-repeat sampling + trajectory generation."""
        # Use self.adapter.inference() for trajectory generation
        pass

    def optimize(self, samples):
        """Stage 6: Policy update."""
        # Use self.adapter.forward() for single-step denoising.
        # Per-forward autocast — never one outer autocast around the loop (#20a).
        # Compute loss, backprop, step
        pass
```

> **Note**: `AdvantageProcessor` is auto-instantiated in `BaseTrainer._init_reward_model()`.
> Reward-based trainers delegate via `self.advantage_processor.compute_advantages()` — see `architecture.md` "Advantage Computation". (Pure-distillation trainers like `diffusion-opd` skip rewards/advantages with a no-op `prepare_feedback()`.)

### Step 4 — Register in Trainer Registry

Add to `_TRAINER_REGISTRY` in `src/flow_factory/trainers/registry.py`:

```python
'my_algo': 'flow_factory.trainers.my_algo.MyAlgoTrainer',
```

## Phase 4: Configuration & Examples

Create example config `examples/my_algo/lora/flux1/default.yaml`:

```yaml
model:
  model_type: "flux1"
  model_name_or_path: "black-forest-labs/FLUX.1-dev"
  finetune_type: "lora"
  target_components: ["transformer"]

train:
  trainer_type: "my_algo"
  my_specific_param: 0.1
  learning_rate: 1e-6          # Flat fields configure a single-optimizer run
  group_size: 4

  num_inference_steps: 28

scheduler:
  dynamics_type: "ODE"          # Or appropriate dynamics

data:
  datasets:
    - name: default
      dataset_dir: "path/to/dataset"   # Folder with train.jsonl / test.jsonl
      train:
        weight: 1
        max_dataset_size: 1024
      eval: {}

rewards:
  - name: "pickscore"

# An algorithm that trains several variants at once declares one optimizer per
# variant instead of the flat fields above, and is the only way to select Muon:
#
# optimizers:
#   - name: generator
#     optimizer: muon
#     learning_rate: 2e-5
#   - name: fake
#     learning_rate: 1e-5
#     update_frequency: 5
    reward_model: "pickscore"
    weight: 1.0
    batch_size: 16
```

## Phase 5: Verification

- [ ] `MyAlgoTrainingArguments` correctly parsed from YAML
- [ ] `get_training_args_class('my_algo')` returns correct subclass
- [ ] `get_trainer_class('my_algo')` loads `MyAlgoTrainer`
- [ ] Training runs end-to-end for ≥2 epochs without errors
- [ ] Loss values are numerically reasonable (not NaN, decreasing)
- [ ] Rewards improve over training
- [ ] Checkpoint save/load works correctly
- [ ] Works with at least two different model adapters
- [ ] Coupled algorithms only use SDE dynamics
- [ ] Decoupled algorithms work with both SDE and ODE dynamics

## Common Pitfalls

1. **Not subclassing `TrainingArguments`** — algorithm-specific params won't be parsed from YAML
2. **Forgetting `_registry.py` + `__init__.py` updates** — falls back to base `TrainingArguments`, losing custom params
3. **Using ODE with coupled paradigm** — no log-probabilities available, silent incorrect gradients
4. **Not calling `self.should_continue_training()`** — infinite loop if `max_epochs` is set
5. **Duplicating `_initialization()` logic** — already called in `BaseTrainer.__init__`; don't re-prepare modules
6. **Reimplementing advantage gather/scatter** — use `self.advantage_processor.compute_advantages()` instead; it handles both sampler topologies automatically
7. **Extending `GRPOTrainer` unnecessarily** — unless your algorithm extends GRPO's PPO-clipped loss, extend `BaseTrainer` directly (as NFT and AWM do)
8. **Optimizer-time CFG without `get_preprocess_guidance_scale()`** — if your algorithm calls `adapter.forward(guidance_scale=X)` where X > 1.0 but `training_args.guidance_scale` ≤ 1.0, negative prompts won't be encoded at preprocessing time and CFG silently falls back to no-CFG. Override `get_preprocess_guidance_scale()` in your TrainingArguments subclass to return `max(guidance_scale, your_optimize_cfg)`. See DGPO's `kl_cfg` for a real example.
