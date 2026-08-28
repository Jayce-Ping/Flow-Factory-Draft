---
name: ff-new-algorithm
description: "Complete workflow for adding an online or offline training algorithm. Covers execution-contract and paradigm selection, TrainingArguments subclass, trainer implementation, registry, example config, and verification. Trigger: 'add algorithm', 'new trainer', 'new training method', 'implement algorithm'."
---

# New Training Algorithm Integration

> **Authoritative reference**: `guidance/algorithms.md`

## Prerequisites

Determine your algorithm's characteristics:
- **Acquisition**: Generated rollouts or a finite dataset? (`generation` / `dataset`)
- **Feedback**: Runtime reward/advantage or none? (`runtime_reward` / `none`)
- **Paradigm**: Coupled or Decoupled? (`constraints.md` #7)
- **Dynamics**: Which SDE/ODE formulation? (`Flow-SDE`, `Dance-SDE`, `CPS`, `ODE`)
- **Supervision**: Prompt-only generation, demonstrations, preference pairs, or a new typed record?
- **Advantage**: If feedback is enabled, how are advantages computed? (Most reward-based algorithms can delegate to `AdvantageProcessor`)
- **Loss**: What is the policy optimization objective?

## Phase 1: Design

1. **Study existing implementations**:
   - Coupled example: `trainers/rl/grpo.py` (GRPO)
   - Decoupled example: `trainers/rl/nft.py` (DiffusionNFT) or `trainers/rl/awm.py` (AWM)
   - Finite demonstration example: `trainers/offline/sft.py` (SFT)
   - Finite preference example: `trainers/offline/offline_dpo.py` (offline DPO)
2. **Identify what's shared vs unique** (`constraints.md` #11):
   - Shared: the cycle loop (`BaseTrainer.start`), acquisition dispatch, progress counters,
     adapter interface, checkpoint/eval boundaries, role optimization, and exact-resume identity
   - Conditional: runtime rewards, `AdvantageProcessor`, `prepare_feedback`, and
     `compute_advantages` exist only when the feedback contract requests them
   - Unique: the loss function and the algorithm-specific hyperparameters. Never restate the loop
   - Generation hook order: `sample()` → optional `prepare_feedback()` → `optimize()`
   - Dataset hook order: official finite loader traversal → `optimize_batch(batch)`
3. **Declare one immutable execution contract**:
   - Online RL: `ONLINE_EXECUTION_CONTRACT` (`generation + runtime_reward`)
   - Generation without rewards: `ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT`
   - Finite offline training: `OFFLINE_EXECUTION_CONTRACT` (`dataset + none`)

Keep execution semantics orthogonal to `PipelineIOContract`. The algorithm owns how examples are
acquired and optimized; the adapter owns accepted input/output media, geometry, and output-state
encoding. A new offline record shape belongs in the typed data layer, never in model-specific loss
branches.

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

For a generated online algorithm:

```python
# src/flow_factory/trainers/rl/my_online_algo.py
from ...contracts import ONLINE_EXECUTION_CONTRACT
from ..abc import BaseTrainer
from ..registry import register_trainer

@register_trainer("my-online-algo")
class MyOnlineAlgoTrainer(BaseTrainer):
    """My generated-acquisition algorithm."""

    execution_contract = ONLINE_EXECUTION_CONTRACT

    # Do NOT define start(). BaseTrainer.start() owns the acquisition loop: reseed,
    # periodic boundaries, acquisition dispatch, EMA, and _after_acquisition_cycle().
    # evaluate(), prepare_feedback(), and compute_advantages() are concrete base methods.
    # A generation trainer implements sample() and optimize(samples).
    #
    # Vary behavior through hooks instead of restating the loop:
    #   sampling_context()       - wrap the rollout (e.g. install a snapshot's weights)
    #   _run_training_step()     - replace the sample -> feedback -> optimize middle
    #   _after_gradient_step()   - run right after each optimizer step
    #   _after_acquisition_cycle() - run once per rollout iteration or data epoch
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

For a finite offline algorithm:

```python
# src/flow_factory/trainers/offline/my_offline_algo.py
from ...contracts import OFFLINE_EXECUTION_CONTRACT
from ..abc import BaseTrainer
from ..registry import register_trainer

@register_trainer("my-offline-algo")
class MyOfflineAlgoTrainer(BaseTrainer):
    """My finite-dataset algorithm."""

    paradigm = "decoupled"
    execution_contract = OFFLINE_EXECUTION_CONTRACT

    def _build_train_dataloader(self):
        """Build a finite official-DistributedSampler loader for one typed schema."""
        # Reuse build_offline_train_dataloader when the supervision type matches;
        # otherwise extend the typed schema/collator first.
        ...

    def optimize_batch(self, batch):
        """Apply one gradient-accumulation microstep from a dataset batch."""
        # Decode output media in the dataset and encode it on demand through
        # adapter.encode_output_state(); never add target VAE latents to the cache.
        ...
```

> **Note**: `AdvantageProcessor` is relevant only to `runtime_reward` feedback.
> Reward-based trainers delegate via `self.advantage_processor.compute_advantages()` — see
> `architecture.md` "Advantage Computation". `none` feedback bypasses rewards structurally; do not
> emulate that by overriding reward methods with incidental no-ops.

### Step 4 — Register in Trainer Registry

Add to `_TRAINER_REGISTRY` in `src/flow_factory/trainers/registry.py`:

```python
'my_algo': 'flow_factory.trainers.rl.my_algo.MyAlgoTrainer',
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
    reward_model: "pickscore"
    weight: 1.0
    batch_size: 16

optimizers:
  - name: default
    learning_rate: 1e-6
    weight_decay: 1e-4
    max_grad_norm: 1.0
```

## Phase 5: Verification

- [ ] `MyAlgoTrainingArguments` correctly parsed from YAML
- [ ] `get_training_args_class('my_algo')` returns correct subclass
- [ ] `get_trainer_class('my_algo')` loads `MyAlgoTrainer`
- [ ] `execution_contract` matches the argument class and implemented optimization hook
- [ ] Training runs end-to-end for ≥2 acquisition cycles without errors
- [ ] Dataset acquisition defines one epoch as one complete finite dataloader traversal
- [ ] Loss values are numerically reasonable (not NaN, decreasing)
- [ ] Rewards improve over training when feedback is `runtime_reward`
- [ ] Offline supervision media is encoded on the fly and excluded from preprocessing caches
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
9. **Inferring online/offline behavior from batch keys** — declare `ExecutionContract`; keep acquisition and feedback independent from the model I/O schema.
10. **Using `optimize()` for finite data** — dataset acquisition calls `optimize_batch(batch)` and advances `data_epoch` only after clean loader exhaustion.
11. **Caching target/chosen/rejected latents** — cache prompt/input conditions only; output media is decoded and encoded on demand through the adapter output codec.
