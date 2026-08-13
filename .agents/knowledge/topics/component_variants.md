# Component Variants

**Read when**: Adding an algorithm that trains more than one copy of the model at once
(distillation, adversarial critics), or changing parameter ownership, per-variant LoRA/full
storage, variant-scoped checkpointing, or the multi-role optimizer contract.

## The layering rule

`BaseAdapter` is infrastructure. It supplies named mechanisms and holds no opinion about what an
algorithm calls things or why. The algorithm's trainer owns the vocabulary.

| Layer | Speaks | Owns |
|---|---|---|
| `models/` | component variants | the mechanism: create a copy, activate one, collect its parameters, save a scope |
| `trainers/` | roles | the meaning: which variants exist, what they are called, which is trained when, which one ships |

Concretely: the model layer never contains the words *generator*, *fake* or *surrogate*. A DMD
trainer names its own variants and keeps a one-line `_add_generator()` if it wants one. Two
distillation trainers duplicating that line is fine and preferred; a shared abstraction that
teaches the adapter what a generator is, is not.

This mirrors how other frameworks draw the line: verl's `Role` enum lives in the PPO trainer, not
in `FSDPWorker`; diffusers exposes `set_adapters(names)` with no privileged adapter name; TRL's
`DPOTrainer` owns `ref_model` rather than the model class owning it.

## Two mechanisms, and how to choose

Both live on `BaseAdapter`. They answer different questions.

**Named parameter snapshots** are *temporal*: one set of weights installed at a time.

```python
adapter.add_named_parameters("old_policy")          # snapshot the live weights
with adapter.use_named_parameters("old_policy"):    # temporarily install them
    ...
adapter.update_named_parameters("old_policy", decay)  # blend toward live weights
```

Use this for anything that is *the same model at another point in time*: a reference policy, an
EMA, an old snapshot, a sampling model. CRD's old and sampling models are exactly this.

**Component variants** are *spatial*: several copies live at once, each with its own optimizer
group.

```python
adapter.declare_component_variants(("generator", "fake"))   # names are yours
with adapter.use_component_variant("fake"):
    velocity = adapter.forward_state(...)
params = adapter.variant_parameters("fake")                  # build your own param group
```

Use this only when copies must coexist and hold gradients simultaneously. That is the one thing a
trainer cannot build for itself, because it needs PEFT adapters on a shared base, one prepared
`ModelBundle` root, and forward routing.

## Declaring variants

`declare_component_variants(trainable_variants)` runs before `accelerator.prepare` so the bundle
sees every member; reconfiguring an existing registry raises rather than silently rebuilding
ownership under a prepared root.

The **base variant is positional**: whichever name comes first owns the adapter's canonical
components, and every later variant is layered on it. The registry never recognises a particular
name, so an algorithm may call its base `generator` and the model layer stays ignorant of the word.

| `ComponentVariantSpec` field | Meaning |
|---|---|
| `storage_mode` | `lora`, `full`, or `snapshot` |
| `adapter_name` | PEFT adapter under `lora`; `None` under `full` |
| `component_routes` | canonical component name -> the module this variant uses |
| `trainable` | whether the variant contributes optimizer parameters |

The storage modes give the memory trade-off directly. Under `lora`, N variants cost one base plus
N adapters and the base takes the `default` adapter. Under `full`, N variants cost N copies routed
as `{variant}__{component}`. `RoutedComponentProxy` resolves a canonical name through the active
variant, so adapter code keeps saying `self.transformer`.

## Checkpointing

Scope is the caller's, passed explicitly:

```python
adapter.save_checkpoint(path, variant=None)        # the base components
adapter.save_checkpoint(path, variant="generator")  # one named variant
```

There is no rule inside the adapter about which variant ships. An algorithm that exports one
variant's EMA composes it from primitives, which is the whole point of the split:

```python
tensors = adapter.variant_parameter_ema_tensors("generator_ema")
with adapter.use_variant_parameter_ema("generator_ema"):
    adapter.save_checkpoint(path, variant=None, model_only=True)
```

Variant metadata is written before `accelerator.save_state` and validated before
`accelerator.load_state`. Both orderings matter: accelerate mutates optimizer state in place, so a
resume that discovers a changed layout afterwards has already restored state onto the wrong groups.
The on-disk key stays `roles` because the checkpoint is the trainer's artifact; the registry's own
key is `variants`, and the two are translated at the boundary.

## Optimizer and backend contract

`trainers/role_optimization.py` is a utility a trainer composes, not something `BaseTrainer` wires
automatically. `RoleOptimizationCoordinator` drives disjoint role updates through one physical
optimizer, one parameter group per role. `_validate_multirole_backend` rejects the layouts
multi-role cannot support:

- more than one prepared model root or more than one prepared optimizer;
- a prepared root or optimizer that is not the tracked `model_bundle` / `optimizer`;
- an optimizer group role mapping that changed during `prepare`;
- DeepSpeed outside ZeRO-1/2 (see [`constraints.md` #10](../constraints.md));
- FSDP2 with `use_orig_params=False`, or optimizer parameters that are not identities from the
  prepared root.

The primary role, whose optimizer step paces the public `self.step`, is simply the first declared.
Single-policy runs skip all of it: one variant, one group, and the flat `train.learning_rate`
family describes it.

## Review checklist

- [ ] No algorithm word (`generator`, `fake`, `surrogate`) appears under `src/flow_factory/models/`.
- [ ] A time-shifted copy uses named parameter snapshots, not a variant.
- [ ] Variants are declared once, before `accelerator.prepare`.
- [ ] `lora` variants declare an `adapter_name`; `full` variants do not.
- [ ] Forwards that depend on a variant run inside `use_component_variant`.
- [ ] Checkpoint scope is passed explicitly rather than inferred.
- [ ] Variant metadata is written before, and validated ahead of, accelerate's state mutation.

## Cross-refs

- UP: [`constraints.md` #10](../constraints.md), [`architecture.md` Component Management](../architecture.md#component-management)
- PEER: [Component Runtime](component_runtime.md), [Structured Trajectory](structured_trajectory.md), [Autocast and Parameter Swaps](autocast_param_swap.md)
