# Model Roles

**Read when**: Adding a multi-role algorithm (distillation, adversarial critics), or changing
parameter ownership, per-role LoRA/full storage, role-scoped checkpointing, or the multi-role
optimizer contract.

## Why roles exist

A single-policy algorithm trains one copy of the trainable components, so `target_module_map`
alone describes parameter ownership: one map, one optimizer, one exported artifact. Distillation
trains several copies at once — a generator, a fake score, a surrogate critic — and each needs its
own optimizer group, its own storage, and its own routing to the components it owns.

`ModelRoleRegistry` generalizes that ownership rather than sitting beside it. It is built on the
canonical component identity contract, so read [Component Runtime](component_runtime.md) first:
role ownership cannot be trusted if component membership is not.

Implementation: `src/flow_factory/models/roles.py`,
`src/flow_factory/trainers/role_optimization.py`, and the role methods on `BaseAdapter` /
`BaseTrainer`.

## Declaring roles

`configure_model_roles(required_trainable_roles)` runs before `accelerator.prepare`, so the bundle
can see every member. Reconfiguring an existing registry raises rather than silently rebuilding
ownership under a prepared root.

| Role | Trainable | Present when |
|---|---|---|
| `generator` | yes | always; the artifact that ships |
| `fake` | yes | the algorithm asks for it |
| `surrogate` | yes | the algorithm asks for it |
| `reference` | no | always, as a snapshot |

`ModelRoleSpec` carries the ownership decision:

| Field | Meaning |
|---|---|
| `storage_mode` | `lora`, `full`, or `snapshot` |
| `adapter_name` | PEFT adapter under `lora`; `None` under `full` |
| `component_routes` | canonical component name -> the module this role actually uses |
| `trainable` | whether the role contributes optimizer parameters |

The two storage modes give the memory trade-off directly. Under `lora` every role is a separate
adapter name on one shared base, so N roles cost one base plus N adapters; the generator takes the
`default` adapter. Under `full` every non-generator role gets its own component copy, routed as
`{role}__{component}`, so N roles cost N copies.

`RoutedComponentProxy` resolves a canonical name through the active role, which is what lets
adapter code keep saying `self.transformer` while the module it reaches depends on the active
role. Constructing it with a static inner module still works for single-role runs.

## Scoping a forward

```python
with adapter.use_model_role("fake"):
    velocity = adapter.forward_state(...)
```

Distillation objectives are written against the score function while every adapter predicts a
velocity under its own direction convention, so convert with
`adapter.project_velocity_to_score_state(...)` rather than spelling the schedule inline. It is a
non-overridable wrapper: it applies the adapter's velocity direction first, then delegates the
schedule-specific clean-to-score step to `_project_clean_to_score_state`.

## Checkpointing

The exported artifact is the generator alone — the fake-score and surrogate roles are training
scaffolding and never ship. `save_checkpoint(model_only=True)` therefore scopes itself to the
generator role, and `_save_lora` passes `selected_adapters` so a shared base does not write every
role's adapter. `save_official_generator_ema` exports the generator's EMA, optionally alongside
raw `ema.ckpt` tensors.

Role metadata is written before `accelerator.save_state` and validated before
`accelerator.load_state`. Both orderings matter: accelerate mutates optimizer state in place, so a
resume that discovers a changed role layout afterwards has already restored state onto the wrong
groups.

## Optimizer and backend contract

`RoleOptimizationCoordinator` drives disjoint role updates through one physical optimizer, one
parameter group per role. `_validate_multirole_backend` rejects the layouts multi-role cannot
support:

- more than one prepared model root or more than one prepared optimizer;
- a prepared root or optimizer that is not the tracked `model_bundle` / `optimizer`;
- an optimizer group role mapping that changed during `prepare`;
- DeepSpeed outside ZeRO-1/2 (see [`constraints.md` #10](../constraints.md));
- FSDP2 with `use_orig_params=False`, or optimizer parameters that are not identities from the
  prepared root.

Single-role runs skip all of it: `_required_trainable_roles()` defaults to `("generator",)` and the
contract returns immediately.

## Review checklist

- [ ] Roles are configured before `accelerator.prepare`, once.
- [ ] Every trainable role has its own optimizer group and its own `component_routes`.
- [ ] `lora` roles declare an `adapter_name`; `full` roles do not.
- [ ] Forwards that depend on a role run inside `use_model_role`.
- [ ] Score-space objectives go through `project_velocity_to_score_state`.
- [ ] Exports contain generator weights only.
- [ ] Role metadata is written before, and validated ahead of, accelerate's state mutation.

## Cross-refs

- UP: [`constraints.md` #10](../constraints.md), [`architecture.md` Component Management](../architecture.md#component-management)
- PEER: [Component Runtime](component_runtime.md), [Structured Trajectory](structured_trajectory.md), [Autocast and Parameter Swaps](autocast_param_swap.md)
