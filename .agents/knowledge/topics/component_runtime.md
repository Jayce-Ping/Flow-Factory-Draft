# Component Runtime

**Read when**: Changing component discovery, loading, lifecycle, scheduler groups, or distributed
preparation.

## Runtime matrix

| Backend | Runtime | Loading contract |
|---|---|---|
| Eager diffusers pipeline | `ClassicPipelineRuntime` | `load_pipeline()` returns materialized components |
| Lazy modular pipeline | `ModularPipelineRuntime` | specs are declared first; explicit names materialize modules |
| Explicit non-diffusers container | `PseudoPipelineRuntime` | canonical roots and non-enumerated aliases are declared explicitly |

All adapters implement `load_pipeline()`. Non-classic adapters also override
`build_component_runtime()`. `adapter.pipeline` remains the backend compatibility alias.

## Lookup and enumeration

- **Canonical lookup** returns the backend-owned component or declaration.
- **Override lookup** returns a prepared proxy or LoRA/checkpoint replacement without changing
  canonical ownership.
- **Declared names/specs** are available for explicit lookup and role discovery.
- **Materialized names** include canonical `torch.nn.Module` instances only.
- Device/dtype lifecycle enumeration excludes declared-only specs, optional `None` entries,
  pseudo aliases, and prepared/replacement overrides.
- `materialize_components(None)` means already-materialized modules. Lazy specs require explicit
  names.

Implementation: `src/flow_factory/models/runtime/` and
`src/flow_factory/models/abc.py::BaseAdapter`.

## Canonical identity

The runtime is the only authority on which components exist. Never test membership with
`hasattr(adapter, name)`: a component whose canonical name is not also an adapter attribute — H3
Ref2VA's `transformer_ref` is the live case — silently drops out of every `hasattr`-gated loop,
which previously produced an empty optimizer parameter list rather than an error. Resolve through
`get_component(name)` and let a missing name raise. Aliasing the component onto `self.transformer`
does not fix this: it registers an extra override named `transformer` while the real component
stays invisible to the gated loops.

Overrides must name a declared component, and `_load_full_model` writes through `set_component()`
so a replacement stays visible to the runtime instead of shadowing it with a bare attribute.

## Scheduler and distributed boundaries

- `adapter.scheduler` is the canonical primary scheduler.
- `SchedulerGroup.names` is immutable and equals `trajectory_component_order`.
- `SchedulerGroup` owns ordered mode and seed dispatch; mapping iteration never defines RNG order.
- `ModelBundle` plus `RoutedComponentProxy` is the sole distributed preparation runtime.
- Trainer stages call public adapter lifecycle methods so model-specific overrides remain active.

## Failure modes

- Expanding omitted materialization to all declarations loads tokenizers, configs, and weights
  unexpectedly.
- Enumerating aliases or overrides as lifecycle roots moves or wraps the same parameters twice.
- Including optional `None` declarations in role groups fails during freeze/device operations.
- Preparing components outside `ModelBundle` breaks FSDP/DeepSpeed root ownership.
- Building scheduler order from mappings breaks deterministic multi-component RNG dispatch.

## Review checklist

- [ ] Canonical, override, declared, and materialized paths remain distinct.
- [ ] Membership resolves through the runtime, never `hasattr` on the adapter.
- [ ] Lazy loading names every required component explicitly.
- [ ] Lifecycle enumeration contains canonical modules only.
- [ ] `adapter.pipeline`, `adapter.scheduler`, and public lifecycle hooks remain compatible.
- [ ] Scheduler names equal `trajectory_component_order`.
- [ ] Distributed preparation still routes through `ModelBundle`/`RoutedComponentProxy`.

## Cross-refs

- UP: [`constraints.md` #5](../constraints.md#5-adapter-component-runtime-contract), [`architecture.md` Component Management](../architecture.md#component-management)
- PEER: [Structured Trajectory](structured_trajectory.md), [Adapter Conventions](adapter_conventions.md)
