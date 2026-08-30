# Agent Knowledge Base

| Trigger | Read |
|---------|------|
| Session start | `philosophy.md`, `constraints.md`, `architecture.md` |
| Touching `trainers/*.optimize`, `adapter.forward`/`inference`, `scheduler.step` | `topics/train_inference_consistency.md` |
| Touching dtype/precision, mixed precision config, debugging NaN/overflow | `topics/dtype_precision.md` |
| Editing a trainer `optimize()` loop / autocast scope, ref/EMA/named param swaps | `topics/autocast_param_swap.md` |
| Adding or modifying a model adapter | `topics/adapter_conventions.md` |
| Changing offline `PipelineIOContract`, condition-state preparation, output codecs, or output geometry | `topics/adapter_conventions.md`, `topics/component_runtime.md` |
| Changing component discovery, loading, lifecycle, or distributed preparation | `topics/component_runtime.md` |
| Changing FSDP loading or activation-checkpoint ownership | `topics/component_runtime.md` |
| Touching rollout collection, replay bridges, index maps, or multi-component order | `topics/structured_trajectory.md` |
| Adding an algorithm that trains several model copies at once, or changing per-variant LoRA/full storage or variant checkpointing | `topics/component_variants.md` |
| Changing optimizer roles, Muon, `CompositeOptimizer`, or optimizer/backend compatibility | `topics/component_variants.md`, `dependencies.md` |
| Adding adapter, upgrading diffusers, debugging output quality | `topics/parity_testing.md` |
| Touching MiniMax H3 workflows, dependency pins, ordered references, or H3 memory policies | `topics/minimax_h3.md` |
| Touching `TimeSampler`, `adapter.forward(t=...)`, `timestep_range`, `flow_match_sigma` | `topics/timestep_sigma.md` |
| Editing `data_utils/sampler*`, hparams sampler/batch fields | `topics/samplers.md` |
| Touching `sample()`/`optimize()` data flow, debugging `sample()`/`optimize()` OOM, adding high-resolution / video example configs | `topics/sample_lifecycle.md` |
| Changing `BaseSample`, partial sample gathering, or concrete-sample reconstruction | `topics/sample_lifecycle.md` |
| After completing a bug fix | `topics/fix_patterns.md` |
| Changing `pyproject.toml`, deps, install commands | `dependencies.md` |
| Adding or editing `.agents/` documentation | `docs_maintenance.md` |
