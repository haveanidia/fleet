# Fleet v0.9 architecture

## Standalone model

```text
FleetForCausalLM
├── full Qwen3 backbone
├── FleetRouterState
└── Qwen blocks
    └── FleetLinear projections
```

The published native checkpoint contains the Qwen backbone and all currently installed Fleet state.
PEFT is used by the compiler pipeline, not by the intended native inference path.

## Compiler

A source LoRA contributes:

\[
\Delta W = sBA
\]

Fleet compiles heterogeneous source ranks into one canonical rank ABI using exact padding when possible and QR + small-core SVD for rank reduction.

## Sparse dynamic FleetLinear

Each supported Qwen projection is wrapped once. The base projection parameter path stays stable.

A FleetLinear keeps only local experts that actually target it:

```text
weight
bias
expert_A
expert_B
expert_scaling
weight_prototypes
expert_ids
```

`expert_ids` maps local slots to global semantic expert indices.

Installing an expert grows only its target modules. Uninstalling removes those local rows and compacts IDs.

## Dynamic semantic bank

The model-level router stores:

```text
centers                [L, H]
null_prototypes        [L, H]
semantic_prototypes    [L, E, H]
semantic_thresholds    [L, E]
```

The semantic center is derived only from the fixed NULL/base anchor set. It is therefore independent of the installed expert list.

Each FleetAdapter supplies one semantic prototype and threshold per layer. `install()` appends them; `uninstall()` removes them.

## Routing

For each request/layer:

1. build a prompt semantic direction at q_proj,
2. compute each expert's margin over the NULL prototype,
3. require each expert to clear its own threshold,
4. combine semantic score with a small LoRA weight-geometry signal,
5. choose dynamic `k=0/1/2`,
6. apply only local experts present in each FleetLinear.

`k=0` is a first-class base-Qwen route.

## Dynamic API

```python
model.experts()
model.install("robotics.fleet")
model.uninstall("robotics")
model.export_adapter("math_reasoning", "math_reasoning.fleet")
```

The mutation lock prevents install/uninstall from racing with generation.

## Serialization

Runtime mutations update FleetConfig eligibility and expert metadata. `save_pretrained()` therefore serializes the current expert layout and reloads matching local bank tensor shapes before state-dict loading.
