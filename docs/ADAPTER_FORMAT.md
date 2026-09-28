# FleetAdapter format — v0.9

A portable Fleet expert is a single `.fleet` ZIP:

```text
manifest.json
weights.safetensors
```

## Manifest

Required fields include:

```json
{
  "format": "fleet-adapter",
  "version": "0.9.0",
  "name": "math_reasoning",
  "base_model": "Qwen/Qwen3-4B-Instruct-2507",
  "rank": 32,
  "targets": [
    "layers.0.self_attn.q_proj"
  ],
  "num_layers": 36,
  "hidden_size": 2560,
  "metadata": {
    "source_repo": "...",
    "source_revision": "...",
    "license": "...",
    "semantic_space_sha256": "..."
  }
}
```

## Tensor keys

For each active module:

```text
module.<semantic_id>.A
module.<semantic_id>.B
module.<semantic_id>.scaling
module.<semantic_id>.weight_prototype
```

Semantic routing state:

```text
semantic.prototypes    # [layers, hidden]
semantic.thresholds    # [layers]
```

The adapter does not contain the base Qwen weights.

## Build

```bash
fleet adapter-pack \
  --bank artifacts/fleet-bank \
  --expert math_reasoning \
  --output math_reasoning.fleet
```

## Install

```python
model.install("math_reasoning.fleet")
```

or from the Hub:

```python
model.install("hf://org/repo/math_reasoning.fleet")
```

## Compatibility

`install()` validates:

- canonical rank,
- base model identity,
- layer/hidden dimensions,
- module A/B shapes,
- semantic-space fingerprint,
- duplicate expert names.

A mismatched package is rejected before model mutation.
