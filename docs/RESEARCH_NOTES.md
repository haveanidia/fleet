# Research notes

## Claims v0.5.1 supports

Fleet implements a post-hoc pipeline for:

- semantic normalization of independently saved LoRA adapters,
- rank normalization,
- union target-module normalization,
- preservation of source module availability,
- mask-aware token routing over a canonical bank.

## Claims not yet established

The repository does not claim:

- state-of-the-art routing,
- better quality than retraining homogeneous experts,
- universal PEFT-method compatibility,
- production-stable router internals,
- that every third-party adapter improves the base model.

## Next experiments

1. benchmark single source adapter vs compiled adapter,
2. standard Arrow vs Fleet masked Arrow,
3. rank 16/24/32/48/64 storage-quality frontier,
4. union ABI vs common-subset ABI,
5. calibration/distillation after rank truncation,
6. independent third-party adapters from unrelated authors,
7. native Fleet router instead of monkey-patching Arrow.
