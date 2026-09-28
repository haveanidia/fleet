# Contributing

Contributions are welcome, especially around:

- heterogeneous LoRA compilation;
- routing telemetry and benchmarks;
- semantic NULL calibration;
- FleetAdapter compatibility;
- native Transformers integration;
- tests for install/uninstall and offload behavior.

## Setup

```bash
pip install -e ".[dev]"
```

## Tests

```bash
pytest -q
```

Please avoid committing model checkpoints, Hugging Face tokens, or generated
`artifacts/` directories.
