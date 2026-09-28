# Deployment

## 1. Build the bank

```bash
pip install -e .
fleet build --config configs/experts.json --output artifacts/fleet-bank
fleet validate artifacts/fleet-bank
```

A production promotion pipeline should build once, validate once, and deploy the immutable
compiled bank rather than running Hub discovery on every server startup.

## 2. Run locally

```bash
pip install -e ".[server]"

fleet serve \
  --bank artifacts/fleet-bank \
  --host 0.0.0.0 \
  --port 8000
```

## 3. Docker

```bash
docker build -t fleet .
docker run --gpus all \
  -p 8000:8000 \
  -v "$(pwd)/artifacts/fleet-bank:/models/fleet-bank:ro" \
  fleet
```

NVIDIA Container Toolkit is required for `--gpus all`.

## 4. Health checks

```bash
curl http://localhost:8000/health
```

The service loads the full model during startup. Container orchestration should allow
enough startup time for model download/load.

## 5. Production recommendations

Before production:

- pin the exact PEFT / Transformers versions tested on your GPU image,
- source adapter revisions are resolved and recorded at build time; promote only banks whose recorded commit SHAs you reviewed,
- record source artifact hashes,
- turn off automatic expert discovery after promotion,
- add benchmark gates for every expert capability,
- add load and concurrency tests,
- consider a dedicated inference engine once Fleet's router is native rather than patched Arrow.

The included service is deployment-ready scaffolding around a research router, not a claim
that the routing algorithm itself has completed production qualification.
