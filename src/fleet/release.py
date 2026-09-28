from __future__ import annotations

from pathlib import Path
import json
import os

from huggingface_hub import HfApi

from .benchmark import run_benchmark
from .export_model import export_single_model
from .validation import validate_bank


ALLOWED_PUBLIC_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    try:
        from google.colab import userdata  # type: ignore
        token = userdata.get("HF_TOKEN")
    except Exception:
        token = None

    if not token:
        raise RuntimeError(
            "No Hugging Face write token found. Put a rotated write token "
            "in HF_TOKEN or Colab Secrets. Never hard-code it in Fleet."
        )

    return token


def _check_public_release(bank: dict) -> None:
    for item in bank["experts"]:
        license_id = str(item.get("license") or "").lower()

        if not item.get("redistributable"):
            raise RuntimeError(
                f"Refusing release: {item['name']} is not redistributable."
            )

        if license_id not in ALLOWED_PUBLIC_LICENSES:
            raise RuntimeError(
                f"Refusing release: {item['name']} has license={license_id!r}."
            )

        if not item.get("revision"):
            raise RuntimeError(
                f"Refusing release: {item['name']} has no pinned source revision."
            )


def _write_model_card(
    model_dir: Path,
    *,
    bank: dict,
    benchmark: dict,
) -> None:
    rows = []

    for item in bank["experts"]:
        rows.append(
            f"| `{item['name']}` | `{item['repo']}` | "
            f"`{item.get('license')}` | `{str(item.get('revision'))[:12]}` | "
            f"{float(item.get('retention', 0.0)):.4f} |"
        )

    card = f"""---
library_name: transformers
base_model: {bank["base_model"]}
license: other
pipeline_tag: text-generation
tags:
- qwen3
- peft
- lora
- semantic-routing
- fleet
---

# Fleet Semantic 4B v0.7.1

**One Fleet model artifact.**

This repository contains one consolidated Fleet weight file:

```text
fleet_model.safetensors
```

It includes all compiled expert tensors plus the semantic-router tensors.
There are no per-expert adapter directories in this release.

The Qwen backbone `{bank["base_model"]}` is referenced and loaded at runtime rather
than duplicated into this repository.

## Expert bank

| Capability | Source | License | Revision | Retention |
| --- | --- | --- | --- | ---: |
{chr(10).join(rows)}

## Files

```text
config.json
fleet_model.safetensors
fleet_manifest.json
configuration_fleet.py
modeling_fleet.py
benchmark_results.json
README.md
```

## Load

Fleet currently requires the `fleet` Python package because the model runtime uses
Fleet's semantic-routing implementation:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "noumenon-labs/Fleet-Semantic-4B",
    trust_remote_code=True,
    device_map="auto",
)
```

## Release benchmark

Passed: `{benchmark.get("passed")}`

This remains a research prototype.
"""

    (model_dir / "README.md").write_text(
        card,
        encoding="utf-8",
    )

    (model_dir / "benchmark_results.json").write_text(
        json.dumps(
            benchmark,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def release_to_hub(
    *,
    bank_dir: str | Path,
    repo_id: str,
    release_dir: str | Path | None = None,
    private: bool = False,
    run_release_benchmark: bool = True,
    max_benchmark_failures: int = 0,
    num_workers: int = 4,
) -> dict:
    bank_dir = Path(bank_dir)

    release_dir = Path(
        release_dir
        or bank_dir.parent / "fleet-single-model"
    )

    validation = validate_bank(bank_dir)

    if not validation["ok"]:
        raise RuntimeError(
            "Bank validation failed:\n"
            + "\n".join(validation["errors"])
        )

    bank = _read_json(bank_dir / "fleet_bank.json")
    _check_public_release(bank)

    benchmark_path = bank_dir / "benchmark_results.json"

    if run_release_benchmark:
        benchmark = run_benchmark(
            bank_dir,
            output_path=benchmark_path,
            max_failures=max_benchmark_failures,
        )
    elif benchmark_path.exists():
        benchmark = _read_json(benchmark_path)
    else:
        raise RuntimeError(
            "No benchmark_results.json and release benchmarking is disabled."
        )

    if not benchmark.get("passed"):
        raise RuntimeError(
            "Release benchmark failed; nothing uploaded.\n"
            + "\n".join(benchmark.get("failures", []))
        )

    exported = export_single_model(
        bank_dir=bank_dir,
        output_dir=release_dir,
    )

    _write_model_card(
        release_dir,
        bank=bank,
        benchmark=benchmark,
    )

    token = _resolve_token()
    api = HfApi(token=token)

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(release_dir),
        num_workers=int(num_workers),
    )

    return {
        "ok": True,
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "single_model_weights": exported["weights"],
        "expert_count": exported["expert_count"],
        "benchmark_passed": True,
    }
