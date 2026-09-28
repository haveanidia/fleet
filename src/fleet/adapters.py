from __future__ import annotations

from pathlib import Path
import json
import shutil

from huggingface_hub import HfApi, snapshot_download

from .compiler import (
    build_schema,
    compile_expert,
    load_pairs,
    lowrank_retention,
)
from .config import BuildPolicy, load_json
from .preflight import inspect_adapter


def select_candidates(
    config: dict,
    policy: BuildPolicy,
) -> tuple[list[dict], list[dict]]:
    api = HfApi()
    inspections: list[dict] = []
    selected: list[dict] = []

    for slot in config.get("experts", []):
        valid: list[dict] = []

        for repo_id in slot.get("candidates", []):
            try:
                info = inspect_adapter(
                    repo_id,
                    expected_base_model=policy.base_model,
                    api=api,
                )

                inspections.append({
                    "slot": slot["name"],
                    **{
                        k: v
                        for k, v in info.items()
                        if k != "config"
                    },
                })

                if not info["problems"]:
                    valid.append({
                        "name": slot["name"],
                        "description": slot.get("description", ""),
                        **info,
                    })

            except Exception as exc:
                inspections.append({
                    "slot": slot["name"],
                    "repo": repo_id,
                    "problems": [
                        f"{type(exc).__name__}: {exc}"
                    ],
                })

        if not valid:
            continue

        valid.sort(
            key=lambda x: (
                not x["redistributable"],
                x["rank"],
            )
        )

        chosen = valid[0]

        if (
            not chosen["redistributable"]
            and not policy.allow_unknown_license_local
        ):
            inspections.append({
                "slot": slot["name"],
                "repo": chosen["repo"],
                "license": chosen.get("license"),
                "revision": chosen.get("revision"),
                "problems": [
                    "not admitted: license is missing or not in the public-release allowlist"
                ],
            })
            continue

        selected.append(chosen)

    return inspections, selected[: policy.max_active_experts]


def download_selected(
    selected: list[dict],
    root: str | Path,
) -> dict[str, Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    for item in selected:
        local = Path(
            snapshot_download(
                repo_id=item["repo"],
                revision=item.get("revision"),
                allow_patterns=[
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "adapter_model.bin",
                    "README.md",
                    "LICENSE",
                    "LICENSE*",
                ],
            )
        )

        if not (
            (local / "adapter_model.safetensors").exists()
            or (local / "adapter_model.bin").exists()
        ):
            raise FileNotFoundError(
                f"No adapter weights found in {item['repo']}"
            )

        paths[item["name"]] = local

    return paths


def build_bank(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict:
    config = load_json(config_path)

    from .config import load_policy
    policy = load_policy(config)

    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)
    source_root = output_dir / "_sources"
    compiled_root = output_dir / "experts"

    source_root.mkdir()
    compiled_root.mkdir()

    inspections, selected = select_candidates(
        config,
        policy,
    )

    if len(selected) < policy.minimum_experts:
        raise RuntimeError(
            f"Only {len(selected)} valid experts survived preflight; "
            f"minimum={policy.minimum_experts}"
        )

    source_paths = download_selected(
        selected,
        source_root,
    )

    source_configs: dict[str, dict] = {}
    source_pairs: dict[str, dict] = {}

    for item in selected:
        name = item["name"]
        source = source_paths[name]

        source_configs[name] = json.loads(
            (source / "adapter_config.json").read_text(
                encoding="utf-8"
            )
        )
        source_pairs[name] = load_pairs(source)

    retention = {
        expert: lowrank_retention(
            source_pairs[expert],
            source_configs[expert],
            policy.canonical_rank,
        )
        for expert in source_pairs
    }

    dropped = [
        expert
        for expert, value in retention.items()
        if value < policy.min_structural_retention
    ]

    for expert in dropped:
        source_pairs.pop(expert, None)
        source_configs.pop(expert, None)
        source_paths.pop(expert, None)

    selected = [
        item
        for item in selected
        if item["name"] in source_pairs
    ]

    if len(selected) < policy.minimum_experts:
        raise RuntimeError(
            f"Only {len(selected)} experts survive retention gate; "
            f"minimum={policy.minimum_experts}"
        )

    schema = build_schema(source_pairs)
    targets = sorted(
        {
            semantic_id.split(".")[-1]
            for semantic_id in schema
        }
    )

    origin_masks = {
        expert: {
            semantic_id: semantic_id in source_pairs[expert]
            for semantic_id in schema
        }
        for expert in source_pairs
    }

    dead = [
        expert
        for expert, mask in origin_masks.items()
        if not any(mask.values())
    ]
    if dead:
        raise RuntimeError(
            f"Experts with zero source-active modules: {dead}"
        )

    reports = {}

    for item in selected:
        name = item["name"]

        reports[name] = compile_expert(
            expert_name=name,
            source_dir=source_paths[name],
            output_dir=compiled_root / name,
            base_model=policy.base_model,
            canonical_rank=policy.canonical_rank,
            module_schema=schema,
            canonical_targets=targets,
            source_metadata={
                "repo": item["repo"],
                "license": item["license"],
                "revision": item.get("revision"),
                "redistributable": item["redistributable"],
                "description": item["description"],
            },
        )

    bank = {
        "format": "fleet-bank",
        "version": "0.7.0",
        "base_model": policy.base_model,
        "canonical_rank": policy.canonical_rank,
        "canonical_targets": targets,
        "top_k": policy.top_k,
        "router_temperature": policy.router_temperature,
        "target_experts": policy.target_experts,
        "minimum_experts": policy.minimum_experts,
        "experts": [
            {
                "name": item["name"],
                "description": item["description"],
                "repo": item["repo"],
                "license": item["license"],
                "revision": item.get("revision"),
                "redistributable": item["redistributable"],
                "retention": retention[item["name"]],
                "path": f"experts/{item['name']}",
            }
            for item in selected
        ],
        "dropped_for_retention": dropped,
        "inspections": inspections,
    }

    (output_dir / "fleet_bank.json").write_text(
        json.dumps(bank, indent=2),
        encoding="utf-8",
    )

    # Sources are cache references, not part of the production bank.
    shutil.rmtree(source_root, ignore_errors=True)

    return bank
