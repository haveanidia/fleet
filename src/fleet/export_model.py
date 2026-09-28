from __future__ import annotations

from pathlib import Path
import json
import shutil

import torch
from safetensors.torch import load_file, save_file


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def export_single_model(
    *,
    bank_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """
    Convert a compiled Fleet bank into ONE Fleet model artifact.

    Output:
      config.json
      fleet_model.safetensors   <- every expert tensor + semantic router tensor
      fleet_manifest.json       <- masks/config/provenance
      configuration_fleet.py
      modeling_fleet.py
      README.md

    No expert subdirectories are emitted.
    """
    bank_dir = Path(bank_dir)
    output_dir = Path(output_dir)

    bank_path = bank_dir / "fleet_bank.json"
    semantic_path = bank_dir / "semantic_router.safetensors"
    semantic_meta_path = bank_dir / "semantic_router.json"

    for required in [bank_path, semantic_path, semantic_meta_path]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required Fleet artifact: {required}")

    bank = _read_json(bank_path)
    semantic_meta = _read_json(semantic_meta_path)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    packed: dict[str, torch.Tensor] = {}
    manifest_experts = []

    # --------------------------------------------------------------
    # Pack every compiled expert into ONE tensor namespace.
    # --------------------------------------------------------------
    for expert_index, item in enumerate(bank["experts"]):
        name = item["name"]
        expert_dir = bank_dir / item["path"]

        weights_path = expert_dir / "adapter_model.safetensors"
        config_path = expert_dir / "adapter_config.json"
        mask_path = expert_dir / "fleet_origin_mask.json"
        report_path = expert_dir / "fleet_compile_report.json"

        for required in [
            weights_path,
            config_path,
            mask_path,
            report_path,
        ]:
            if not required.exists():
                raise FileNotFoundError(
                    f"{name}: required compiled artifact missing: {required}"
                )

        state = load_file(str(weights_path), device="cpu")

        for key, tensor in state.items():
            packed[
                f"experts.{expert_index}.{key}"
            ] = tensor.contiguous()

        manifest_experts.append({
            "index": expert_index,
            "name": name,
            "repo": item.get("repo"),
            "revision": item.get("revision"),
            "license": item.get("license"),
            "redistributable": bool(item.get("redistributable")),
            "retention": item.get("retention"),
            "adapter_config": _read_json(config_path),
            "origin_mask": _read_json(mask_path),
            "compile_report": _read_json(report_path),
        })

    # --------------------------------------------------------------
    # Pack semantic-router tensors into same safetensors file.
    # --------------------------------------------------------------
    semantic_state = load_file(
        str(semantic_path),
        device="cpu",
    )

    for key, tensor in semantic_state.items():
        packed[
            f"semantic.{key}"
        ] = tensor.contiguous()

    if not packed:
        raise RuntimeError("Fleet single-model export produced no tensors.")

    model_weights = output_dir / "fleet_model.safetensors"
    save_file(
        packed,
        str(model_weights),
        metadata={
            "format": "fleet-single-model",
            "version": "0.7.1",
            "base_model": str(bank["base_model"]),
        },
    )

    manifest = {
        "format": "fleet-single-model",
        "version": "0.7.1",
        "base_model": bank["base_model"],
        "expert_order": [x["name"] for x in bank["experts"]],
        "canonical_rank": bank.get("canonical_rank"),
        "canonical_targets": bank.get("canonical_targets"),
        "router": bank.get("router"),
        "semantic_router": semantic_meta,
        "experts": manifest_experts,
        "weights_file": "fleet_model.safetensors",
    }

    _write_json(
        output_dir / "fleet_manifest.json",
        manifest,
    )

    config = {
        "model_type": "fleet",
        "architectures": ["FleetForCausalLM"],
        "base_model_name_or_path": bank["base_model"],
        "fleet_manifest": "fleet_manifest.json",
        "fleet_weights": "fleet_model.safetensors",
        "auto_map": {
            "AutoConfig": "configuration_fleet.FleetConfig",
            "AutoModelForCausalLM": "modeling_fleet.FleetForCausalLM",
        },
    }

    _write_json(
        output_dir / "config.json",
        config,
    )

    # Copy load/runtime code shipped specifically with the model repo.
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent.parent

    for source_name, dest_name in [
        ("hub_model/configuration_fleet.py", "configuration_fleet.py"),
        ("hub_model/modeling_fleet.py", "modeling_fleet.py"),
    ]:
        src = package_root / source_name
        if not src.exists():
            raise FileNotFoundError(f"Hub model runtime source missing: {src}")
        shutil.copy2(src, output_dir / dest_name)

    return {
        "output_dir": str(output_dir),
        "weights": str(model_weights),
        "expert_count": len(manifest_experts),
        "expert_order": manifest["expert_order"],
        "base_model": bank["base_model"],
    }
