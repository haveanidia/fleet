from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
from peft import LoraConfig, TaskType
from safetensors.torch import load_file, save_file


A_SUFFIX = ".lora_A.weight"
B_SUFFIX = ".lora_B.weight"

SEMANTIC_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    r"(?:\.base_layer)?$"
)


def semantic_module_id(prefix: str) -> str:
    """Normalize wrapper-specific PEFT/Unsloth paths to Qwen layer semantics."""
    match = SEMANTIC_RE.search(prefix)

    if not match:
        raise ValueError(f"Cannot canonicalize LoRA module path: {prefix}")

    layer, block, target = match.groups()
    return f"layers.{int(layer)}.{block}.{target}"


def storage_prefix(semantic_id: str) -> str:
    """Canonical PEFT storage path for Qwen3ForCausalLM adapters."""
    return f"base_model.model.model.{semantic_id}"


def source_scaling(config: dict) -> float:
    r = int(config["r"])
    alpha = float(config["lora_alpha"])

    if config.get("use_rslora", False):
        return alpha / math.sqrt(r)

    return alpha / r


def load_adapter_state(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    safe = path / "adapter_model.safetensors"
    legacy = path / "adapter_model.bin"

    if safe.exists():
        return load_file(str(safe), device="cpu")

    if legacy.exists():
        state = torch.load(
            legacy,
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(state, dict):
            return state

    raise FileNotFoundError(f"No supported adapter weights in {path}")


def load_pairs(path: str | Path) -> dict:
    state = load_adapter_state(path)
    pairs = {}

    for key, tensor in state.items():
        if not key.endswith(A_SUFFIX):
            continue

        raw_prefix = key[:-len(A_SUFFIX)]
        b_key = raw_prefix + B_SUFFIX

        if b_key not in state:
            raise KeyError(f"Missing B matrix for {raw_prefix}")

        semantic_id = semantic_module_id(raw_prefix)

        if semantic_id in pairs:
            raise ValueError(
                f"Duplicate semantic module after normalization: {semantic_id}"
            )

        pairs[semantic_id] = {
            "A": tensor,
            "B": state[b_key],
            "raw_prefix": raw_prefix,
        }

    if not pairs:
        raise RuntimeError(f"No standard LoRA A/B pairs found in {path}")

    return pairs


def build_schema(source_pairs: dict[str, dict]) -> dict[str, tuple[int, int]]:
    schema: dict[str, tuple[int, int]] = {}

    for expert, pairs in source_pairs.items():
        for semantic_id, pair in pairs.items():
            shape = (
                int(pair["B"].shape[0]),
                int(pair["A"].shape[1]),
            )

            if semantic_id in schema and schema[semantic_id] != shape:
                raise ValueError(
                    f"Shape conflict at {semantic_id}: "
                    f"{schema[semantic_id]} vs {shape} ({expert})"
                )

            schema[semantic_id] = shape

    return schema


def refactor_lora_delta(
    A: torch.Tensor,
    B: torch.Tensor,
    source_scale: float,
    target_rank: int,
):
    """
    Represent source_scale * B @ A at a target LoRA rank.

    - target_rank >= source rank: exact up to floating error
    - target_rank < source rank: best truncated low-rank approximation
      obtained via thin QR + SVD of the small rank-space core
    """
    A = A.float()
    B = B.float()

    source_rank = int(A.shape[0])
    out_dim = int(B.shape[0])
    in_dim = int(A.shape[1])

    if target_rank >= source_rank:
        mag = math.sqrt(abs(float(source_scale)))
        sign = 1.0 if source_scale >= 0 else -1.0

        A_out = torch.zeros(
            target_rank,
            in_dim,
            dtype=torch.float32,
        )
        B_out = torch.zeros(
            out_dim,
            target_rank,
            dtype=torch.float32,
        )

        A_out[:source_rank] = A * mag
        B_out[:, :source_rank] = B * (mag * sign)

        return A_out, B_out, {
            "source_rank": source_rank,
            "target_rank": target_rank,
            "retained_energy": 1.0,
            "mode": "exact" if target_rank == source_rank else "exact_pad",
        }

    Qb, Rb = torch.linalg.qr(B, mode="reduced")
    Qa, Ra = torch.linalg.qr(A.T, mode="reduced")

    core = float(source_scale) * (Rb @ Ra.T)
    U, S, Vh = torch.linalg.svd(core, full_matrices=False)

    k = min(target_rank, int(S.numel()))
    sqrt_s = torch.sqrt(torch.clamp(S[:k], min=0))

    Bk = (Qb @ U[:, :k]) * sqrt_s.unsqueeze(0)
    Ak = sqrt_s.unsqueeze(1) * (Vh[:k, :] @ Qa.T)

    A_out = torch.zeros(
        target_rank,
        in_dim,
        dtype=torch.float32,
    )
    B_out = torch.zeros(
        out_dim,
        target_rank,
        dtype=torch.float32,
    )

    A_out[:k] = Ak
    B_out[:, :k] = Bk

    total = float(torch.sum(S**2).item())
    kept = float(torch.sum(S[:k]**2).item())

    return A_out, B_out, {
        "source_rank": source_rank,
        "target_rank": target_rank,
        "retained_energy": 1.0 if total == 0 else kept / total,
        "mode": "truncate_svd",
    }


def lowrank_retention(
    pairs: dict,
    config: dict,
    target_rank: int,
) -> float:
    scale = source_scaling(config)
    total = 0.0
    kept = 0.0

    for pair in pairs.values():
        A = pair["A"].float()
        B = pair["B"].float()

        Qb, Rb = torch.linalg.qr(B, mode="reduced")
        Qa, Ra = torch.linalg.qr(A.T, mode="reduced")

        core = float(scale) * (Rb @ Ra.T)
        singular = torch.linalg.svdvals(core)
        energy = singular.square()

        total += float(energy.sum().item())
        k = min(target_rank, int(singular.numel()))
        kept += float(energy[:k].sum().item())

    return 1.0 if total == 0 else kept / total


def compile_expert(
    *,
    expert_name: str,
    source_dir: str | Path,
    output_dir: str | Path,
    base_model: str,
    canonical_rank: int,
    module_schema: dict[str, tuple[int, int]],
    canonical_targets: list[str],
    source_metadata: dict | None = None,
) -> dict:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    cfg = json.loads(
        (source_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    pairs = load_pairs(source_dir)
    scale = source_scaling(cfg)

    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)

    state = {}
    origin_mask = {}
    module_report = []

    for semantic_id, (out_dim, in_dim) in module_schema.items():
        existed = semantic_id in pairs
        origin_mask[semantic_id] = existed

        if existed:
            pair = pairs[semantic_id]
            A_new, B_new, info = refactor_lora_delta(
                pair["A"],
                pair["B"],
                source_scale=scale,
                target_rank=canonical_rank,
            )
        else:
            A_new = torch.zeros(
                canonical_rank,
                in_dim,
                dtype=torch.float32,
            )
            B_new = torch.zeros(
                out_dim,
                canonical_rank,
                dtype=torch.float32,
            )
            info = {
                "source_rank": 0,
                "target_rank": canonical_rank,
                "retained_energy": 1.0,
                "mode": "zero_slot",
            }

        state[storage_prefix(semantic_id) + A_SUFFIX] = A_new.contiguous()
        state[storage_prefix(semantic_id) + B_SUFFIX] = B_new.contiguous()

        module_report.append({
            "module": semantic_id,
            "origin_active": existed,
            **info,
        })

    save_file(
        state,
        str(output_dir / "adapter_model.safetensors"),
    )

    compiled_cfg = LoraConfig(
        r=canonical_rank,
        lora_alpha=canonical_rank,
        lora_dropout=0.0,
        target_modules=canonical_targets,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
    )
    compiled_cfg.base_model_name_or_path = base_model
    compiled_cfg.save_pretrained(output_dir)

    (output_dir / "fleet_origin_mask.json").write_text(
        json.dumps(origin_mask, indent=2),
        encoding="utf-8",
    )

    real = [x for x in module_report if x["origin_active"]]

    report = {
        "expert": expert_name,
        "source_dir": str(source_dir),
        "source_rank": int(cfg["r"]),
        "canonical_rank": canonical_rank,
        "canonical_targets": canonical_targets,
        "origin_active_modules": int(sum(origin_mask.values())),
        "canonical_slots": len(origin_mask),
        "mean_retained_energy": (
            float(sum(x["retained_energy"] for x in real) / len(real))
            if real
            else 0.0
        ),
        "min_retained_energy": (
            float(min(x["retained_energy"] for x in real))
            if real
            else 0.0
        ),
        "source_metadata": source_metadata or {},
        "module_report": module_report,
    }

    (output_dir / "fleet_compile_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report
