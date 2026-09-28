from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass(slots=True)
class BuildPolicy:
    base_model: str = DEFAULT_BASE_MODEL
    canonical_rank: int = 32
    min_structural_retention: float = 0.95
    max_active_experts: int = 14
    target_experts: int = 10
    minimum_experts: int = 4
    top_k: int = 2
    router_temperature: float = 1.0
    allow_unknown_license_local: bool = True


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_policy(config: dict) -> BuildPolicy:
    policy = config.get("policy", {})
    return BuildPolicy(
        base_model=config.get("base_model", DEFAULT_BASE_MODEL),
        canonical_rank=int(policy.get("canonical_rank", 32)),
        min_structural_retention=float(policy.get("min_structural_retention", 0.95)),
        max_active_experts=int(policy.get("max_active_experts", 14)),
        target_experts=int(policy.get("target_experts", 10)),
        minimum_experts=int(policy.get("minimum_experts", 4)),
        top_k=int(policy.get("top_k", 2)),
        router_temperature=float(policy.get("router_temperature", 1.0)),
        allow_unknown_license_local=bool(
            policy.get("allow_unknown_license_local", True)
        ),
    )
