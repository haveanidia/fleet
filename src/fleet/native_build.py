from __future__ import annotations

from pathlib import Path
import ast
import json
import math
import shutil
import threading

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import (
    AutoTokenizer,
    Qwen3ForCausalLM,
)

from .compiler import (
    A_SUFFIX,
    B_SUFFIX,
    storage_prefix,
)
from .native.configuration_fleet import (
    FleetConfig,
)
from .native.modeling_fleet import (
    FleetForCausalLM,
    FleetLinear,
)


def _read_json(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _top_right_proto(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    iters: int = 15,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Arrow-compatible dominant right singular direction without materializing BA.
    """
    A = A.float()
    B = B.float()

    C = B.T @ B

    # Deterministic start instead of RNG so export is reproducible.
    v = torch.ones(
        A.shape[1],
        dtype=torch.float32,
    )

    v = v / (
        v.norm()
        + eps
    )

    for _ in range(
        int(iters)
    ):
        w = (
            A.T
            @ (
                C
                @ (
                    A
                    @ v
                )
            )
        )

        n = w.norm()

        if float(
            n.item()
        ) <= eps:
            return torch.zeros_like(
                v
            )

        v = w / n

    return v


def _build_config(
    base_config,
    bank: dict,
    semantic_meta: dict,
    origin_masks: dict,
) -> FleetConfig:
    raw = base_config.to_dict()
    for key in [
        "model_type",
        "architectures",
        "auto_map",
        "_name_or_path",
    ]:
        raw.pop(key, None)

    expert_names = [item["name"] for item in bank["experts"]]
    metadata = []
    for item in bank["experts"]:
        metadata.append({
            "name": item["name"],
            "description": item.get("description"),
            "source_repo": item.get("repo"),
            "source_revision": item.get("revision"),
            "license": item.get("license"),
            "redistributable": bool(item.get("redistributable")),
            "structural_retention": item.get("retention"),
        })

    all_semantic_ids = sorted({
        semantic_id
        for mask in origin_masks.values()
        for semantic_id in mask
    })
    eligibility = {}
    for semantic_id in all_semantic_ids:
        eligibility[semantic_id] = [
            bool(origin_masks[expert].get(semantic_id, False))
            for expert in expert_names
        ]

    return FleetConfig(
        **raw,
        fleet_version="0.9.0",
        fleet_base_model=bank["base_model"],
        fleet_rank=int(bank["canonical_rank"]),
        fleet_expert_names=expert_names,
        fleet_expert_metadata=metadata,
        fleet_targets=all_semantic_ids,
        fleet_eligibility=eligibility,
        fleet_router_policy=dict(semantic_meta["policy"]),
        fleet_semantic_mode="per_expert_thresholds",
        fleet_semantic_space_sha256=semantic_meta.get("semantic_space_sha256"),
    )

def _module_from_semantic(
    model,
    semantic_id: str,
):
    parts = (
        semantic_id.split(".")
    )

    if len(parts) != 4:
        raise ValueError(
            f"Unexpected semantic module id: {semantic_id}"
        )

    _, layer_str, block_name, target_name = parts

    layer = (
        model.model.layers[
            int(layer_str)
        ]
    )

    block = getattr(
        layer,
        block_name,
    )

    module = getattr(
        block,
        target_name,
    )

    if not isinstance(
        module,
        FleetLinear,
    ):
        raise TypeError(
            f"{semantic_id} is not FleetLinear after native conversion."
        )

    return module


def _load_semantic_router(
    model: FleetForCausalLM,
    *,
    bank_dir: Path,
):
    state = load_file(str(bank_dir / "semantic_router.safetensors"), device="cpu")
    num_experts = len(model.config.fleet_expert_names)

    for layer_id in range(model.config.num_hidden_layers):
        model.fleet_router.centers[layer_id].copy_(
            state[f"center.layer_{layer_id}"].to(model.fleet_router.centers.dtype)
        )
        model.fleet_router.semantic_prototypes[layer_id].copy_(
            state[f"prototype.layer_{layer_id}"].to(
                model.fleet_router.semantic_prototypes.dtype
            )
        )
        model.fleet_router.null_prototypes[layer_id].copy_(
            state[f"null_prototype.layer_{layer_id}"].to(
                model.fleet_router.null_prototypes.dtype
            )
        )

        per_expert_key = f"expert_thresholds.layer_{layer_id}"
        if per_expert_key in state:
            thresholds = state[per_expert_key].reshape(-1).float()
        else:
            # v0.6.1/v0.8 migration: replicate the bank-wide threshold.
            threshold = state[f"null_threshold.layer_{layer_id}"].reshape(()).float()
            thresholds = threshold.repeat(num_experts)

        if thresholds.numel() != num_experts:
            raise RuntimeError(
                f"Layer {layer_id} semantic thresholds have {thresholds.numel()} values "
                f"for {num_experts} experts."
            )
        model.fleet_router.semantic_thresholds[layer_id].copy_(thresholds)

def _load_experts(
    model: FleetForCausalLM,
    *,
    bank_dir: Path,
    bank: dict,
):
    expert_names = [x["name"] for x in bank["experts"]]
    num_experts = len(expert_names)

    for expert_index, item in enumerate(bank["experts"]):
        print(
            f"[native] loading expert {expert_index + 1}/{num_experts}: {item['name']}"
        )
        expert_dir = bank_dir / item["path"]
        state = load_file(str(expert_dir / "adapter_model.safetensors"), device="cpu")
        mask = _read_json(expert_dir / "fleet_origin_mask.json")

        for semantic_id, active in mask.items():
            if not active:
                continue
            module = _module_from_semantic(model, semantic_id)
            local_slot = module.local_slot(expert_index)
            if local_slot is None:
                raise RuntimeError(
                    f"Config eligibility did not allocate {item['name']} in {semantic_id}."
                )
            a_key = storage_prefix(semantic_id) + A_SUFFIX
            b_key = storage_prefix(semantic_id) + B_SUFFIX
            if a_key not in state or b_key not in state:
                raise KeyError(
                    f"{item['name']}: missing canonical tensors for {semantic_id}"
                )
            A = state[a_key]
            B = state[b_key]
            module.expert_A[local_slot].copy_(A.to(dtype=module.expert_A.dtype))
            module.expert_B[local_slot].copy_(B.to(dtype=module.expert_B.dtype))
            module.expert_scaling[local_slot] = 1.0
            module.weight_prototypes[local_slot].copy_(
                _top_right_proto(A, B).to(dtype=module.weight_prototypes.dtype)
            )

def build_native_checkpoint(
    *,
    bank_dir: str | Path,
    output_dir: str | Path,
    base_model: str | None = None,
    dtype: str = "float16",
    max_shard_size: str = "4GB",
) -> dict:
    bank_dir = Path(
        bank_dir
    )

    output_dir = Path(
        output_dir
    )

    bank = _read_json(
        bank_dir
        / "fleet_bank.json"
    )

    semantic_meta = _read_json(
        bank_dir
        / "semantic_router.json"
    )

    if semantic_meta.get("version") != "0.9.0" or not semantic_meta.get(
        "semantic_space_sha256"
    ):
        raise RuntimeError(
            "Fleet v0.9 native build requires a v0.9 dynamic semantic router. "
            "Delete semantic_router.json/.safetensors and run `fleet semantic-build` again."
        )

    base_model = (
        base_model
        or bank[
            "base_model"
        ]
    )

    torch_dtype = {
        "float16": (
            torch.float16
        ),
        "bfloat16": (
            torch.bfloat16
        ),
        "float32": (
            torch.float32
        ),
    }[
        dtype
    ]

    origin_masks = {}

    for item in (
        bank["experts"]
    ):
        origin_masks[
            item["name"]
        ] = _read_json(
            bank_dir
            / item["path"]
            / "fleet_origin_mask.json"
        )

    print(
        "[native] loading base model:",
        base_model,
    )

    # CPU load is intentional. The standalone builder needs full precision
    # Qwen weights available for save_pretrained and should not quantize the
    # released checkpoint.
    base = (
        Qwen3ForCausalLM
        .from_pretrained(
            base_model,
            dtype=torch_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    )

    fleet_config = _build_config(
        base.config,
        bank,
        semantic_meta,
        origin_masks,
    )

    # Convert the already-loaded Qwen object in-place so there is no second
    # 4B backbone allocated in RAM.
    base.__class__ = (
        FleetForCausalLM
    )

    base.config = (
        fleet_config
    )
    object.__setattr__(base, "_fleet_mutation_lock", threading.RLock())

    # Install Fleet modules around the existing loaded projection params.
    base._install_native_fleet()

    model = base

    _load_semantic_router(
        model,
        bank_dir=bank_dir,
    )

    _load_experts(
        model,
        bank_dir=bank_dir,
        bank=bank,
    )

    if output_dir.exists():
        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            base_model
        )
    )

    tokenizer.save_pretrained(
        output_dir
    )

    # Ship the custom architecture with the checkpoint.
    here = Path(
        __file__
    ).resolve().parent

    shutil.copy2(
        here
        / "native"
        / "configuration_fleet.py",
        output_dir
        / "configuration_fleet.py",
    )

    shutil.copy2(
        here
        / "native"
        / "modeling_fleet.py",
        output_dir
        / "modeling_fleet.py",
    )

    shutil.copy2(
        here
        / "native"
        / "fleet_adapter.py",
        output_dir
        / "fleet_adapter.py",
    )

    # Recursively validate every one-level relative import used by
    # Transformers remote-code loading before the expensive checkpoint save.
    def _check_remote_code(entry: Path, root: Path):
        pending = [entry]
        seen = set()

        while pending:
            current = pending.pop().resolve()
            if current in seen:
                continue
            seen.add(current)

            tree = ast.parse(
                current.read_text(encoding="utf-8"),
                filename=str(current),
            )

            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level == 0:
                    continue

                if node.level != 1:
                    raise RuntimeError(
                        f"{current.name} contains unsupported parent-relative "
                        f"import level={node.level}: {node.module!r}"
                    )

                if not node.module:
                    continue

                target = root / (node.module.replace(".", "/") + ".py")
                if not target.exists():
                    raise FileNotFoundError(
                        f"Remote-code dependency missing: {target}; "
                        f"imported by {current.name}"
                    )

                pending.append(target)

        return seen

    _check_remote_code(
        output_dir / "modeling_fleet.py",
        output_dir,
    )

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=(
            max_shard_size
        ),
    )

    # save_pretrained writes config after code copy. Rewrite to guarantee
    # the remote AutoClass mapping survived.
    model.config.save_pretrained(
        output_dir
    )

    if getattr(
        model,
        "generation_config",
        None,
    ) is not None:
        model.generation_config.save_pretrained(
            output_dir
        )

    return {
        "output_dir": str(
            output_dir
        ),
        "architecture": (
            "FleetForCausalLM"
        ),
        "base_model": (
            base_model
        ),
        "expert_count": len(
            bank["experts"]
        ),
        "standalone": True,
        "requires_base_download_at_inference": False,
    }
