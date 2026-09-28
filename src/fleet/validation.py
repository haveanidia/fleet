from __future__ import annotations

from pathlib import Path
import json


def validate_bank(
    bank_dir: str | Path,
) -> dict:
    bank_dir = Path(
        bank_dir
    )

    errors = []
    warnings = []

    manifest = (
        bank_dir
        / "fleet_bank.json"
    )

    if not manifest.exists():
        return {
            "ok": False,
            "errors": [
                "fleet_bank.json missing"
            ],
            "warnings": [],
        }

    bank = json.loads(
        manifest.read_text(
            encoding="utf-8"
        )
    )

    experts = bank.get(
        "experts",
        [],
    )

    if not experts:
        errors.append(
            "No experts in bank."
        )

    ranks = set()
    targets = set()

    for item in experts:
        d = (
            bank_dir
            / item["path"]
        )

        for required in [
            "adapter_config.json",
            "adapter_model.safetensors",
            "fleet_origin_mask.json",
            "fleet_compile_report.json",
        ]:
            if not (
                d / required
            ).exists():
                errors.append(
                    f"{item['name']}: "
                    f"missing {required}"
                )

        cfg_path = (
            d
            / "adapter_config.json"
        )

        mask_path = (
            d
            / "fleet_origin_mask.json"
        )

        if cfg_path.exists():
            cfg = json.loads(
                cfg_path.read_text(
                    encoding="utf-8"
                )
            )

            ranks.add(
                int(cfg["r"])
            )

            modules = (
                cfg.get(
                    "target_modules"
                )
                or []
            )

            if isinstance(
                modules,
                str,
            ):
                modules = [
                    modules
                ]

            targets.add(
                tuple(
                    sorted(modules)
                )
            )

        if mask_path.exists():
            mask = json.loads(
                mask_path.read_text(
                    encoding="utf-8"
                )
            )

            if not any(
                mask.values()
            ):
                errors.append(
                    f"{item['name']}: "
                    "zero active modules"
                )

    if len(ranks) > 1:
        errors.append(
            "Multiple canonical ranks: "
            + str(
                sorted(ranks)
            )
        )

    if len(targets) > 1:
        errors.append(
            "Canonical target-module "
            "sets differ."
        )

    minimum = int(
        bank.get(
            "minimum_experts",
            1,
        )
    )

    target = int(
        bank.get(
            "target_experts",
            len(experts),
        )
    )

    if len(experts) < minimum:
        errors.append(
            f"Expert count "
            f"{len(experts)} "
            f"below minimum {minimum}."
        )

    if len(experts) < target:
        warnings.append(
            f"Expert count "
            f"{len(experts)} "
            f"below target {target}."
        )

    router = bank.get(
        "router",
        {},
    )

    if (
        router.get("type")
        in {"fleet-semantic-v0.6", "fleet-semantic-v0.6.1", "fleet-semantic-v0.9"}
    ):
        for required in [
            "semantic_router.json",
            "semantic_router.safetensors",
        ]:
            if not (
                bank_dir
                / required
            ).exists():
                errors.append(
                    f"Semantic router "
                    f"declared but "
                    f"{required} missing."
                )

        meta_path = (
            bank_dir
            / "semantic_router.json"
        )

        if meta_path.exists():
            meta = json.loads(
                meta_path.read_text(
                    encoding="utf-8"
                )
            )

            bank_order = [
                x["name"]
                for x in experts
            ]

            if (
                meta.get(
                    "expert_order"
                )
                != bank_order
            ):
                errors.append(
                    "Semantic router expert "
                    "order differs from bank."
                )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "expert_count": (
            len(experts)
        ),
        "rank": (
            next(iter(ranks))
            if len(ranks) == 1
            else None
        ),
        "targets": (
            list(
                next(
                    iter(targets)
                )
            )
            if len(targets) == 1
            else None
        ),
        "router": (
            router.get("type")
            or "arrow-baseline"
        ),
    }
