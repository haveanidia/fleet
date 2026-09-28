from __future__ import annotations

from pathlib import Path
import json

from huggingface_hub import HfApi, hf_hub_download


ALLOWED_REDIS_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
}


def canonical_base_family(value: str | None) -> str | None:
    if not value:
        return None

    value = str(value).lower().replace("\\", "/").rstrip("/")

    if "qwen3-4b-instruct-2507" in value:
        return "qwen3-4b-instruct-2507"

    return value.split("/")[-1]


def _read_card_license(card_data):
    if card_data is None:
        return None

    if isinstance(card_data, dict):
        value = card_data.get("license")
    else:
        value = getattr(card_data, "license", None)

    if isinstance(value, (list, tuple)):
        value = value[0] if value else None

    return str(value).lower() if value else None


def get_repo_metadata(repo_id: str, api: HfApi | None = None) -> dict:
    api = api or HfApi()

    try:
        info = api.model_info(repo_id)

        license_id = _read_card_license(
            getattr(info, "card_data", None)
        )

        if not license_id:
            for tag in getattr(info, "tags", None) or []:
                tag = str(tag)

                if tag.startswith("license:"):
                    license_id = tag.split(":", 1)[1].strip().lower()
                    break

        return {
            "license": license_id,
            "revision": getattr(info, "sha", None),
        }

    except Exception:
        return {
            "license": None,
            "revision": None,
        }


def get_repo_license(repo_id: str, api: HfApi | None = None) -> str | None:
    return get_repo_metadata(repo_id, api=api)["license"]


def inspect_adapter(
    repo_id: str,
    *,
    expected_base_model: str,
    api: HfApi | None = None,
) -> dict:
    api = api or HfApi()
    hub_meta = get_repo_metadata(
        repo_id,
        api=api,
    )

    cfg_path = hf_hub_download(
        repo_id=repo_id,
        filename="adapter_config.json",
        revision=hub_meta["revision"],
    )

    cfg = json.loads(
        Path(cfg_path).read_text(encoding="utf-8")
    )

    problems: list[str] = []

    if str(cfg.get("peft_type", "")).upper() != "LORA":
        problems.append(
            f"peft_type={cfg.get('peft_type')}"
        )

    expected = canonical_base_family(expected_base_model)
    actual = canonical_base_family(
        cfg.get("base_model_name_or_path")
    )

    if actual != expected:
        problems.append(
            f"base family={actual}; expected={expected}"
        )

    if cfg.get("use_dora", False):
        problems.append("DoRA")
    if cfg.get("bias", "none") != "none":
        problems.append("LoRA bias")
    if cfg.get("modules_to_save"):
        problems.append("modules_to_save")
    if cfg.get("rank_pattern"):
        problems.append("rank_pattern")
    if cfg.get("alpha_pattern"):
        problems.append("alpha_pattern")
    if cfg.get("layers_to_transform") is not None:
        problems.append("layers_to_transform")
    if cfg.get("target_parameters"):
        problems.append("target_parameters")
    if cfg.get("fan_in_fan_out", False):
        problems.append("fan_in_fan_out=True")
    if cfg.get("layer_replication"):
        problems.append("layer_replication")

    targets = cfg.get("target_modules") or []
    if isinstance(targets, str):
        targets = [targets]

    license_id = hub_meta["license"]

    return {
        "repo": repo_id,
        "config": cfg,
        "rank": int(cfg.get("r", 0)),
        "alpha": float(cfg.get("lora_alpha", 0)),
        "targets": sorted(str(x) for x in targets),
        "license": license_id,
        "revision": hub_meta["revision"],
        "redistributable": license_id in ALLOWED_REDIS_LICENSES,
        "problems": problems,
    }
