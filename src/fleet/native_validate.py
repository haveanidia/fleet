from __future__ import annotations

from pathlib import Path
import ast
import json
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _relative_import_closure(entry: Path, root: Path) -> set[Path]:
    pending = [entry]
    seen: set[Path] = set()

    while pending:
        current = pending.pop()
        current = current.resolve()

        if current in seen:
            continue
        seen.add(current)

        source = current.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(current))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level == 0:
                continue

            if node.level != 1:
                raise RuntimeError(
                    f"{current.name} contains unsupported parent-relative import: "
                    f"level={node.level}, module={node.module!r}"
                )

            if not node.module:
                continue

            target = root / (node.module.replace(".", "/") + ".py")
            if not target.exists():
                raise FileNotFoundError(
                    f"Remote-code dependency missing: {target}. "
                    f"Imported by {current.name}."
                )

            pending.append(target)

    return seen


def validate_runtime_sources(model_dir: str | Path) -> dict:
    model_dir = Path(model_dir)

    required = [
        model_dir / "configuration_fleet.py",
        model_dir / "modeling_fleet.py",
        model_dir / "fleet_adapter.py",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Standalone Fleet remote-code files are incomplete: "
            + ", ".join(missing)
        )

    closure = _relative_import_closure(
        model_dir / "modeling_fleet.py",
        model_dir,
    )

    return {
        "ok": True,
        "files": sorted(p.name for p in closure),
    }


@torch.inference_mode()
def validate_native_checkpoint(model_dir: str | Path) -> dict:
    model_dir = Path(model_dir)

    validate_runtime_sources(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto",
    )

    if model.__class__.__name__ != "FleetForCausalLM":
        raise RuntimeError(
            "AutoModelForCausalLM did not instantiate FleetForCausalLM."
        )

    required_methods = ["install", "uninstall", "experts", "export_adapter"]
    missing = [name for name in required_methods if not hasattr(model, name)]
    if missing:
        raise RuntimeError(f"FleetForCausalLM is missing dynamic APIs: {missing}")

    offloaded_fleet_parameters = sum(
        1
        for _, module in model._iter_fleet_linears()
        for tensor in (module.expert_A, module.expert_B)
        if bool(getattr(tensor, "is_meta", False))
    )

    initial = model.experts()
    initial_names = [x["name"] for x in initial]
    initial_semantic_count = int(model.fleet_router.num_experts)
    dynamic_report = {
        "initial_experts": initial_names,
        "initial_semantic_count": initial_semantic_count,
        "offloaded_fleet_parameters": offloaded_fleet_parameters,
        "roundtrip_expert": None,
        "uninstall_ok": None,
        "reinstall_ok": None,
    }

    if initial_names:
        # Use the final expert so re-installation restores the original ordering.
        name = initial_names[-1]
        dynamic_report["roundtrip_expert"] = name
        with tempfile.TemporaryDirectory(prefix="fleet_dynamic_validate_") as td:
            package_path = Path(td) / f"{name}.fleet"
            model.export_adapter(name, package_path)
            if not package_path.exists():
                raise RuntimeError("FleetAdapter export did not produce a package.")

            removed = model.uninstall(name)
            after_remove = model.experts()
            if any(x["name"] == name for x in after_remove):
                raise RuntimeError("uninstall() did not remove the expert registry entry.")
            if int(model.fleet_router.num_experts) != initial_semantic_count - 1:
                raise RuntimeError("Dynamic semantic prototype bank did not shrink.")
            dynamic_report["uninstall_ok"] = True
            dynamic_report["removed_metadata"] = removed

            installed_name = model.install(package_path)
            if installed_name != name:
                raise RuntimeError("install() returned the wrong expert name.")
            after_install = model.experts()
            after_names = [x["name"] for x in after_install]
            if after_names != initial_names:
                raise RuntimeError(
                    f"Expert order changed after uninstall/install: {after_names} != {initial_names}"
                )
            if int(model.fleet_router.num_experts) != initial_semantic_count:
                raise RuntimeError("Dynamic semantic prototype bank did not grow back.")
            dynamic_report["reinstall_ok"] = True
            dynamic_report["final_experts"] = after_names

    messages = [{
        "role": "user",
        "content": (
            "Solve carefully: 8 pens cost $12. At the same rate, "
            "how much do 14 pens cost?"
        ),
    }]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    text = tokenizer.decode(
        out[0, inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    report = {
        "ok": bool(text) and (not initial_names or (
            dynamic_report["uninstall_ok"] and dynamic_report["reinstall_ok"]
        )),
        "architecture": model.__class__.__name__,
        "generated_text": text,
        "standalone_load": True,
        "dynamic_api": dynamic_report,
    }
    (model_dir / "native_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report
