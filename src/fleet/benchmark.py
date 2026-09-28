from __future__ import annotations

from pathlib import Path
import json

from .semantic_runtime import FleetSemanticRuntime


PROBES = {
    "math_reasoning": {
        "prompt": (
            "Solve carefully: A store sells 8 pens for $12. "
            "At the same rate, how much do 14 pens cost?"
        ),
        "max_new_tokens": 120,
    },
    "medical_reasoning": {
        "prompt": (
            "For education only, explain clearly what insulin does in the body."
        ),
        "max_new_tokens": 120,
    },
    "legal_ops": {
        "prompt": (
            'Extract JSON with the key "governing_law" from: '
            '"This Agreement is governed by the laws of California."'
        ),
        "max_new_tokens": 90,
    },
    "intent": {
        "prompt": (
            'Classify the user intent concisely: '
            '"Can you move my meeting from Monday to Wednesday?"'
        ),
        "max_new_tokens": 60,
    },
    "agent": {
        "prompt": (
            "Plan a concrete sequence of steps to debug a Python API test "
            "that passes locally but fails only in CI."
        ),
        "max_new_tokens": 130,
    },
    "function_calling": {
        "prompt": (
            'Available function: weather_lookup(city: str). '
            'Produce the function/tool call for: "What is the weather in Cebu?"'
        ),
        "max_new_tokens": 80,
    },
    "structured": {
        "prompt": (
            'Return JSON only with keys "task", "steps", and "risk" for: '
            "inspect a failing web scraper."
        ),
        "max_new_tokens": 100,
    },
    "summary": {
        "prompt": (
            "Summarize in two sentences: A research team built a routing "
            "system that combines specialist adapters while keeping a shared "
            "language-model backbone. The system decides when a specialist "
            "should influence generation and can fall back to the base model."
        ),
        "max_new_tokens": 100,
    },
}

NULL_PROBE = {
    "prompt": "What is the capital of Japan?",
    "max_new_tokens": 40,
}


def _prompt_gate_counts(routing: dict) -> tuple[int, int, int]:
    gate = routing["gating"]["prompt"]
    return (
        int(gate["null"]),
        int(gate["k1"]),
        int(gate["k2"]),
    )


def _top_prompt_route(routing: dict) -> dict | None:
    rows = sorted(
        routing["experts"],
        key=lambda row: (
            float(row["prompt_selection_rate"]),
            int(row["prompt_selected"]),
        ),
        reverse=True,
    )
    return rows[0] if rows else None


def run_benchmark(
    bank_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    max_failures: int = 0,
) -> dict:
    bank_dir = Path(bank_dir)
    bank = json.loads(
        (bank_dir / "fleet_bank.json").read_text(encoding="utf-8")
    )
    active = [x["name"] for x in bank["experts"]]

    runtime = FleetSemanticRuntime(bank_dir)

    cases = []
    failures = []

    for expert in active:
        probe = PROBES.get(expert)
        if probe is None:
            failures.append(
                f"No benchmark probe is defined for active expert '{expert}'."
            )
            continue

        result = runtime.generate(
            probe["prompt"],
            max_new_tokens=int(probe["max_new_tokens"]),
        )
        routing = result["routing"]
        top = _top_prompt_route(routing)
        semantic = routing["semantic_request"]

        top_name = top["expert"] if top else None
        top_rate = float(top["prompt_selection_rate"]) if top else 0.0
        semantic_votes = semantic.get("top_expert_votes", {})
        semantic_top = (
            max(semantic_votes, key=semantic_votes.get)
            if semantic_votes
            else None
        )

        # Route gate: the expected expert must win either the normalized
        # prompt-selection metric or the semantic layer vote, and at least
        # some layers must accept specialist routing.
        accepted = int(semantic.get("accepted_layers", 0))
        route_ok = (
            (top_name == expert or semantic_top == expert)
            and accepted > 0
        )

        case = {
            "expected": expert,
            "prompt": probe["prompt"],
            "output": result["text"],
            "top_prompt_route": top,
            "semantic_top": semantic_top,
            "semantic_request": semantic,
            "gating": routing["gating"],
            "route_ok": route_ok,
        }
        cases.append(case)

        if not route_ok:
            failures.append(
                f"{expert}: top_route={top_name}, semantic_top={semantic_top}, "
                f"accepted_layers={accepted}, top_rate={top_rate:.4f}"
            )

    # Explicit NULL/base test.
    null_result = runtime.generate(
        NULL_PROBE["prompt"],
        max_new_tokens=int(NULL_PROBE["max_new_tokens"]),
    )
    n0, n1, n2 = _prompt_gate_counts(null_result["routing"])
    denom = max(1, n0 + n1 + n2)
    null_rate = n0 / denom
    null_ok = null_rate >= 0.50

    null_case = {
        "expected": None,
        "prompt": NULL_PROBE["prompt"],
        "output": null_result["text"],
        "gating": null_result["routing"]["gating"],
        "semantic_request": null_result["routing"]["semantic_request"],
        "prompt_null_rate": null_rate,
        "route_ok": null_ok,
    }
    cases.append(null_case)

    if not null_ok:
        failures.append(
            f"NULL probe: prompt null rate {null_rate:.3f} < 0.50"
        )

    report = {
        "format": "fleet-routing-benchmark",
        "version": "0.7.0",
        "bank_version": bank.get("version"),
        "active_experts": active,
        "passed": len(failures) <= int(max_failures),
        "failure_count": len(failures),
        "failures": failures,
        "cases": cases,
    }

    if output_path is not None:
        Path(output_path).write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return report
