from __future__ import annotations

from collections import defaultdict
import types

import pandas as pd
import torch

from .compiler import semantic_module_id


def _resolve_semantic_id(host_name: str) -> str | None:
    try:
        return semantic_module_id(host_name)
    except Exception:
        return None


def fleet_masked_arrow_forward(
    self,
    x,
    lora_A,
    lora_B,
    dropout,
    scaling,
):
    """
    Arrow-compatible token routing with Fleet source-origin masking.

    The patch filters invalid experts before top-k selection.
    """
    x = self._cast_input_dtype(
        x,
        lora_A[self.task_adapter_names[0]].weight.dtype,
    )

    batch, *rest, in_features = x.shape
    tok = x.view(-1, in_features)

    valid_global = torch.nonzero(
        self.fleet_valid_expert_mask.to(tok.device),
        as_tuple=False,
    ).flatten()

    if valid_global.numel() == 0:
        out_features = lora_B[
            self.task_adapter_names[0]
        ].weight.shape[0]

        return tok.new_zeros(
            batch,
            *rest,
            out_features,
        )

    valid_names = [
        self.task_adapter_names[int(i)]
        for i in valid_global.tolist()
    ]

    valid_prototypes = self.prototypes.index_select(
        0,
        valid_global,
    )

    similarity = torch.abs(
        tok @ valid_prototypes.T
    )

    effective_k = min(
        self.top_k,
        len(valid_names),
    )

    top_values, local_indices = torch.topk(
        similarity,
        effective_k,
        dim=1,
    )

    full_score = tok.new_full(
        (tok.shape[0], len(valid_names)),
        float("-inf"),
    )
    full_score.scatter_(
        1,
        local_indices,
        top_values,
    )

    coefficients = torch.softmax(
        full_score / self.temperature,
        dim=1,
    )

    scales = torch.tensor(
        [scaling[name] for name in valid_names],
        device=tok.device,
        dtype=tok.dtype,
    )

    A_stack = torch.stack(
        [lora_A[name].weight for name in valid_names],
        dim=0,
    )
    B_stack = torch.stack(
        [lora_B[name].weight for name in valid_names],
        dim=0,
    )

    low = torch.einsum(
        "tf,erf->ter",
        tok,
        A_stack,
    )

    projected = torch.einsum(
        "ter,eor->teo",
        low,
        B_stack,
    )

    projected = projected * scales.view(
        1,
        -1,
        1,
    )

    delta_flat = torch.einsum(
        "te,teo->to",
        coefficients,
        projected,
    )

    chosen_global = valid_global[
        local_indices
    ]

    counts = torch.bincount(
        chosen_global.flatten(),
        minlength=len(self.task_adapter_names),
    ).detach().cpu()

    self.fleet_usage_counts += counts
    self.fleet_tokens_seen += int(tok.shape[0])

    # Canonical Fleet adapters are compiled with lora_dropout=0.
    delta = dropout(delta_flat)

    return delta.view(
        batch,
        *rest,
        delta_flat.size(-1),
    )


def install_masked_router(
    model,
    *,
    expert_order: list[str],
    origin_masks: dict[str, dict[str, bool]],
) -> dict:
    patched = 0
    unresolved: list[str] = []
    eligibility = {
        expert: 0
        for expert in expert_order
    }

    for host_name, module in model.named_modules():
        if not hasattr(module, "lora_arrow"):
            continue

        if "arrow_router" not in module.lora_arrow:
            continue

        arrow = module.lora_arrow["arrow_router"]
        semantic_id = _resolve_semantic_id(host_name)

        if semantic_id is None:
            unresolved.append(host_name)
            continue

        valid = torch.tensor(
            [
                bool(
                    origin_masks[expert].get(
                        semantic_id,
                        False,
                    )
                )
                for expert in expert_order
            ],
            dtype=torch.bool,
        )

        if not valid.any():
            unresolved.append(
                f"{host_name}:EMPTY_MASK"
            )
            continue

        arrow.fleet_valid_expert_mask = valid
        arrow.fleet_schema_key = semantic_id
        arrow.fleet_usage_counts = torch.zeros(
            len(expert_order),
            dtype=torch.long,
        )
        arrow.fleet_tokens_seen = 0

        if not hasattr(
            arrow,
            "_fleet_original_forward",
        ):
            arrow._fleet_original_forward = (
                arrow.forward
            )

        arrow.forward = types.MethodType(
            fleet_masked_arrow_forward,
            arrow,
        )

        for i, expert in enumerate(expert_order):
            eligibility[expert] += int(
                bool(valid[i])
            )

        patched += 1

    if patched == 0:
        raise RuntimeError(
            "No Arrow layers were patched. "
            "PEFT internals may have changed."
        )

    dead = [
        expert
        for expert, count in eligibility.items()
        if count == 0
    ]

    if dead:
        raise RuntimeError(
            "Experts have zero runtime eligibility: "
            + ", ".join(dead)
        )

    return {
        "patched_layers": patched,
        "unresolved": unresolved,
        "eligibility": eligibility,
    }


def reset_route_stats(model):
    for _, module in model.named_modules():
        if not hasattr(module, "lora_arrow"):
            continue
        if "arrow_router" not in module.lora_arrow:
            continue

        arrow = module.lora_arrow["arrow_router"]

        if hasattr(
            arrow,
            "fleet_usage_counts",
        ):
            arrow.fleet_usage_counts.zero_()
            arrow.fleet_tokens_seen = 0


def collect_route_stats(
    model,
    expert_order: list[str],
):
    totals = torch.zeros(
        len(expert_order),
        dtype=torch.long,
    )

    by_target = defaultdict(
        lambda: torch.zeros(
            len(expert_order),
            dtype=torch.long,
        )
    )

    for _, module in model.named_modules():
        if not hasattr(module, "lora_arrow"):
            continue
        if "arrow_router" not in module.lora_arrow:
            continue

        arrow = module.lora_arrow["arrow_router"]

        if not hasattr(
            arrow,
            "fleet_usage_counts",
        ):
            continue

        counts = arrow.fleet_usage_counts
        totals += counts

        target = arrow.fleet_schema_key.split(".")[-1]
        by_target[target] += counts

    total = int(totals.sum())

    overall = [
        {
            "expert": expert,
            "selections": int(totals[i]),
            "share": (
                int(totals[i]) / total
                if total
                else 0.0
            ),
        }
        for i, expert in enumerate(expert_order)
    ]

    targets = []

    for target, counts in sorted(by_target.items()):
        subtotal = int(counts.sum())

        for i, expert in enumerate(expert_order):
            targets.append({
                "target": target,
                "expert": expert,
                "selections": int(counts[i]),
                "share": (
                    int(counts[i]) / subtotal
                    if subtotal
                    else 0.0
                ),
            })

    return {
        "overall": overall,
        "by_target": targets,
    }
