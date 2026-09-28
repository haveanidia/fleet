from __future__ import annotations

from collections import defaultdict
import types

import torch

from .compiler import semantic_module_id
from .semantic import (
    SemanticState,
    dynamic_topk_coefficients,
    relative_weight_zscore,
)


def _semantic_parts(
    host_name: str,
):
    try:
        semantic_id = (
            semantic_module_id(
                host_name
            )
        )
    except Exception:
        return (
            None,
            None,
            None,
        )

    parts = (
        semantic_id.split(".")
    )

    layer_id = int(
        parts[1]
    )

    target = parts[-1]

    return (
        semantic_id,
        layer_id,
        target,
    )


def fleet_semantic_forward(
    self,
    x,
    lora_A,
    lora_B,
    dropout,
    scaling,
):
    x = self._cast_input_dtype(
        x,
        lora_A[
            self.task_adapter_names[
                0
            ]
        ].weight.dtype,
    )

    if (
        x.ndim != 3
        or x.shape[0] != 1
    ):
        raise RuntimeError(
            "Fleet Semantic v0.9 "
            "currently supports batch size 1."
        )

    batch, seq_len, in_features = (
        x.shape
    )

    tok = x.view(
        -1,
        in_features,
    )

    valid_global = torch.nonzero(
        self.fleet_valid_expert_mask.to(
            tok.device
        ),
        as_tuple=False,
    ).flatten()

    if (
        valid_global.numel()
        == 0
    ):
        out_features = (
            lora_B[
                self.task_adapter_names[
                    0
                ]
            ].weight.shape[0]
        )

        return tok.new_zeros(
            batch,
            seq_len,
            out_features,
        )

    valid_names = [
        self.task_adapter_names[
            int(i)
        ]
        for i
        in valid_global.tolist()
    ]

    # q_proj establishes semantic state for the transformer layer.
    q_input = (
        x
        if self.fleet_target
        == "q_proj"
        else None
    )

    (
        semantic_full,
        null_threshold,
    ) = (
        self.fleet_semantic_state
        .update_or_get_prior(
            layer_id=(
                self.fleet_layer_id
            ),
            q_input=q_input,
            sequence_length=(
                seq_len
            ),
        )
    )

    if semantic_full is None:
        semantic_full = (
            torch.zeros(
                len(
                    self
                    .fleet_semantic_state
                    .expert_order
                ),
                device=tok.device,
                dtype=torch.float32,
            )
        )

    if null_threshold is None:
        raise RuntimeError(
            "Semantic NULL threshold unavailable "
            f"for layer {self.fleet_layer_id}."
        )

    semantic_valid = (
        semantic_full.index_select(
            0,
            valid_global.to(
                semantic_full.device
            ),
        ).float()
    )

    semantic_score = (
        semantic_valid
        .view(
            1,
            -1,
        )
        .expand(
            tok.shape[0],
            -1,
        )
    )

    # Secondary LoRA geometry signal.
    weight_prototypes = (
        self.prototypes
        .index_select(
            0,
            valid_global,
        )
    )

    weight_score = (
        relative_weight_zscore(
            tok,
            weight_prototypes,
        )
    )

    policy = (
        self.fleet_semantic_state
        .policy
    )

    combined = (
        policy.semantic_weight
        * semantic_score
        + policy.weight_weight
        * weight_score
    )

    (
        coeff,
        gates,
    ) = (
        dynamic_topk_coefficients(
            rank_scores=combined,
            semantic_gate_scores=(
                semantic_score
            ),
            null_threshold=(
                float(
                    null_threshold
                )
            ),
            single_margin=(
                policy
                .single_margin
            ),
            temperature=(
                policy
                .router_temperature
            ),
        )
    )

    scales = torch.tensor(
        [
            scaling[
                name
            ]
            for name
            in valid_names
        ],
        device=tok.device,
        dtype=tok.dtype,
    )

    A_stack = torch.stack(
        [
            lora_A[
                name
            ].weight
            for name
            in valid_names
        ],
        dim=0,
    )

    B_stack = torch.stack(
        [
            lora_B[
                name
            ].weight
            for name
            in valid_names
        ],
        dim=0,
    )

    z = torch.einsum(
        "tf,erf->ter",
        tok,
        A_stack,
    )

    y = torch.einsum(
        "ter,eor->teo",
        z,
        B_stack,
    )

    y = (
        y
        * scales.view(
            1,
            -1,
            1,
        )
    )

    delta_flat = torch.einsum(
        "te,teo->to",
        coeff.to(
            y.dtype
        ),
        y,
    )

    # --------------------------------------------------------------
    # Corrected telemetry: selected / eligible.
    # --------------------------------------------------------------
    is_prompt = (
        seq_len > 1
    )

    eligible_increment = (
        torch.zeros(
            len(
                self
                .task_adapter_names
            ),
            dtype=torch.long,
        )
    )

    eligible_increment[
        valid_global.cpu()
    ] = tok.shape[0]

    selected_local = (
        coeff.detach()
        .gt(0)
        .to(
            torch.long
        )
        .sum(
            dim=0
        )
        .cpu()
    )

    selected_increment = (
        torch.zeros(
            len(
                self
                .task_adapter_names
            ),
            dtype=torch.long,
        )
    )

    selected_increment[
        valid_global.cpu()
    ] = selected_local

    if is_prompt:
        self.fleet_prompt_eligible += (
            eligible_increment
        )

        self.fleet_prompt_selected += (
            selected_increment
        )

        self.fleet_prompt_null += int(
            gates[
                "k0"
            ].sum().item()
        )

        self.fleet_prompt_k1 += int(
            gates[
                "k1"
            ].sum().item()
        )

        self.fleet_prompt_k2 += int(
            gates[
                "k2"
            ].sum().item()
        )
    else:
        self.fleet_decode_eligible += (
            eligible_increment
        )

        self.fleet_decode_selected += (
            selected_increment
        )

        self.fleet_decode_null += int(
            gates[
                "k0"
            ].sum().item()
        )

        self.fleet_decode_k1 += int(
            gates[
                "k1"
            ].sum().item()
        )

        self.fleet_decode_k2 += int(
            gates[
                "k2"
            ].sum().item()
        )

    delta = dropout(
        delta_flat
    )

    return delta.view(
        batch,
        seq_len,
        delta_flat.size(-1),
    )


def install_semantic_router(
    model,
    *,
    expert_order: list[str],
    origin_masks: dict[
        str,
        dict[
            str,
            bool,
        ],
    ],
    semantic_state: SemanticState,
) -> dict:
    patched = 0
    unresolved = []

    eligibility = {
        expert: 0
        for expert
        in expert_order
    }

    for host_name, module in (
        model.named_modules()
    ):
        if not hasattr(
            module,
            "lora_arrow",
        ):
            continue

        if (
            "arrow_router"
            not in module.lora_arrow
        ):
            continue

        arrow = (
            module
            .lora_arrow[
                "arrow_router"
            ]
        )

        (
            semantic_id,
            layer_id,
            target,
        ) = _semantic_parts(
            host_name
        )

        if semantic_id is None:
            unresolved.append(
                host_name
            )
            continue

        valid = torch.tensor(
            [
                bool(
                    origin_masks[
                        expert
                    ].get(
                        semantic_id,
                        False,
                    )
                )
                for expert
                in expert_order
            ],
            dtype=torch.bool,
        )

        if not valid.any():
            unresolved.append(
                host_name
                + ":EMPTY_MASK"
            )
            continue

        arrow.fleet_valid_expert_mask = (
            valid
        )

        arrow.fleet_schema_key = (
            semantic_id
        )

        arrow.fleet_layer_id = (
            layer_id
        )

        arrow.fleet_target = (
            target
        )

        arrow.fleet_semantic_state = (
            semantic_state
        )

        n = len(
            expert_order
        )

        for attr in [
            "fleet_prompt_eligible",
            "fleet_prompt_selected",
            "fleet_decode_eligible",
            "fleet_decode_selected",
        ]:
            setattr(
                arrow,
                attr,
                torch.zeros(
                    n,
                    dtype=torch.long,
                ),
            )

        for attr in [
            "fleet_prompt_null",
            "fleet_prompt_k1",
            "fleet_prompt_k2",
            "fleet_decode_null",
            "fleet_decode_k1",
            "fleet_decode_k2",
        ]:
            setattr(
                arrow,
                attr,
                0,
            )

        if not hasattr(
            arrow,
            "_fleet_original_forward",
        ):
            arrow._fleet_original_forward = (
                arrow.forward
            )

        arrow.forward = (
            types.MethodType(
                fleet_semantic_forward,
                arrow,
            )
        )

        for i, expert in (
            enumerate(
                expert_order
            )
        ):
            eligibility[
                expert
            ] += int(
                bool(
                    valid[i]
                )
            )

        patched += 1

    if patched == 0:
        raise RuntimeError(
            "No Arrow layers patched."
        )

    dead = [
        expert
        for expert, count
        in eligibility.items()
        if count == 0
    ]

    if dead:
        raise RuntimeError(
            "Zero-eligibility experts: "
            + ", ".join(
                dead
            )
        )

    return {
        "router": (
            "fleet-semantic-v0.9"
        ),
        "patched_layers": (
            patched
        ),
        "unresolved": (
            unresolved
        ),
        "eligibility_layers": (
            eligibility
        ),
    }


def reset_semantic_stats(
    model,
    semantic_state: SemanticState,
):
    semantic_state.reset_request()

    for _, module in (
        model.named_modules()
    ):
        if not hasattr(
            module,
            "lora_arrow",
        ):
            continue

        if (
            "arrow_router"
            not in module.lora_arrow
        ):
            continue

        arrow = (
            module
            .lora_arrow[
                "arrow_router"
            ]
        )

        for attr in [
            "fleet_prompt_eligible",
            "fleet_prompt_selected",
            "fleet_decode_eligible",
            "fleet_decode_selected",
        ]:
            tensor = getattr(
                arrow,
                attr,
                None,
            )

            if tensor is not None:
                tensor.zero_()

        for attr in [
            "fleet_prompt_null",
            "fleet_prompt_k1",
            "fleet_prompt_k2",
            "fleet_decode_null",
            "fleet_decode_k1",
            "fleet_decode_k2",
        ]:
            if hasattr(
                arrow,
                attr,
            ):
                setattr(
                    arrow,
                    attr,
                    0,
                )


def collect_semantic_stats(
    model,
    expert_order: list[str],
    semantic_state: SemanticState,
) -> dict:
    n = len(
        expert_order
    )

    prompt_eligible = (
        torch.zeros(
            n,
            dtype=torch.long,
        )
    )

    prompt_selected = (
        torch.zeros(
            n,
            dtype=torch.long,
        )
    )

    decode_eligible = (
        torch.zeros(
            n,
            dtype=torch.long,
        )
    )

    decode_selected = (
        torch.zeros(
            n,
            dtype=torch.long,
        )
    )

    prompt_null = 0
    prompt_k1 = 0
    prompt_k2 = 0
    decode_null = 0
    decode_k1 = 0
    decode_k2 = 0

    by_target = defaultdict(
        lambda: {
            "eligible": torch.zeros(
                n,
                dtype=torch.long,
            ),
            "selected": torch.zeros(
                n,
                dtype=torch.long,
            ),
        }
    )

    for _, module in (
        model.named_modules()
    ):
        if not hasattr(
            module,
            "lora_arrow",
        ):
            continue

        if (
            "arrow_router"
            not in module.lora_arrow
        ):
            continue

        arrow = (
            module
            .lora_arrow[
                "arrow_router"
            ]
        )

        if not hasattr(
            arrow,
            "fleet_prompt_eligible",
        ):
            continue

        prompt_eligible += (
            arrow
            .fleet_prompt_eligible
        )

        prompt_selected += (
            arrow
            .fleet_prompt_selected
        )

        decode_eligible += (
            arrow
            .fleet_decode_eligible
        )

        decode_selected += (
            arrow
            .fleet_decode_selected
        )

        prompt_null += (
            arrow
            .fleet_prompt_null
        )

        prompt_k1 += (
            arrow
            .fleet_prompt_k1
        )

        prompt_k2 += (
            arrow
            .fleet_prompt_k2
        )

        decode_null += (
            arrow
            .fleet_decode_null
        )

        decode_k1 += (
            arrow
            .fleet_decode_k1
        )

        decode_k2 += (
            arrow
            .fleet_decode_k2
        )

        target = (
            arrow
            .fleet_target
        )

        by_target[
            target
        ]["eligible"] += (
            arrow
            .fleet_prompt_eligible
            + arrow
            .fleet_decode_eligible
        )

        by_target[
            target
        ]["selected"] += (
            arrow
            .fleet_prompt_selected
            + arrow
            .fleet_decode_selected
        )

    total_eligible = (
        prompt_eligible
        + decode_eligible
    )

    total_selected = (
        prompt_selected
        + decode_selected
    )

    expert_rows = []

    for i, expert in (
        enumerate(
            expert_order
        )
    ):
        eligible = int(
            total_eligible[i]
        )

        selected = int(
            total_selected[i]
        )

        p_eligible = int(
            prompt_eligible[i]
        )

        p_selected = int(
            prompt_selected[i]
        )

        d_eligible = int(
            decode_eligible[i]
        )

        d_selected = int(
            decode_selected[i]
        )

        expert_rows.append({
            "expert": expert,
            "selected": selected,
            "eligible": eligible,
            "selection_rate": (
                selected
                / eligible
                if eligible
                else 0.0
            ),
            "prompt_selected": (
                p_selected
            ),
            "prompt_eligible": (
                p_eligible
            ),
            "prompt_selection_rate": (
                p_selected
                / p_eligible
                if p_eligible
                else 0.0
            ),
            "decode_selected": (
                d_selected
            ),
            "decode_eligible": (
                d_eligible
            ),
            "decode_selection_rate": (
                d_selected
                / d_eligible
                if d_eligible
                else 0.0
            ),
        })

    target_rows = []

    for target, data in sorted(
        by_target.items()
    ):
        for i, expert in (
            enumerate(
                expert_order
            )
        ):
            eligible = int(
                data[
                    "eligible"
                ][i]
            )

            selected = int(
                data[
                    "selected"
                ][i]
            )

            target_rows.append({
                "target": target,
                "expert": expert,
                "selected": selected,
                "eligible": eligible,
                "selection_rate": (
                    selected
                    / eligible
                    if eligible
                    else 0.0
                ),
            })

    return {
        "experts": (
            expert_rows
        ),
        "by_target": (
            target_rows
        ),
        "gating": {
            "prompt": {
                "null": (
                    prompt_null
                ),
                "k1": (
                    prompt_k1
                ),
                "k2": (
                    prompt_k2
                ),
            },
            "decode": {
                "null": (
                    decode_null
                ),
                "k1": (
                    decode_k1
                ),
                "k2": (
                    decode_k2
                ),
            },
        },
        "semantic_request": (
            semantic_state
            .request_summary()
        ),
    }
