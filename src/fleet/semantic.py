from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .compiler import semantic_module_id


@dataclass(slots=True)
class SemanticPolicy:
    anchor_token_window: int = 32
    semantic_weight: float = 0.90
    weight_weight: float = 0.10
    prompt_prior_weight: float = 0.95
    single_margin: float = 0.18
    router_temperature: float = 0.70
    score_clip: float = 3.0
    minimum_calibration_gap: float = 0.02


def load_semantic_config(
    path: str | Path,
) -> tuple[SemanticPolicy, dict[str, list[str]], list[str]]:
    raw = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    p = raw.get("policy", {})

    policy = SemanticPolicy(
        anchor_token_window=int(
            p.get("anchor_token_window", 32)
        ),
        semantic_weight=float(
            p.get("semantic_weight", 0.90)
        ),
        weight_weight=float(
            p.get("weight_weight", 0.10)
        ),
        prompt_prior_weight=float(
            p.get("prompt_prior_weight", 0.95)
        ),
        single_margin=float(
            p.get("single_margin", 0.18)
        ),
        router_temperature=float(
            p.get("router_temperature", 0.70)
        ),
        score_clip=float(
            p.get("score_clip", 3.0)
        ),
        minimum_calibration_gap=float(
            p.get("minimum_calibration_gap", 0.02)
        ),
    )

    anchors = {
        str(name): [str(x) for x in values]
        for name, values
        in raw.get("anchors", {}).items()
    }

    null_anchors = [
        str(x)
        for x in raw.get("null_anchors", [])
    ]

    if not null_anchors:
        raise ValueError(
            "semantic_anchors.json must define null_anchors."
        )

    return (
        policy,
        anchors,
        null_anchors,
    )


def pooled_direction(
    x: torch.Tensor,
    window: int = 32,
) -> torch.Tensor:
    """
    One normalized request direction from a [1, seq, hidden] module input.

    We average normalized token directions, then normalize the aggregate.
    """
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(
            "Fleet Semantic currently expects [1, seq, hidden]."
        )

    n = min(
        int(window),
        int(x.shape[1]),
    )

    token_vectors = (
        x[0, -n:]
        .float()
    )

    token_vectors = F.normalize(
        token_vectors,
        dim=-1,
    )

    pooled = token_vectors.mean(
        dim=0
    )

    return F.normalize(
        pooled,
        dim=-1,
    )


def centered_direction(
    vector: torch.Tensor,
    center: torch.Tensor,
) -> torch.Tensor:
    """
    Remove the common chat/prompt direction shared by all anchor classes.

    This is the key v0.6.1 change: generic instruction/template geometry no
    longer dominates semantic class differences.
    """
    return F.normalize(
        vector.float()
        - center.float(),
        dim=-1,
    )


def semantic_margin_scores(
    query: torch.Tensor,
    *,
    center: torch.Tensor,
    expert_prototypes: torch.Tensor,
    null_prototype: torch.Tensor,
    score_clip: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      expert_margin: [experts], expert cosine minus NULL/base cosine
      null_similarity: scalar

    Positive margin means "this looks more like this specialist than NULL".
    """
    q = centered_direction(
        F.normalize(
            query.float(),
            dim=-1,
        ),
        center,
    )

    experts = F.normalize(
        expert_prototypes.float(),
        dim=-1,
    )

    null = F.normalize(
        null_prototype.float(),
        dim=-1,
    )

    expert_similarity = (
        q
        @ experts.T
    )

    null_similarity = (
        q
        @ null
    )

    margin = (
        expert_similarity
        - null_similarity
    )

    return (
        margin.clamp(
            min=-float(score_clip),
            max=float(score_clip),
        ),
        null_similarity,
    )


def relative_weight_zscore(
    token_vectors: torch.Tensor,
    weight_prototypes: torch.Tensor,
) -> torch.Tensor:
    """
    Arrow-like LoRA-geometry score, cosine-normalized and standardized
    across currently valid experts per token.
    """
    x = F.normalize(
        token_vectors.float(),
        dim=-1,
    )

    p = F.normalize(
        weight_prototypes.float(),
        dim=-1,
    )

    score = torch.abs(
        x
        @ p.T
    )

    if score.shape[1] <= 1:
        return torch.zeros_like(
            score
        )

    mean = score.mean(
        dim=1,
        keepdim=True,
    )

    std = score.std(
        dim=1,
        keepdim=True,
        unbiased=False,
    ).clamp_min(0.05)

    return (
        score - mean
    ) / std


def dynamic_topk_coefficients(
    rank_scores: torch.Tensor,
    semantic_gate_scores: torch.Tensor,
    *,
    null_threshold: float,
    single_margin: float,
    temperature: float,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    """
    rank_scores:
      [tokens, valid_experts]
      used to choose/mix specialists.

    semantic_gate_scores:
      [tokens, valid_experts]
      specialist-vs-NULL semantic margins.

    k=0 is decided ONLY by the semantic-vs-NULL competition. LoRA weight
    geometry is not allowed to force an adapter onto a generic request.
    """
    tokens, experts = (
        rank_scores.shape
    )

    coeff = torch.zeros_like(
        rank_scores
    )

    if experts == 0:
        null = torch.ones(
            tokens,
            dtype=torch.bool,
            device=rank_scores.device,
        )
        off = torch.zeros_like(
            null
        )
        return coeff, {
            "k0": null,
            "k1": off,
            "k2": off,
        }

    semantic_best = (
        semantic_gate_scores
        .max(dim=1)
        .values
    )

    k0 = (
        semantic_best
        < float(null_threshold)
    )

    if experts == 1:
        k1 = ~k0
        k2 = torch.zeros_like(
            k0
        )

        coeff[k1, 0] = 1.0

        return coeff, {
            "k0": k0,
            "k1": k1,
            "k2": k2,
        }

    top_values, top_indices = (
        torch.topk(
            rank_scores,
            k=2,
            dim=1,
        )
    )

    rank_margin = (
        top_values[:, 0]
        - top_values[:, 1]
    )

    k1 = (
        (~k0)
        & (
            rank_margin
            >= float(
                single_margin
            )
        )
    )

    k2 = (
        (~k0)
        & (~k1)
    )

    rows = torch.arange(
        tokens,
        device=rank_scores.device,
    )

    if k1.any():
        coeff[
            rows[k1],
            top_indices[k1, 0],
        ] = 1.0

    if k2.any():
        pair = torch.softmax(
            top_values[k2]
            / float(
                temperature
            ),
            dim=1,
        )

        coeff[
            rows[k2].unsqueeze(1),
            top_indices[k2],
        ] = pair

    return coeff, {
        "k0": k0,
        "k1": k1,
        "k2": k2,
    }


def _layer_id_from_qproj_name(
    module_name: str,
) -> int | None:
    try:
        semantic = (
            semantic_module_id(
                module_name
            )
        )
    except Exception:
        return None

    if not semantic.endswith(
        ".self_attn.q_proj"
    ):
        return None

    return int(
        semantic.split(".")[1]
    )


def _quantile(
    values: torch.Tensor,
    q: float,
) -> float:
    if values.numel() == 0:
        return 0.0

    return float(
        torch.quantile(
            values.float(),
            torch.tensor(
                q,
                dtype=torch.float32,
            ),
        ).item()
    )


@torch.inference_mode()
def build_semantic_prototypes(
    *,
    model,
    tokenizer,
    expert_order: list[str],
    anchors: dict[str, list[str]],
    null_anchors: list[str],
    output_dir: str | Path,
    policy: SemanticPolicy,
) -> dict:
    """
    Build per-layer semantic classes from the frozen base model.

    v0.6.1 calibration:
      1. capture anchor request directions,
      2. remove the global/common prompt direction,
      3. construct specialist and explicit NULL prototypes,
      4. learn a per-layer NULL threshold from held-in anchor statistics.

    Still zero gradients and zero parameter updates.
    """
    output_dir = Path(
        output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    missing = [
        expert
        for expert in expert_order
        if not anchors.get(
            expert
        )
    ]

    if missing:
        raise ValueError(
            "Missing semantic anchors for active experts: "
            + ", ".join(
                missing
            )
        )

    captured = {}
    hooks = []

    def make_hook(
        layer_id: int,
    ):
        def hook(
            _module,
            inputs,
        ):
            captured[
                layer_id
            ] = pooled_direction(
                inputs[0].detach(),
                window=(
                    policy
                    .anchor_token_window
                ),
            ).cpu()

        return hook

    layer_ids = []

    for name, module in (
        model.named_modules()
    ):
        layer_id = (
            _layer_id_from_qproj_name(
                name
            )
        )

        if layer_id is None:
            continue

        layer_ids.append(
            layer_id
        )

        hooks.append(
            module.register_forward_pre_hook(
                make_hook(
                    layer_id
                )
            )
        )

    layer_ids = sorted(
        set(layer_ids)
    )

    if not layer_ids:
        raise RuntimeError(
            "Could not find Qwen self_attn.q_proj modules."
        )

    class_names = (
        list(expert_order)
        + ["__null__"]
    )

    all_anchors = {
        **{
            expert: anchors[
                expert
            ]
            for expert in expert_order
        },
        "__null__": list(
            null_anchors
        ),
    }

    vectors = {
        name: {
            layer: []
            for layer
            in layer_ids
        }
        for name
        in class_names
    }

    embedding_device = (
        model
        .get_input_embeddings()
        .weight
        .device
    )

    try:
        for class_name in (
            class_names
        ):
            prompts = (
                all_anchors[
                    class_name
                ]
            )

            print(
                f"[semantic] "
                f"{class_name}: "
                f"{len(prompts)} anchors"
            )

            for prompt in prompts:
                captured.clear()

                inputs = (
                    tokenizer
                    .apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                )

                inputs = {
                    k: v.to(
                        embedding_device
                    )
                    for k, v
                    in inputs.items()
                }

                model(
                    **inputs,
                    use_cache=False,
                )

                absent = [
                    layer
                    for layer
                    in layer_ids
                    if layer
                    not in captured
                ]

                if absent:
                    raise RuntimeError(
                        "Missing captured "
                        "q_proj inputs for "
                        f"layers: "
                        f"{absent[:8]}"
                    )

                for layer in (
                    layer_ids
                ):
                    vectors[
                        class_name
                    ][layer].append(
                        captured[
                            layer
                        ]
                    )

    finally:
        for hook in hooks:
            hook.remove()

    tensors = {}
    layer_calibration = {}

    for layer in layer_ids:
        raw_by_class = {}

        for class_name in (
            class_names
        ):
            raw_by_class[
                class_name
            ] = torch.stack(
                vectors[
                    class_name
                ][layer],
                dim=0,
            ).float()

        # v0.9 dynamic ABI: the semantic coordinate system must not depend on
        # which experts happen to be installed. Build the common center only
        # from the fixed NULL/base anchor set, never from specialist anchors.
        # A FleetAdapter built elsewhere therefore lives in the same semantic
        # coordinate system as every other adapter for this base model/anchor set.
        global_center = raw_by_class["__null__"].mean(dim=0)

        centered_by_class = {}

        for class_name in (
            class_names
        ):
            rows = (
                raw_by_class[
                    class_name
                ]
            )

            centered_by_class[
                class_name
            ] = F.normalize(
                rows
                - global_center
                .view(
                    1,
                    -1,
                ),
                dim=-1,
            )

        expert_prototypes = []

        for expert in (
            expert_order
        ):
            proto = (
                centered_by_class[
                    expert
                ]
                .mean(
                    dim=0
                )
            )

            expert_prototypes.append(
                F.normalize(
                    proto,
                    dim=-1,
                )
            )

        expert_prototypes = (
            torch.stack(
                expert_prototypes,
                dim=0,
            )
        )

        null_prototype = (
            centered_by_class[
                "__null__"
            ]
            .mean(
                dim=0
            )
        )

        null_prototype = (
            F.normalize(
                null_prototype,
                dim=-1,
            )
        )

        # ------------------------------------------------------------
        # Learn a NULL gate threshold for this layer.
        #
        # Positive examples:
        #   correct specialist margin over NULL for specialist anchors.
        #
        # Negative examples:
        #   strongest specialist margin over NULL for generic NULL anchors.
        # ------------------------------------------------------------
        positive_margins = []

        for expert_index, expert in (
            enumerate(
                expert_order
            )
        ):
            rows = (
                centered_by_class[
                    expert
                ]
            )

            expert_sim = (
                rows
                @ expert_prototypes[
                    expert_index
                ]
            )

            null_sim = (
                rows
                @ null_prototype
            )

            positive_margins.append(
                expert_sim
                - null_sim
            )

        positive_margins = (
            torch.cat(
                positive_margins,
                dim=0,
            )
        )

        null_rows = (
            centered_by_class[
                "__null__"
            ]
        )

        null_expert_sim = (
            null_rows
            @ expert_prototypes.T
        )

        null_self_sim = (
            null_rows
            @ null_prototype
        )

        negative_margins = (
            null_expert_sim.max(
                dim=1
            ).values
            - null_self_sim
        )

        # v0.9: calibrate each expert independently so semantic columns can be
        # installed/uninstalled without recalibrating the entire bank. The old
        # bank-wide threshold is still emitted below for v0.6-v0.8 compatibility.
        per_expert_thresholds = []
        per_expert_calibration = {}
        for expert_index, expert in enumerate(expert_order):
            expert_rows = centered_by_class[expert]
            pos = (
                expert_rows @ expert_prototypes[expert_index]
                - expert_rows @ null_prototype
            )
            neg = (
                null_expert_sim[:, expert_index]
                - null_self_sim
            )
            p20 = _quantile(pos, 0.20)
            n80 = _quantile(neg, 0.80)
            pmean = float(pos.mean().item())
            nmean = float(neg.mean().item())
            if p20 > n80 + policy.minimum_calibration_gap:
                et = (p20 + n80) / 2.0
                emode = "quantile_gap"
            else:
                et = (pmean + nmean) / 2.0
                emode = "overlap_midpoint"
            et = float(max(-1.0, min(1.0, et)))
            per_expert_thresholds.append(et)
            per_expert_calibration[expert] = {
                "threshold": et,
                "mode": emode,
                "positive_mean": pmean,
                "negative_mean": nmean,
                "positive_q20": p20,
                "negative_q80": n80,
            }

        pos_q20 = _quantile(
            positive_margins,
            0.20,
        )

        neg_q80 = _quantile(
            negative_margins,
            0.80,
        )

        positive_mean = float(
            positive_margins
            .mean()
            .item()
        )

        negative_mean = float(
            negative_margins
            .mean()
            .item()
        )

        if (
            pos_q20
            > neg_q80
            + policy.minimum_calibration_gap
        ):
            threshold = (
                pos_q20
                + neg_q80
            ) / 2.0

            mode = (
                "quantile_gap"
            )
        else:
            threshold = (
                positive_mean
                + negative_mean
            ) / 2.0

            mode = (
                "overlap_midpoint"
            )

        threshold = float(
            max(
                -1.0,
                min(
                    1.0,
                    threshold,
                ),
            )
        )

        tensors[
            f"center.layer_{layer}"
        ] = (
            global_center
            .to(
                dtype=torch.float16
            )
            .contiguous()
        )

        tensors[
            f"prototype.layer_{layer}"
        ] = (
            expert_prototypes
            .to(
                dtype=torch.float16
            )
            .contiguous()
        )

        tensors[
            f"null_prototype.layer_{layer}"
        ] = (
            null_prototype
            .to(
                dtype=torch.float16
            )
            .contiguous()
        )

        tensors[
            f"null_threshold.layer_{layer}"
        ] = torch.tensor(
            [threshold],
            dtype=torch.float32,
        )

        tensors[
            f"expert_thresholds.layer_{layer}"
        ] = torch.tensor(
            per_expert_thresholds,
            dtype=torch.float32,
        )

        layer_calibration[
            str(layer)
        ] = {
            "threshold": threshold,
            "mode": mode,
            "positive_mean": (
                positive_mean
            ),
            "negative_mean": (
                negative_mean
            ),
            "positive_q20": (
                pos_q20
            ),
            "negative_q80": (
                neg_q80
            ),
            "per_expert": per_expert_calibration,
            "gap_q20_q80": (
                pos_q20
                - neg_q80
            ),
        }

    weight_path = (
        output_dir
        / "semantic_router.safetensors"
    )

    save_file(
        tensors,
        str(weight_path),
    )

    anchor_blob = json.dumps(
        {
            "anchors": {
                expert: anchors[
                    expert
                ]
                for expert
                in expert_order
            },
            "null_anchors": (
                null_anchors
            ),
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode(
        "utf-8"
    )

    semantic_space_blob = json.dumps(
        {
            "base_semantic_space": "null-anchor-only-center-v0.9",
            "null_anchors": null_anchors,
            "anchor_token_window": policy.anchor_token_window,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    meta = {
        "format": (
            "fleet-semantic-router"
        ),
        "version": "0.9.0",
        "expert_order": (
            expert_order
        ),
        "layers": layer_ids,
        "anchor_sha256": (
            hashlib.sha256(
                anchor_blob
            ).hexdigest()
        ),
        "semantic_space_sha256": hashlib.sha256(semantic_space_blob).hexdigest(),
        "policy": {
            "anchor_token_window": (
                policy
                .anchor_token_window
            ),
            "semantic_weight": (
                policy
                .semantic_weight
            ),
            "weight_weight": (
                policy
                .weight_weight
            ),
            "prompt_prior_weight": (
                policy
                .prompt_prior_weight
            ),
            "single_margin": (
                policy
                .single_margin
            ),
            "router_temperature": (
                policy
                .router_temperature
            ),
            "score_clip": (
                policy
                .score_clip
            ),
            "minimum_calibration_gap": (
                policy
                .minimum_calibration_gap
            ),
        },
        "layer_calibration": (
            layer_calibration
        ),
        "notes": (
            "v0.9 uses a NULL-anchor-only common center so semantic "
            "coordinates are independent of the installed expert set. "
            "Each expert also receives an independent per-layer activation "
            "threshold, enabling hot install/uninstall without bank-wide "
            "recalibration. No gradient-based training is performed."
        ),
    }

    (
        output_dir
        / "semantic_router.json"
    ).write_text(
        json.dumps(
            meta,
            indent=2,
        ),
        encoding="utf-8",
    )

    return meta


class SemanticState:
    """
    Per-runtime semantic store plus request-local prompt priors.

    The router stores semantic class margins for each layer. v0.6.1 also
    exposes per-layer debug state so a benchmark can show whether NULL won
    because it should have, or because calibration failed.
    """

    def __init__(
        self,
        *,
        bank_dir: str | Path,
        expert_order: list[str],
    ):
        bank_dir = Path(
            bank_dir
        )

        self.meta = json.loads(
            (
                bank_dir
                / "semantic_router.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        if (
            self.meta[
                "expert_order"
            ]
            != expert_order
        ):
            raise ValueError(
                "Semantic router expert order "
                "does not match compiled bank."
            )

        if self.meta.get("version") not in {"0.6.1", "0.9.0"}:
            raise ValueError(
                "Unsupported Fleet semantic router version. "
                "Run `fleet semantic-build` again."
            )

        self.expert_order = (
            expert_order
        )

        self.policy = (
            SemanticPolicy(
                **self.meta[
                    "policy"
                ]
            )
        )

        self.tensors = load_file(
            str(
                bank_dir
                / "semantic_router.safetensors"
            ),
            device="cpu",
        )

        self._device_cache = {}
        self.request_prior = {}
        self.request_debug = {}

    def reset_request(
        self,
    ):
        self.request_prior.clear()
        self.request_debug.clear()

    def layer_tensors(
        self,
        layer_id: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        key = (
            int(layer_id),
            str(device),
            dtype,
        )

        if key not in (
            self._device_cache
        ):
            center = self.tensors[
                f"center.layer_{layer_id}"
            ].to(
                device=device,
                dtype=dtype,
            )

            prototypes = (
                self.tensors[
                    f"prototype.layer_{layer_id}"
                ].to(
                    device=device,
                    dtype=dtype,
                )
            )

            null_proto = (
                self.tensors[
                    f"null_prototype.layer_{layer_id}"
                ].to(
                    device=device,
                    dtype=dtype,
                )
            )

            threshold = float(
                self.tensors[
                    f"null_threshold.layer_{layer_id}"
                ][0].item()
            )

            self._device_cache[
                key
            ] = (
                center,
                prototypes,
                null_proto,
                threshold,
            )

        return (
            self._device_cache[
                key
            ]
        )

    def _score_query(
        self,
        *,
        layer_id: int,
        query: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        float,
        float,
    ]:
        (
            center,
            prototypes,
            null_proto,
            threshold,
        ) = self.layer_tensors(
            layer_id,
            device=query.device,
            dtype=query.dtype,
        )

        margins, null_similarity = (
            semantic_margin_scores(
                query,
                center=center,
                expert_prototypes=(
                    prototypes
                ),
                null_prototype=(
                    null_proto
                ),
                score_clip=(
                    self.policy
                    .score_clip
                ),
            )
        )

        return (
            margins,
            float(
                null_similarity.item()
            ),
            float(
                threshold
            ),
        )

    def update_or_get_prior(
        self,
        *,
        layer_id: int,
        q_input: torch.Tensor | None,
        sequence_length: int,
    ) -> tuple[
        torch.Tensor | None,
        float | None,
    ]:
        """
        Returns:
          semantic expert-vs-NULL margins [all experts]
          calibrated NULL threshold for this layer
        """
        if q_input is None:
            prior = (
                self.request_prior.get(
                    layer_id
                )
            )

            threshold = (
                self.request_debug.get(
                    layer_id,
                    {}
                ).get(
                    "threshold"
                )
            )

            return (
                prior,
                threshold,
            )

        if (
            sequence_length > 1
            or layer_id
            not in self.request_prior
        ):
            query = pooled_direction(
                q_input,
                window=(
                    self.policy
                    .anchor_token_window
                ),
            )

            (
                margins,
                null_similarity,
                threshold,
            ) = self._score_query(
                layer_id=layer_id,
                query=query,
            )

            self.request_prior[
                layer_id
            ] = margins.detach()

            top_index = int(
                margins.argmax().item()
            )

            top_margin = float(
                margins[
                    top_index
                ].item()
            )

            self.request_debug[
                layer_id
            ] = {
                "layer": (
                    int(layer_id)
                ),
                "top_expert": (
                    self.expert_order[
                        top_index
                    ]
                ),
                "top_margin": (
                    top_margin
                ),
                "threshold": (
                    threshold
                ),
                "accepted": (
                    top_margin
                    >= threshold
                ),
                "null_similarity": (
                    null_similarity
                ),
            }

            return (
                margins,
                threshold,
            )

        current_query = F.normalize(
            q_input[
                0,
                -1,
            ].float(),
            dim=-1,
        )

        (
            current,
            _,
            threshold,
        ) = self._score_query(
            layer_id=layer_id,
            query=current_query,
        )

        prompt = (
            self.request_prior[
                layer_id
            ].to(
                current.device
            )
        )

        alpha = (
            self.policy
            .prompt_prior_weight
        )

        mixed = (
            alpha
            * prompt
            + (
                1.0
                - alpha
            )
            * current
        )

        return (
            mixed,
            threshold,
        )

    def request_summary(
        self,
    ) -> dict:
        rows = [
            self.request_debug[
                layer
            ]
            for layer
            in sorted(
                self.request_debug
            )
        ]

        votes = {}

        for row in rows:
            name = row[
                "top_expert"
            ]

            votes[name] = (
                votes.get(
                    name,
                    0,
                )
                + 1
            )

        accepted = [
            row
            for row
            in rows
            if row[
                "accepted"
            ]
        ]

        return {
            "layers": rows,
            "top_expert_votes": (
                dict(
                    sorted(
                        votes.items(),
                        key=lambda kv: (
                            -kv[1],
                            kv[0],
                        ),
                    )
                )
            ),
            "accepted_layers": (
                len(accepted)
            ),
            "total_layers": (
                len(rows)
            ),
            "acceptance_rate": (
                len(accepted)
                / len(rows)
                if rows
                else 0.0
            ),
            "mean_top_margin": (
                sum(
                    row[
                        "top_margin"
                    ]
                    for row in rows
                )
                / len(rows)
                if rows
                else 0.0
            ),
            "mean_threshold": (
                sum(
                    row[
                        "threshold"
                    ]
                    for row in rows
                )
                / len(rows)
                if rows
                else 0.0
            ),
        }
