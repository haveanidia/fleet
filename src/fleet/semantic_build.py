from __future__ import annotations

from pathlib import Path
import gc
import json

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .semantic import (
    build_semantic_prototypes,
    load_semantic_config,
)


def build_semantic_router(
    *,
    bank_dir: str | Path,
    anchors_path: str | Path,
    load_in_4bit: bool = True,
) -> dict:
    bank_dir = Path(bank_dir)

    bank_path = (
        bank_dir
        / "fleet_bank.json"
    )

    bank = json.loads(
        bank_path.read_text(
            encoding="utf-8"
        )
    )

    expert_order = [
        item["name"]
        for item in bank["experts"]
    ]

    policy, anchors, null_anchors = (
        load_semantic_config(
            anchors_path
        )
    )

    missing = [
        expert
        for expert in expert_order
        if expert not in anchors
    ]

    if missing:
        raise ValueError(
            "Active experts missing anchors: "
            + ", ".join(missing)
        )

    bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    dtype = (
        torch.bfloat16
        if bf16
        else torch.float16
    )

    quantization_config = None

    if load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit semantic calibration requires CUDA."
            )

        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            bank["base_model"],
            use_fast=True,
        )
    )

    if (
        tokenizer.pad_token_id
        is None
    ):
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            bank["base_model"],
            quantization_config=(
                quantization_config
            ),
            dtype=dtype,
            device_map="auto",
        )
    )

    model.eval()

    meta = build_semantic_prototypes(
        model=model,
        tokenizer=tokenizer,
        expert_order=expert_order,
        anchors=anchors,
        null_anchors=null_anchors,
        output_dir=bank_dir,
        policy=policy,
    )

    bank["router"] = {
        "type": "fleet-semantic-v0.9",
        "semantic_weights": (
            "semantic_router.safetensors"
        ),
        "semantic_metadata": (
            "semantic_router.json"
        ),
        "anchors_source": str(
            anchors_path
        ),
        "policy": meta["policy"],
    }

    bank_path.write_text(
        json.dumps(
            bank,
            indent=2,
        ),
        encoding="utf-8",
    )

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return meta
