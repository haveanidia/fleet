from __future__ import annotations

from pathlib import Path
import json
import threading

import torch
from peft import (
    ArrowConfig,
    create_arrow_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .semantic import SemanticState
from .semantic_router import (
    collect_semantic_stats,
    install_semantic_router,
    reset_semantic_stats,
)


class FleetSemanticRuntime:
    """
    Frozen-backbone semantic prior + layer-local LoRA geometry.

    The semantic prior is established from prompt-prefill hidden states inside
    q_proj. No external classifier or keyword router is used.
    """

    def __init__(
        self,
        bank_dir: str | Path,
        *,
        load_in_4bit: bool = True,
        device_map="auto",
    ):
        self.bank_dir = Path(
            bank_dir
        )

        self.bank = json.loads(
            (
                self.bank_dir
                / "fleet_bank.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        self.base_model_name = (
            self.bank["base_model"]
        )

        self.expert_order = [
            item["name"]
            for item in self.bank[
                "experts"
            ]
        ]

        if not (
            self.bank_dir
            / "semantic_router.json"
        ).exists():
            raise FileNotFoundError(
                "semantic_router.json missing. "
                "Run `fleet semantic-build` first."
            )

        if not (
            self.bank_dir
            / "semantic_router.safetensors"
        ).exists():
            raise FileNotFoundError(
                "semantic_router.safetensors missing. "
                "Run `fleet semantic-build` first."
            )

        self.semantic_state = (
            SemanticState(
                bank_dir=self.bank_dir,
                expert_order=(
                    self.expert_order
                ),
            )
        )

        self.origin_masks = {}

        for item in self.bank[
            "experts"
        ]:
            expert = item["name"]

            self.origin_masks[
                expert
            ] = json.loads(
                (
                    self.bank_dir
                    / item["path"]
                    / "fleet_origin_mask.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

        dtype = (
            torch.bfloat16
            if (
                torch.cuda.is_available()
                and torch.cuda.is_bf16_supported()
            )
            else torch.float16
        )

        quantization_config = None

        if load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "4-bit Fleet runtime requires CUDA."
                )

            quantization_config = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                )
            )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                self.base_model_name,
                use_fast=True,
            )
        )

        if (
            self.tokenizer.pad_token_id
            is None
        ):
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        base = (
            AutoModelForCausalLM
            .from_pretrained(
                self.base_model_name,
                quantization_config=(
                    quantization_config
                ),
                dtype=dtype,
                device_map=device_map,
            )
        )

        base.config.use_cache = True

        expert_paths = [
            str(
                self.bank_dir
                / item["path"]
            )
            for item
            in self.bank["experts"]
        ]

        # Arrow remains the LoRA container / weight-prototype baseline.
        # Fleet replaces its routing forward.
        arrow_config = (
            ArrowConfig(
                top_k=min(
                    2,
                    len(
                        expert_paths
                    ),
                ),
                router_temperature=1.0,
                rng_seed=42,
            )
        )

        self.model = (
            create_arrow_model(
                base_model=base,
                task_specific_adapter_paths=(
                    expert_paths
                ),
                arrow_config=(
                    arrow_config
                ),
            )
        )

        self.model.set_adapter(
            "arrow_router"
        )

        self.router_info = (
            install_semantic_router(
                self.model,
                expert_order=(
                    self.expert_order
                ),
                origin_masks=(
                    self.origin_masks
                ),
                semantic_state=(
                    self.semantic_state
                ),
            )
        )

        # Current v0.6 semantic state is request-local and batch-size 1.
        # Serialize model use in the API until state is moved into a batched
        # model-native cache.
        self._generation_lock = (
            threading.Lock()
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 180,
        temperature: float = 0.0,
    ) -> dict:
        with self._generation_lock:
            reset_semantic_stats(
                self.model,
                self.semantic_state,
            )

            inputs = (
                self.tokenizer
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

            device = (
                self.model
                .get_input_embeddings()
                .weight
                .device
            )

            inputs = {
                k: v.to(device)
                for k, v
                in inputs.items()
            }

            do_sample = (
                temperature > 0
            )

            generate_kwargs = {
                "max_new_tokens": (
                    max_new_tokens
                ),
                "do_sample": do_sample,
                "pad_token_id": (
                    self.tokenizer
                    .pad_token_id
                ),
                "eos_token_id": (
                    self.tokenizer
                    .eos_token_id
                ),
            }

            if do_sample:
                generate_kwargs[
                    "temperature"
                ] = temperature

            output = (
                self.model.generate(
                    **inputs,
                    **generate_kwargs,
                )
            )

            new_tokens = output[
                0,
                inputs[
                    "input_ids"
                ].shape[-1]:
            ]

            text = (
                self.tokenizer
                .decode(
                    new_tokens,
                    skip_special_tokens=True,
                )
                .strip()
            )

            routing = (
                collect_semantic_stats(
                    self.model,
                    self.expert_order,
                    self.semantic_state,
                )
            )

            return {
                "text": text,
                "routing": routing,
            }

    def experts(
        self,
    ) -> list[dict]:
        return list(
            self.bank["experts"]
        )
