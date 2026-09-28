from __future__ import annotations

from pathlib import Path
import json

import torch
from peft import ArrowConfig, create_arrow_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .router import (
    collect_route_stats,
    install_masked_router,
    reset_route_stats,
)


class FleetRuntime:
    def __init__(
        self,
        bank_dir: str | Path,
        *,
        load_in_4bit: bool = True,
        device_map="auto",
    ):
        self.bank_dir = Path(bank_dir)

        self.bank = json.loads(
            (self.bank_dir / "fleet_bank.json").read_text(
                encoding="utf-8"
            )
        )

        self.base_model_name = self.bank["base_model"]
        self.expert_order = [
            item["name"]
            for item in self.bank["experts"]
        ]

        self.origin_masks = {
            expert: json.loads(
                (
                    self.bank_dir
                    / next(
                        item["path"]
                        for item in self.bank["experts"]
                        if item["name"] == expert
                    )
                    / "fleet_origin_mask.json"
                ).read_text(
                    encoding="utf-8"
                )
            )
            for expert in self.expert_order
        }

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        quantization_config = None

        if load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "4-bit Fleet runtime currently expects CUDA."
                )

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quantization_config,
            dtype=dtype,
            device_map=device_map,
        )

        base.config.use_cache = True

        expert_paths = [
            str(self.bank_dir / item["path"])
            for item in self.bank["experts"]
        ]

        arrow_cfg = ArrowConfig(
            top_k=min(
                int(self.bank.get("top_k", 2)),
                len(expert_paths),
            ),
            router_temperature=float(
                self.bank.get(
                    "router_temperature",
                    1.0,
                )
            ),
            rng_seed=42,
        )

        self.model = create_arrow_model(
            base_model=base,
            task_specific_adapter_paths=expert_paths,
            arrow_config=arrow_cfg,
        )

        self.model.set_adapter(
            "arrow_router"
        )

        self.router_info = install_masked_router(
            self.model,
            expert_order=self.expert_order,
            origin_masks=self.origin_masks,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 180,
        temperature: float = 0.0,
    ) -> dict:
        reset_route_stats(self.model)

        inputs = self.tokenizer.apply_chat_template(
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

        device = next(
            self.model.parameters()
        ).device

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        do_sample = temperature > 0

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature

        output = self.model.generate(
            **inputs,
            **generation_kwargs,
        )

        new_tokens = output[
            0,
            inputs["input_ids"].shape[-1]:
        ]

        text = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        return {
            "text": text,
            "routing": collect_route_stats(
                self.model,
                self.expert_order,
            ),
        }

    def experts(self) -> list[dict]:
        return list(self.bank["experts"])
