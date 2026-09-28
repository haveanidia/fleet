from __future__ import annotations

from pathlib import Path
import json
import tempfile

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, PreTrainedModel
from peft import ArrowConfig, create_arrow_model

from .configuration_fleet import FleetConfig


class FleetForCausalLM(PreTrainedModel):
    """
    Hugging Face single-model wrapper for Fleet.

    Hub storage is consolidated:
      - one `fleet_model.safetensors`
      - one `fleet_manifest.json`

    At load time, expert adapter files are materialized into a temporary runtime
    cache because PEFT's current Arrow loader expects adapter directories.
    This is an implementation detail; the published Fleet artifact remains one
    model repository / one consolidated Fleet weights file.
    """

    config_class = FleetConfig
    base_model_prefix = "fleet"

    def __init__(self, config: FleetConfig):
        super().__init__(config)
        self.fleet = None
        self._fleet_tmp = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        **kwargs,
    ):
        from huggingface_hub import snapshot_download

        load_in_4bit = kwargs.pop("load_in_4bit", True)
        device_map = kwargs.pop("device_map", "auto")
        torch_dtype = kwargs.pop("torch_dtype", None)
        dtype = kwargs.pop("dtype", torch_dtype)

        local = Path(
            snapshot_download(
                repo_id=str(pretrained_model_name_or_path),
                allow_patterns=[
                    "config.json",
                    "fleet_manifest.json",
                    "fleet_model.safetensors",
                    "configuration_fleet.py",
                    "modeling_fleet.py",
                ],
            )
        )

        config = FleetConfig.from_pretrained(local)
        wrapper = cls(config)

        manifest = json.loads(
            (local / config.fleet_manifest).read_text(encoding="utf-8")
        )

        packed = load_file(
            str(local / config.fleet_weights),
            device="cpu",
        )

        # Import Fleet package if installed. The source project provides the
        # semantic runtime helpers. A native PEFT-free runtime is a future step.
        from fleet.semantic import SemanticState
        from fleet.semantic_router import install_semantic_router

        if dtype is None:
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )

        quantization_config = None

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )

        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            quantization_config=quantization_config,
            dtype=dtype,
            device_map=device_map,
        )

        tmp = tempfile.TemporaryDirectory(prefix="fleet_model_")
        tmp_root = Path(tmp.name)

        expert_paths = []
        origin_masks = {}

        for expert in manifest["experts"]:
            idx = int(expert["index"])
            name = expert["name"]

            d = tmp_root / name
            d.mkdir(parents=True)

            expert_state = {}

            prefix = f"experts.{idx}."

            for key, tensor in packed.items():
                if key.startswith(prefix):
                    expert_state[key[len(prefix):]] = tensor

            if not expert_state:
                raise RuntimeError(
                    f"Consolidated Fleet weights contain no tensors for expert {name!r}"
                )

            save_file(
                expert_state,
                str(d / "adapter_model.safetensors"),
            )

            (d / "adapter_config.json").write_text(
                json.dumps(expert["adapter_config"], indent=2),
                encoding="utf-8",
            )

            origin_masks[name] = expert["origin_mask"]
            expert_paths.append(str(d))

        # Reconstruct semantic-router files in runtime cache.
        semantic_state = {}

        for key, tensor in packed.items():
            if key.startswith("semantic."):
                semantic_state[key[len("semantic."):]] = tensor

        save_file(
            semantic_state,
            str(tmp_root / "semantic_router.safetensors"),
        )

        (tmp_root / "semantic_router.json").write_text(
            json.dumps(manifest["semantic_router"], indent=2),
            encoding="utf-8",
        )

        top_k = min(2, len(expert_paths))

        fleet_model = create_arrow_model(
            base_model=base,
            task_specific_adapter_paths=expert_paths,
            arrow_config=ArrowConfig(
                top_k=top_k,
                router_temperature=1.0,
                rng_seed=42,
            ),
        )

        fleet_model.set_adapter("arrow_router")

        expert_order = [
            x["name"]
            for x in manifest["experts"]
        ]

        semantic = SemanticState(
            bank_dir=tmp_root,
            expert_order=expert_order,
        )

        install_semantic_router(
            fleet_model,
            expert_order=expert_order,
            origin_masks=origin_masks,
            semantic_state=semantic,
        )

        wrapper.fleet = fleet_model
        wrapper.semantic_state = semantic
        wrapper.expert_order = expert_order
        wrapper._fleet_tmp = tmp

        return wrapper

    def forward(self, *args, **kwargs):
        return self.fleet(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.fleet.generate(*args, **kwargs)

    def get_input_embeddings(self):
        if self.fleet is None:
            return None
        return self.fleet.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.fleet.set_input_embeddings(value)
