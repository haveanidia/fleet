from __future__ import annotations

from transformers import Qwen3Config


class FleetConfig(Qwen3Config):
    model_type = "fleet"

    def __init__(
        self,
        fleet_version="0.9.0",
        fleet_base_model="Qwen/Qwen3-4B-Instruct-2507",
        fleet_rank=32,
        fleet_expert_names=None,
        fleet_expert_metadata=None,
        fleet_targets=None,
        fleet_eligibility=None,
        fleet_router_policy=None,
        fleet_semantic_mode="per_expert_thresholds",
        fleet_semantic_space_sha256=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fleet_version = str(fleet_version)
        self.fleet_base_model = str(fleet_base_model)
        self.fleet_rank = int(fleet_rank)
        self.fleet_expert_names = list(fleet_expert_names or [])
        self.fleet_expert_metadata = list(fleet_expert_metadata or [])
        self.fleet_targets = list(fleet_targets or [])
        self.fleet_eligibility = dict(fleet_eligibility or {})
        self.fleet_router_policy = dict(fleet_router_policy or {})
        self.fleet_semantic_mode = str(fleet_semantic_mode)
        self.fleet_semantic_space_sha256 = (
            None if fleet_semantic_space_sha256 is None else str(fleet_semantic_space_sha256)
        )

        if self.fleet_expert_metadata and (
            len(self.fleet_expert_metadata) != len(self.fleet_expert_names)
        ):
            raise ValueError(
                "fleet_expert_metadata must be empty or match fleet_expert_names length."
            )

        self.architectures = ["FleetForCausalLM"]
        self.auto_map = {
            "AutoConfig": "configuration_fleet.FleetConfig",
            "AutoModelForCausalLM": "modeling_fleet.FleetForCausalLM",
        }
