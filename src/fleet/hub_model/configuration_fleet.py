from transformers import PretrainedConfig


class FleetConfig(PretrainedConfig):
    model_type = "fleet"

    def __init__(
        self,
        base_model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
        fleet_manifest="fleet_manifest.json",
        fleet_weights="fleet_model.safetensors",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.fleet_manifest = fleet_manifest
        self.fleet_weights = fleet_weights
