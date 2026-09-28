from pathlib import Path

import torch

from fleet.fleet_adapter import FleetAdapter


def _demo_adapter():
    semantic_id = "layers.0.self_attn.q_proj"
    return FleetAdapter(
        manifest={
            "format": "fleet-adapter",
            "version": "0.9.0",
            "name": "demo",
            "description": "test",
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "rank": 2,
            "targets": [semantic_id],
            "num_layers": 2,
            "hidden_size": 4,
            "metadata": {"semantic_space_sha256": "demo-space"},
        },
        tensors={
            f"module.{semantic_id}.A": torch.randn(2, 4),
            f"module.{semantic_id}.B": torch.randn(3, 2),
            f"module.{semantic_id}.scaling": torch.tensor([1.0]),
            f"module.{semantic_id}.weight_prototype": torch.randn(4),
            "semantic.prototypes": torch.randn(2, 4),
            "semantic.thresholds": torch.tensor([0.1, 0.2]),
        },
    )


def test_fleet_adapter_roundtrip(tmp_path: Path):
    adapter = _demo_adapter()
    path = adapter.save(tmp_path / "demo.fleet")
    loaded = FleetAdapter.load(path)
    assert loaded.name == "demo"
    assert loaded.rank == 2
    assert loaded.targets == ["layers.0.self_attn.q_proj"]
    assert tuple(loaded.semantic_prototypes.shape) == (2, 4)
    assert tuple(loaded.semantic_thresholds.shape) == (2,)
