import torch

from fleet.compiler import (
    refactor_lora_delta,
    semantic_module_id,
)


def test_semantic_module_normalization():
    a = (
        "base_model.model.model.layers.4."
        "self_attn.q_proj"
    )
    b = (
        "base_model.model.model.model.layers.4."
        "self_attn.q_proj"
    )

    expected = "layers.4.self_attn.q_proj"

    assert semantic_module_id(a) == expected
    assert semantic_module_id(b) == expected


def test_exact_rank_expansion():
    torch.manual_seed(42)

    A = torch.randn(8, 31)
    B = torch.randn(37, 8)

    Ac, Bc, info = refactor_lora_delta(
        A,
        B,
        source_scale=2.0,
        target_rank=16,
    )

    source = 2.0 * (B @ A)
    compiled = Bc @ Ac

    rel_error = (
        torch.linalg.norm(source - compiled)
        / torch.linalg.norm(source)
    ).item()

    assert info["retained_energy"] == 1.0
    assert rel_error < 1e-5


def test_rank_reduction_shapes():
    torch.manual_seed(42)

    A = torch.randn(32, 17)
    B = torch.randn(19, 32)

    Ac, Bc, info = refactor_lora_delta(
        A,
        B,
        source_scale=1.0,
        target_rank=8,
    )

    assert Ac.shape == (8, 17)
    assert Bc.shape == (19, 8)
    assert 0.0 <= info["retained_energy"] <= 1.0
