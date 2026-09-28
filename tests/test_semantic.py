import torch

from fleet.semantic import (
    centered_direction,
    dynamic_topk_coefficients,
    semantic_margin_scores,
)


def test_centering_removes_common_direction():
    common = torch.tensor(
        [10.0, 0.0, 0.0]
    )

    x = torch.tensor(
        [10.0, 1.0, 0.0]
    )

    centered = centered_direction(
        x,
        common,
    )

    assert centered[1] > 0.99


def test_null_margin_prefers_matching_expert():
    center = torch.zeros(
        3
    )

    experts = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    null = torch.tensor(
        [0.0, 0.0, 1.0]
    )

    query = torch.tensor(
        [1.0, 0.0, 0.0]
    )

    margin, _ = (
        semantic_margin_scores(
            query,
            center=center,
            expert_prototypes=experts,
            null_prototype=null,
        )
    )

    assert margin[0] > margin[1]
    assert margin[0] > 0


def test_dynamic_gate_can_choose_null():
    ranking = torch.tensor([
        [2.0, 1.0],
    ])

    semantic = torch.tensor([
        [-0.1, -0.2],
    ])

    coeff, gate = (
        dynamic_topk_coefficients(
            ranking,
            semantic,
            null_threshold=0.0,
            single_margin=0.2,
            temperature=0.7,
        )
    )

    assert coeff.sum() == 0
    assert gate["k0"].item()


def test_dynamic_gate_one():
    ranking = torch.tensor([
        [2.0, 0.5],
    ])

    semantic = torch.tensor([
        [0.5, -0.3],
    ])

    coeff, gate = (
        dynamic_topk_coefficients(
            ranking,
            semantic,
            null_threshold=0.0,
            single_margin=0.2,
            temperature=0.7,
        )
    )

    assert torch.allclose(
        coeff,
        torch.tensor([
            [1.0, 0.0]
        ]),
    )
    assert gate["k1"].item()


def test_dynamic_gate_two():
    ranking = torch.tensor([
        [1.0, 0.9],
    ])

    semantic = torch.tensor([
        [0.5, 0.4],
    ])

    coeff, gate = (
        dynamic_topk_coefficients(
            ranking,
            semantic,
            null_threshold=0.0,
            single_margin=0.2,
            temperature=0.7,
        )
    )

    assert torch.isclose(
        coeff.sum(),
        torch.tensor(1.0),
    )
    assert (
        coeff > 0
    ).sum() == 2
    assert gate["k2"].item()
