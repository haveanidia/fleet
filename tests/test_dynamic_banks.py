import torch

from fleet.native.configuration_fleet import FleetConfig
from fleet.native.modeling_fleet import FleetLinear, FleetRouterState, _dynamic_coefficients


def test_dynamic_sparse_local_and_semantic_banks():
    config = FleetConfig(
        hidden_size=4,
        num_hidden_layers=2,
        fleet_rank=2,
        fleet_expert_names=[],
        fleet_eligibility={},
    )
    router = FleetRouterState(config)
    layer = FleetLinear(
        4,
        3,
        bias=False,
        rank=2,
        layer_id=0,
        semantic_id="layers.0.self_attn.q_proj",
        router=router,
        expert_ids=[],
    )

    i0 = router.append_expert(torch.randn(2, 4), torch.tensor([0.1, 0.2]))
    layer.append_expert(
        global_index=i0,
        A=torch.randn(2, 4),
        B=torch.randn(3, 2),
        weight_prototype=torch.randn(4),
    )
    i1 = router.append_expert(torch.randn(2, 4), torch.tensor([0.2, 0.3]))
    layer.append_expert(
        global_index=i1,
        A=torch.randn(2, 4),
        B=torch.randn(3, 2),
        weight_prototype=torch.randn(4),
    )

    assert layer.expert_ids.tolist() == [0, 1]
    assert router.semantic_prototypes.shape == (2, 2, 4)
    assert router.semantic_thresholds.shape == (2, 2)

    layer.remove_global_expert(0)
    router.remove_expert(0)
    assert layer.expert_ids.tolist() == [0]
    assert router.semantic_prototypes.shape == (2, 1, 4)


def test_per_expert_semantic_gate():
    ranking = torch.tensor([[0.9, 0.8], [0.5, 0.4]])
    semantic = torch.tensor([[0.3, 0.1], [0.05, 0.2]])
    thresholds = torch.tensor([0.2, 0.15])
    coeff = _dynamic_coefficients(
        ranking,
        semantic,
        thresholds,
        single_margin=0.2,
        temperature=0.7,
    )
    assert torch.allclose(coeff.sum(dim=1), torch.ones(2))
    assert coeff[0, 0] == 1
    assert coeff[1, 1] == 1
