def test_native_checkpoint_layout_contract():
    expected = {
        "config.json",
        "configuration_fleet.py",
        "modeling_fleet.py",
    }

    assert "config.json" in expected
    assert "modeling_fleet.py" in expected


def test_native_architecture_name():
    assert "FleetForCausalLM" == "FleetForCausalLM"
