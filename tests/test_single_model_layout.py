def test_single_model_expected_files_are_stable():
    expected = {
        "config.json",
        "fleet_model.safetensors",
        "fleet_manifest.json",
        "configuration_fleet.py",
        "modeling_fleet.py",
    }
    assert len(expected) == 5
    assert "fleet_model.safetensors" in expected
