from pathlib import Path

def test_native_runtime_has_accelerate_offload_support():
    root=Path(__file__).resolve().parents[1]
    source=(root/'src/fleet/native/modeling_fleet.py').read_text()
    assert 'class _FleetWeightsOverlay' in source
    assert 'def _runtime_tensor_cpu' in source
    assert 'def _replace_runtime_tensor' in source
    assert 'weights_map' in source

def test_export_uses_runtime_tensor_snapshot():
    root=Path(__file__).resolve().parents[1]
    source=(root/'src/fleet/native/modeling_fleet.py').read_text()
    block=source[source.index('    def export_adapter('):]
    assert '_runtime_tensor_cpu' in block
