from pathlib import Path


def test_sync_config_has_meta_guard():
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "src" / "fleet" / "native" / "modeling_fleet.py"
    ).read_text(encoding="utf-8")

    assert 'getattr(module.expert_ids, "is_meta", False)' in source
    assert "return" in source
