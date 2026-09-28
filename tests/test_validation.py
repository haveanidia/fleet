from pathlib import Path
import json

from fleet.validation import validate_bank


def test_missing_manifest(tmp_path: Path):
    result = validate_bank(tmp_path)

    assert result["ok"] is False
    assert "fleet_bank.json missing" in result["errors"]
