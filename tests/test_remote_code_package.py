from pathlib import Path
import ast


def _relative_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        (node.level, node.module)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level
    ]


def test_native_model_only_has_one_level_relative_imports():
    root = Path(__file__).resolve().parents[1]
    model = root / "src" / "fleet" / "native" / "modeling_fleet.py"

    imports = _relative_modules(model)
    assert imports
    assert all(level == 1 for level, _ in imports)


def test_native_fleet_adapter_has_no_fleet_relative_dependencies():
    root = Path(__file__).resolve().parents[1]
    adapter = root / "src" / "fleet" / "native" / "fleet_adapter.py"

    assert adapter.exists()
    assert _relative_modules(adapter) == []

    source = adapter.read_text(encoding="utf-8")
    assert "from_compiled_bank" not in source
    assert "package_bank_expert" not in source
    assert ".compiler" not in source


def test_build_time_adapter_still_has_packaging_api():
    root = Path(__file__).resolve().parents[1]
    adapter = root / "src" / "fleet" / "fleet_adapter.py"
    source = adapter.read_text(encoding="utf-8")

    assert "from_compiled_bank" in source
    assert "package_bank_expert" in source
