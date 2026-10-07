from __future__ import annotations

import ast
from pathlib import Path

import pytest
pytestmark = pytest.mark.gate


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "sglib"
STAGES = {"dataoverview", "generator", "experiment", "analysis", "report"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_core_does_not_import_stage_packages() -> None:
    for path in (PACKAGE / "core").rglob("*.py"):
        imports = _imports(path)
        assert not any(name.startswith(tuple(f"sglib.{stage}" for stage in STAGES)) for name in imports), path


def test_algorithms_do_not_import_infrastructure() -> None:
    for path in (PACKAGE / "core" / "algorithms").rglob("*.py"):
        assert not any(name.startswith("sglib.core.infra") for name in _imports(path)), path


def test_stage_packages_are_mutually_isolated() -> None:
    for stage in STAGES:
        for path in (PACKAGE / stage).rglob("*.py"):
            imports = _imports(path)
            forbidden = STAGES - {stage}
            assert not any(name.startswith(tuple(f"sglib.{item}" for item in forbidden)) for name in imports), path


def test_sglib_never_imports_runtime_or_legacy_packages() -> None:
    for path in PACKAGE.rglob("*.py"):
        imports = _imports(path)
        assert not any(name.startswith("casestudy.runtime") for name in imports), path
        assert not any(name.startswith("SpatialGranularity") for name in imports), path
        assert not any(name.startswith("data.fetcher") for name in imports), path


def test_only_learned_weighter_may_import_torch() -> None:
    allowed = (PACKAGE / "generator" / "weighter" / "learned").resolve()
    for path in PACKAGE.rglob("*.py"):
        if not any(name == "torch" or name.startswith("torch.") for name in _imports(path)):
            continue
        assert path.resolve().is_relative_to(allowed), path
