"""No-training static validation of the active sglib and casestudy sources."""

from __future__ import annotations

import json
from pathlib import Path
import py_compile
import warnings

import nbformat

import pytest
pytestmark = pytest.mark.gate



ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOTS = (ROOT / "sglib", ROOT / "casestudy")
CASE_STUDY = ROOT / "casestudy"


def _caches() -> set[Path]:
    return {path for root in ACTIVE_ROOTS for path in root.rglob("__pycache__")}


def test_all_active_python_compiles_outside_the_source_tree(tmp_path: Path):
    sources = sorted(path for root in ACTIVE_ROOTS for path in root.rglob("*.py"))
    assert sources
    before = _caches()
    for source in sources:
        relative = source.relative_to(ROOT).with_suffix(".pyc")
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        py_compile.compile(str(source), cfile=str(target), doraise=True)
    assert len(list(tmp_path.rglob("*.pyc"))) == len(sources)
    assert _caches() == before


def test_all_case_study_notebooks_are_nbformat_valid_and_source_clean():
    paths = sorted(CASE_STUDY.rglob("*.ipynb"))
    assert paths
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["nbformat"] == 4 and raw["nbformat_minor"] >= 4
        if raw["nbformat_minor"] >= 5:
            assert all(cell.get("id") for cell in raw["cells"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
        for cell in raw["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
