from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.produce


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "casestudy/2_Generator/5_NZ/run_smoke.py"


def test_nz_smoke_runner_is_structural_and_not_formal() -> None:
    source = (ROOT / "sglib/examples/nz.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "pre_hpc_structural_fixture" in source
    assert '"formal": False' in source
    assert '"hpc_submission_authorized": False' in source
    assert "MPLBACKEND" in source
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("CaseStudy")
        for node in ast.walk(tree)
    )


def _runner_module():
    from sglib.examples import nz
    return nz


def test_nz_smoke_reconciles_ids_instead_of_historical_counts() -> None:
    runner = _runner_module()
    source = (ROOT / "sglib/examples/nz.py").read_text(encoding="utf-8")
    assert "expected_n_" not in source
    runner.reconcile_source_ids("r", ["a", "b"], ["b", "a"])
    for rebuilt, registered, kind in (
        (["a"], ["a", "b"], "missing"),
        (["a", "c"], ["a"], "unregistered"),
        (["a", "a"], ["a"], "duplicate_rebuilt"),
        (["a"], ["a", "a"], "duplicate_registered"),
    ):
        with pytest.raises(ValueError, match=kind):
            runner.reconcile_source_ids("r", rebuilt, registered)
    import geopandas as gpd

    sites = gpd.GeoDataFrame({"station_id": ["s1", "s2"]})
    runner.reconcile_site_coverage(sites, ["s2", "s1"])
    with pytest.raises(ValueError, match="uncovered"):
        runner.reconcile_site_coverage(sites, ["s1"])
    with pytest.raises(ValueError, match="multiply_covered"):
        runner.reconcile_site_coverage(sites, ["s1", "s2", "s2"])


def test_nz_display_notebooks_are_numbered_chain_steps_without_side_writes() -> None:
    directory = ROOT / "casestudy/1_DataOverview/5_NZ"
    notebooks = sorted(directory.glob("0[1-9]_*.ipynb"))
    assert [path.name for path in notebooks] == [
        "02_regions_stations.ipynb",
        "03_grid.ipynb",
        "05_features_overview.ipynb",
        "06_inventory.ipynb",
    ]
    forbidden = ("to_file(", "to_csv(", "to_parquet(", "train(", "download(")
    for path in notebooks:
        document = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in document["cells"]
            if cell["cell_type"] == "code"
        )
        assert 'COUNTRY = "nz"' in source
        if path.name == "06_inventory.ipynb":
            assert "report = run_gate(context)" in source
        else:
            assert "stage.run_step(" in source or "stage.require_done(" in source
        assert not any(token in source for token in forbidden)
