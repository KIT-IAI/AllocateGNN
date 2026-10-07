"""Current-paper regression contracts, independent of the historical package."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from SpatialPlacement import current_paper as paper


@pytest.fixture(scope="module")
def current_reproduction(tmp_path_factory):
    output = tmp_path_factory.mktemp("current_reproduction")
    return paper.reproduce(output=output, verify=True), output


def test_current_statistics_and_ledger_match_frozen_exports(current_reproduction):
    data, output = current_reproduction
    assert data["release"]["id"] == "fixedload_20260925_r2"
    assert data["release"]["verification"] == "PASS"
    assert data["release"]["verified_statistical_tables"] == 12
    assert data["release"]["verified_source_files"] == 558
    actual = pd.read_csv(output / "ledger/claim_ledger.csv")
    frozen = pd.read_csv(paper.RELEASE_ROOT / "frozen/ledger/claim_ledger.csv")
    assert data["release"]["claim_count"] == len(frozen)
    assert list(actual.claim_id) == list(frozen.claim_id)
    assert len(actual.claim_id) == len(set(actual.claim_id))
    assert "No model training" in data["release"]["reproducibility_boundary"]
    assert any("Voronoi" in item for item in data["release"]["reused_frozen_metadata"])


def test_current_british_results_replace_legacy_values(current_reproduction):
    data, _ = current_reproduction
    gb = data["tasks"]["GB"]
    assert gb["reconstruction"]["base_mean"] == pytest.approx(10.462268014399758, rel=1e-12)
    assert gb["reconstruction"]["gnn_mean"] == pytest.approx(9.635992845183765, rel=1e-12)
    assert gb["connection"]["base_mean"] == pytest.approx(11592140.985882353, rel=1e-12)
    assert gb["connection"]["gnn_mean"] == pytest.approx(8629419.950455109, rel=1e-12)
    assert gb["connection"]["n_regions"] == gb["connection"]["regions_improved"] == 16
    assert set(gb["connection"]["per_seed"]) == {"42", "123", "456"}


def test_australian_connection_direction_and_four_module_families(current_reproduction):
    data, _ = current_reproduction
    for country in ("GB", "AU"):
        tasks = data["tasks"][country]
        family = {module for module, row in tasks.items() if row["in_family"]}
        assert family == {"reconstruction", "siting", "sizing", "connection"}
        assert not tasks["sizing_1to1"]["in_family"]
        for module in family:
            assert tasks[module]["p_holm_four_module"] >= tasks[module]["p_signflip"]
    connection = data["tasks"]["AU"]["connection"]
    assert connection["gnn_mean"] > connection["base_mean"]
    assert connection["median_change_pct"] == pytest.approx(17.926037848320426)
    assert connection["n_pct_defined"] == 10
    assert connection["n_regions"] == 12


def test_manifest_rejects_tampering(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    source = frozen / "input.csv"
    content = b"value\n123\n"
    source.write_bytes(content)
    manifest = {"files": [{"path": "frozen/input.csv", "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    paper.verify_manifest(tmp_path)
    source.write_bytes(b"value\n124\n")
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        paper.verify_manifest(tmp_path)


def test_manifest_rejects_unlisted_frozen_files(tmp_path):
    (tmp_path / "frozen").mkdir()
    (tmp_path / "manifest.json").write_text('{"files": []}', encoding="utf-8")
    (tmp_path / "frozen/unlisted.csv").write_text("value\n1\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="frozen allowlist mismatch"):
        paper.verify_manifest(tmp_path)


def test_source_verification_rejects_modified_calculation_code(tmp_path):
    (tmp_path / "upstream").mkdir()
    path = tmp_path / "upstream/calculation.py"
    code = b"RESULT = 1\n"
    path.write_bytes(code)
    source = {"upstream_files": [{"path": "calculation.py", "sha256": hashlib.sha256(code).hexdigest()}], "paper_scripts": []}
    (tmp_path / "SOURCE.json").write_text(json.dumps(source), encoding="utf-8")
    assert paper.verify_source(tmp_path) == 1
    path.write_bytes(b"RESULT = 2\n")
    with pytest.raises(AssertionError, match="source SHA-256 mismatch"):
        paper.verify_source(tmp_path)
