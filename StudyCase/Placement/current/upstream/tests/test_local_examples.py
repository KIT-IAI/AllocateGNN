"""Local, synthetic examples require neither installed data nor formal models."""
from pathlib import Path
import json

import numpy as np
import pytest
from sglib.examples.synthetic import run
from sglib.examples import nl, nz
from sglib.examples.common import check_smoke_output

pytestmark = pytest.mark.consume
ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_example_keeps_two_contexts_and_output_roots_separate(tmp_path):
    first = run(ROOT, tmp_path / "first", demand_scale=1)
    second = run(ROOT, tmp_path / "second", demand_scale=3)
    assert json.loads(first.read_text())["demand_total"] == pytest.approx(12)
    assert json.loads(second.read_text())["demand_total"] == pytest.approx(36)
    for name, expected in (("first", 12), ("second", 36)):
        root = tmp_path / name / "results/2_Generator/1_UK"
        receipt = json.loads((root / "inputs/receipt.json").read_text())
        assert receipt["dataoverview_inventory"]["path_base"] == "generator_root"
        with np.load(root / "static/uniform/Fixture.npz") as field:
            assert field["data"].sum() == pytest.approx(expected)
    with pytest.raises(FileExistsError, match="empty output"):
        run(ROOT, tmp_path / "first")


def _files(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("runner", (nl.run, nz.run_smoke))
@pytest.mark.parametrize("relative", (
    "results", "results/1_DataOverview/test", "results/2_Generator/test",
    "results/3_Experiment/test", "results/4_Analysis/test", "results/5_Report/test",
))
def test_country_examples_reject_formal_outputs_before_reading_inputs(tmp_path, monkeypatch, runner, relative):
    root = tmp_path / relative
    root.mkdir(parents=True)
    (root / "formal-sentinel.bin").write_bytes(b"original formal bytes\r\n")
    before = _files(tmp_path)

    def unexpected_input_read(*args, **kwargs):
        raise AssertionError("output must be rejected before source inputs are read")

    monkeypatch.setattr(nl, "_handoff", unexpected_input_read)
    monkeypatch.setattr(nz, "build_core9_dataoverview_from_fresh", unexpected_input_read)
    with pytest.raises(ValueError, match="formal results"):
        runner(tmp_path, root)
    assert _files(tmp_path) == before


@pytest.mark.parametrize("runner", (nl.run, nz.run_smoke))
@pytest.mark.parametrize("relative", ("release", "release/nested/new"))
@pytest.mark.parametrize("closure_relative", ("_closures", "2_Generator/_closures", "5_Report/_closures"))
def test_country_examples_reject_sealed_roots_and_descendants(tmp_path, runner, relative, closure_relative):
    release = tmp_path / "release"
    closure = release / closure_relative
    closure.mkdir(parents=True)
    (closure / "original.json").write_bytes(b'{"sealed":true}\n')
    (release / "sealed-sentinel.bin").write_bytes(b"preserved bytes")
    before = _files(tmp_path)
    with pytest.raises(ValueError, match="sealed results"):
        runner(tmp_path, tmp_path / relative)
    assert _files(tmp_path) == before
    assert not (release / "nested").exists()


@pytest.mark.parametrize("runner", (nl.run, nz.run_smoke))
@pytest.mark.parametrize("refresh", (False, True))
def test_country_examples_never_overwrite_partial_outputs(tmp_path, runner, refresh):
    root = tmp_path / "partial"
    root.mkdir()
    (root / "unfinished.bin").write_bytes(b"incomplete computation must be preserved")
    before = _files(tmp_path)
    with pytest.raises(FileExistsError, match="empty output"):
        runner(tmp_path, root, refresh=refresh)
    assert _files(tmp_path) == before


@pytest.mark.parametrize("runner,country,schema,relative", (
    (nl.run, "nl", "sg_nl_pre_hpc_smoke_v2", "pre_hpc_smoke.json"),
    (nz.run_smoke, "nz", "sg_nz_pre_hpc_smoke_receipt_v1", "2_Generator/5_NZ/smoke_receipt.json"),
))
def test_status_only_receipts_are_rejected_without_writes(tmp_path, runner, country, schema, relative):
    root = tmp_path / "results/_smoke/finished"
    receipt = root / relative
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"status": "PASS", "formal": False, "country": country,
                                   "schema_version": schema}), encoding="utf-8")
    # Formal siblings do not make the independent auxiliary root read-only.
    closure = tmp_path / "results/2_Generator/_closures"
    closure.mkdir(parents=True)
    (closure / "formal.json").write_bytes(b'{"original":true}')
    before = _files(tmp_path)
    with pytest.raises(ValueError, match="artifacts"):
        runner(tmp_path, root)
    assert _files(tmp_path) == before
    with pytest.raises(FileExistsError, match="empty output"):
        runner(tmp_path, root, refresh=True)
    assert _files(tmp_path) == before


@pytest.mark.parametrize("change", (
    {"formal": True}, {"formal": None}, {"status": "FAIL"},
    {"country": "uk"}, {"schema_version": "unrelated"},
))
def test_a_nonformal_pass_from_the_same_example_is_required_for_reuse(tmp_path, change):
    root = tmp_path / "finished"
    root.mkdir()
    receipt = root / "pre_hpc_smoke.json"
    receipt.write_text(json.dumps({"status": "PASS", "formal": False, "country": "nl",
                                   "schema_version": "sg_nl_pre_hpc_smoke_v2", **change}))
    before = _files(tmp_path)
    with pytest.raises(ValueError, match="nonformal evidence"):
        nl.run(tmp_path, root)
    assert _files(tmp_path) == before


def test_guard_does_not_create_an_admitted_empty_root(tmp_path):
    root = tmp_path / "new"
    assert check_smoke_output(tmp_path, root) is None
    assert not root.exists()
