"""Setup snapshots: a results root records its own configuration, commit and code checks,
and closed roots are judged by that record instead of by the evolving checkout."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from sglib.core.infra import setup_snapshot
from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.content_chain import derive_chain_commitment, derive_chain_receipt
from sglib.core.infra.hashing import sha256_file

pytestmark = pytest.mark.gate


ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A tiny committed repository with one configuration file and no pip freeze cost."""

    repo = tmp_path / "repo"
    (repo / "casestudy/config").mkdir(parents=True)
    (repo / "casestudy/config/a.toml").write_text('x = 1\n', encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "seed")
    monkeypatch.setattr(setup_snapshot, "environment_listing", lambda: "pkg==1.0\n")
    return repo


def test_snapshot_records_config_commit_and_files_and_verifies_itself(checkout, tmp_path):
    root = tmp_path / "results/2_Generator/1_UK"
    root.mkdir(parents=True)
    written = setup_snapshot.write_setup(root, repo_root=checkout, stage="2_Generator", country="uk",
                                         profile="formal", config_files=["casestudy/config/a.toml"])
    record = json.loads((written / "setup.json").read_text(encoding="utf-8"))
    assert record["schema_version"] == "sg_setup_snapshot_v1"
    assert record["source"] == "sealed" and record["git"]["dirty"] is False
    assert len(record["git"]["commit"]) == 40
    assert record["config_files"] == ["casestudy/config/a.toml"]
    assert (written / "config/casestudy/config/a.toml").read_text(encoding="utf-8") == "x = 1\n"
    assert record["files"]["config/casestudy/config/a.toml"] == sha256_file(checkout / "casestudy/config/a.toml")
    assert record["dependencies_sha256"] == sha256_file(written / "env.txt")
    assert "dirty.patch" not in record["files"]
    assert setup_snapshot.verify_setup(root)["status"] == "PASS"
    assert setup_snapshot.config_root(root) == written / "config"
    assert setup_snapshot.setup_files(root) == sorted(
        f"setup/{name}" for name in (*record["files"], "setup.json"))
    # write once
    with pytest.raises(setup_snapshot.SetupSnapshotError, match="already exists"):
        setup_snapshot.write_setup(root, repo_root=checkout, stage="2_Generator", country="uk",
                                   profile="formal", config_files=[])
    # any later change to the record's files is reported
    (written / "config/casestudy/config/a.toml").write_text("x = 2\n", encoding="utf-8")
    report = setup_snapshot.verify_setup(root)
    assert report["status"] == "FAIL" and report["changed"] == ["config/casestudy/config/a.toml"]
    assert setup_snapshot.verify_setup(tmp_path / "nowhere")["status"] == "ABSENT"


def test_formal_snapshot_refuses_a_dirty_worktree_while_smoke_keeps_the_patch(checkout, tmp_path):
    (checkout / "casestudy/config/a.toml").write_text("x = 3\n", encoding="utf-8")
    (checkout / "stray.txt").write_text("", encoding="utf-8")
    formal = tmp_path / "formal"
    formal.mkdir()
    with pytest.raises(setup_snapshot.SetupSnapshotError, match="clean worktree"):
        setup_snapshot.write_setup(formal, repo_root=checkout, stage="s", country="c", profile="formal",
                                   config_files=["casestudy/config/a.toml"])
    assert not (formal / "setup").exists()
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    written = setup_snapshot.write_setup(smoke, repo_root=checkout, stage="s", country="c", profile="smoke",
                                         config_files=["casestudy/config/a.toml"], allow_dirty=True)
    record = json.loads((written / "setup.json").read_text(encoding="utf-8"))
    assert record["git"]["dirty"] is True
    assert record["git"]["changed_files"] == ["casestudy/config/a.toml"]
    assert record["git"]["untracked_files"] == ["stray.txt"]
    assert "x = 3" in (written / "dirty.patch").read_text(encoding="utf-8")
    assert record["files"]["dirty.patch"] == sha256_file(written / "dirty.patch")
    # the snapshot copies the file as it was read (dirty content), not the committed one
    assert (written / "config/casestudy/config/a.toml").read_text(encoding="utf-8") == "x = 3\n"


def test_dirty_policy_follows_profile_and_results_location(tmp_path):
    repo = tmp_path / "repo"
    assert setup_snapshot.dirty_allowed(repo, repo / "results", "smoke")
    assert not setup_snapshot.dirty_allowed(repo, repo / "results", "formal")
    assert not setup_snapshot.dirty_allowed(repo, repo / "results/_staging/x", "preflight")
    assert setup_snapshot.dirty_allowed(repo, tmp_path / "elsewhere", "formal")


def test_receipt_code_checks_report_every_receipt_without_silence(tmp_path):
    root = tmp_path / "root"
    for node, code in (("stage.node.a", "a" * 64), ("stage.node.b", "b" * 64), ("stage.other.c", "c" * 64)):
        commitment = derive_chain_commitment(node, inputs={}, scientific_parameters={"k": 1}, code_sha256=code)
        atomic_json(derive_chain_receipt(commitment, outputs={"out": "d" * 64}), root / node / "receipt.json")
    atomic_json({"schema_version": "unrelated"}, root / "x/receipt.json")
    atomic_json({"schema_version": "sg_setup_snapshot_v1"}, root / "setup/setup.json")
    rows = setup_snapshot.receipt_code_checks(
        root, lambda node_id, receipt: None if ".other." in node_id else "a" * 64)
    assert [(row["node_id"], row["match"]) for row in rows] == [
        ("stage.node.a", True), ("stage.node.b", False), ("stage.other.c", None)]
    assert rows[1]["receipt_code_sha256"] == "b" * 64 and rows[1]["current_code_sha256"] == "a" * 64
    assert rows[2]["current_code_sha256"] is None


def test_closed_experiment_root_is_judged_by_its_own_recorded_registrations(tmp_path, monkeypatch):
    """Registrations in Git may evolve; a closed root with setup/config keeps its own."""

    from sglib.experiment import stage
    from sglib.experiment.config import load_experiment_config

    results = tmp_path / "results"
    root = results / "3_Experiment/1_UK"
    (results / "3_Experiment/_closures").mkdir(parents=True)
    atomic_json({"schema_version": "sg_006_planning_completion_v1"}, results / "3_Experiment/_closures/c.json")
    current = load_experiment_config(ROOT, "uk")
    files = stage.setup_config_files(stage.country_context(ROOT, "uk", profile="formal", results_root=tmp_path / "w"))
    assert "casestudy/3_Experiment/general/registrations.json" in files
    assert "casestudy/config/countries/uk.toml" in files
    monkeypatch.setattr(setup_snapshot, "environment_listing", lambda: "")
    setup_snapshot.write_setup(root, repo_root=ROOT, stage="3_Experiment", country="uk", profile="formal",
                               config_files=files, allow_dirty=True)
    recorded = root / "setup/config/casestudy/3_Experiment/general/registrations.json"
    registrations = json.loads(recorded.read_text(encoding="utf-8"))
    registrations["bounds"]["etas"] = [0.0]
    recorded.write_text(json.dumps(registrations), encoding="utf-8")
    ctx = stage.country_context(ROOT, "uk", profile="formal", results_root=results)
    assert ctx.read_only
    assert ctx.loaded.registrations["bounds"]["etas"] == [0.0]
    assert ctx.loaded.registrations["bounds"]["etas"] != current.registrations["bounds"]["etas"]
    assert ctx.loaded.sources["registrations"] == recorded
    writable = stage.country_context(ROOT, "uk", profile="formal", results_root=tmp_path / "w")
    assert writable.loaded.registrations["bounds"]["etas"] == current.registrations["bounds"]["etas"]


def test_closed_generator_root_reads_its_recorded_authorities(tmp_path, monkeypatch):
    from sglib.generator import stage

    results = tmp_path / "results"
    root = results / "2_Generator/1_UK"
    (results / "2_Generator/_closures").mkdir(parents=True)
    atomic_json({"status": "PASS", "countries": [], "chain_closure": {
        "schema_version": "sg_content_chain_closure_v1", "node_count": 1,
        "nodes": [{"node_id": "n", "commitment_sha256": "a" * 64, "receipt_sha256": "b" * 64,
                   "outputs": {"o": "c" * 64}}]}}, results / "2_Generator/_closures/x.json")
    # a closure must verify; derive a real one instead
    from sglib.core.infra.content_chain import derive_chain_closure
    commitment = derive_chain_commitment("n", inputs={}, scientific_parameters={}, code_sha256="a" * 64)
    receipt = derive_chain_receipt(commitment, outputs={"o": "c" * 64})
    atomic_json({"status": "PASS", "countries": ["uk"], "chain_closure": derive_chain_closure([receipt])},
                results / "2_Generator/_closures/x.json")
    loaded = stage.country_config(ROOT, "uk")
    files = stage.setup_config_files(loaded, ROOT)
    assert "casestudy/2_Generator/general/generator.toml" in files
    assert "casestudy/config/authority/evidence/indexed_landuse_equivalence.json" in files
    monkeypatch.setattr(setup_snapshot, "environment_listing", lambda: "")
    setup_snapshot.write_setup(root, repo_root=ROOT, stage="2_Generator", country="uk", profile="formal",
                               config_files=files, allow_dirty=True)
    overlay = root / "setup/config/casestudy/2_Generator/1_UK/uk.toml"
    overlay.write_text(overlay.read_text(encoding="utf-8").replace("civd_enabled = true", "civd_enabled = false"),
                       encoding="utf-8")
    ctx = stage.country_context(ROOT, "uk", profile="formal", results_root=results)
    assert ctx.read_only
    assert ctx.loaded.values["execution"]["civd_enabled"] is False
    assert "uk.civd.civd" not in ctx.units
    assert "uk.civd.civd" in stage.country_context(ROOT, "uk", profile="formal", results_root=tmp_path / "w").units


def test_snapshot_checks_embedded_inference_receipts(tmp_path):
    commitment = derive_chain_commitment("generator.inference.fixture", inputs={},
        scientific_parameters={}, code_sha256="a" * 64)
    receipt = derive_chain_receipt(commitment, outputs={"field.npz": "b" * 64})
    atomic_json({"chain": receipt}, tmp_path / "inference_completion.json")
    checks = setup_snapshot.receipt_code_checks(tmp_path, lambda *args: "c" * 64)
    assert len(checks) == 1
    assert checks[0]["receipt"] == "inference_completion.json#/chain"
    assert checks[0]["match"] is False
