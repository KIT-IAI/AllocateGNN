"""Plan 003b: byte comparison, partition boundaries, and first-generation gates."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from sglib.core.infra import leaf_manifest
from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.hashing import sha256_json
from sglib.dataoverview import manifest as data_manifest
from sglib.dataoverview.overview import inventory as inventory_writer
from sglib.generator import manifest as generator_manifest
from sglib.generator import stage
from sglib.generator.registry import build_registry

pytestmark = pytest.mark.gate


ROOT = Path(__file__).resolve().parents[1]


def write(root, relative, payload=b"original"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        atomic_json(payload, path)
    else:
        path.write_bytes(payload)
    return path


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def generator_context(tmp_path):
    groups = ["B-UK-GNN", "B-UK-MLP", "T-UK", "L-UK-N", "L-UK-P",
              *(f"{kind}-UK-{signal}" for kind in ("P", "F") for signal in ("N", "P", "NP"))]
    candidates = json.loads((ROOT / "casestudy/2_Generator/general/candidate_registry.json").read_text(encoding="utf-8"))
    config = {"task_groups": groups, "sweeps": {name: [] for name in ("alpha", "beta", "gamma", "kappa", "lambda", "tau")}}
    root = tmp_path / "results/2_Generator/1_UK"
    ctx = SimpleNamespace(repo=tmp_path, results=tmp_path / "results", root=root, country="uk",
                          loaded=SimpleNamespace(country_profile=SimpleNamespace(directory="1_UK")),
                          units=build_registry("uk", config, candidates), candidates=candidates, read_only=True)
    for path in ("inputs/bundle.pkl", "static/gpm/index.json", "candidates/GNN/R.npz",
                 "candidates/Uni/R.npz", "candidates/index_GNN.json",
                 "candidates/candidate_index.json", "candidates/candidate_qa_index.json",
                 "idr_fixed/R.npz", "idr_matched/R.npz", "civd/index.json", "audit.json"):
        write(root, path)
    for name in config["sweeps"]:
        write(root, f"sweeps/{name}/R.npz")
        write(root, f"sweeps/index_{name}.json")
    for group in groups:
        task_path = f"training/tasks/{group}/task.json"
        write(root, f"training/tasks/{group}/index.json", {"tasks": [{"path": task_path}]})
        write(root, task_path, {"output_results_relative": f"2_Generator/1_UK/2_Weighter/{group}/task"})
        for path in (f"2_Weighter/{group}/task/model.pth", f"training/receipts/{group}.json",
                     f"training/verify/{group}.json", f"inference/tasks/{group}/task.json",
                     f"inference/outputs/{group}/field.npz", f"inference/receipts/{group}.json",
                     f"inference/verify/{group}.json"):
            write(root, path)
    return ctx


def baseline(ctx):
    current, _ = generator_manifest.enumerate_leaves(ctx.root, ctx.units, ctx.candidates)
    document = leaf_manifest.build("2_Generator", ctx.country, leaf_manifest.read_current(current, ctx.root))
    atomic_json(document, ctx.root / "manifest.json")
    return document


def failed_leaves(report):
    return {name for name, leaf in report["leaves"].items() if leaf["status"] == "FAIL"}


def test_canonical_fingerprints_and_inventory_adapter_do_not_read_disk(monkeypatch):
    artifacts = [{"path": "absent/data.bin", "bytes": 3, "sha256": "b" * 64}]
    original = deepcopy(artifacts)
    monkeypatch.setattr(leaf_manifest, "sha256_file", lambda *a: pytest.fail("must use recorded artifacts"))
    document = data_manifest.from_inventory({"country": "uk", "artifacts": artifacts})
    canonical = json.dumps(artifacts, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    assert document["schema_version"] == "sg_leaf_manifest_v1"
    assert document["leaves"]["inventory"]["fingerprint"] == fingerprint
    assert document["root_fingerprint"] == sha256_json({"inventory": fingerprint})
    assert artifacts == original


@pytest.mark.parametrize("kind", ["changed", "missing", "extra"])
def test_generator_failures_identify_leaf_and_notebook_exits_without_product_writes(generator_context, kind, capsys):
    ctx = generator_context
    baseline(ctx)
    field = ctx.root / "candidates/GNN/R.npz"
    relative = field.relative_to(ctx.root).as_posix()
    if kind == "changed":
        field.write_bytes(b"modified")  # Same byte count; hashing, not size, must detect this.
    elif kind == "missing":
        field.unlink()
    else:
        relative = "candidates/GNN/new.npz"
        write(ctx.root, relative)
    before = snapshot(ctx.repo)
    notebook = json.loads((ROOT / "casestudy/2_Generator/1_UK/07_audit_handoff.ipynb").read_text(encoding="utf-8"))
    namespace = {"ctx": ctx, "run_gate": generator_manifest.run_gate, "write_setup": generator_manifest.write_setup}
    with pytest.raises(SystemExit) as exc:
        exec("".join(notebook["cells"][-1]["source"]), namespace)
    assert exc.value.code == 1
    report = namespace["report"]
    assert report["status"] == "FAIL" and failed_leaves(report) == {"materialize/GNN"}
    assert report["leaves"]["materialize/GNN"][kind] == [relative]
    gate_path = "results/_views/2_Generator/1_UK/gate.json"
    after = snapshot(ctx.repo)
    assert set(after) - set(before) == {gate_path}
    assert {k: v for k, v in after.items() if k != gate_path} == before
    assert json.loads(after[gate_path]) == report
    assert "FAIL materialize/GNN" in capsys.readouterr().out


def test_materialized_field_changes_only_its_family_not_training_or_finalize(generator_context):
    ctx = generator_context
    original = baseline(ctx)
    (ctx.root / "candidates/GNN/R.npz").write_bytes(b"changed demand")
    report = generator_manifest.run_gate(ctx)
    assert failed_leaves(report) == {"materialize/GNN"}
    assert report["root_fingerprint"] != original["root_fingerprint"]
    assert report["baseline_root_fingerprint"] == original["root_fingerprint"]
    for name, leaf in original["leaves"].items():
        if name.startswith("train/") or name == "finalize":
            assert report["leaves"][name]["fingerprint"] == leaf["fingerprint"]


@pytest.mark.parametrize("filename", ["candidate_index.json", "candidate_qa_index.json"])
def test_finalize_fingerprint_changes_only_with_index_contents(generator_context, filename):
    ctx = generator_context
    original = baseline(ctx)
    index = ctx.root / "candidates" / filename
    index.write_bytes(index.read_bytes())
    unchanged = generator_manifest.run_gate(ctx)
    assert unchanged["status"] == "PASS"
    assert unchanged["root_fingerprint"] == original["root_fingerprint"]
    index.write_bytes(b"new index content")
    changed = generator_manifest.run_gate(ctx)
    assert failed_leaves(changed) == {"finalize"}
    assert changed["leaves"]["finalize"]["fingerprint"] != original["leaves"]["finalize"]["fingerprint"]
    assert [a["path"] for a in original["leaves"]["finalize"]["artifacts"]] == [
        "candidates/candidate_index.json", "candidates/candidate_qa_index.json"]


def test_registry_partitions_own_referenced_checkpoints_and_exclude_views(generator_context):
    ctx = generator_context
    baseline(ctx)
    unclaimed = ["2_Weighter/orphan/model.pth", "training/graphs/cache.pkl", "candidates/unknown/R.npz"]
    for path in unclaimed + ["_views/cache.json", "candidates/GNN/_views/plot.png"]:
        write(ctx.root, path)
    current, extras = generator_manifest.enumerate_leaves(ctx.root, ctx.units, ctx.candidates)
    assert len(current) == 40
    assert extras == sorted(unclaimed)
    assert sum(map(len, current.values())) == len(set().union(*map(set, current.values())))
    assert generator_manifest.run_gate(ctx)["status"] == "PASS"
    checkpoint = "2_Weighter/B-UK-GNN/task/additional.pth"
    write(ctx.root, checkpoint)
    report = generator_manifest.run_gate(ctx)
    assert failed_leaves(report) == {"train/B-UK-GNN"}
    assert report["leaves"]["train/B-UK-GNN"]["extra"] == [checkpoint]


def test_missing_training_index_stays_a_training_leaf_failure(generator_context):
    ctx = generator_context
    baseline(ctx)
    (ctx.root / "training/tasks/B-UK-GNN/index.json").unlink()
    report = generator_manifest.run_gate(ctx)
    assert failed_leaves(report) == {"train/B-UK-GNN"}
    assert "2_Weighter/B-UK-GNN/task/model.pth" in report["unclaimed"]


def test_closed_generator_first_generation_never_becomes_baseline(generator_context, monkeypatch, capsys):
    ctx = generator_context
    monkeypatch.setattr(generator_manifest, "verify", lambda *a: pytest.fail("new manifest is not a baseline"))
    for _ in range(2):
        report = generator_manifest.run_gate(ctx)
        assert report["status"] == "GENERATED"
        assert report["baseline_root_fingerprint"] is None
        assert not (ctx.root / "manifest.json").exists()
        assert (stage.figures_root(ctx) / "manifest.json").is_file()
        assert {leaf["status"] for leaf in report["leaves"].values()} == {"GENERATED"}
    output = capsys.readouterr().out
    assert output.count("首次生成，无比对") == 2 and "PASS" not in output


def test_writable_generator_first_generation_places_baseline(generator_context, monkeypatch):
    ctx = generator_context
    ctx.read_only = False
    with monkeypatch.context() as patch:
        patch.setattr(generator_manifest, "verify", lambda *a: pytest.fail("new manifest is not a baseline"))
        first = generator_manifest.run_gate(ctx)
    assert first["status"] == "GENERATED" and first["baseline_root_fingerprint"] is None
    assert (ctx.root / "manifest.json").is_file()
    assert not (stage.figures_root(ctx) / "manifest.json").exists()
    second = generator_manifest.run_gate(ctx)
    assert second["status"] == "PASS"
    assert second["baseline_root_fingerprint"] == first["root_fingerprint"]


@pytest.fixture
def inventory_context(tmp_path, monkeypatch):
    merged = {"country": {"directory": "1_UK", "evaluation_scope": "test"},
              "temporal": {"protocol": "test"}, "station_contract": {
                  "demand_column": "demand", "region_column": "region", "capacity_basis": "test"}}
    ctx = SimpleNamespace(repo_root=tmp_path, country_code="uk", merged=merged, region_items=[],
                          derived_root=tmp_path / "data/datasets/2_derived/uk")
    write(ctx.derived_root, "features_bplus/features_receipt.json")
    canonical = {"regions": write(tmp_path, "data/regions.bin"),
                 "stations": write(tmp_path, "results/stations.bin")}
    monkeypatch.setattr(inventory_writer, "canonical_paths", lambda *a, **kw: canonical)
    monkeypatch.setattr(inventory_writer, "read_artifact", lambda *a: pd.DataFrame({"demand": [1], "region": ["R"]}))
    monkeypatch.setattr(inventory_writer, "expected_feature_artifacts", lambda ctx: [])
    monkeypatch.setattr(inventory_writer, "load_handoff_evidence", lambda *a, **kw: {})
    return ctx


def test_inventory_writer_optional_destination_preserves_default(inventory_context):
    ctx = inventory_context
    view = ctx.repo_root / "results/_views/1_DataOverview/1_UK/data_inventory.json"
    formal = ctx.repo_root / "results/1_DataOverview/1_UK/data_inventory.json"
    assert inventory_writer.write_inventory(ctx, output_path=view) == view
    assert not formal.exists()
    assert inventory_writer.write_inventory(ctx) == formal
    assert formal.read_bytes() == view.read_bytes()
    inventory = json.loads(formal.read_bytes())
    assert inventory["fingerprint"] == sha256_json(inventory["artifacts"])


@pytest.mark.parametrize("kind", ["changed", "missing", "extra"])
def test_dataoverview_checks_only_inventory_paths_without_scan(inventory_context, monkeypatch, kind):
    ctx = inventory_context
    path = inventory_writer.write_inventory(ctx)
    baseline_bytes = path.read_bytes()
    relative = "data/regions.bin"
    if kind == "changed":
        (ctx.repo_root / relative).write_bytes(b"modified")
    elif kind == "missing":
        (ctx.repo_root / relative).unlink()
    else:
        write(ctx.repo_root, "data/new.bin")
    before = snapshot(ctx.repo_root)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "rglob", lambda *a, **kw: pytest.fail("inventory gate must not scan"))
        patch.setattr(inventory_writer, "write_inventory", lambda *a, **kw: pytest.fail("must not refresh existing inventory"))
        report = data_manifest.run_gate(ctx)
    assert path.read_bytes() == baseline_bytes
    assert report["status"] == ("PASS" if kind == "extra" else "FAIL")
    assert report["leaves"]["inventory"]["extra"] == []
    if kind != "extra":
        assert failed_leaves(report) == {"inventory"}
        assert report["leaves"]["inventory"][kind] == [relative]
    after = snapshot(ctx.repo_root)
    gate_path = "results/_views/1_DataOverview/1_UK/gate.json"
    assert set(after) - set(before) == {gate_path}
    assert {k: v for k, v in after.items() if k != gate_path} == before


def test_dataoverview_first_generation_stays_in_view_and_never_compares(inventory_context, monkeypatch, capsys):
    ctx = inventory_context
    monkeypatch.setattr(data_manifest, "verify", lambda *a: pytest.fail("fresh inventory is not a baseline"))
    for _ in range(2):
        report = data_manifest.run_gate(ctx)
        assert report["status"] == "GENERATED"
        assert report["baseline_root_fingerprint"] is None
        assert not (ctx.repo_root / "results/1_DataOverview/1_UK/data_inventory.json").exists()
        assert (ctx.repo_root / "results/_views/1_DataOverview/1_UK/data_inventory.json").is_file()
    output = capsys.readouterr().out
    assert output.count("首次生成，无比对") == 2 and "PASS" not in output


def test_dataoverview_notebook_failure_exits_nonzero(inventory_context, monkeypatch):
    from sglib.core.infra import paths
    from sglib.dataoverview import stage as data_stage

    ctx = inventory_context
    inventory_writer.write_inventory(ctx)
    (ctx.repo_root / "data/regions.bin").write_bytes(b"modified")
    monkeypatch.setattr(paths, "find_repo_root", lambda: ctx.repo_root)
    monkeypatch.setattr(data_stage, "country_context", lambda *a: ctx)
    document = json.loads((ROOT / "casestudy/1_DataOverview/1_UK/06_inventory.ipynb").read_text(encoding="utf-8"))
    namespace = {}
    with pytest.raises(SystemExit) as exc:
        for cell in document["cells"]:
            if cell["cell_type"] == "code":
                exec("".join(cell["source"]), namespace)
    assert exc.value.code == 1
    assert namespace["report"]["status"] == "FAIL"
