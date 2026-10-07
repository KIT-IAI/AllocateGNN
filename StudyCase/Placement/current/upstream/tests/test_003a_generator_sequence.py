from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from sglib.generator import stage
from sglib.generator.registry import GeneratorUnit, unit_status

pytestmark = pytest.mark.gate

ROOT = Path(__file__).resolve().parents[1]


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_training_states_follow_preparation_and_completion(tmp_path):
    unit = GeneratorUnit("uk.train.G", "train", "uk", "G", ())
    assert unit_status(unit, tmp_path) == "PENDING"
    index = tmp_path / "training/tasks/G/index.json"
    write(index, {"schema_version": "sg_prepared_training_group_v1", "tasks": [{"path": "task.json"}]})
    assert unit_status(unit, tmp_path) == "PREPARED"
    marker = tmp_path / "training/receipts/G.json"
    write(marker, {"schema_version": "sg_training_group_receipt_v1", "complete": True})
    assert unit_status(unit, tmp_path) == "DONE"
    marker.write_text("{", encoding="utf-8")
    assert unit_status(unit, tmp_path) == "INVALID"


def test_inference_requires_verification_in_addition_to_group_receipt(tmp_path):
    unit = GeneratorUnit("uk.infer.G", "infer", "uk", "G", ())
    write(tmp_path / "inference/tasks/G/index.json", {"schema_version": "sg_prepared_inference_group_v1", "tasks": [{}]})
    write(tmp_path / "inference/receipts/G.json", {"schema_version": "sg_inference_group_view_v2", "tasks": [{}]})
    assert unit_status(unit, tmp_path) == "PREPARED"
    write(tmp_path / "inference/verify/G.json", {"schema_version": "sg_inference_verify_v1", "tasks": [{}]})
    assert unit_status(unit, tmp_path) == "DONE"


def test_wrong_schema_is_invalid_even_for_nonempty_file(tmp_path):
    unit = GeneratorUnit("uk.inputs.bundle", "inputs", "uk", "bundle", ())
    write(tmp_path / "inputs/receipt.json", {"schema_version": "unrelated_schema"})
    assert unit_status(unit, tmp_path) == "INVALID"


def test_numbered_step_never_executes_missing_predecessor(tmp_path, monkeypatch):
    predecessor = GeneratorUnit("uk.inputs.bundle", "inputs", "uk", "bundle", ())
    child = GeneratorUnit("uk.static.gpm", "static", "uk", "gpm", (predecessor.id,))
    ctx = SimpleNamespace(units={u.id: u for u in (predecessor, child)}, root=tmp_path)
    monkeypatch.setattr(stage, "_execute", lambda *args: pytest.fail("execution must not start"))
    with pytest.raises(stage.StageOrderError, match="earlier numbered steps"):
        stage.run_units(ctx, [child.id])
    assert not list(tmp_path.iterdir())


def test_hpc_preparation_remains_prepared_and_skip_has_reason(tmp_path, monkeypatch, capsys):
    unit = GeneratorUnit("uk.train.G", "train", "uk", "G", ())
    ctx = SimpleNamespace(units={unit.id: unit}, root=tmp_path, results=tmp_path,
        read_only=False, backend="hpc", profile="formal")
    def prepare(context, target):
        write(tmp_path / "training/tasks/G/index.json", {"schema_version": "sg_prepared_training_group_v1", "tasks": [{}]})
        return False
    monkeypatch.setattr(stage, "_execute", prepare)
    result = stage.run_units(ctx, [unit.id])
    assert result.prepared == (unit.id,) and result.ran == ()
    write(tmp_path / "training/receipts/G.json", {"schema_version": "sg_training_group_receipt_v1", "complete": True})
    monkeypatch.setattr(stage, "_execute", lambda *args: pytest.fail("DONE must skip"))
    assert stage.run_units(ctx, [unit.id]).skipped == (unit.id,)
    assert "SKIP uk.train.G: marker contract is DONE:" in capsys.readouterr().out


@pytest.mark.parametrize("document", [{"schema_version": "unknown"}, {}])
def test_unknown_closure_fails_closed(tmp_path, document):
    write(tmp_path / "2_Generator/_closures/example.json", document)
    with pytest.raises(stage.StageOrderError, match="invalid closure"):
        stage._closed_results(tmp_path)


def test_closed_root_rejects_refresh_without_executing(tmp_path, monkeypatch):
    unit = GeneratorUnit("uk.inputs.bundle", "inputs", "uk", "bundle", ())
    ctx = SimpleNamespace(units={unit.id: unit}, root=tmp_path, read_only=True)
    monkeypatch.setattr(stage, "_execute", lambda *args: pytest.fail("closed root must not execute"))
    with pytest.raises(stage.StageOrderError, match="read-only"):
        stage.run_units(ctx, [unit.id], refresh=True)


def test_writable_root_with_inputs_from_another_configuration_is_rejected(tmp_path):
    inputs = tmp_path / "2_Generator/1_UK/inputs"
    inputs.mkdir(parents=True)
    stale = {"schema_version": "sg_generator_inputs_receipt_v3", "country": "uk",
             "config_fingerprint": "8aea09709d0e8935f2b5b0594868487a6b80c82cf1230b43403ccfbc89c34a48"}
    (inputs / "receipt.json").write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(stage.StageOrderError, match="another scientific configuration"):
        stage.country_context(ROOT, "uk", profile="smoke", results_root=tmp_path)
    # a v3 receipt carrying the current fingerprint is accepted
    from sglib.generator.config import scientific_config_fingerprint
    current = {"schema_version": "sg_generator_inputs_receipt_v3", "country": "uk",
               "config_fingerprint": scientific_config_fingerprint(stage.country_config(ROOT, "uk"), repo_root=ROOT)}
    (inputs / "receipt.json").write_text(json.dumps(current), encoding="utf-8")
    assert not stage.country_context(ROOT, "uk", profile="smoke", results_root=tmp_path).read_only
    # a receipt carrying a predecessor fingerprint is rejected: closed roots are judged by their own setup
    predecessor = {"schema_version": "sg_generator_inputs_receipt_v2", "country": "uk",
                   "config_fingerprint": "fbf7923b88afe35d5f1b20c3c42afff28029e345f23da7e587365fc871e75a27"}
    (inputs / "receipt.json").write_text(json.dumps(predecessor), encoding="utf-8")
    with pytest.raises(stage.StageOrderError, match="another scientific configuration"):
        stage.country_context(ROOT, "uk", profile="smoke", results_root=tmp_path)


def test_default_formal_root_is_separate_and_accepts_local_backend(tmp_path):
    ctx = stage.country_context(ROOT, "uk", profile="formal", backend="local", results_root=tmp_path)
    assert ctx.root == tmp_path / "2_Generator/1_UK"
    assert not ctx.read_only
    assert stage.resolve_results_root(ROOT, "formal") == ROOT / "results"
    assert stage.resolve_results_root(ROOT, "smoke") == ROOT / "results/_smoke"


def test_archived_results_roots_are_refused(tmp_path):
    # archived batches are records only: no stage context and no Generator handoff may read them
    from sglib.generator.downstream import load_handoff

    archived = tmp_path / "results" / "_archive" / "rounds" / "v3_final"
    archived.mkdir(parents=True)
    with pytest.raises(ValueError, match="archived records"):
        stage.resolve_results_root(ROOT, "formal", archived)
    with pytest.raises(ValueError, match="archived records"):
        stage.country_context(ROOT, "uk", profile="formal", results_root=archived)
    with pytest.raises(ValueError, match="archived records"):
        load_handoff(ROOT, "uk", results_root=archived)
    assert stage.resolve_results_root(ROOT, "formal", tmp_path / "results/_staging/x") == tmp_path / "results/_staging/x"


def test_civd_unit_accepts_both_index_schemas(tmp_path):
    unit = GeneratorUnit("uk.civd.civd", "civd", "uk", "civd", ())
    index = tmp_path / "civd/index.json"
    index.parent.mkdir(parents=True)
    for schema, expected in (("sg_civd_index_v1", "DONE"), ("sg_civd_extension_index_v1", "DONE"), ("sg_civd_index_v9", "INVALID")):
        index.write_text(json.dumps({"schema_version": schema, "entries": []}), encoding="utf-8")
        assert unit_status(unit, tmp_path) == expected
