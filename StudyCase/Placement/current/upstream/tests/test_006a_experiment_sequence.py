"""Plan 006a: Experiment registry, completion contracts, numbered entrypoints and leaf gate."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.content_chain import derive_chain_commitment, derive_chain_receipt
from sglib.core.infra.hashing import sha256_json
from sglib.experiment import manifest as experiment_manifest
from sglib.experiment import stage
from sglib.experiment.config import load_experiment_config, registration_projection, registered_values_match
from sglib.experiment.registry import (ExperimentUnit, build_registry, coordinate_root, expected_node_id, planning_progress,
                                       topological_order, unit_status)

pytestmark = pytest.mark.gate


ROOT = Path(__file__).resolve().parents[1]
COUNTRIES = {"uk": "1_UK", "au": "2_AU", "nl": "4_NL", "nz": "5_NZ"}
NOTEBOOKS = ("01_preflight.ipynb", "06_audit.ipynb")
SCRIPTS = ("02_observe.py", "03_planning.py", "04_bounds.py", "05_defense.py")


@pytest.fixture(scope="module")
def loaded():
    return {cc: load_experiment_config(ROOT, cc) for cc in COUNTRIES}


@pytest.fixture(scope="module")
def candidates():
    return json.loads((ROOT / "casestudy/2_Generator/general/candidate_registry.json").read_text(encoding="utf-8"))


def receipt_for(unit, config, root, field=None, seed=None, parameters=None, outputs=None):
    node = expected_node_id(unit, config, field, seed)
    kind = {"preflight": {"connection": "preflight_connection", "planning_pool": "preflight_planning_pool"}.get(unit.member),
            "observe": "observe", "planning": "planning", "bounds": "bounds", "defense": "defense"}[unit.step]
    registered = deepcopy(config.registrations[kind]) if parameters is None else parameters
    commitment = derive_chain_commitment(node, inputs={"x": "a" * 64}, scientific_parameters=registered, code_sha256="b" * 64)
    target = coordinate_root(unit, root, config, field, seed) if unit.step == "planning" else \
        (root / "observations" / unit.member if unit.step == "observe" else root / unit.step if unit.step != "preflight" else root / "preflight" / unit.member)
    target.mkdir(parents=True, exist_ok=True)
    outputs = outputs or {"table.csv": "c" * 64}
    for name in outputs:
        (target / name).write_bytes(b"x")
    atomic_json(derive_chain_receipt(commitment, outputs=outputs), target / "receipt.json")
    return target / "receipt.json"


def test_registry_matches_sealed_matrix(loaded, candidates):
    expected = {"uk": (16, 368), "au": (12, 276), "nl": (16, 368), "nz": (9, 207)}
    for cc, config in loaded.items():
        units = build_registry(cc, config, candidates)
        observe = [u for u in units.values() if u.step == "observe"]
        planning = [u for u in units.values() if u.step == "planning"]
        assert len(observe) == expected[cc][0]
        assert sum(len(u.coordinates) for u in planning) == expected[cc][1]
        order = [u.id for u in topological_order(units)]
        assert order[-1] == f"{cc}.audit.audit"
        assert order.index(f"{cc}.preflight.connection") < order.index(f"{cc}.observe.{config.values['regions'][0]}")
        assert order.index(f"{cc}.bounds.c6") > max(order.index(u.id) for u in observe)


def test_registered_values_are_country_invariant_and_projected(loaded):
    for kind in ("observe", "planning", "bounds"):
        values = {cc: sha256_json({k: v for k, v in c.registrations[kind].items() if k != "connection_unit"}) for cc, c in loaded.items()}
        assert len(set(values.values())) == 1, kind
    big = {"specification": {"expected_coordinates": list(range(100))}}
    projected = registration_projection("defense", big)
    assert "expected_coordinates" not in projected["specification"]
    assert projected["specification"]["expected_coordinates_sha256"] == sha256_json(list(range(100)))


def test_unit_states_from_receipts(tmp_path, loaded, candidates):
    config = loaded["nz"]
    units = build_registry("nz", config, candidates)
    region = config.values["regions"][0]
    observe = units[f"nz.observe.{region}"]
    assert unit_status(observe, tmp_path, config) == "PENDING"
    path = receipt_for(observe, config, tmp_path)
    assert unit_status(observe, tmp_path, config) == "DONE"
    (path.parent / "table.csv").unlink()
    assert unit_status(observe, tmp_path, config) == "INVALID"
    path.write_text("{", encoding="utf-8")
    assert unit_status(observe, tmp_path, config) == "INVALID"


def test_registration_mismatch_and_wrong_node_are_invalid(tmp_path, loaded, candidates):
    config = loaded["nz"]
    units = build_registry("nz", config, candidates)
    bounds = units["nz.bounds.c6"]
    parameters = deepcopy(config.registrations["bounds"])
    parameters["etas"] = [0.1]
    receipt_for(bounds, config, tmp_path, parameters=parameters)
    assert registered_values_match("bounds", config.registrations, parameters) == ["etas"]
    assert unit_status(bounds, tmp_path, config) == "INVALID"
    other = ExperimentUnit("nz.bounds.c6", "bounds", "uk", "c6", ())
    receipt_for(other, config, tmp_path / "wrong")
    assert unit_status(bounds, tmp_path / "wrong", config) == "INVALID"


def test_planning_unit_completes_only_with_every_coordinate(tmp_path, loaded, candidates):
    config = loaded["nz"]
    units = build_registry("nz", config, candidates)
    planning = next(u for u in units.values() if u.step == "planning")
    assert unit_status(planning, tmp_path, config) == "PENDING"
    for field, seed in planning.coordinates[:-1]:
        receipt_for(planning, config, tmp_path, field, seed)
    assert unit_status(planning, tmp_path, config) == "PENDING"
    assert planning_progress(planning, tmp_path, config) == (len(planning.coordinates) - 1, len(planning.coordinates))
    field, seed = planning.coordinates[-1]
    receipt_for(planning, config, tmp_path, field, seed)
    assert unit_status(planning, tmp_path, config) == "DONE"


def test_numbered_step_never_executes_missing_predecessor(tmp_path, loaded, candidates, monkeypatch):
    config = loaded["nz"]
    units = build_registry("nz", config, candidates)
    ctx = SimpleNamespace(units=units, root=tmp_path, results=tmp_path, loaded=config, read_only=False,
                          backend="local", profile="smoke", regions=None, limit=None)
    monkeypatch.setattr(stage, "_execute", lambda *args: pytest.fail("execution must not start"))
    with pytest.raises(stage.StageOrderError, match="earlier numbered steps"):
        stage.run_units(ctx, ["nz.bounds.c6"])
    assert not list(tmp_path.iterdir())


def test_hpc_backend_only_prepares(tmp_path, loaded, candidates, capsys):
    config = loaded["nz"]
    units = build_registry("nz", config, candidates)
    region = config.values["regions"][0]
    receipt_for(units["nz.preflight.connection"], config, tmp_path)
    ctx = SimpleNamespace(units=units, root=tmp_path, results=tmp_path, loaded=config, read_only=False,
                          backend="hpc", profile="smoke", regions=[region], limit=None, upstream={})
    report = stage.run_units(ctx, [f"nz.observe.{region}"])
    assert report.prepared == (f"nz.observe.{region}",) and report.ran == ()
    assert unit_status(units[f"nz.observe.{region}"], tmp_path, config) == "PREPARED"
    assert "PREPARED nz.observe" in capsys.readouterr().out


def test_gate_leaf_is_the_unit(tmp_path, loaded, candidates):
    config = loaded["nz"]
    units = build_registry("nz", config, candidates)
    region = config.values["regions"][0]
    receipt_for(units[f"nz.observe.{region}"], config, tmp_path)
    receipt_for(units["nz.bounds.c6"], config, tmp_path)
    (tmp_path / "stray.txt").write_bytes(b"?")
    current, unclaimed = experiment_manifest.enumerate_leaves(tmp_path, config.values["regions"])
    assert current[f"observe/{region}"] == [f"observations/{region}/receipt.json", f"observations/{region}/table.csv"]
    assert current["bounds"] == ["bounds/receipt.json", "bounds/table.csv"]
    assert unclaimed == ["stray.txt"]


@pytest.mark.parametrize("country", sorted(COUNTRIES))
def test_entrypoints_are_numbered_and_inject_upstream(country):
    folder = ROOT / "casestudy/3_Experiment" / COUNTRIES[country]
    for name in NOTEBOOKS:
        document = json.loads((folder / name).read_text(encoding="utf-8"))
        code = "".join("".join(cell["source"]) for cell in document["cells"] if cell["cell_type"] == "code")
        assert f'COUNTRY = "{country}"' in code and "stage.run_step(" in code and "upstream=UPSTREAM" in code
        assert all(cell.get("outputs", []) == [] for cell in document["cells"])
        if name == "06_audit.ipynb":
            assert "run_gate(ctx)" in code and "raise SystemExit(1)" in code
    for name in SCRIPTS:
        source = (folder / name).read_text(encoding="utf-8")
        assert f'"{country}"' in source and "stage.run_step(" in source and "upstream=UPSTREAM" in source
    assert sorted(p.name for p in folder.iterdir() if p.suffix in {".py", ".ipynb"}) == sorted(NOTEBOOKS + SCRIPTS)


def test_experiment_package_imports_no_other_stage():
    import ast
    for path in (ROOT / "sglib/experiment").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [node.module] if isinstance(node, ast.ImportFrom) and node.module else \
                [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            assert not any(n.startswith(("sglib.dataoverview", "sglib.generator", "sglib.analysis", "sglib.report")) for n in names), path
