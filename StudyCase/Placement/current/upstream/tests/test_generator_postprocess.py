from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from sglib.generator import execution
from sglib.generator.config import load_generator_config, scientific_config_fingerprint, scientific_config_projection
from sglib.generator.registry import build_registry
from sglib.generator.weighter.candidates import (
    CandidateRegistryError,
    load_candidate_registry,
    validate_candidate_registry,
)
from sglib.core.infra.hashing import sha256_file
pytestmark = pytest.mark.consume
ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "casestudy/2_Generator"
REGISTRY = GENERATOR / "general/candidate_registry.json"
MATRIX = GENERATOR / "general/training_task_matrix_scientific_v2.csv"
LEGACY_MATRIX = GENERATOR / "general/training_task_matrix.csv"


def test_candidate_country_contract_accepts_four_unique_uppercase_codes() -> None:
    document = load_candidate_registry(REGISTRY)
    document["active_countries"] = ["UK", "AU", "NL", "NZ"]
    assert validate_candidate_registry(document)["active_countries"] == [
        "UK",
        "AU",
        "NL",
        "NZ",
    ]

    for invalid in (["UK", "UK"], ["UK", "nl"], ["UK", "NLD"]):
        document["active_countries"] = invalid
        with pytest.raises(CandidateRegistryError, match="unique uppercase"):
            validate_candidate_registry(document)


def _country_config(code: str, directory: str):
    return load_generator_config(
        ROOT,
        GENERATOR / "general/generator.toml",
        ROOT / f"casestudy/config/countries/{code}.toml",
        GENERATOR / f"{directory}/{code}.toml",
    )


def test_registry_omits_civd_and_its_audit_dependency_when_disabled() -> None:
    candidates = load_candidate_registry(REGISTRY)
    units = build_registry(
        "nl",
        {
            "execution": {"civd_enabled": False},
            "task_groups": [],
            "sweeps": [],
        },
        candidates,
    )
    assert "nl.civd.civd" not in units
    assert all(".civd." not in item for item in units["nl.audit.audit"].depends_on)


def test_postprocess_dag_separates_family_writes_and_actual_sweep_inputs() -> None:
    from sglib.generator.chain import context, STEPS

    _, _, units, _ = context(ROOT, ROOT / "results", "nl")
    assert len([u for u in units.values() if u.step in STEPS]) == 21  # CIVD enabled for NL since 2026-09-18
    assert "nl.static.gpm" in units["nl.materialize.MLP"].depends_on
    assert "nl.infer.T-NL" not in units["nl.materialize.GNN"].depends_on
    assert set(units["nl.sweeps.lambda"].depends_on) == {
        "nl.inputs.bundle", "nl.infer.L-NL-N", "nl.infer.L-NL-P", "nl.infer.P-NL-N", "nl.infer.P-NL-P"}
    assert "nl.finalize.candidate_index" in units["nl.idr_matched.idr_matched"].depends_on
    assert len(units["nl.finalize.candidate_index"].depends_on) == 5


def test_postprocess_output_identity_ignores_locator_but_rejects_changed_content(tmp_path: Path) -> None:
    from sglib.generator.chain import output_view
    from sglib.generator.registry import GeneratorUnit

    root = tmp_path / "root"
    marker = root / "static/uniform/index.json"
    marker.parent.mkdir(parents=True)
    artifact = root / "one.npz"
    np.savez(artifact, data=np.array([1., 2.]))
    record = {"region": "r", "path": "one.npz", "sha256": sha256_file(artifact), "bytes": artifact.stat().st_size}
    marker.write_text(json.dumps({"regions": [record]}), encoding="utf-8")
    unit = GeneratorUnit("uk.static.uniform", "static", "uk", "uniform", ())
    before, _ = output_view(root, unit)
    artifact.rename(root / "relocated.npz")
    record["path"] = "relocated.npz"
    marker.write_text(json.dumps({"regions": [record]}), encoding="utf-8")
    assert output_view(root, unit)[0] == before
    (root / "relocated.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="内容不符"):
        output_view(root, unit)


def test_static_identifiers_are_readable_without_pickle(tmp_path: Path, monkeypatch) -> None:
    import pandas as pd

    grid = pd.DataFrame({"source": ["s", "s"]})
    bundle = SimpleNamespace(country="uk", regions=("r",), source_column="source",
        params={"features": {}, "crs": {"working": "EPSG:27700"}},
        grids={"r": (grid, 1.)}, source_regions={"r": pd.DataFrame()},
        stations={"r": pd.DataFrame({"station_id": ["001", "002"]})})
    monkeypatch.setattr(execution, "_load_bundle", lambda root: bundle)
    monkeypatch.setattr(execution, "materialize_assignment", lambda *a, **k: np.array([0, 1]))
    monkeypatch.setattr(execution, "public_activity_field", lambda *a, **k: (np.array([1., 1.]), 0.))
    (tmp_path / "static/gpm").mkdir(parents=True)
    np.savez(tmp_path / "static/gpm/r.npz", data=np.array([1., 1.]))
    for component, name in (("assignments", "station_id"), ("public_activity", "source_keys")):
        execution.generate_static_component(tmp_path, component)
        with np.load(tmp_path / "static" / component / "r.npz", allow_pickle=False) as arrays:
            assert arrays[name].dtype.kind in "US"


def test_static_and_idr_indexes_record_working_crs(tmp_path: Path, monkeypatch) -> None:
    class Bundle:
        country = "nl"
        regions = ()
        params = {
            "crs": {"working": "EPSG:28992"},
            "features": {},
            "station_contract": {},
            "idr": {"b_tv": 0.10},
        }

    monkeypatch.setattr(execution, "_load_bundle", lambda root: Bundle())
    (tmp_path / "candidates").mkdir()
    (tmp_path / "candidates/candidate_index.json").write_text(
        '{"entries": []}', encoding="utf-8"
    )
    paths = (
        execution.generate_static_component(tmp_path, "uniform"),
        execution.run_civd(tmp_path),
        execution.run_idr_fixed(tmp_path),
        execution.run_idr_matched(tmp_path),
    )
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["working_crs"]
        == "EPSG:28992"
        for path in paths
    )


def test_civd_protocol_is_registered_and_enters_the_scientific_identity() -> None:
    from sglib.generator.handoff import CIVD_METHOD_KEYS, GeneratorHandoffError, civd_protocol, verify_civd_index

    loaded = _country_config("nl", "4_NL")
    values = dict(loaded.values)
    assert values["execution"]["civd_enabled"] is True
    protocol = civd_protocol(values)
    assert set(CIVD_METHOD_KEYS) < set(protocol)
    assert protocol["country"] == "nl"
    assert protocol["working_crs"] == values["crs"]["working"]
    assert protocol["capacity_column"] == values["station_contract"]["capacity_column"]
    assert protocol["capacity_basis"] == values["station_contract"]["capacity_basis"]
    assert "civd" in scientific_config_projection(loaded, repo_root=ROOT)
    changed = deepcopy(values)
    changed["civd"] = {**changed["civd"], "noise_rule": "drop_noise_stations"}
    assert scientific_config_fingerprint(changed, repo_root=ROOT) != scientific_config_fingerprint(loaded, repo_root=ROOT)

    entry = {"region": "R", "path": "civd/R.npz", "sha256": "0" * 64, "protocol": {**protocol, "region": "R"}}
    index = {"schema_version": "sg_civd_extension_index_v1", "country": "nl",
             "working_crs": values["crs"]["working"], "protocol": protocol, "entries": [entry]}
    assert verify_civd_index(index, values) == "sg_civd_extension_index_v1"
    legacy = {"schema_version": "sg_civd_index_v1", "country": "nl", "working_crs": values["crs"]["working"], "entries": []}
    assert verify_civd_index(legacy, values) == "sg_civd_index_v1"
    with pytest.raises(GeneratorHandoffError, match=r"registered protocol at \['weight_rule'\]"):
        verify_civd_index({**index, "protocol": {**protocol, "weight_rule": "uniform"}}, values)
    with pytest.raises(GeneratorHandoffError, match="entry R protocol differs"):
        verify_civd_index({**index, "entries": [{**entry, "protocol": {**entry["protocol"], "capacity_floor": 0.0}}]}, values)
    with pytest.raises(GeneratorHandoffError, match="working CRS"):
        verify_civd_index({**index, "working_crs": "EPSG:4326"}, values)
    with pytest.raises(GeneratorHandoffError, match="unknown CIVD index schema"):
        verify_civd_index({**index, "schema_version": "sg_civd_index_v9"}, values)


def test_run_civd_writes_the_extension_index_when_a_protocol_is_registered(tmp_path: Path, monkeypatch) -> None:
    import numpy as np
    from sglib.generator.handoff import civd_protocol

    loaded = _country_config("uk", "1_UK")

    import pandas as pd
    from sglib.core.infra.schema import validate_artifact

    class Bundle:
        country = "uk"
        regions = ("R",)
        params = dict(loaded.values)
        grids = {"R": (object(),)}
        stations = {"R": pd.DataFrame({"station_id": ["s1", "s2"]})}

    result = {"assignment": np.zeros(3, dtype=np.int64), "grid_cluster": np.zeros(3, dtype=np.int64),
              "station_cluster": np.zeros(2, dtype=np.int64), "raw_labels": np.zeros(2, dtype=np.int64),
              "probabilities": np.ones(2), "n_clusters": 1, "n_noise": 0}
    monkeypatch.setattr(execution, "_load_bundle", lambda root: Bundle())
    monkeypatch.setattr(execution, "materialize_civd", lambda *a, **k: result)
    document = json.loads(execution.run_civd(tmp_path).read_text(encoding="utf-8"))
    assert document["schema_version"] == "sg_civd_extension_index_v1"
    assert document["protocol"] == civd_protocol(loaded.values)
    assert document["entries"][0]["protocol"] == {**civd_protocol(loaded.values), "region": "R"}
    validate_artifact(tmp_path / "civd/R.npz", GENERATOR / "schemas/civd.toml")
    with np.load(tmp_path / "civd/R.npz", allow_pickle=False) as archive:
        assert archive["station_id"].tolist() == ["s1", "s2"]
