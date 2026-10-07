from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point
from sglib.core.infra.content_chain import derive_chain_commitment
from sglib.generator import execution
from sglib.generator.config import load_generator_config, scientific_task_matrix_projection, training_params
from sglib.generator.weighter.learned.inference.engine import (
    PreparedInferenceTask,
    _validate_inference_relative_identity,
)
from sglib.generator.stage_chain import training_preflight
from sglib.generator.weighter.learned.training.engine import (
    EXPECTED_GROUPS,
    EXPECTED_LOGICAL_COORDINATES,
    EXPECTED_REUSE_COORDINATES,
    EXPECTED_TRAIN_COORDINATES,
    load_prepared_task,
    load_task_matrix,
    prepare_tasks,
    tasks_for_group,
    validate_task_matrix,
)
from sglib.core.infra.hashing import sha256_file
pytestmark = pytest.mark.consume
ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "casestudy/2_Generator"
REGISTRY = GENERATOR / "general/candidate_registry.json"
MATRIX = GENERATOR / "general/training_task_matrix_scientific_v2.csv"
LEGACY_MATRIX = GENERATOR / "general/training_task_matrix.csv"


def test_matrix_supports_two_country_transition_and_four_country_contract() -> None:
    formal_matrix = load_task_matrix(MATRIX)
    matrix = tuple(
        record for record in formal_matrix if record.country in {"uk", "au"}
    )
    params = training_params(
        load_generator_config(
            ROOT,
            GENERATOR / "general/generator.toml",
            ROOT / "casestudy/config/countries/uk.toml",
            GENERATOR / "1_UK/uk.toml",
        )
    )
    transition = validate_task_matrix(
        matrix,
        frozen_params_by_country={"uk": params, "au": params},
    )
    assert (
        transition.groups,
        transition.logical_coordinates,
        transition.reuse_coordinates,
        transition.train_coordinates,
    ) == (22, 252, 12, 240)

    formal = validate_task_matrix(
        formal_matrix,
        frozen_params_by_country={code: params for code in ("uk", "au", "nl", "nz")},
    )
    assert (
        EXPECTED_GROUPS,
        EXPECTED_LOGICAL_COORDINATES,
        EXPECTED_REUSE_COORDINATES,
        EXPECTED_TRAIN_COORDINATES,
    ) == (44, 504, 24, 480)
    assert (
        formal.groups,
        formal.logical_coordinates,
        formal.reuse_coordinates,
        formal.train_coordinates,
    ) == (44, 504, 24, 480)


def test_g1_r2_matrix_projection_excludes_operations_without_changing_science() -> None:
    assert scientific_task_matrix_projection(
        LEGACY_MATRIX
    ) == scientific_task_matrix_projection(MATRIX)
    header = MATRIX.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    assert not {
        "gres",
        "cpus",
        "time_limit",
        "max_concurrent",
        "execution_policy",
        "array_formula",
        "output_pattern",
    } & set(header)
    legacy = load_task_matrix(LEGACY_MATRIX)
    current = load_task_matrix(MATRIX)
    assert [item.output_pattern for item in legacy] == [
        item.output_pattern for item in current
    ]
    assert [item.array_formula for item in legacy] == [
        item.array_formula for item in current
    ]


def test_inference_country_validation_accepts_any_lowercase_two_letter_code() -> None:
    fingerprints = {
        name: "a" * 64
        for name in ("checkpoint", "graph_cache", "frozen_params")
    }
    parameters = {
        "training_task_id": "B-NL-GNN-S42-F1",
        "country": "nl",
        "group": "B-NL-GNN",
        "family": "gnn",
        "config": "baseline",
        "signal": "base",
        "parameter": "fixed",
        "value": "base",
        "seed": 42,
        "fold": 1,
        "feature_set": "lu5",
        "selector": "outer_train_min_training_loss",
        "regions": ["R1"],
        "compute_threads": 8,
        "run_fingerprint": "b" * 64,
    }
    commitment = derive_chain_commitment(
        "generator.inference.B-NL-GNN-S42-F1",
        inputs=fingerprints,
        scientific_parameters=parameters,
        code_sha256="c" * 64,
    )
    task = PreparedInferenceTask(
        country="nl",
        group="B-NL-GNN",
        training_task_id="B-NL-GNN-S42-F1",
        family="gnn",
        config="baseline",
        signal="base",
        parameter="fixed",
        value="base",
        seed=42,
        fold=1,
        feature_set="lu5",
        repo_root=str(ROOT),
        results_root=str(ROOT / "results"),
        training_task_path=str(ROOT / "results/task.json"),
        checkpoint_path=str(ROOT / "results/model.pth"),
        frozen_params_path=str(ROOT / "params.json"),
        graph_cache_path=str(ROOT / "results/cache.pkl"),
        output_path=str(ROOT / "results/out"),
        training_task_results_relative="task.json",
        checkpoint_results_relative="model.pth",
        frozen_params_repo_relative="params.json",
        cache_results_relative="cache.pkl",
        output_results_relative="out",
        selector="outer_train_min_training_loss",
        fingerprints=fingerprints,
        regions=("R1",),
        execution_backend="local",
        execution_identity={"run_identity": {"run_fingerprint": "b" * 64}},
        chain_commitment=commitment,
    )
    _validate_inference_relative_identity(task)
    assert not {
        "repo_root",
        "results_root",
        "training_task_path",
        "checkpoint_path",
        "frozen_params_path",
        "graph_cache_path",
        "output_path",
        "execution_backend",
    } & set(task.to_dict())
    with pytest.raises(Exception, match="country/backend"):
        _validate_inference_relative_identity(replace(task, country="nld"))


def test_local_inference_preflight_is_read_only_and_requires_full_training(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results/v3_final"
    before = list(tmp_path.rglob("*"))
    status = training_preflight(ROOT, results)
    assert status["ready"] is False
    assert status["expected_training_groups"] == 44
    assert status["expected_training_coordinates"] == 480
    assert status["completed_training_coordinates"] == 0
    assert list(tmp_path.rglob("*")) == before


def test_prepared_training_task_serialization_is_machine_portable(
    tmp_path: Path,
) -> None:
    matrix = load_task_matrix(MATRIX)
    coordinate = tasks_for_group(
        matrix, country="au", group="B-AU-GNN"
    )[0]
    repo = tmp_path / "repo"
    results = repo / "results/v3"
    params = repo / "params.json"
    graph = results / "2_Generator/2_AU/training/graphs/lu5.pkl"
    params.parent.mkdir(parents=True)
    graph.parent.mkdir(parents=True)
    params.write_text("{}", encoding="utf-8")
    graph.write_bytes(b"graph")
    prepared = prepare_tasks(
        [coordinate],
        frozen_params_path=params,
        graph_cache_by_feature_set={"lu5": graph},
        repo_root=repo,
        results_root=results,
        config_fingerprint=sha256_file(params),
        input_fingerprints={"inputs_scientific_fingerprint": "a" * 64},
        execution_backend="hpc",
    )[0]
    document = prepared.to_dict()
    absolute_fields = {
        "repo_root",
        "results_root",
        "frozen_params_path",
        "graph_cache_path",
        "output_path",
    }
    assert not absolute_fields & set(document)
    task_path = results / "task.json"
    task_path.write_text(json.dumps(document), encoding="utf-8")
    rebased = load_prepared_task(
        task_path, repo_root=repo, results_root=results
    )
    assert Path(rebased.repo_root) == repo.resolve()
    assert Path(rebased.output_path) == (
        results / coordinate.output_relative
    ).resolve()


def test_training_preparation_validates_country_and_exact_overlay_groups(
    tmp_path: Path, monkeypatch
) -> None:
    from sglib.generator.weighter.learned.training import matrix as task_api

    bundle = SimpleNamespace(
        country="nl",
        params={
            "authorities": {"training_task_matrix": "matrix.csv"},
            "task_groups": ["B-NL-GNN", "B-NL-MLP"],
        },
    )
    matrix = (
        SimpleNamespace(country="nl", task_group="B-NL-GNN"),
        SimpleNamespace(country="nl", task_group="P-NL-N"),
    )
    observed = {}
    monkeypatch.setattr(execution, "_load_bundle", lambda root: bundle)
    monkeypatch.setattr(task_api, "load_task_matrix", lambda path: matrix)
    monkeypatch.setattr(
        task_api,
        "validate_country_task_matrix",
        lambda loaded, country, params: observed.update(
            matrix=loaded, country=country, params=Path(params)
        ),
    )
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/worker_params.json").write_text("{}", encoding="utf-8")

    with pytest.raises(execution.GeneratorExecutionError, match="overlay task_groups"):
        execution.prepare_training_group(
            tmp_path,
            repo_root=tmp_path,
            results_root=tmp_path,
            group="B-NL-GNN",
            execution_backend="local",
        )
    assert observed == {
        "matrix": matrix,
        "country": "nl",
        "params": tmp_path / "inputs/worker_params.json",
    }


def _formal_bundle_fixture(tmp_path: Path, n_sources: int):
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "data/metadata.toml").write_text("schema_version='fixture'", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0'", encoding="utf-8")
    profile_path = tmp_path / "casestudy/config/countries/nl.toml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("fixture=true", encoding="utf-8")
    authority_path = tmp_path / "casestudy/config/authority/engineering_admission.toml"
    authority_path.parent.mkdir(parents=True)
    shutil.copy2(
        ROOT / "casestudy/config/authority/engineering_admission.toml",
        authority_path,
    )
    contract_path = tmp_path / "casestudy/1_DataOverview/4_NL/admission.toml"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(
        """schema_version = "sg_country_engineering_admission_contract_v2"
status = "FROZEN"
[shared_authority]
path = "casestudy/config/authority/engineering_admission.toml"
schema_version = "sg_engineering_admission_authority_v2"
[idr_matched]
formal_candidates = 38
seeds = 3
max_realizations_all_regions = 2000
""",
        encoding="utf-8",
    )
    source_ids = [f"S{index}" for index in range(n_sources)]
    grid_ids = np.repeat(source_ids, 4)
    grid = gpd.GeoDataFrame(
        {"source": grid_ids},
        geometry=[Point(float(index % 10), float(index // 10)) for index in range(len(grid_ids))],
        crs="EPSG:3857",
    )
    sources = gpd.GeoDataFrame(
        {"source": source_ids},
        geometry=[Point(0, 0)] * n_sources,
        crs="EPSG:3857",
    )
    targets = gpd.GeoDataFrame(
        {"station_id": ["T0"]}, geometry=[Point(0, 0)], crs="EPSG:3857"
    )
    dry = {
        "region": "R",
        "n_sources": n_sources,
        "n_targets": 1,
        "n_cells": len(grid),
        "min_cells_per_source": 4,
        "min_cells_per_target": len(grid),
        "empty_targets": 0,
        "total_edges_directed": len(grid) * 2,
        "active_agent_nodes_upper_bound": len(grid),
        "cdist_peak_workspace_mib": 1.0,
    }
    evidence = SimpleNamespace(
        document={"regions": [dry]},
        repo_relative="data/admission.json",
        sha256="e" * 64,
    )
    handoff = SimpleNamespace(
        profile=SimpleNamespace(source_path=profile_path),
        evidence={"engineering_admission": evidence},
    )
    bundle = SimpleNamespace(
        country="nl",
        regions=("R",),
        source_column="source",
        grids={"R": (grid, 1.0)},
        source_regions={"R": sources},
        stations={"R": targets},
        region_inputs={
            "R": SimpleNamespace(covered_mask=np.ones(len(grid), dtype=bool))
        },
        params={"crs": {"working": "EPSG:3857"}},
    )
    config = {
        "checks": {
            "requires_engineering_admission": True,
            "admission_contract": "casestudy/1_DataOverview/4_NL/admission.toml",
            "engineering_admission_evidence": "engineering_admission",
        }
    }
    bundle_path = tmp_path / "bundle.pkl"
    bundle_path.write_bytes(b"bundle")
    return handoff, bundle, config, bundle_path


def test_local_formal_real_active_preflight_passes_and_fails_dense_limit(
    tmp_path: Path,
) -> None:
    handoff, bundle, config, bundle_path = _formal_bundle_fixture(tmp_path / "pass", 1)
    result = execution._formal_engineering_pre_submission(
        handoff, bundle, config, bundle_path=bundle_path
    )
    assert result and result["status"] == "PASS"
    assert result["regions"][0]["active_agent_nodes_observed"] == 4

    handoff, bundle, config, bundle_path = _formal_bundle_fixture(
        tmp_path / "over_limit", 1000
    )
    with pytest.raises(execution.GeneratorExecutionError, match="max_dense_tensor_real_active_mib"):
        execution._formal_engineering_pre_submission(
            handoff, bundle, config, bundle_path=bundle_path
        )
