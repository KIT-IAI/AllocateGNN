"""Run the NZ core-9 structural pre-HPC smoke.

Real 2024/2025/2023 core-9 handoff tables are used.  Only the high-resolution
OSM/GHSL/NTL layer is replaced by an explicitly labelled lightweight fixture.
Nothing produced by this module is formal evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import geopandas as gpd
import numpy as np

from .common import check_smoke_output, write_fixture_inventory, publish_static_candidate_indexes
from .verification import artifact_manifest, verify_smoke_products

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.config import load_dataoverview_config
from sglib.core.infra.hashing import sha256_file
from sglib.core.infra.paths import resolve_case_path
from sglib.dataoverview.handoff import DataOverviewBundle, RegionData
from sglib.dataoverview.processing.derive.nz.core9 import (
    REGION_ORDER,
    SOURCE_FEATURES,
    build_core9_dataoverview_from_fresh,
    build_size_gate,
)
from sglib.generator.config import load_generator_config, training_params
from sglib.generator.execution import (
    generate_static_component,
    materialize_family,
    prepare_inputs,
    run_audit,
    run_idr_fixed,
    run_civd,
    run_idr_matched,
)
from sglib.generator.generation import input_identity_binding, derive_run_identity
from sglib.generator.weighter.candidates import load_candidate_registry
from sglib.generator.weighter.learned.inputs import build_graphs, save_graph_cache
from sglib.generator.weighter.learned.training import TrainingTask, run_training_task
from sglib.generator.weighter.learned.training.preparation import prepare_tasks
from sglib.generator.weighter.learned.training.verify import verify_task_artifacts


FIXTURE_SCHEMA = "sg_nz_pre_hpc_structural_fixture_v1"
SMOKE_REGIONS = REGION_ORDER[:4]


def _load_dataoverview_config(repo_root: Path):
    return load_dataoverview_config(
        repo_root / "casestudy/1_DataOverview/general/general.toml",
        repo_root / "casestudy/config/countries/nz.toml",
        repo_root / "casestudy/1_DataOverview/5_NZ/nz.toml",
    )


def _source_row_features(row) -> np.ndarray:
    # Generator agent columns use residential, commercial, industrial,
    # agricultural, others, unknown.  Census shares are real, but distributing
    # them over this tiny grid is only a structural fixture.
    values = np.asarray(
        [
            row.residential_percent,
            row.commercial_percent,
            row.industrial_percent,
            row.agricultural_percent,
            row.others_percent,
        ],
        dtype=np.float64,
    )
    total = float(values.sum())
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("NZ source features must be finite and non-negative")
    if total <= 0:
        values = np.asarray([0, 0, 0, 0, 1], dtype=np.float64)
    else:
        values /= total
    return np.append(values, 0.0)


def _fixture_region(
    name: str,
    source: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    source_order: list[str],
) -> tuple[RegionData, dict]:
    source = source.copy()
    source["SA22023_V1_00"] = source["SA22023_V1_00"].astype(str)
    source_by_key = source.set_index("SA22023_V1_00", drop=False)
    if list(source_by_key.loc[source_order, "SA22023_V1_00"]) != source_order:
        raise ValueError(f"{name}: source-order authority differs from the real handoff")
    projected = source_by_key.loc[source_order].to_crs("EPSG:2193")
    representatives = projected.geometry.representative_point()
    representatives = gpd.GeoSeries(representatives, crs="EPSG:2193").to_crs("EPSG:4326")
    targets = targets.copy()
    targets["SA22023_V1_00"] = targets["SA22023_V1_00"].astype(str)

    records: list[dict] = []
    feature_rows: list[np.ndarray] = []
    ntl_rows: list[float] = []
    for source_index, key in enumerate(source_order):
        source_row = source_by_key.loc[key]
        feature = _source_row_features(source_row)
        population = max(float(source_row["population_2023"]), 0.0)
        records.append(
            {
                "index_region": source_index,
                "SA22023_V1_00": key,
                "SA22023_V1_00_NAME": str(source_row["SA22023_V1_00_NAME"]),
                "analysis_region": name,
                "fixture_point_role": "source_representative",
                "geometry": representatives.iloc[source_index],
            }
        )
        feature_rows.append(feature)
        ntl_rows.append(float(np.log1p(population)))
        for target in targets.loc[targets["SA22023_V1_00"].eq(key)].itertuples(index=False):
            records.append(
                {
                    "index_region": source_index,
                    "SA22023_V1_00": key,
                    "SA22023_V1_00_NAME": str(source_row["SA22023_V1_00_NAME"]),
                    "analysis_region": name,
                    "fixture_point_role": "site_occupancy_probe",
                    "geometry": target.geometry,
                }
            )
            feature_rows.append(feature.copy())
            ntl_rows.append(float(np.log1p(population)))
    grid = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    features = np.asarray(feature_rows, dtype=np.float64)
    source_keys = grid["SA22023_V1_00"].astype(str).to_numpy()
    support = {
        "features": features,
        "built_fraction": np.ones(len(grid), dtype=np.float64),
        "covered_mask": np.ones(len(grid), dtype=bool),
        "unknown_mask": np.zeros(len(grid), dtype=bool),
        "zero_mask": np.zeros(len(grid), dtype=bool),
        "source_keys": source_keys,
        "source_key_order": np.asarray(source_order, dtype=str),
    }
    metadata = {
        "schema_version": FIXTURE_SCHEMA,
        "grid_policy": "pre_hpc_structural_fixture_not_formal",
        "working_crs": "EPSG:2193",
        "generation_crs": "EPSG:3857",
        "storage_crs": "EPSG:4326",
        "target_ground_step_m": 100.0,
        "projected_step_m": 100.0,
        "n_cells": len(grid),
        "n_sources": len(source),
        "n_targets": len(targets),
        "source_key": "SA22023_V1_00",
        "source_key_order": source_order,
        "formal_feature_layers": False,
        "fixture_point_roles": ["source_representative", "site_occupancy_probe"],
    }
    region = RegionData(
        region=name,
        grid=grid,
        grid_metadata=metadata,
        landuse={"data": features[:, :5]},
        built_surface={"data": np.ones((len(grid), 1), dtype=np.float64)},
        cuz_support=support,
        ntl={"data": np.asarray(ntl_rows, dtype=np.float64).reshape(-1, 1)},
    )
    record = {
        "region": name,
        "real_sources": len(source),
        "real_site_targets": len(targets),
        "fixture_cells": len(grid),
        "fixture_source_representatives": len(source),
        "fixture_site_occupancy_probes": len(targets),
    }
    return region, record


def reconcile_source_ids(name: str, rebuilt: list[str], registered: list[str]) -> None:
    """Reconcile a region's rebuilt source IDs with its registered source order.

    Every registered ID must be rebuilt and every rebuilt ID registered, each
    exactly once; counts of an earlier materialisation are not consulted.
    """

    problems = {
        "missing": sorted(set(registered) - set(rebuilt)),
        "unregistered": sorted(set(rebuilt) - set(registered)),
        "duplicate_rebuilt": sorted({key for key in rebuilt if rebuilt.count(key) > 1}),
        "duplicate_registered": sorted({key for key in registered if registered.count(key) > 1}),
    }
    problems = {kind: keys for kind, keys in problems.items() if keys}
    if problems:
        raise ValueError(f"{name}: rebuilt source IDs do not reconcile with source_key_order: {problems}")


def reconcile_site_coverage(sites: gpd.GeoDataFrame, covered: list[str]) -> None:
    """Every rebuilt site must be a target of exactly one registered region."""

    problems = {
        "uncovered": sorted(set(sites["station_id"].astype(str)) - set(covered)),
        "multiply_covered": sorted({key for key in covered if covered.count(key) > 1}),
    }
    problems = {kind: keys for kind, keys in problems.items() if keys}
    if problems:
        raise ValueError(f"rebuilt sites do not reconcile with the registered regions: {problems}")


def observed_handoff(gate_path: Path) -> dict:
    """Counts observed in this rebuild, read from its acceptance gate."""

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    return {
        "station_sites": int(gate["formal_truth_rows"]),
        "station_sections": int(gate["section_lineage_rows"]),
        "source_regions": int(gate["source_rows"]),
        "analysis_regions": int(gate["analysis_regions"]),
    }


def build_structural_handoff(repo_root: Path, dataoverview_root: Path) -> tuple[DataOverviewBundle, dict]:
    artifacts = build_core9_dataoverview_from_fresh(repo_root, dataoverview_root)
    sources = gpd.read_file(artifacts.sources, layer="source_regions")
    sites = gpd.read_file(artifacts.sites, layer="station_sites")
    loaded = _load_dataoverview_config(repo_root)
    items = {str(item["id"]): dict(item) for item in loaded.values["regions"]["items"]}
    unregistered_regions = sorted(set(sources["analysis_region"].astype(str)) - set(REGION_ORDER))
    if unregistered_regions:
        raise ValueError(f"rebuilt sources fall in unregistered regions: {unregistered_regions}")
    regions: list[RegionData] = []
    records: list[dict] = []
    covered: list[str] = []
    for name in REGION_ORDER:
        item = items[name]
        source = sources.loc[sources["analysis_region"].eq(name)].copy().reset_index(drop=True)
        registered = list(map(str, item["source_key_order"]))
        reconcile_source_ids(name, source["SA22023_V1_00"].astype(str).tolist(), registered)
        keys = set(registered)
        target = sites.loc[sites["SA22023_V1_00"].astype(str).isin(keys)].copy().reset_index(drop=True)
        covered.extend(target["station_id"].astype(str))
        region, record = _fixture_region(name, source, target, registered)
        regions.append(region)
        records.append(record)
    reconcile_site_coverage(sites, covered)
    inventory = {
        "schema_version": "sg_dataoverview_inventory_v1",
        "country": "nz",
        "country_directory": "5_NZ",
        "profile": "pre_hpc_structural_fixture",
        "formal": False,
        "real_handoff": {
            **observed_handoff(artifacts.gate),
            "gate_sha256": sha256_file(artifacts.gate),
        },
        "fixture_limit": "OSM/GHSL/NTL and high-resolution grid are not formal artifacts",
        "regions": records,
    }
    bundle = DataOverviewBundle(
        country="nz",
        profile=loaded.country_profile,
        regions_table=sources,
        stations_table=sites,
        regions=tuple(regions),
        inventory=inventory,
        evidence={},
        configuration=loaded,
    )
    return bundle, inventory


def _generator_config(repo_root: Path):
    return load_generator_config(
        repo_root,
        repo_root / "casestudy/2_Generator/general/generator.toml",
        repo_root / "casestudy/config/countries/nz.toml",
        repo_root / "casestudy/2_Generator/5_NZ/nz.toml",
    )


def run_smoke(repo_root: Path, results_root: Path, *, refresh: bool = False) -> Path:
    repo_root = repo_root.resolve()
    results_root = Path(results_root).resolve()
    dataoverview_root = results_root / "1_DataOverview/5_NZ"
    output_root = results_root / "2_Generator/5_NZ"
    receipt_path = output_root / "smoke_receipt.json"
    completed = check_smoke_output(
        repo_root, results_root, receipt_relative=receipt_path.relative_to(results_root).as_posix(),
        receipt_schema="sg_nz_pre_hpc_smoke_receipt_v1", country="nz", reuse=not refresh,
    )
    if completed is not None:
        return completed

    core9 = build_core9_dataoverview_from_fresh(repo_root, dataoverview_root, force=refresh)
    size_gate_path = build_size_gate(repo_root, dataoverview_root, force=refresh)
    handoff, fixture_inventory = build_structural_handoff(repo_root, dataoverview_root)
    atomic_json(fixture_inventory, dataoverview_root / "structural_fixture_manifest.json")
    handoff, inventory_path = write_fixture_inventory(handoff, output_root)

    config = _generator_config(repo_root)
    worker = training_params(config)
    worker["regions"] = list(SMOKE_REGIONS)
    worker["run_contract"] = {
        **worker["run_contract"],
        "profile": "pre_hpc_structural_fixture",
        "formal": False,
        "feature_scope": "real Census/source/station handoff; lightweight grid and feature fixture",
    }
    for spec in worker["config_map"].values():
        spec["epochs"] = 2
    prepare_inputs(
        handoff,
        config.values,
        output_root,
        selected_regions=list(SMOKE_REGIONS),
        worker_params=worker,
        require_formal_evidence=False,
        inventory_path=inventory_path,
    )
    with (output_root / "inputs/bundle.pkl").open("rb") as stream:
        import pickle

        generator_bundle = pickle.load(stream)
    for component in (
        "assignments",
        "uniform",
        "gpm",
        "proximity",
        "public_activity",
    ):
        generate_static_component(output_root, component)
    run_idr_fixed(output_root)

    candidates = load_candidate_registry(
        resolve_case_path(repo_root / config.values["authorities"]["candidate_registry"])
    )
    for family in ("Uni", "GPM", "Equal"):
        materialize_family(output_root, family, candidates)
    publish_static_candidate_indexes(output_root)
    run_idr_matched(output_root)
    run_civd(output_root)

    graphs = build_graphs(generator_bundle, feature_set="lu5", inject_priors=True)
    input_receipt = output_root / "inputs/receipt.json"
    input_binding = input_identity_binding(input_receipt)
    graph_path = save_graph_cache(
        output_root / "training/graphs/lu5.pkl",
        generator_bundle,
        graphs,
        feature_set="lu5",
        input_fingerprints={
            input_binding["field"]: input_binding["fingerprint"]
        },
    )
    task = TrainingTask(
        group="B-NZ-GNN",
        country="nz",
        family="gnn",
        config="baseline",
        signal="none",
        parameter="fixed",
        value="default",
        seed=42,
        fold=1,
        output_relative="2_Generator/5_NZ/2_Weighter/base/gnn/seed_42/fold1",
        feature_set="lu5",
    )
    worker_path = output_root / "inputs/worker_params.json"
    prepared = prepare_tasks(
        [task],
        frozen_params_path=worker_path,
        graph_cache_by_feature_set={"lu5": graph_path},
        repo_root=results_root,
        results_root=results_root,
        config_fingerprint=sha256_file(worker_path),
        input_fingerprints={
            input_binding["field"]: input_binding["fingerprint"]
        },
        execution_backend="local",
        execution_identity={
            "run_identity": derive_run_identity(output_root, config.values, repo_root=repo_root),
            "profile": "pre_hpc_structural_fixture",
            "formal": False,
            "real_core9_handoff": True,
        },
    )[0]
    task_path = atomic_json(
        prepared.to_dict(),
        output_root / "training/tasks/B-NZ-GNN/B-NZ-GNN-S42-F1.json",
    )
    atomic_json(
        {
            "schema_version": "sg_prepared_training_group_v1",
            "country": "nz",
            "group": "B-NZ-GNN",
            "execution_backend": "local",
            "profile": "pre_hpc_structural_fixture",
            "tasks": [{"task_id": prepared.task_id, "path": task_path.relative_to(output_root).as_posix(), "sha256": sha256_file(task_path)}],
        },
        output_root / "training/tasks/B-NZ-GNN/index.json",
    )
    result = run_training_task(prepared, backend="local", device="cpu")
    completion = verify_task_artifacts(prepared)
    atomic_json(
        {
            "schema_version": "sg_training_group_receipt_v1",
            "group": "B-NZ-GNN",
            "profile": "pre_hpc_structural_fixture",
            "tasks": [{"task_id": prepared.task_id, "selected_training_loss": completion["selected_training_loss"]}],
            "complete": True,
        },
        output_root / "training/receipts/B-NZ-GNN.json",
    )
    audit_path = run_audit(output_root, profile="smoke")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    size_gate = json.loads(size_gate_path.read_text(encoding="utf-8"))
    receipt = {
        "schema_version": "sg_nz_pre_hpc_smoke_receipt_v1",
        "status": "PASS",
        "country": "nz",
        "profile": "pre_hpc_structural_fixture",
        "formal": False,
        "real_core9_handoff": True,
        "real_inputs": {
            **observed_handoff(core9.gate),
            "nz_gate": {"path": core9.gate.relative_to(results_root).as_posix(), "sha256": sha256_file(core9.gate)},
        },
        "fixture_inputs": {
            "grid": "one real source representative plus one occupancy probe per real site",
            "landuse_and_built_surface": "deterministic structural fixture derived from real Census source shares",
            "ntl": "deterministic log1p(population_2023) proxy",
            "formal_osm_ghsl_ntl": False,
        },
        "selected_regions": list(SMOKE_REGIONS),
        "training": {
            "task_id": prepared.task_id,
            "epochs": completion["epochs_observed"],
            "device": completion["runtime"]["device_resolved"],
            "selected_training_loss": completion["selected_training_loss"],
            "completion_path": (result / "task_completion.json").relative_to(results_root).as_posix(),
        },
        "idr": {
            "fixed_entries": len(json.loads((output_root / "idr_fixed/index.json").read_text(encoding="utf-8"))["entries"]),
            "matched_entries": len(json.loads((output_root / "idr_matched/index.json").read_text(encoding="utf-8"))["entries"]),
            "version": "v1",
        },
        "generator_audit": {"status": audit["status"], "path": audit_path.relative_to(results_root).as_posix(), "sha256": sha256_file(audit_path)},
        "size_gate": {
            "status": size_gate["status"],
            "admission_decision": size_gate["admission_decision"],
            "authority_status": size_gate["authority_status"],
            "path": size_gate_path.relative_to(results_root).as_posix(),
            "sha256": sha256_file(size_gate_path),
        },
        "hpc_submission_authorized": False,
        "limitations": [
            "formal OSM/GHSL/NTL and full grids have not been built on HAICORE",
            "the dense-memory hard gate must be repeated with observed active cells before formal training submission",
            "this smoke does not authorize formal training submission",
        ],
    }
    receipt["artifacts"] = artifact_manifest(results_root, receipt_path)
    verify_smoke_products(repo_root, results_root, receipt, country="nz")
    return atomic_json(receipt, receipt_path)
