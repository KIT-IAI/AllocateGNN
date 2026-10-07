"""Run the NL pre-HPC smoke without claiming full-feature admission.

Source polygons and equipment rows are real Gate-A outputs.  Grid features are a
small deterministic fixture so CPU smoke can exercise shared Generator algorithms
before OSM/GHSL/NTL are produced on HPC.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import geopandas as gpd
import numpy as np
import pandas as pd

from .common import check_smoke_output, write_fixture_inventory, publish_static_candidate_indexes
from .verification import artifact_manifest, verify_smoke_products

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.hashing import sha256_file
from sglib.core.infra.paths import resolve_case_path
from sglib.core.infra.terms import load_country_profile
from sglib.dataoverview.handoff import DataOverviewBundle, RegionData
from sglib.generator.config import load_generator_config, training_params
from sglib.generator.execution import (
    generate_static_component,
    materialize_family,
    prepare_graph_cache,
    prepare_inputs,
    run_idr_fixed,
    run_civd,
    run_idr_matched,
    verify_inference_group,
)
from sglib.generator.generation import input_identity_binding, derive_run_identity
from sglib.generator.weighter.candidates import load_candidate_registry
from sglib.generator.weighter.learned.training import run_training_task
from sglib.generator.weighter.learned.training.tasks import TrainingTask
from sglib.generator.weighter.learned.training.preparation import prepare_tasks
from sglib.generator.weighter.learned.training.verify import verify_task_artifacts
from sglib.generator.weighter.learned.inference import run_inference_task, verify_inference_artifacts
from sglib.generator.weighter.learned.inference.preparation import prepare_inference_tasks


def _smoke_config(repo_root: Path, profile) -> dict:
    """Load the four-country authority, then narrow only the smoke regions."""

    loaded = load_generator_config(
        repo_root,
        repo_root / "casestudy/2_Generator/general/generator.toml",
        profile.source_path,
        repo_root / "casestudy/2_Generator/4_NL/nl.toml",
    )
    return deepcopy(dict(loaded.values))


def _fixture_region(
    region: str,
    source: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, RegionData]:
    source = source.sort_values("wijk_code", kind="stable").head(6).copy()
    codes = set(source["wijk_code"].astype(str))
    selected = (
        stations.loc[stations["wijk_code"].astype(str).isin(codes)]
        .sort_values(["wijk_code", "station_id"], kind="stable")
        .groupby("wijk_code", sort=False, group_keys=False)
        .head(2)
        .copy()
    )
    present = set(selected["wijk_code"].astype(str))
    source = source.loc[source["wijk_code"].astype(str).isin(present)].copy().reset_index(drop=True)
    selected = selected.loc[selected["wijk_code"].astype(str).isin(present)].copy().reset_index(drop=True)
    demand = selected.groupby("wijk_code", observed=True)["peak_kw"].sum()
    source["demand_peak_kw"] = source["wijk_code"].astype(str).map(demand).astype(float)
    working = selected.to_crs("EPSG:28992")
    rows: list[dict] = []
    feature_rows: list[np.ndarray] = []
    ntl: list[float] = []
    feature_names = [
        "residential_percent",
        "commercial_percent",
        "industrial_percent",
        "agricultural_percent",
        "others_percent",
    ]
    for target_index, target in working.iterrows():
        code = str(target["wijk_code"])
        base = target.geometry
        for offset_index, (dx, dy) in enumerate(
            ((0.0, 0.0), (35.0, 0.0), (0.0, 35.0), (-35.0, 0.0), (0.0, -35.0))
        ):
            rows.append(
                {
                    "index_region": len(rows),
                    "wijk_code": code,
                    "analysis_region": region,
                    "geometry": type(base)(base.x + dx, base.y + dy),
                }
            )
            cell_features = np.zeros(6, dtype=np.float64)
            cell_features[offset_index] = 1.0
            feature_rows.append(cell_features)
            ntl.append(float(1 + target_index + offset_index) / 10.0)
    grid = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:28992").to_crs("EPSG:4326")
    features = np.asarray(feature_rows, dtype=np.float64)
    n_cells = len(grid)
    source_order = source["wijk_code"].astype(str).tolist()
    support = {
        "features": features,
        "built_fraction": np.full(n_cells, 0.5, dtype=np.float64),
        "covered_mask": np.ones(n_cells, dtype=bool),
        "unknown_mask": np.zeros(n_cells, dtype=bool),
        "zero_mask": np.zeros(n_cells, dtype=bool),
        "source_keys": grid["wijk_code"].astype(str).to_numpy(),
        "source_key_order": np.asarray(source_order, dtype=np.str_),
    }
    arrays = {
        "data": np.asarray(ntl, dtype=np.float64).reshape(-1, 1),
        "columns": np.asarray(["ntl"], dtype=np.str_),
    }
    item = RegionData(
        region=region,
        grid=grid,
        grid_metadata={
            "schema_version": "sg_grid_bundle_v2",
            "projected_step_m": 35.0,
            "target_ground_step_m": 35.0,
            "source_key": "wijk_code",
            "source_key_order": source_order,
            "working_crs": "EPSG:28992",
            "fixture": True,
        },
        landuse={"data": features[:, :5], "columns": np.asarray(feature_names)},
        built_surface={"data": np.full((n_cells, 1), 0.5), "columns": np.asarray(["built_fraction"])},
        cuz_support=support,
        ntl=arrays,
    )
    return source, selected, item


def _handoff(repo_root: Path, derived_root: Path | None = None) -> tuple[DataOverviewBundle, list[str]]:
    derived = Path(derived_root) if derived_root is not None else repo_root / "data/datasets/2_derived/nl"
    gate = json.loads((derived / "audit/gate_a.json").read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise RuntimeError("NL Gate A must pass before the Generator smoke")
    inventory = json.loads((derived / "authority/analysis_region_inventory.json").read_text(encoding="utf-8"))
    region_names = [row["id"] for row in inventory["regions"] if row["n_sources"] >= 2 and row["n_targets"] >= 2][:4]
    if len(region_names) != 4:
        raise RuntimeError("NL smoke needs four non-empty provisional regions")
    all_sources = gpd.read_file(derived / "bplus/source_regions.gpkg", layer="source_regions")
    all_stations = gpd.read_file(derived / "bplus/buurt_pseudo_stations.gpkg", layer="buurt_pseudo_stations")
    source_parts: list[gpd.GeoDataFrame] = []
    station_parts: list[gpd.GeoDataFrame] = []
    region_data: list[RegionData] = []
    for name in region_names:
        source, stations, item = _fixture_region(
            name,
            all_sources.loc[all_sources["analysis_region"].eq(name)].copy(),
            all_stations.loc[all_stations["analysis_region"].eq(name)].copy(),
        )
        source_parts.append(source)
        station_parts.append(stations)
        region_data.append(item)
    sources = gpd.GeoDataFrame(
        pd.concat(source_parts, ignore_index=True),
        geometry="geometry",
        crs=source_parts[0].crs,
    )
    stations = gpd.GeoDataFrame(
        pd.concat(station_parts, ignore_index=True),
        geometry="geometry",
        crs=station_parts[0].crs,
    )
    profile = load_country_profile(repo_root / "casestudy/config/countries/nl.toml")
    bundle = DataOverviewBundle(
        country="nl",
        profile=profile,
        regions_table=sources,
        stations_table=stations,
        regions=tuple(region_data),
        inventory={"schema_version": "sg_dataoverview_inventory_v1", "country": "nl", "fixture_features": True},
        evidence={},
        configuration=None,
    )
    return bundle, region_names


def _run_training_smoke(root: Path, repo_root: Path, *, family: str, config: dict) -> dict:
    graph_path = root / "training/graphs/lu5.pkl"
    if not graph_path.is_file():
        prepare_graph_cache(root, "lu5")
    task = TrainingTask(
        group=f"B-NL-{family.upper()}",
        country="nl",
        family=family,
        config="baseline",
        signal="none",
        parameter="fixed",
        value="default",
        seed=42,
        fold=1,
        output_relative=f"2_Weighter/base/{family}/seed_42/fold1",
        feature_set="lu5",
    )
    params = root / "inputs/worker_params.json"
    receipt = root / "inputs/receipt.json"
    input_binding = input_identity_binding(receipt)
    run_identity = derive_run_identity(root, config, repo_root=repo_root)
    prepared = prepare_tasks(
        [task],
        frozen_params_path=params,
        graph_cache_by_feature_set={"lu5": graph_path},
        repo_root=root,
        results_root=root,
        config_fingerprint=sha256_file(params),
        input_fingerprints={
            input_binding["field"]: input_binding["fingerprint"]
        },
        execution_backend="local",
        execution_identity={
            "run_identity": run_identity,
            "profile": "nl_pre_hpc_real_rows_light_feature_fixture",
            "formal": False,
        },
    )[0]
    task_path = atomic_json(prepared.to_dict(), root / "training/tasks" / task.group / f"{task.task_id}.json")
    run_training_task(prepared, backend="local", device="cpu")
    completion = verify_task_artifacts(prepared)
    training_completion_path = Path(prepared.output_path) / "task_completion.json"
    training_verify_path = atomic_json(
        {
            "schema_version": "sg_nl_smoke_training_verify_v1",
            "status": "PASS",
            "formal": False,
            "task_id": task.task_id,
            "completion_sha256": sha256_file(training_completion_path),
        },
        root / "training/verify" / f"{task.group}.json",
    )
    inference = prepare_inference_tasks(
        [task_path],
        output_root=root / "inference/outputs" / task.group,
        repo_root=root,
        results_root=root,
        execution_backend="local",
        execution_identity={"profile": "nl_pre_hpc_smoke", "formal": False},
    )[0]
    inference_task_path = atomic_json(
        inference.to_dict(),
        root / "inference/tasks" / task.group / f"{inference.task_id}.json",
    )
    atomic_json(
        {
            "schema_version": "sg_prepared_inference_group_v1",
            "group": task.group,
            "execution_backend": "local",
            "tasks": [
                {
                    "task_id": inference.task_id,
                    "path": inference_task_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(inference_task_path),
                    "bytes": inference_task_path.stat().st_size,
                }
            ],
        },
        root / "inference/tasks" / task.group / "index.json",
    )
    inference_output = run_inference_task(
        inference,
        backend="local",
        device="cpu",
        execution_identity={"formal": False},
    )
    inference_completion = verify_inference_artifacts(inference)
    inference_completion_path = inference_output / "inference_completion.json"
    standard_verify_path = verify_inference_group(root, task.group, repo_root=root, results_root=root)
    standard_verify = json.loads(standard_verify_path.read_text(encoding="utf-8"))
    inference_verify_path = atomic_json(
        {
            **standard_verify,
            "status": "PASS",
            "formal": False,
        },
        standard_verify_path,
    )
    return {
        "family": family,
        "task_id": task.task_id,
        "formal": False,
        "epochs": completion["epochs_observed"],
        "selected_training_loss": completion["selected_training_loss"],
        "completion": str((Path(prepared.output_path) / "task_completion.json").relative_to(root)).replace("\\", "/"),
        "training_verify": training_verify_path.relative_to(root).as_posix(),
        "inference_task_id": inference.task_id,
        "inference_completion": str(
            (inference_output / "inference_completion.json").relative_to(root)
        ).replace("\\", "/"),
        "inference_verify": inference_verify_path.relative_to(root).as_posix(),
        "inference_regions": inference_completion["n_regions"],
    }


def run(repo_root: Path, output_root: Path, *, derived_root: Path | None = None, refresh: bool = False) -> Path:
    repo_root = Path(repo_root).resolve()
    root = Path(output_root).resolve()
    final = root / "pre_hpc_smoke.json"
    completed = check_smoke_output(
        repo_root, root, receipt_relative=final.name,
        receipt_schema="sg_nl_pre_hpc_smoke_v2", country="nl", reuse=not refresh,
    )
    if completed is not None:
        return completed
    handoff, regions = _handoff(repo_root, derived_root)
    handoff, inventory_path = write_fixture_inventory(handoff, root)
    values = _smoke_config(repo_root, handoff.profile)
    values["regions"] = regions
    worker = training_params(values)
    worker["seeds"] = [42]
    for config in worker["config_map"].values():
        config["epochs"] = 2
    prepare_inputs(
        handoff,
        values,
        root,
        selected_regions=regions,
        worker_params=worker,
        require_formal_evidence=False,
        inventory_path=inventory_path,
    )
    for component in ("assignments", "uniform", "gpm", "proximity", "public_activity"):
        generate_static_component(root, component)
    run_idr_fixed(root)
    candidates = load_candidate_registry(resolve_case_path(repo_root / values["authorities"]["candidate_registry"]))
    for family in ("Uni", "GPM", "Equal"):
        materialize_family(root, family, candidates)
    publish_static_candidate_indexes(root)
    run_idr_matched(root)
    run_civd(root)
    training = [_run_training_smoke(root, repo_root, family=family, config=values) for family in ("gnn", "mlp")]
    document = {
        "schema_version": "sg_nl_pre_hpc_smoke_v2",
        "status": "PASS",
        "country": "nl",
        "formal": False,
        "regions": regions,
        "real_gate_a_sources_and_targets": True,
        "source_level": "wijk",
        "target_level": "buurt_pseudo_station",
        "feature_fixture": True,
        "scientific_admission": False,
        "admission_note": "Full OSM/GHSL/NTL feature materialization and formal pre-submission memory recheck remain required before HPC submission.",
        "idr": {"version": "v1", "fixed": True, "matched": True, "v2": "withdrawn_not_executed"},
        "static_components": ["assignments", "uniform", "gpm", "proximity", "public_activity"],
        "training": training,
    }
    document["artifacts"] = artifact_manifest(root, final)
    verify_smoke_products(repo_root, root, document, country="nl")
    return atomic_json(document, final)
