"""Local production of every Experiment unit (plan 006a Â§4).

Each ``run_*`` function reproduces exactly what the sealed 006 workers computed
for that unit, using the same ``sglib.experiment`` numerical functions, and
publishes a content-chain receipt whose scientific parameters equal the
registered ones. Upstream objects (DataOverview bundle, Generator handoff,
variant-grid helpers) arrive through ``ctx.upstream``; this module imports no
other stage.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
import sklearn
from threadpoolctl import threadpool_limits

from sglib.core.infra.artifacts import atomic_json, atomic_npz
from sglib.core.infra.content_chain import derive_chain_commitment, derive_chain_receipt, verify_chain
from sglib.core.infra.hashing import sha256_file, sha256_json

from . import (allocator_observations, boundary_diagnostics, conditional_bounds, connection, connection_observations,
               correction, defenses, planning_pool, planning_tasks, reconstruction, sweep_observations)
from .registry import PREPARED_MARKER, PREPARED_SCHEMA, AUDIT_SCHEMA, ExperimentUnit, coordinate_root, expected_node_id, planning_progress, topological_order, unit_output_path, unit_status, unit_type


GATE_KEYS = ("selected_mode", "g0_pass", "g1_pass", "tv_mass", "source_total_basis", "transport_budget", "candidate_sha256", "fallback_reason")


class ProductionError(RuntimeError):
    pass


# ---------------------------------------------------------------- shared helpers

def data_handoff(ctx):
    from .stage import load_upstream
    return load_upstream(ctx, "data")


def generator_handoff(ctx):
    from .stage import load_upstream
    return load_upstream(ctx, "generator")


def station_ledger(data):
    contract = data.profile.station_contract
    table = data.stations_table.copy()
    table[contract["id_column"]] = table[contract["id_column"]].astype(str)
    return table.set_index(contract["id_column"], verify_integrity=True)


def _region_sources(data, region_data):
    contract = data.profile.station_contract
    source_ids = list(map(str, region_data.grid_metadata["source_key_order"]))
    return data.regions_table[data.regions_table[contract["source_key"]].astype(str).isin(source_ids)], source_ids


def frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype={"country": str, "region": str, "target_id": str, "source_id": str, "station_id": str},
                           float_precision="round_trip", low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _registered(ctx, kind: str) -> dict[str, Any]:
    return dict(ctx.loaded.registrations[kind])


def _prepare_target(root: Path, receipt_name: str = "receipt.json") -> None:
    if (root / receipt_name).exists():
        raise ProductionError(f"{root}: receipt already exists; the registry should have reported DONE or INVALID")
    if root.exists() and any(root.iterdir()):
        raise ProductionError(f"{root}: unfinished products present; investigate before rerunning")
    root.mkdir(parents=True, exist_ok=True)


def _publish(root: Path, commitment: Mapping[str, Any], outputs: Mapping[str, str], observations: Mapping[str, Any]) -> dict:
    receipt = derive_chain_receipt(commitment, outputs=dict(outputs), observations={
        **observations, "backend": "local_cpu", "completed_at": datetime.now(timezone.utc).isoformat()})
    atomic_json(receipt, root / "receipt.json")
    return receipt


def write_tables(root: Path, tables: Mapping[str, pd.DataFrame], outputs: dict[str, str], prefix: str = "") -> dict[str, int]:
    rows = {}
    for name, table in tables.items():
        relative = f"{prefix}/{name}.csv" if prefix else f"{name}.csv"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
        outputs[relative] = sha256_file(path)
        rows[relative] = len(table)
    return rows


def serialize_context(region_data, stations: gpd.GeoDataFrame, sources: gpd.GeoDataFrame, assignment: np.ndarray,
                      contract: Mapping[str, Any], profile, station_name_column: str | None) -> dict[str, np.ndarray]:
    """The portable region context of the sealed observations (same arrays, same order)."""

    grid_xy = region_data.grid.to_crs(profile.crs["working"])
    station_xy = stations.to_crs(profile.crs["working"])
    names = stations[station_name_column].astype(str).to_numpy(dtype=str) if station_name_column else stations.index.to_numpy(dtype=str)
    return {"grid_xy": np.column_stack([grid_xy.geometry.x, grid_xy.geometry.y]),
            "station_xy": np.column_stack([station_xy.geometry.x, station_xy.geometry.y]),
            "station_ids": stations.index.to_numpy(dtype=str), "assignment": assignment,
            "observed": stations[contract["demand_column"]].to_numpy(float),
            "capacity": stations[contract["capacity_column"]].to_numpy(float),
            "station_sources": stations[contract["region_column"]].astype(str).to_numpy(dtype=str),
            "grid_sources": region_data.grid[contract["source_key"]].astype(str).to_numpy(dtype=str),
            "source_names": sources[contract["source_key"]].astype(str).to_numpy(dtype=str),
            "source_totals": sources[contract["region_demand_column"]].to_numpy(float),
            "sources_wkt": sources.to_crs(profile.crs["working"]).geometry.to_wkt().to_numpy(dtype=str),
            "station_names": names,
            "cu_support": region_data.cuz_support["covered_mask"] | region_data.cuz_support["unknown_mask"]}


def region_context(ctx, region: str):
    data, generator = data_handoff(ctx), generator_handoff(ctx)
    region_data = next(item for item in data.regions if item.region == region)
    vd = next(item for item in generator.bundle.vd if item.region == region)
    ledger = station_ledger(data)
    stations = ledger.loc[vd.target_ids.astype(str)]
    sources, source_ids = _region_sources(data, region_data)
    t1 = ctx.loaded.values.get("t1", {})
    arrays = serialize_context(region_data, stations, sources, vd.assignment, data.profile.station_contract, data.profile,
                               t1.get("station_name_column") if region in ctx.loaded.values["t1_regions"] else None)
    return data, generator, region_data, vd, stations, sources, source_ids, arrays


# ---------------------------------------------------------------- preflight units

def run_preflight(ctx, unit: ExperimentUnit) -> None:
    {"connection": _preflight_connection, "planning_pool": _preflight_planning_pool}[unit.member](ctx, unit)


def _preflight_connection(ctx, unit: ExperimentUnit) -> None:
    data, generator = data_handoff(ctx), generator_handoff(ctx)
    profile, contract = data.profile, data.profile.station_contract
    protocol = _registered(ctx, "preflight_connection")
    working = str(profile.crs["working"])
    crs = CRS.from_user_input(working)
    if not crs.is_projected or any(axis.unit_conversion_factor != 1. for axis in crs.axis_info[:2]):
        raise ProductionError("the working CRS is not a metric projection; register a distance contract first")
    unit_label = str(ctx.loaded.registrations["observe"]["connection_unit"])
    ledger = station_ledger(data)
    vd_by_region = {item.region: item for item in generator.bundle.vd}
    prepared, inputs = [], {}
    for region_data in data.regions:
        vd = vd_by_region[region_data.region]
        station_ids = vd.target_ids.astype(str)
        selected = ledger.loc[station_ids].copy()
        if not selected[contract["region_column"]].astype(str).isin(list(map(str, region_data.grid_metadata["source_key_order"]))).all():
            raise ProductionError("canonical VD ledger members are outside the analysis region")
        selected = selected.to_crs(working)
        grid = region_data.grid.to_crs(working)
        region = connection.ConnectionRegion(ctx.country, region_data.region, unit_label, working, str(contract["capacity_basis"]),
            np.column_stack([grid.geometry.x, grid.geometry.y]), np.column_stack([selected.geometry.x, selected.geometry.y]),
            station_ids, selected[contract["demand_column"]].to_numpy(float), selected[contract["capacity_column"]].to_numpy(float), vd.assignment)
        projection = {"grid_xy": region.grid_xy.tolist(), "station_xy": region.station_xy.tolist(), "station_ids": station_ids.tolist(),
                      "demand": region.demand.tolist(), "capacity": region.capacity.tolist(), "working_crs": working,
                      "unit": unit_label, "capacity_basis": region.capacity_basis}
        inputs[f"{ctx.country}:VD:{region_data.region}"] = vd.lineage["sha256"]
        inputs[f"{ctx.country}:context:{region_data.region}"] = sha256_json(projection)
        prepared.append(region)
    target = unit_output_path(unit, ctx.root).parent
    _prepare_target(target)
    commitment = derive_chain_commitment(expected_node_id(unit, ctx.loaded), inputs=inputs,
                                         scientific_parameters=protocol, code_sha256=_code(ctx))
    scenarios, members, surfaces, incomplete = [], [], [], []
    for region in prepared:
        if len(region.grid_xy) < protocol["candidate_count"]:
            incomplete.append({"country": ctx.country, "region": region.region, "status": "METRIC_NOT_ASSESSABLE",
                               "reason": "RAW_GRID_SMALLER_THAN_2000", "n_grid": len(region.grid_xy)})
            continue
        surface, membership = connection.reference_surface(region, tuple(protocol["radii_km"]), protocol["candidate_count"])
        scenario = connection.scenario_preflight(region, surface, tuple(protocol["lambdas"]), protocol["reference_radius_km"], protocol["ref_threshold"])
        surfaces.append(surface)
        members.append(membership)
        scenarios.append(scenario)
        print(f"W0 {ctx.country}/{region.region}: Ref={scenario.ref_score.iloc[0]} eligible={bool(scenario.ref_eligible.iloc[0])}", flush=True)
    if not scenarios:
        raise ProductionError("no region admits the fixed candidate pool; preflight cannot close")
    scenarios = pd.concat(scenarios, ignore_index=True)
    summary = []
    for (load, radius), frame in scenarios.groupby(["lambda", "radius_km"], sort=True):
        valid = frame[frame.scenario_defined]
        quantiles = np.quantile(valid.X, [0., .25, .5, .75, 1.], method="linear") if len(valid) else [None] * 5
        summary.append({"country": ctx.country, "unit": prepared[0].unit, "lambda": load, "radius_km": radius,
                        "n_expected": len(prepared), "n_produced": len(frame), "n_scenario_defined": len(valid),
                        "n_c4_c5_expected_eligible": int(frame.c4_c5_expected_eligible.sum()),
                        "n_c6_expected_eligible": int(frame.c6_expected_eligible.sum()),
                        **dict(zip(["X_min", "X_p25", "X_median", "X_p75", "X_max"], quantiles))})
    tables = {"connection_scenario_preflight.csv": scenarios, "connection_scenario_summary.csv": pd.DataFrame(summary),
              "connection_reference_surface.csv": pd.concat(surfaces, ignore_index=True),
              "connection_ledger_membership.csv": pd.concat(members, ignore_index=True),
              "not_assessable.csv": pd.DataFrame(incomplete, columns=["country", "region", "status", "reason", "n_grid"])}
    outputs = {}
    for name, table in tables.items():
        table.to_csv(target / name, index=False, float_format="%.17g", lineterminator="\n")
        outputs[name] = sha256_file(target / name)
    _publish(target, commitment, outputs, {"preregistration_status": "W0_preflight_not_W1", "candidate_performance_consumed": False,
                                           "country": ctx.country, "region_count": len(prepared)})
    print(f"PASS W0 {ctx.country}: {len(prepared)} regions, {len(scenarios)} scenario rows", flush=True)


def _preflight_planning_pool(ctx, unit: ExperimentUnit) -> None:
    data, generator = data_handoff(ctx), generator_handoff(ctx)
    contract = data.profile.station_contract
    parameters = {**_registered(ctx, "preflight_planning_pool"), "country": ctx.country, "working_crs": data.profile.crs["working"]}
    if parameters["sklearn_version"] != sklearn.__version__:
        raise ProductionError(f"registered sklearn {parameters['sklearn_version']} differs from installed {sklearn.__version__}")
    target = unit_output_path(unit, ctx.root).parent
    _prepare_target(target)
    commitment = derive_chain_commitment(expected_node_id(unit, ctx.loaded), inputs={"data": generator.run_fingerprint},
                                         scientific_parameters=parameters, code_sha256=_code(ctx))
    records, outputs = [], {}
    for region_data in data.regions:
        source_ids = list(map(str, region_data.grid_metadata["source_key_order"]))
        k = int(data.stations_table[contract["region_column"]].astype(str).isin(source_ids).sum())
        grid = region_data.grid.copy()
        grid["zero_mask"] = region_data.cuz_support["zero_mask"]
        print(f"Planning pool {ctx.country}/{region_data.region}: start k={k}", flush=True)
        with threadpool_limits(limits=1):
            pool, record = planning_pool.build_pool(grid, k, data.profile.crs["working"])
        base = {"country": ctx.country, "region": region_data.region, "sklearn_version": sklearn.__version__, **record}
        if pool is not None:
            path = target / f"{region_data.region}.npz"
            atomic_npz(path, candidate_grid_rows=pool.candidate_indices, candidate_lonlat=pool.candidate_coords,
                       buildable_grid_rows=pool.buildable_indices, labels=pool.labels)
            outputs[region_data.region] = sha256_file(path)
            base["candidate_pool_sha256"] = sha256_json({"grid_rows": pool.candidate_indices.tolist(), "lonlat": pool.candidate_coords.tolist(),
                                                       "buildable": pool.buildable_indices.tolist(), "labels": pool.labels.tolist()})
            base["artifact_sha256"] = outputs[region_data.region]
        records.append(base)
    table = pd.DataFrame(records)
    table.to_csv(target / "planning_pool_preflight.csv", index=False, lineterminator="\n")
    outputs["planning_pool_preflight"] = sha256_file(target / "planning_pool_preflight.csv")
    _publish(target, commitment, outputs, {"country": ctx.country})
    print(f"PASS W0 planning pool {ctx.country}: {len(table)} measured regions, {int(table.status.eq('VALID').sum())} admitted", flush=True)


# ---------------------------------------------------------------- observations

def t1_tables(ctx, region, arrays, allocators, protocol):
    t1 = ctx.loaded.values["t1"]
    reference_path = ctx.repo / str(t1["reference"])
    if sha256_file(reference_path) != str(t1["reference_sha256"]):
        raise ProductionError("the independent boundary reference differs from its registered hash")
    reference = gpd.read_file(reference_path)
    reference = reference[reference.DNO_LICENCE_AREA_NAME.str.startswith(tuple(protocol["reference_filter"]))].copy().to_crs(protocol["working_crs"])
    reference = reference.rename(columns={"UPID": "reference_id", "PRIMARY_NAME": "reference_name", "DNO_LICENCE_AREA_NAME": "provenance"})
    stations = gpd.GeoDataFrame({"station_name": arrays["station_names"]},
                                geometry=gpd.points_from_xy(arrays["station_xy"][:, 0], arrays["station_xy"][:, 1]), crs=protocol["working_crs"])
    views = []
    for kind, label in (("vd", "VD"), ("fixed", "IDR-fixed"), ("civd", "CIVD"), ("matched", "IDR-matched")):
        for view in allocators[kind]:
            item = {"allocator": label, "candidate": "PUBLIC_GPM_SHAPE" if kind == "fixed" else view["candidate"],
                    "seed": view["seed"], "assignment": view["assignment"], "assignment_sha256": view["sha256"],
                    "assignment_kind": "cluster" if kind == "civd" else "station"}
            if kind == "civd":
                item["station_cluster"] = view["object"].station_cluster
            views.append(item)
    returned = boundary_diagnostics.evaluate(region.grid_xy, arrays["cu_support"], stations, reference, views)
    result = {}
    for name, table in zip(("crosswalk", "station_metrics", "region_metrics"), returned, strict=True):
        table["country"] = region.country
        table["region"] = region.region
        result[name] = table
    return result, sha256_file(reference_path)


def allocator_views(generator, region: str):
    """Allocator products of one region in the sealed role order: vd, fixed, matched, civd."""

    bundle = generator.bundle
    vd = [item for item in bundle.vd if item.region == region]
    fixed = [item for item in bundle.idr_fixed if item.region == region]
    matched = [item for item in bundle.idr_matched if item.region == region]
    civd = [item for item in bundle.civd if item.region == region]
    if len(vd) != 1 or len(fixed) != 1:
        raise ProductionError(f"{region}: canonical VD and fixed IDR must be unique")
    views = {"vd": [{"object": v, "candidate": None, "seed": None, "assignment": v.assignment, "sha256": v.lineage["sha256"],
                     "gate": {}} for v in vd],
             "fixed": [{"object": v, "candidate": None, "seed": None, "assignment": v.assignment, "sha256": v.gate["sha256"],
                        "gate": {k: v.gate[k] for k in GATE_KEYS if k in v.gate}} for v in fixed],
             "matched": [{"object": v, "candidate": v.candidate, "seed": v.seed or None, "assignment": v.assignment, "sha256": v.gate["sha256"],
                          "gate": {k: v.gate[k] for k in GATE_KEYS if k in v.gate}} for v in matched],
             "civd": [{"object": v, "candidate": None, "seed": None, "assignment": v.assignment, "sha256": v.metadata["sha256"],
                       "gate": {k: v.metadata[k] for k in GATE_KEYS if k in v.metadata}} for v in civd]}
    return views


def run_observe(ctx, unit: ExperimentUnit) -> None:
    region_name = unit.member
    data, generator, region_data, vd, stations, sources, source_ids, arrays = region_context(ctx, region_name)
    profile, contract = data.profile, data.profile.station_contract
    registered = _registered(ctx, "observe")
    fields = [f for f in generator.bundle.candidates if f.region == region_name and not f.qa_only]
    sweeps = [s for s in generator.bundle.sweeps if s.region == region_name]
    views = allocator_views(generator, region_name)
    preflight = ctx.root / "preflight/connection"
    surface_all = pd.read_csv(preflight / "connection_reference_surface.csv", dtype={"region": str}, float_precision="round_trip")
    scenarios_all = pd.read_csv(preflight / "connection_scenario_preflight.csv", dtype={"region": str}, float_precision="round_trip")
    t1_protocol = None
    if region_name in ctx.loaded.values["t1_regions"]:
        t1_protocol = {**ctx.loaded.values["t1"]["protocol"], "regions": list(ctx.loaded.values["t1_regions"])}
    params = {**registered, "country": ctx.country, "region": region_name, "unit": profile.units["demand"],
              "working_crs": profile.crs["working"], "capacity_basis": contract["capacity_basis"],
              "input_fingerprint": generator.inputs_receipt_sha256,
              "map_selected": region_name == ctx.loaded.values["representative_region"],
              "fields": [{"role": f"field_{i}", "label": f.label, "family": f.family, "seed": f.seed, "fold": f.fold} for i, f in enumerate(fields)],
              "sweeps": [{"role": f"sweep_{i}", "parameter": s.parameter, "signal": s.signal, "value": s.value, "seed": s.seed, "fold": s.fold}
                         for i, s in enumerate(sweeps)],
              "allocators": [{"role": f"allocator_{kind}_{i}", "kind": kind, "candidate": v["candidate"], "seed": v["seed"], "gate": v["gate"]}
                             for kind in ("vd", "fixed", "matched", "civd") for i, v in enumerate(views[kind])],
              "t1": t1_protocol}
    inputs = {"data": generator.inputs_receipt_sha256, "context": sha256_json({k: v.tolist() for k, v in arrays.items()}),
              "connection_reference_surface": sha256_file(preflight / "connection_reference_surface.csv"),
              "connection_scenario_preflight": sha256_file(preflight / "connection_scenario_preflight.csv")}
    inputs.update({f"field_{i}": f.lineage["sha256"] for i, f in enumerate(fields)})
    inputs.update({f"sweep_{i}": s.lineage["sha256"] for i, s in enumerate(sweeps)})
    inputs.update({f"allocator_{kind}_{i}": v["sha256"] for kind in ("vd", "fixed", "matched", "civd") for i, v in enumerate(views[kind])})
    if t1_protocol is not None:
        inputs["t1_reference"] = str(ctx.loaded.values["t1"]["reference_sha256"])
    destination = unit_output_path(unit, ctx.root).parent
    _prepare_target(destination)
    commitment = derive_chain_commitment(expected_node_id(unit, ctx.loaded), inputs=inputs, scientific_parameters=params, code_sha256=_code(ctx))
    region = reconstruction.ReconstructionRegion(ctx.country, region_name, params["unit"], arrays["station_ids"], arrays["station_sources"],
        arrays["observed"], dict(zip(arrays["source_names"].tolist(), arrays["source_totals"].tolist(), strict=True)), arrays["grid_sources"],
        arrays["assignment"], tuple(fields), params["input_fingerprint"], arrays["station_xy"], arrays["grid_xy"],
        tuple(arrays["sources_wkt"].tolist()), params["map_selected"])
    started = time.monotonic()
    outputs, rows = {}, {}
    with threadpool_limits(limits=int(params["compute_threads"])):
        recon = reconstruction.observe([region])
        rows.update(write_tables(destination, recon, outputs, "reconstruction"))
        corr = correction.observe(recon["predictions"], params["correction_pairs"], {region.region: sum(region.source_totals.values())})
        rows.update(write_tables(destination, {"coordinates": corr}, outputs, "correction"))
        rows.update(write_tables(destination, sweep_observations.observe(region, sweeps), outputs, "sweeps"))
        allocated = allocator_observations.observe([region], [v["object"] for v in views["fixed"]], [v["object"] for v in views["matched"]],
                                                   [v["object"] for v in views["civd"]], [v["object"] for v in views["vd"]])
        baseline = allocated["metrics"][allocated["metrics"].allocator.eq("VD")].merge(
            recon["metrics"], on=["candidate", "seed", "metric"], suffixes=("_allocator", "_recon"), validate="one_to_one")
        if not ((baseline.value_allocator == baseline.value_recon) | (baseline.value_allocator.isna() & baseline.value_recon.isna())).all():
            raise ProductionError("the allocator canonical VD does not reproduce the reconstruction on the same machine")
        rows.update(write_tables(destination, allocated, outputs, "allocator"))
        typed = connection.ConnectionRegion(region.country, region.region, params["connection_unit"], params["working_crs"], params["capacity_basis"],
            region.grid_xy, region.station_xy, region.station_ids, region.observed, arrays["capacity"], region.assignment)
        surface = surface_all[surface_all.region.eq(region_name)].reset_index(drop=True)
        scenarios = scenarios_all[scenarios_all.region.eq(region_name)].reset_index(drop=True)
        packed, tables = connection_observations.observe_region(typed, region.fields, sum(region.source_totals.values()), surface, scenarios, params["planning_labels"])
        rows.update(write_tables(destination, tables, outputs, "connection"))
        atomic_npz(destination / "connection/surfaces.npz", **packed)
        outputs["connection/surfaces.npz"] = sha256_file(destination / "connection/surfaces.npz")
        if t1_protocol is not None:
            t1, _ = t1_tables(ctx, region, arrays, views, t1_protocol)
            rows.update(write_tables(destination, t1, outputs, "T1"))
        context = pd.DataFrame([{"country": region.country, "region": region.region, "n_source": len(region.source_totals),
            "n_target": len(region.observed), "granularity_ratio": len(region.observed) / len(region.source_totals),
            "n_grid": len(region.grid_xy), "capacity_basis": params["capacity_basis"], "unit": region.unit,
            "source_total": sum(region.source_totals.values()), "target_total": float(region.observed.sum()),
            "evidence": "fixed_input_metadata_not_performance_selected"}])
        rows.update(write_tables(destination, {"regions": context}, outputs, "context"))
        counts = {"reconstruction/metrics.csv": (len(region.fields) + 2) * 5,
                  "correction/coordinates.csv": sum(len(p["seeds"]) for p in params["correction_pairs"]) * 6,
                  "sweeps/metrics.csv": len(sweeps) * 5, "allocator/metrics.csv": len(region.fields) * (4 if views["civd"] else 3) * 5,
                  "connection/connection_metrics.csv": 23 * 6, "connection/panel_metrics.csv": 9 * 3 * 6,
                  "connection/scale_metrics.csv": 4 * 8 * 2 * 2 * 6}
        wrong = {name: (rows[name], n) for name, n in counts.items() if rows[name] != n}
        if wrong:
            raise ProductionError(f"observation coordinate counts differ from the approved matrix: {wrong}")
    _publish(destination, commitment, outputs, {"elapsed_seconds": time.monotonic() - started, "rows": rows})
    print(f"PASS observe {ctx.country}/{region_name}: {len(outputs)} outputs", flush=True)


# ---------------------------------------------------------------- planning

def run_planning(ctx, unit: ExperimentUnit) -> None:
    region_name = unit.member
    data, generator, region_data, vd, stations, sources, source_ids, arrays = region_context(ctx, region_name)
    contract = data.profile.station_contract
    registered = _registered(ctx, "planning")
    if registered["sklearn_version"] != sklearn.__version__:
        raise ProductionError(f"registered sklearn {registered['sklearn_version']} differs from installed {sklearn.__version__}")
    pool_path = ctx.root / "preflight/planning_pool" / f"{region_name}.npz"
    if not pool_path.is_file():
        raise ProductionError(f"{region_name}: no admitted planning pool (region ineligible by design or preflight missing)")
    pool_hash = sha256_file(pool_path)
    with np.load(pool_path, allow_pickle=False) as blob:
        pool = {key: blob[key].copy() for key in blob.files}
    grid = region_data.grid.to_crs("EPSG:4326")
    grid_lonlat = np.column_stack([grid.geometry.x, grid.geometry.y])
    if not np.array_equal(grid_lonlat[pool["candidate_grid_rows"]], pool["candidate_lonlat"]):
        raise ProductionError("the fixed pool grid rows differ from the current grid identity")
    stations_lonlat = stations.to_crs("EPSG:4326")
    station_lonlat = np.column_stack([stations_lonlat.geometry.x, stations_lonlat.geometry.y])
    fields = {(f.label, f.seed): f for f in generator.bundle.candidates if f.region == region_name and not f.qa_only}
    count = 0
    for label, seed in unit.coordinates:
        root = coordinate_root(unit, ctx.root, ctx.loaded, label, seed)
        node = expected_node_id(unit, ctx.loaded, label, seed)
        if (root / "receipt.json").exists():
            continue
        if ctx.limit is not None and count >= ctx.limit:
            break
        field = fields[label, seed]
        scientific = {**registered, "country": ctx.country, "region": region_name, "candidate": field.label, "seed": seed,
                      "fold": field.fold, "k": len(stations), "unit": data.profile.units["demand"], "capacity_basis": contract["capacity_basis"]}
        commitment = derive_chain_commitment(node, inputs={"field": field.lineage["sha256"], "pool": pool_hash,
                                                           "canonical_vd": vd.lineage["sha256"], "data": generator.inputs_receipt_sha256},
                                             scientific_parameters=scientific, code_sha256=_code(ctx))
        _prepare_target(root)
        print(f"Planning {ctx.country}/{region_name}/{label}/seed{seed or 0}: start k={len(stations)} M={len(pool['candidate_grid_rows'])}", flush=True)
        start = time.monotonic()
        with threadpool_limits(limits=int(registered["compute_threads"])):
            tables, selection = planning_tasks.observe_task(grid_lonlat, station_lonlat, arrays["observed"], arrays["capacity"], vd.assignment,
                field, pool, country=ctx.country, region=region_name, unit=scientific["unit"], capacity_basis=scientific["capacity_basis"],
                solver=registered["solver"])
        outputs = {}
        for name, table in tables.items():
            table.to_csv(root / f"{name}.csv", index=False, float_format="%.17g", lineterminator="\n")
            outputs[f"{name}.csv"] = sha256_file(root / f"{name}.csv")
        atomic_npz(root / "selection.npz", **selection)
        outputs["selection.npz"] = sha256_file(root / "selection.npz")
        _publish(root, commitment, outputs, {"elapsed_seconds": time.monotonic() - start})
        count += 1
        print(f"PASS Planning {ctx.country}/{region_name}/{label}/seed{seed or 0} elapsed={time.monotonic() - start:.2f}s", flush=True)
    done, expected = planning_progress(unit, ctx.root, ctx.loaded)
    print(f"Planning {ctx.country}/{region_name}: {done}/{expected} coordinates (new={count})", flush=True)


# ---------------------------------------------------------------- bounds and defenses

def _observation_receipts(ctx) -> dict[str, dict]:
    receipts = {}
    for region in ctx.loaded.values["regions"]:
        path = ctx.root / "observations" / region / "receipt.json"
        receipts[region] = verify_chain(json.loads(path.read_text(encoding="utf-8")))
    return receipts


def _surfaces(ctx, receipts: Mapping[str, dict], keys: tuple[str, ...] | None) -> tuple[dict, dict]:
    packed, digests = {}, {}
    for region, receipt in receipts.items():
        path = ctx.root / "observations" / region / "connection/surfaces.npz"
        digest = receipt["outputs"]["connection/surfaces.npz"]
        if sha256_file(path) != digest:
            raise ProductionError(f"{region}: the connection surfaces differ from their receipt")
        with np.load(path, allow_pickle=False) as blob:
            packed[region] = {name: blob[name].copy() for name in (keys or blob.files)}
        digests[region] = digest
    return packed, digests


def _collect(ctx, receipts: Mapping[str, dict], prefix: str, names: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    parts = {name: [] for name in names}
    for region, receipt in receipts.items():
        for name in names:
            relative = f"{prefix}/{name}.csv"
            path = ctx.root / "observations" / region / relative
            if relative not in receipt["outputs"] or sha256_file(path) != receipt["outputs"][relative]:
                raise ProductionError(f"{region}/{relative}: observation table missing or changed")
            parts[name].append(frame(path))
    return {name: pd.concat(values, ignore_index=True) if values else pd.DataFrame() for name, values in parts.items()}


def run_bounds(ctx, unit: ExperimentUnit) -> None:
    registered = _registered(ctx, "bounds")
    receipts = _observation_receipts(ctx)
    packed, digests = _surfaces(ctx, receipts, ("field_labels", "field_seeds", "source_total", "radii_km", "lambdas", "X", "G", "F", "Ghat", "unit"))
    target = unit_output_path(unit, ctx.root).parent
    _prepare_target(target)
    commitment = derive_chain_commitment(expected_node_id(unit, ctx.loaded), inputs=digests, scientific_parameters=registered, code_sha256=_code(ctx))
    with threadpool_limits(limits=1):
        bounds = conditional_bounds.observe(ctx.country, packed, etas=tuple(registered["etas"]))
    bounds.to_csv(target / "bounds.csv", index=False, float_format="%.17g", lineterminator="\n")
    _publish(target, commitment, {"bounds.csv": sha256_file(target / "bounds.csv")}, {"rows": len(bounds)})
    print(f"PASS C6 bounds {ctx.country}: {len(bounds)} rows", flush=True)


def run_defense(ctx, unit: ExperimentUnit) -> None:
    data = data_handoff(ctx)
    registered = _registered(ctx, "defense")
    spec = registered["specification"]
    receipts = _observation_receipts(ctx)
    surfaces, digests = _surfaces(ctx, receipts, None)
    grid_xy = {}
    for region_data in data.regions:
        grid = region_data.grid.to_crs(data.profile.crs["working"])
        grid_xy[region_data.region] = np.column_stack([grid.geometry.x, grid.geometry.y])
    inputs = {f"surfaces:{region}": digest for region, digest in digests.items()}
    target = unit_output_path(unit, ctx.root).parent
    _prepare_target(target)
    commitment = derive_chain_commitment(expected_node_id(unit, ctx.loaded), inputs=inputs,
                                         scientific_parameters={**registered, "country": ctx.country}, code_sha256=_code(ctx))
    with threadpool_limits(limits=int(registered["compute_threads"])):
        fixed = defenses.fixed_load_observations(ctx.country, surfaces, x=300.)
        representative = spec["representative_region"]
        connection_map = defenses.connection_map_observations(ctx.country, representative, surfaces[representative], grid_xy[representative])
        tables = {"C4_fixed_300": fixed, "C4_connection_map": connection_map}
    outputs = {}
    write_tables(target, tables, outputs)
    _publish(target, commitment, outputs, {"completion_scope": registered["completion_scope"]})
    print(f"PASS defenses {ctx.country}: {len(outputs)} tables", flush=True)


# ---------------------------------------------------------------- audit, preparation, code identity

def coverage(ctx) -> dict[str, Any]:
    """Expected versus produced coordinates of every unit, read from markers only."""

    units, families = [], set(ctx.loaded.values["observe"]["families"])
    for unit in topological_order(ctx.units):
        state = unit_status(unit, ctx.root, ctx.loaded)
        row = {"unit": unit.id, "step": unit.step, "member": unit.member, "state": state}
        if unit.step == "planning":
            done, expected = planning_progress(unit, ctx.root, ctx.loaded)
            row.update(expected=expected, produced=done)
        elif unit.step == "observe":
            marker = unit_output_path(unit, ctx.root)
            produced = set()
            if state == "DONE":
                outputs = json.loads(marker.read_text(encoding="utf-8"))["outputs"]
                produced = {name.split("/", 1)[0] for name in outputs}
            wanted = set(families) | ({ctx.loaded.values["observe"]["t1_family"]} if unit.member in ctx.loaded.values["t1_regions"] else set())
            row.update(expected=len(wanted), produced=len(wanted & produced), missing=sorted(wanted - produced))
        else:
            row.update(expected=1, produced=int(state == "DONE"))
        units.append(row)
    complete = all(r["state"] == "DONE" for r in units if r["step"] != "audit") and all(r["produced"] == r["expected"] for r in units if r["step"] != "audit")
    return {"schema_version": AUDIT_SCHEMA, "country": ctx.country, "status": "PASS" if complete else "INCOMPLETE",
            "units": units, "expected": sum(r["expected"] for r in units if r["step"] != "audit"),
            "produced": sum(r["produced"] for r in units if r["step"] != "audit")}


def run_audit(ctx, unit: ExperimentUnit) -> None:
    document = coverage(ctx)
    if document["status"] != "PASS":
        raise ProductionError(f"{ctx.country}: audit cannot PASS with incomplete units")
    atomic_json(document, unit_output_path(unit, ctx.root))
    print(f"PASS audit {ctx.country}: {document['produced']}/{document['expected']} coordinates", flush=True)


def prepare_unit(ctx, unit: ExperimentUnit) -> None:
    """HPC backend: record what the private shell must execute, without running it."""

    marker = (unit_output_path(unit, ctx.root) if unit.step == "planning" else unit_output_path(unit, ctx.root).parent) / PREPARED_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_json({"schema_version": PREPARED_SCHEMA, "unit": unit.id, "node_id": expected_node_id(unit, ctx.loaded) if unit.step == "observe" else None,
                 "coordinates": [[field, seed] for field, seed in unit.coordinates], "backend": ctx.backend,
                 "prepared_at": datetime.now(timezone.utc).isoformat()}, marker)


def _code(ctx) -> str:
    from .stage import numerical_code
    return numerical_code()
