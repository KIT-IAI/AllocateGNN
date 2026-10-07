"""Independent final audit of the complete four-country CIVD correction."""
from __future__ import annotations

import json
from pathlib import Path
import sys


import numpy as np
import pandas as pd

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.content_chain import code_projection, derive_chain_commitment, derive_chain_receipt, derive_chain_closure, verify_chain
from sglib.core.infra.hashing import sha256_file, sha256_json
from sglib.core.infra.paths import resolve_recorded_path


COUNTRIES = {"uk": "1_UK", "au": "2_AU", "nl": "4_NL", "nz": "5_NZ"}
ALLOCATORS = {"VD", "IDR-fixed", "IDR-matched", "CIVD"}
ARRAYS = ("assignment", "grid_cluster", "station_cluster", "raw_labels", "probabilities")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _verify_comparison_sources(repo, provenance, comparison):
    """Hash-check comparison sources; recorded paths resolve against the repository, not the cwd."""
    prior_group = "previous_implementation_sources" if "previous_implementation_sources" in provenance else "original_invalid_implementation_sources"
    for group in ("corrected_sources", prior_group):
        for country, source in provenance[group].items():
            _hash(resolve_recorded_path(source["path"], repo), source["sha256"])
            role = f"analysis:{country}:C3:region_metrics.csv" if group == "corrected_sources" else f"historical:{country}:C3:region_metrics.csv"
            _require(comparison["commitment"]["inputs"][role] == source["sha256"], f"comparison provenance/source receipt link differs: {role}")


def _hash(path, expected):
    path = Path(path)
    _require(path.is_file(), f"missing artifact: {path}")
    observed = sha256_file(path)
    _require(observed == expected, f"artifact hash differs from frozen/committed identity: {path}")
    return observed


def _receipt(path):
    path = Path(path)
    document = verify_chain(_json(path))
    _require(document["schema_version"] == "sg_content_chain_receipt_v1", f"expected receipt: {path}")
    for relative, digest in document["outputs"].items():
        artifact = (path.parent / relative).resolve()
        _require(artifact.is_relative_to(path.parent.resolve()), f"receipt output escapes root: {path}: {relative}")
        _hash(artifact, digest)
    return document


def _npz(path, digest):
    _hash(path, digest)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _frame(path):
    try:
        return pd.read_csv(path, dtype={"country": str, "region": str, "target_id": str, "station_id": str,
            "source_id": str}, float_precision="round_trip", low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _field_key(label, seed):
    return str(label), None if seed is None or pd.isna(seed) else int(seed)


def _groups(table):
    return {_field_key(label, seed): group for (label, seed), group in table.groupby(["candidate", "seed"], sort=False, dropna=False)}


def _independent_metrics(observed, predicted):
    error = predicted-observed
    squared = np.square(observed-observed.mean()).sum()
    total = observed.sum()
    return {"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error))),
        "wape": float(np.abs(error).sum()/total) if total else None,
        "predictive_r2": float(1-np.square(error).sum()/squared) if squared else None,
        "corr": float(np.corrcoef(observed, predicted)[0, 1]) if squared and np.var(predicted) > 0 and len(observed) > 1 else None}


def _preserved_tables(old_root, new_root):
    for family, names in (("allocator", ("predictions", "metrics", "gates", "maps")),
                          ("T1", ("station_metrics", "region_metrics"))):
        for name in names:
            old_path, new_path = old_root / family / f"{name}.csv", new_root / family / f"{name}.csv"
            if not old_path.exists():
                continue
            old, new = _frame(old_path), _frame(new_path)
            if not len(old.columns):
                _require(not len(new.columns), f"unexpected nonempty table: {new_path}")
                continue
            _require(set(old.columns).issubset(new.columns), f"lost baseline columns: {new_path}")
            before = old[old.allocator.ne("CIVD")].reset_index(drop=True)
            after = new.loc[new.allocator.ne("CIVD"), old.columns].reset_index(drop=True)
            pd.testing.assert_frame_equal(before, after, check_dtype=False, check_exact=True, obj=str(new_path))
            if family == "T1":
                civd = new[new.allocator.eq("CIVD")]
                _require(civd.status.eq("METRIC_NOT_ASSESSABLE").all(), f"CIVD station footprint incorrectly assessed: {new_path}")
                _require(civd.reason.eq("CIVD_CLUSTER_ALLOCATION_HAS_NO_UNIQUE_STATION_FOOTPRINT").all(),
                         f"CIVD station footprint reason differs: {new_path}")


def _audit_country(repo, results, country, directory, expected_observation_code, *, country_inputs, correction_id):
    label = f"{country}: final CIVD audit"
    data, config = country_inputs(repo, country)
    expected_regions = list(config.values["regions"])
    formal = repo / "results/2_Generator" / directory
    extension = results / "2_Generator" / directory / "civd"
    country_receipt = _receipt(extension / "receipt.json")
    index = _json(extension / "index.json")
    entries = {item["region"]: item for item in index["entries"]}
    _require(len(entries) == len(index["entries"]) and set(entries) == set(expected_regions), f"{label}: incomplete/duplicate Generator regions")
    country_protocol = country_receipt["commitment"]["scientific_parameters"]
    _require(country_protocol == index["protocol"], f"{label}: country protocol/index differs")
    _require(country_protocol["scope"] == "four_country_extension_after_bug_correction" and not country_protocol["historically_preregistered"],
             f"{label}: incorrect extension scope")
    contract = data.profile.station_contract
    working = data.profile.crs["working"]
    _require(country_protocol["working_crs"] == working and country_protocol["capacity_column"] == contract["capacity_column"],
             f"{label}: capacity/CRS protocol differs from country contract")
    _require(country_protocol["weight_rule"] == "max(max(station_capacity,1e-10)/max(distance_m,1e-10))_within_cluster",
             f"{label}: capacity-weighting protocol differs")
    ledger = data.stations_table.copy()
    ledger[contract["id_column"]] = ledger[contract["id_column"]].astype(str)
    ledger = ledger.set_index(contract["id_column"], verify_integrity=True)
    region_data = {item.region: item for item in data.regions}
    vd_entries = {item["region"]: item for item in _json(formal / "static/assignments/index.json")["regions"]}
    candidate_entries = {(item["region"], *_field_key(item["label"], item.get("seed"))): item
        for item in _json(formal / "candidates/candidate_index.json")["entries"] if not item.get("qa_only", False)}
    allocator_hashes = {}
    for family in ("idr_fixed", "idr_matched"):
        for item in _json(formal / family / "index.json")["entries"]:
            _hash(formal / item["path"], item["sha256"])
            allocator_hashes[item["sha256"]] = item["path"]
    frozen_inputs = _hash(formal / "inputs/receipt.json", country_receipt["commitment"]["inputs"]["formal_generator_inputs"])
    w0 = _receipt(repo / "results/3_Experiment" / directory / "preflight/connection/receipt.json")
    regional_receipts, observations, checks = [], {}, []
    for name in expected_regions:
        tag = f"{country}/{name}"
        item = entries[name]
        regional_path = extension / f"{name}.receipt.json"
        regional = _receipt(regional_path)
        file_hash = _hash(regional_path, item["receipt_sha256"])
        _require(country_receipt["commitment"]["inputs"][f"region:{name}"] == file_hash, f"{tag}: country/regional receipt link differs")
        _require(regional["commitment"]["scientific_parameters"] == item["protocol"], f"{tag}: regional protocol/index differs")
        _require(regional["commitment"]["inputs"]["formal_generator_inputs"] == frozen_inputs, f"{tag}: frozen inputs receipt differs")
        blob = _npz(extension.parent / item["path"], item["sha256"])
        _require(regional["outputs"].get(f"{name}.npz") == item["sha256"], f"{tag}: regional receipt/archive differs")
        vd_entry = vd_entries[name]
        vd = _npz(formal / vd_entry["path"], vd_entry["sha256"])
        station_ids = vd["station_id"].astype(str)
        np.testing.assert_array_equal(blob["station_id"].astype(str), station_ids, err_msg=f"{tag}: station identity order")
        _require(sha256_json(station_ids.tolist()) == item["station_id_sha256"], f"{tag}: station identity digest differs")
        _require(regional["commitment"]["inputs"]["canonical_vd"] == vd_entry["sha256"] == item["canonical_vd_sha256"], f"{tag}: canonical VD link differs")
        stations = ledger.loc[station_ids]
        grid = region_data[name].grid.to_crs(working)
        station_xy = stations.to_crs(working)
        grid_xy = np.column_stack((grid.geometry.x, grid.geometry.y))
        xy = np.column_stack((station_xy.geometry.x, station_xy.geometry.y))
        observed, capacity = stations[contract["demand_column"]].to_numpy(float), stations[contract["capacity_column"]].to_numpy(float)
        old_root = repo / "results/3_Experiment" / directory / "observations" / name
        new_root = results / "3_Experiment" / directory / "observations" / name
        old, new = _receipt(old_root / "receipt.json"), _receipt(new_root / "receipt.json")
        _require(new["observations"].get("correction_id") == correction_id, f"{tag}: unrelated observation correction")
        execution_code = new["commitment"]["code_sha256"]
        _require(execution_code == expected_observation_code, f"{tag}: local observation does not carry the current execution identity")
        params = old["commitment"]["scientific_parameters"]
        projection = {"grid_xy": grid_xy.tolist(), "station_xy": xy.tolist(), "station_ids": station_ids.tolist(),
            "demand": observed.tolist(), "capacity": capacity.tolist(), "working_crs": working,
            "unit": params["connection_unit"], "capacity_basis": contract["capacity_basis"]}
        w0_digest = sha256_json(projection)
        _require(w0_digest == w0["commitment"]["inputs"][f"{country}:context:{name}"],
                 f"{tag}: live coordinates/demand/capacity differ from frozen W0 context projection")
        context_digest = sha256_json({"station_id": station_ids.tolist(), "working_crs": working,
            "grid_xy": grid_xy.tolist(), "station_xy": xy.tolist(), "capacity": capacity.tolist()})
        _require(context_digest == regional["commitment"]["inputs"]["region_context"], f"{tag}: CIVD context differs from audited frozen W0 context")
        _require(new["commitment"]["inputs"]["inherited_observation_receipt"] == old["receipt_sha256"], f"{tag}: inherited observation link differs")
        _require(new["commitment"]["inputs"]["civd_extension_receipt"] == country_receipt["receipt_sha256"], f"{tag}: observation/Generator extension link differs")
        _require(new["commitment"]["inputs"]["allocator_civd_0"] == item["sha256"], f"{tag}: observation CIVD archive differs")
        for role, digest in old["commitment"]["inputs"].items():
            if role.startswith(("allocator_vd_", "allocator_fixed_", "allocator_matched_")):
                _require(new["commitment"]["inputs"].get(role) == digest, f"{tag}: frozen allocator input changed: {role}")
        if country in {"uk", "au"}:
            previous = _npz(formal / "civd" / f"{name}.npz", item["frozen_assignment_sha256"])
            for array in ARRAYS:
                np.testing.assert_array_equal(blob[array], previous[array], err_msg=f"{tag}: frozen CIVD {array}")
        _preserved_tables(old_root, new_root)
        tables = {table: _frame(new_root / "allocator" / f"{table}.csv") for table in ("predictions", "metrics", "gates", "maps")}
        fields = params["fields"]
        keys = {_field_key(field["label"], field["seed"]) for field in fields}
        _require(len(fields) == len(keys) == 78, f"{tag}: expected 78 unique field realizations")
        for table in ("predictions", "metrics", "gates"):
            frame = tables[table]
            _require(set(frame.allocator) == ALLOCATORS, f"{tag}: {table} missing one of four allocators")
            for allocator, subset in frame.groupby("allocator", sort=False):
                _require(set(_groups(subset)) == keys, f"{tag}: incomplete {allocator}/{table} field coordinates")
            factor = len(station_ids) if table == "predictions" else 5 if table == "metrics" else 1
            _require(len(frame) == 78*4*factor, f"{tag}: {table} row count differs")
        predicted = _groups(tables["predictions"].query("allocator == 'CIVD'"))
        metrics = _groups(tables["metrics"].query("allocator == 'CIVD'"))
        gates = _groups(tables["gates"].query("allocator == 'CIVD'"))
        baseline = _groups(tables["predictions"].query("allocator == 'VD'"))
        labels = blob["station_cluster"]
        ordered, station_ordinals = np.unique(labels, return_inverse=True)
        np.testing.assert_array_equal(blob["assignment"], blob["grid_cluster"], err_msg=f"{tag}: cluster assignments disagree")
        _require(labels.shape == station_ids.shape and blob["assignment"].shape == (len(grid),), f"{tag}: cluster array shapes differ")
        _require(np.all((blob["assignment"] >= 0) & (blob["assignment"] < len(ordered))), f"{tag}: cluster ordinal out of range")
        _require(item["n_clusters"] == len(ordered) and item["n_stations"] == len(station_ids), f"{tag}: cluster/station inventory differs")
        grid_masks = [blob["assignment"] == ordinal for ordinal in range(len(ordered))]
        station_masks = [labels == label for label in ordered]
        max_difference, max_mass_difference = 0., 0.
        recorded = {_field_key(row["candidate"], row["seed"]): row for row in new["observations"]["equal_split_checks"]}
        _require(len(recorded) == len(new["observations"]["equal_split_checks"]) == 78 and set(recorded) == keys,
                 f"{tag}: independent-oracle receipt coordinates differ")
        for field in fields:
            key = _field_key(field["label"], field["seed"])
            source = candidate_entries[(name, *key)]
            expected_hash = old["commitment"]["inputs"][field["role"]]
            _require(source["sha256"] == expected_hash == new["commitment"]["inputs"][field["role"]], f"{tag}/{key}: field identity differs")
            values = _npz(formal / source["path"], expected_hash)["data"]
            _require(values.shape == (len(grid),) and np.isfinite(values).all() and np.all(values >= 0), f"{tag}/{key}: invalid frozen field")
            expected = np.zeros(len(station_ids), float)
            for grid_mask, station_mask in zip(grid_masks, station_masks, strict=True):
                expected[station_mask] = values[grid_mask].sum()/station_mask.sum()
            rows = predicted[key]
            np.testing.assert_array_equal(rows.target_id.astype(str), station_ids, err_msg=f"{tag}/{key}: prediction station order")
            np.testing.assert_array_equal(rows.observed.to_numpy(float), observed, err_msg=f"{tag}/{key}: observed station demand")
            actual = rows.predicted.to_numpy(float)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10, err_msg=f"{tag}/{key}: independent equal-split oracle")
            np.testing.assert_allclose(actual.sum(), values.sum(), rtol=1e-12, atol=1e-8, err_msg=f"{tag}/{key}: demand conservation")
            metric_rows = metrics[key].set_index("metric", verify_integrity=True)
            for metric, value in _independent_metrics(observed, actual).items():
                row = metric_rows.loc[metric]
                if value is None:
                    _require(pd.isna(row.value) and row.status == "METRIC_NOT_ASSESSABLE", f"{tag}/{key}/{metric}: undefined metric contract differs")
                else:
                    _require(row.status == "VALID", f"{tag}/{key}/{metric}: expected VALID")
                    np.testing.assert_allclose(row.value, value, rtol=1e-12, atol=1e-10, err_msg=f"{tag}/{key}/{metric}: metric from station predictions")
            gate = gates[key].iloc[0]
            base = baseline[key].predicted.to_numpy(float)
            if actual.sum():
                np.testing.assert_allclose(gate.candidate_tv_mass, np.abs(actual-base).sum()/(2*actual.sum()), rtol=1e-12, atol=1e-12)
            _require(gate.changed_grid_count == np.count_nonzero(blob["assignment"] != station_ordinals[vd["assignment"]]), f"{tag}/{key}: changed-grid cluster basis differs")
            _require(gate.changed_grid_basis == "canonical_vd_station_cluster" and gate.station_prediction_rule == "equal_split_cluster_demand", f"{tag}/{key}: gate protocol missing")
            check = recorded[key]
            _require(check["status"] == "PASS" and check["n_stations"] == len(station_ids) and check["n_clusters"] == len(ordered), f"{tag}/{key}: recorded oracle status/counts differ")
            np.testing.assert_allclose([check["field_total"], check["prediction_total"]], [values.sum(), actual.sum()], rtol=1e-12, atol=1e-8)
            max_difference = max(max_difference, float(np.abs(actual-expected).max()))
            max_mass_difference = max(max_mass_difference, float(abs(actual.sum()-values.sum())))
        if len(tables["maps"]):
            mapped = tables["maps"].query("allocator == 'CIVD'")
            _require(mapped.assignment_kind.eq("cluster_ordinal").all() and len(mapped) == len(grid), f"{tag}: CIVD map semantics/coverage differ")
            np.testing.assert_array_equal(mapped.selected_assignment, blob["assignment"])
            np.testing.assert_array_equal(mapped.raw_assignment, blob["assignment"])
        checks.append({"country": country, "region": name, "fields": 78, "station_count": len(station_ids),
            "cluster_count": len(ordered), "w0_context_sha256": w0_digest, "max_oracle_difference": max_difference,
            "max_mass_difference": max_mass_difference, "baseline_values": "exactly_preserved",
            "recorded_execution_code_sha256": execution_code, "current_orchestration_code_sha256": expected_observation_code,
            "execution_matches_current_orchestration": execution_code == expected_observation_code,
            "numerical_validity_basis": "independent_recomputation_from_verified_frozen_inputs", "status": "PASS"})
        regional_receipts.append(regional)
        observations[f"observation:{country}:{name}"] = new["receipt_sha256"]
        print(f"AUDIT PASS {tag}: 78 independent field checks; frozen W0 context identical", flush=True)
    _require(len({receipt["commitment"]["code_sha256"] for receipt in regional_receipts}) == 1, f"{label}: mixed Generator code identities")
    return checks, [country_receipt, *regional_receipts], observations


def audit_release(repo, results, *, country_inputs, observation_code, scientific_invariants, verify_links, correction_id):
    """Audit complete artifacts and write its receipt without closing the writable root."""
    repo, results = Path(repo).resolve(), Path(results).resolve()
    if any(path for name in ("2_Generator", "3_Experiment", "4_Analysis", "5_Report")
           for path in (results / name / "_closures").glob("*")):
        raise ValueError("sealed results root refuses a new CIVD audit receipt")
    _require(results != repo / "results" and results.is_relative_to(repo / "results"), "audit requires a separate correction root")
    downstream = _json(results / "civd_downstream_validation.json")
    _require(downstream["status"] == "PASS" and downstream["C3_main_contrasts"] == "exactly_equal_to_formal", "downstream scientific validation did not PASS")
    _require(all(gate["status"] == "PASS" and not gate["unclaimed"] for gate in downstream["analysis_gates"].values()), "downstream artifact gates did not PASS")
    report = _json(results / "5_Report/audit.json")
    _require(report == downstream["report_audit"] and report["status"] == "PASS", "report audit differs from downstream validation")
    checks, generator_receipts, inputs = [], [], {}
    for country, directory in COUNTRIES.items():
        _require(_json(results / "3_Experiment" / directory / "audit.json")["status"] == "PASS", f"{country}: Experiment coverage is incomplete")
        regional, receipts, observation_inputs = _audit_country(repo, results, country, directory, observation_code, country_inputs=country_inputs, correction_id=correction_id)
        checks.extend(regional)
        generator_receipts.extend(receipts)
        inputs.update(observation_inputs)
    _require(len(checks) == 53 and sum(row["fields"] for row in checks) == 53*78, "expected 53 regions and 4,134 field realizations")
    _require(len(generator_receipts) == 57, "expected 53 regional and four country Generator receipts")
    invariants = scientific_invariants(repo, results)
    links = verify_links(repo, results)
    _require(links == downstream["verified_parent_links"], "downstream parent links changed after validation")
    comparison_root = results / "civd_comparison"
    comparison = _receipt(comparison_root / "receipt.json")
    provenance = _json(comparison_root / "provenance.json")
    _verify_comparison_sources(repo, provenance, comparison)
    for relative, digest in provenance["outputs"].items():
        _hash(comparison_root / relative, digest)
        _require(comparison["outputs"].get(relative) == digest, f"comparison output is not receipt-bound: {relative}")
    _require(set(provenance["outputs"]) | {"provenance.json"} == set(comparison["outputs"]), "comparison output inventory differs from provenance")
    for country, directory in COUNTRIES.items():
        parent = _receipt(results / "4_Analysis" / directory / "C3/receipt.json")
        _require(comparison["commitment"]["inputs"][f"analysis:{country}:C3"] == parent["receipt_sha256"], f"comparison references stale C3 parent: {country}")
    inputs.update({"comparison": comparison["receipt_sha256"], "downstream_validation": sha256_file(results / "civd_downstream_validation.json"),
                   "correction_scope": sha256_file(results / "correction_scope.json"), "report_audit": sha256_file(results / "5_Report/audit.json")})
    generator_closure = derive_chain_closure(generator_receipts)
    execution_groups = {}
    for row in checks:
        execution_groups.setdefault(row["recorded_execution_code_sha256"], []).append(f"{row['country']}/{row['region']}")
    document = {"status": "PASS", "correction_id": correction_id,
        "production_event": "structural_refactor_regeneration", "regions": 53, "field_realizations": 53*78,
        "generator_receipts": 57, "generator_closure_sha256": generator_closure["closure_sha256"],
        "regional_checks": checks, "scientific_invariant_tables": len(invariants), "verified_parent_links": links,
        "comparison_receipt_sha256": comparison["receipt_sha256"],
        "execution_code_traceability": {"recorded_execution_groups": execution_groups,
            "current_orchestration_code_sha256": observation_code,
            "regions_matching_current_orchestration": sum(row["execution_matches_current_orchestration"] for row in checks),
            "different_execution_hashes": len(execution_groups),
            "receipt_rewriting": False,
            "interpretation": "All locally regenerated observations carry the current execution identity and independently pass frozen-input, station-prediction, metric, gate and exact-baseline checks. Inherited input artifacts retain their original production identities."},
        "frozen_context_basis": "original_W0_projection_including_grid_station_coordinates_demand_capacity_and_identity",
        "oracle": "independent_mask_sum_and_equal_station_split_from_rehashed_frozen_candidate_archives",
        "baseline_check": "all original VD/IDR table columns exactly preserved"}
    audit_path = results / "civd_release_audit.json"
    atomic_json(document, audit_path)
    code = code_projection({name: value for name, value in globals().items() if callable(value) and getattr(value, "__module__", None) == __name__})
    commitment = derive_chain_commitment("CIVD.four_country_release_audit", inputs=inputs,
        scientific_parameters={"correction_id": correction_id, "regions": 53, "fields_per_region": 78,
            "oracle_rtol": 1e-12, "oracle_atol": 1e-10, "mass_atol": 1e-8, "baseline_comparison": "exact"},
        code_sha256=code["code_sha256"])
    atomic_json(derive_chain_receipt(commitment, outputs={audit_path.name: sha256_file(audit_path)}, observations={"code_projection": code}),
        results / "civd_release_audit.receipt.json")
    return document
