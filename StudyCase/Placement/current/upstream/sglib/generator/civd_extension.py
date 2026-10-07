"""Explicit four-country CIVD extension over the sealed formal inputs.

This orchestration helper never changes formal Generator results or defaults.
UK/AU reuse their frozen assignments; NL/NZ run the same capacity-weighted
algorithm in each country's contracted working CRS. All outputs live under
the caller's dedicated staging root and carry auditable station identities.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from sglib.core.infra.artifacts import atomic_json, atomic_npz
from sglib.core.infra.content_chain import code_projection, derive_chain_commitment, derive_chain_receipt, verify_chain
from sglib.core.infra.hashing import sha256_file, sha256_json
from sglib.core.infra.paths import portable_path
from sglib.generator.allocator.civd import CIVDAllocator, hdbscan_station_clusters, influence_matrix
from sglib.generator.handoff import CivdView
from sglib.generator.materialize.civd import materialize_civd


SCOPE = "four_country_extension_after_bug_correction"
WEIGHT_RULE = "max(max(station_capacity,1e-10)/max(distance_m,1e-10))_within_cluster"


def _verified_receipt(path, commitment):
    if not path.is_file():
        return None
    receipt = verify_chain(json.loads(path.read_text(encoding="utf-8")))
    if receipt["commitment"]["commitment_sha256"] != commitment["commitment_sha256"]:
        raise ValueError(f"CIVD rerun input commitment changed: {path}")
    for relative, digest in receipt["outputs"].items():
        artifact = (path.parent / relative).resolve()
        if not artifact.is_relative_to(path.parent.resolve()) or not artifact.is_file() or sha256_file(artifact) != digest:
            raise ValueError(f"CIVD rerun input output differs from its receipt: {artifact}")
    return receipt


def _validated_arrays(arrays, station_ids, n_grid):
    assignment = np.asarray(arrays["assignment"])
    clusters = np.asarray(arrays["station_cluster"])
    if clusters.shape != station_ids.shape or clusters.dtype.kind not in "iu" or np.any(clusters < 0):
        raise ValueError("CIVD station cluster rows differ from canonical VD target identities")
    count = len(np.unique(clusters))
    if assignment.shape != (n_grid,) or assignment.dtype.kind not in "iu" or np.any((assignment < 0) | (assignment >= count)):
        raise ValueError("CIVD assignment must contain ordered cluster indices")
    if not np.array_equal(assignment, arrays["grid_cluster"]):
        raise ValueError("CIVD assignment and grid_cluster disagree")
    if not np.array_equal(np.asarray(arrays["station_id"]).astype(str), station_ids):
        raise ValueError("CIVD saved station identities differ from canonical VD target order")
    if np.asarray(arrays["raw_labels"]).shape != station_ids.shape or np.asarray(arrays["probabilities"]).shape != station_ids.shape:
        raise ValueError("CIVD raw labels/probabilities differ from station identities")
    return count


def extend_country(repo, country, results, *, data, generator, regions):
    """Return a Generator handoff with frozen fields and an isolated CIVD extension.

    The returned Generator keeps its formal root and frozen inputs-receipt hash.
    Each CIVD view points to its own staging artifact and regional receipt;
    ``results/2_Generator/<directory>/civd/receipt.json`` binds the full extension.
    Calling again verifies completed regional products before reusing them.
    """
    repo, results = Path(repo).resolve(), Path(results).resolve()
    formal = repo / "results"
    if any((results / "2_Generator/_closures").glob("*")):
        raise ValueError("sealed Generator root refuses CIVD materialization")
    isolated = any(results.is_relative_to(formal / branch) and results != formal / branch for branch in ("_staging", "_releases"))
    if country not in {"uk", "au", "nl", "nz"} or not isolated:
        raise ValueError("CIVD extension requires a supported country and dedicated results/_staging or results/_releases child")
    directory = data.profile.directory
    if generator.root.resolve() != (formal / "2_Generator" / directory).resolve():
        raise ValueError("CIVD extension must consume the formal frozen Generator handoff")
    root = results / "2_Generator" / directory
    output = root / "civd"
    if not output.resolve().is_relative_to(results):
        raise ValueError("CIVD output escapes its staging root")
    output.mkdir(parents=True, exist_ok=True)
    contract = data.profile.station_contract
    working_crs, capacity_column = str(data.profile.crs["working"]), str(contract["capacity_column"])
    if generator.config["crs"]["working"] != working_crs or generator.config["station_contract"]["capacity_column"] != capacity_column:
        raise ValueError("formal Generator and DataOverview capacity/CRS contracts differ")
    protocol = {"scope": SCOPE, "country": country, "working_crs": working_crs,
        "clustering": "hdbscan(min_cluster_size=2,min_samples=None,eom,core_dist_n_jobs=1)",
        "noise_rule": "each_noise_station_becomes_singleton", "cluster_assignment": "sorted_cluster_label_ordinal",
        "weight_rule": WEIGHT_RULE, "capacity_column": capacity_column, "capacity_basis": contract["capacity_basis"],
        "capacity_floor": 1e-10, "distance_floor_m": 1e-10,
        "station_prediction_rule": "equal_split_cluster_demand", "station_identity": "canonical_VD_target_ids",
        "extension_status": "extension_after_bug_correction", "historically_preregistered": False}
    code = code_projection({"loader": extend_country, "validate": _validated_arrays,
        "materialize": materialize_civd, "clustering": hdbscan_station_clusters,
        "allocator": CIVDAllocator.allocate, "influence": influence_matrix})
    ledger = data.stations_table.copy()
    ledger[contract["id_column"]] = ledger[contract["id_column"]].astype(str)
    ledger = ledger.set_index(contract["id_column"], verify_integrity=True)
    expected_regions = tuple(regions)
    region_data_by_name = {item.region: item for item in data.regions}
    frozen = {item.region: item for item in generator.bundle.civd}
    vds = {item.region: item for item in generator.bundle.vd}
    if set(vds) != set(expected_regions) or set(vds) != set(region_data_by_name):
        raise ValueError("CIVD extension region coverage differs from frozen inputs")
    if country in {"uk", "au"} and set(frozen) != set(vds):
        raise ValueError("UK/AU CIVD extension must reuse every frozen regional assignment")
    if country in {"uk", "au"}:
        old_index = json.loads((generator.root / "civd/index.json").read_text(encoding="utf-8"))
        if old_index["working_crs"] != working_crs:
            raise ValueError("frozen UK/AU CIVD working CRS differs from the country contract")
    entries, views, country_inputs = [], [], {"formal_generator_inputs": generator.inputs_receipt_sha256}
    for region in expected_regions:
        vd, region_data = vds[region], region_data_by_name[region]
        station_ids = vd.target_ids.astype(str)
        stations = ledger.loc[station_ids].copy()
        if capacity_column not in stations or not np.isfinite(stations[capacity_column].to_numpy(float)).all():
            raise ValueError(f"{country}/{region}: invalid contracted station capacities")
        grid_xy, station_xy = region_data.grid.to_crs(working_crs), stations.to_crs(working_crs)
        context_hash = sha256_json({"station_id": station_ids.tolist(), "working_crs": working_crs,
            "grid_xy": np.column_stack((grid_xy.geometry.x, grid_xy.geometry.y)).tolist(),
            "station_xy": np.column_stack((station_xy.geometry.x, station_xy.geometry.y)).tolist(),
            "capacity": stations[capacity_column].to_numpy(float).tolist()})
        reused = country in {"uk", "au"}
        regional_protocol = {**protocol, "region": region,
            "assignment_origin": "reused_frozen_UK_AU" if reused else "new_NL_NZ_extension"}
        inputs = {"formal_generator_inputs": generator.inputs_receipt_sha256,
                  "canonical_vd": vd.lineage["sha256"], "region_context": context_hash}
        if reused:
            inputs["frozen_civd"] = frozen[region].metadata["sha256"]
        commitment = derive_chain_commitment(f"{country}.civd_extension.{region}", inputs=inputs,
            scientific_parameters=regional_protocol, code_sha256=code["code_sha256"])
        path, receipt_path = output / f"{region}.npz", output / f"{region}.receipt.json"
        if path.parent.resolve() != output.resolve():
            raise ValueError("CIVD region name is not a direct artifact filename")
        existing = _verified_receipt(receipt_path, commitment)
        if existing is None:
            if reused:
                view = frozen[region]
                arrays = {name: np.asarray(getattr(view, name)) for name in
                          ("assignment", "grid_cluster", "station_cluster", "raw_labels", "probabilities")}
            else:
                result = materialize_civd(region_data.grid, stations.reset_index(),
                    working_crs=working_crs, capacity_column=capacity_column)
                arrays = {name: result[name] for name in
                          ("assignment", "grid_cluster", "station_cluster", "raw_labels", "probabilities")}
            arrays["station_id"] = station_ids
            _validated_arrays(arrays, station_ids, len(region_data.grid))
            atomic_npz(path, **arrays)
            receipt = derive_chain_receipt(commitment, outputs={path.name: sha256_file(path)},
                observations={"n_grid": len(region_data.grid), "n_stations": len(stations),
                              "scope": SCOPE, "code_projection": code})
            atomic_json(receipt, receipt_path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        count = _validated_arrays(arrays, station_ids, len(region_data.grid))
        entry = {"region": region, "path": path.relative_to(root).as_posix(), "sha256": sha256_file(path),
            "bytes": path.stat().st_size, "n_clusters": count, "n_noise": int(np.count_nonzero(arrays["raw_labels"] < 0)),
            "n_grid": len(region_data.grid), "n_stations": len(stations),
            "artifact_root": root.relative_to(repo).as_posix(), "station_id_sha256": sha256_json(station_ids.tolist()),
            "receipt_sha256": sha256_file(receipt_path), "protocol": regional_protocol,
            "canonical_vd_sha256": vd.lineage["sha256"],
            "frozen_assignment_sha256": frozen[region].metadata["sha256"] if reused else None}
        entries.append(entry)
        views.append(CivdView(region, arrays["grid_cluster"], arrays["station_cluster"], arrays["raw_labels"],
                              arrays["probabilities"], entry, arrays["assignment"]))
        country_inputs[f"region:{region}"] = entry["receipt_sha256"]
        print(f"CIVD INPUT {country}/{region}: clusters={count} stations={len(stations)} origin={regional_protocol['assignment_origin']}", flush=True)
    index_path = output / "index.json"
    index = {"schema_version": "sg_civd_extension_index_v1", "country": country,
             "working_crs": working_crs, "protocol": protocol, "entries": entries}
    country_commitment = derive_chain_commitment(f"{country}.civd_extension.inputs", inputs=country_inputs,
        scientific_parameters=protocol, code_sha256=code["code_sha256"])
    existing = _verified_receipt(output / "receipt.json", country_commitment)
    if existing is None:
        atomic_json(index, index_path)
        atomic_json(derive_chain_receipt(country_commitment, outputs={"index.json": sha256_file(index_path)},
            observations={"scope": SCOPE, "n_regions": len(entries), "formal_generator_root": portable_path(generator.root, repo),
                          "formal_inputs_receipt_sha256": generator.inputs_receipt_sha256}), output / "receipt.json")
    elif json.loads(index_path.read_text(encoding="utf-8")) != index:
        raise ValueError("verified CIVD extension index disagrees with the loaded regional products")
    extended = replace(generator, bundle=replace(generator.bundle, civd=tuple(views)))
    return extended


__all__ = ["SCOPE", "WEIGHT_RULE", "extend_country"]
