"""006 选址 / 定容纯计算：复用一个严格 k 解，逐指标处理零分母。"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from sglib.core.algorithms.planning_candidates import aggregate_weights_to_candidates
from sglib.core.algorithms.planning_geometry import haversine_distance_matrix
from sglib.core.algorithms.planning_metrics import compute_all_siting_metrics
from sglib.core.algorithms.planning_sizing import predict_substation_demand, recommend_capacity, match_to_real_substations, match_to_real_substations_unique
from .connection import reference_field
from .planning_pool import solve_strict


def sizing_scores(predicted, recommended, actual, capacity):
    p, q, d, f = [np.asarray(v, float) for v in (predicted, recommended, actual, capacity)]
    rows = []
    for name, numerator, denominator in (("RSD", np.abs(p-d), d), ("CE", np.abs(q-d), d),
                                       ("TUR", d, q), ("FCE", np.abs(q-f), f)):
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
        values = 100 * numerator[valid] / denominator[valid]
        for aggregate in ("mean", "median"):
            rows.append({"metric": name + "_" + aggregate, "value": float(np.mean(values) if aggregate == "mean" else np.median(values)) if len(values) else None,
                         "n_valid_matches": int(valid.sum()), "n_requested_matches": len(d), "unit": "percent",
                         "status": "VALID" if len(values) else "METRIC_NOT_ASSESSABLE",
                         "reason": "" if len(values) else "NO_MATCH_WITH_POSITIVE_METRIC_DENOMINATOR"})
    valid = np.isfinite(q) & np.isfinite(d) & (d > 0)
    for name, condition in (("UPR", q < d), ("OPR", q > 2 * d)):
        rows.append({"metric": name, "value": float(condition[valid].mean()) if valid.any() else None,
                     "n_valid_matches": int(valid.sum()), "n_requested_matches": len(d), "unit": "fraction",
                     "status": "VALID" if valid.any() else "METRIC_NOT_ASSESSABLE", "reason": "" if valid.any() else "NO_VALID_MATCH"})
    return rows


def observe_task(grid_lonlat, station_lonlat, truth, capacity, vd_assignment, field, pool, *, country, region, unit, capacity_basis, solver):
    values = np.asarray(field.values, float)
    k = len(truth)
    if values.shape != (len(grid_lonlat),) or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("规划场必须为正总量且逐格非负有限")
    if len(pool["candidate_grid_rows"]) < 2 * k:
        raise ValueError("规划池未满足 D3 去重后至少 2k 门")
    compatible = SimpleNamespace(candidate_indices=pool["candidate_grid_rows"], labels=pool["labels"], buildable_indices=pool["buildable_grid_rows"])
    weights = values / values.sum()
    candidate_weights = aggregate_weights_to_candidates(weights, compatible)
    selected, _ = solve_strict(pool["candidate_lonlat"], pool["candidate_lonlat"], candidate_weights, k, solver)
    selected = np.sort(selected)
    selected_coords = pool["candidate_lonlat"][selected]
    assignment = np.empty(len(grid_lonlat), dtype=np.int64)
    # 分块只改变内存占用；每行 haversine 与 first-index 最近邻定义不变。
    for start in range(0, len(grid_lonlat), 2048):
        assignment[start:start+2048] = haversine_distance_matrix(grid_lonlat[start:start+2048], selected_coords).argmin(axis=1)
    reference = reference_field(vd_assignment, truth)
    siting = compute_all_siting_metrics(grid_lonlat, selected_coords, assignment, reference, (1., 2., 5.))
    predicted = predict_substation_demand(assignment, weights, float(values.sum()), k)
    recommended = recommend_capacity(predicted, safety_margin=1.5, discretise_to=None)
    base = {"country": country, "region": region, "candidate": field.label, "seed": field.seed, "fold": field.fold,
            "field_sha256": field.lineage["sha256"], "k": k, "capacity_basis": capacity_basis,
            "native_demand_unit": unit, "capacity_calculation_note": "PF1_MVA_equivalent" if country == "au" else "native_same_unit"}
    metrics = [{**base, "task": "siting", "matching": "not_applicable", "metric": name, "value": value,
                "unit": "km" if name == "WSD" else "km*" + unit if name == "LWFL" else "dimensionless",
                "status": "VALID", "reason": "", "n_valid_matches": k, "n_requested_matches": k} for name, value in siting.items()]
    details = []
    for matching, function in (("many_to_one", match_to_real_substations), ("one_to_one", match_to_real_substations_unique)):
        returned = function(selected_coords, station_lonlat, truth, capacity, 3., 20.)
        actual, firm, distances, low = returned[:4]
        metrics.extend({**base, "task": "sizing", "matching": matching, **row} for row in sizing_scores(predicted, recommended, actual, firm))
        details.append(pd.DataFrame({**base, "matching": matching, "facility_ordinal": np.arange(k),
            "candidate_grid_row": pool["candidate_grid_rows"][selected], "longitude": selected_coords[:, 0], "latitude": selected_coords[:, 1],
            "predicted_demand": predicted, "recommended_capacity": recommended, "matched_actual": actual, "matched_capacity": firm,
            "match_distance_km": distances, "low_confidence": low, "collision_rate": returned[4] if len(returned) > 4 else None}))
    return {"metrics": pd.DataFrame(metrics), "decisions": pd.concat(details, ignore_index=True)}, {"selected_candidate_indices": selected, "grid_assignment": assignment}
