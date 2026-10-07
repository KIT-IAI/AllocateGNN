"""只消费 Generator 的现成 assignment；完整保留 selected / raw / 回退观测。"""

import numpy as np
import pandas as pd

from sglib.core.algorithms.reconstruction_metrics import reconstruction_metrics


def observe(regions, fixed, matched, civd, canonical):
    fixed = {item.region: item for item in fixed}
    matched = {(item.region, item.candidate, item.seed): item for item in matched}
    civd = {item.region: item for item in civd}
    canonical_views = {item.region: item for item in canonical}
    predictions, metrics, gates, maps = [], [], [], []
    for region in regions:
        if region.region not in fixed:
            raise ValueError("C3 缺少 fixed assignment")
        if region.region not in canonical_views:
            raise ValueError("C3 缺少 canonical VD lineage")
        vd = canonical_views[region.region]
        if not np.array_equal(vd.assignment, region.assignment) or not np.array_equal(vd.target_ids.astype(str), region.station_ids.astype(str)):
            raise ValueError("C3 canonical VD 与台账行序不符")
        n = len(region.observed)
        if region.region in civd:
            view = civd[region.region]
            station_cluster = np.asarray(view.station_cluster)
            if station_cluster.shape != (n,) or not n or station_cluster.dtype.kind not in "iu" or np.any(station_cluster < 0):
                raise ValueError("C3 CIVD station_cluster 与站点行序不符")
            if not np.array_equal(view.assignment, view.grid_cluster):
                raise ValueError("C3 CIVD assignment 与 grid_cluster 不符")
            # Generator assignment indexes the sorted cluster labels, not station rows.
            _, station_cluster_index, cluster_counts = np.unique(station_cluster, return_inverse=True, return_counts=True)
        for field in region.fields:
            if field.qa_only:
                continue
            key = (region.region, field.label, field.seed)
            if key not in matched:
                raise ValueError(f"C3 缺少 matched assignment：{key}")
            public, paired = fixed[region.region], matched[key]
            contexts = [("VD", region.assignment, region.assignment, {"selected_mode": "canonical_vd", "sha256": vd.lineage["sha256"]}),
                        ("IDR-fixed", public.assignment, public.raw_assignment, public.gate),
                        ("IDR-matched", paired.assignment, paired.raw_assignment, paired.gate)]
            if region.region in civd:
                view = civd[region.region]
                contexts.append(("CIVD", view.assignment, view.assignment, {**view.metadata, "selected_mode": "civd"}))
            canonical = np.bincount(region.assignment, weights=field.values, minlength=n)
            for allocator, assignment, raw, gate in contexts:
                assignment, raw = np.asarray(assignment), np.asarray(raw)
                n_assignment_targets = len(cluster_counts) if allocator == "CIVD" else n
                if assignment.shape != field.values.shape or assignment.dtype.kind not in "iu" or np.any((assignment < 0) | (assignment >= n_assignment_targets)):
                    raise ValueError("C3 assignment 与场 / target 身份不符")
                if allocator == "CIVD":
                    # Step 4: aggregate cluster demand, then split equally in station row order.
                    cluster_totals = np.bincount(assignment, weights=field.values, minlength=len(cluster_counts))
                    predicted = (cluster_totals / cluster_counts)[station_cluster_index]
                    canonical_assignment = station_cluster_index[region.assignment]
                else:
                    predicted = np.bincount(assignment, weights=field.values, minlength=n)
                    canonical_assignment = region.assignment
                base = {"country": region.country, "region": region.region, "candidate": field.label, "seed": field.seed,
                        "fold": field.fold, "allocator": allocator, "unit": region.unit,
                        "field_sha256": field.lineage["sha256"], "assignment_sha256": gate["sha256"]}
                predictions.append(pd.DataFrame({**base, "target_id": region.station_ids.astype(str), "observed": region.observed,
                                                  "predicted": predicted, "source_id": region.station_sources.astype(str)}))
                for metric, record in reconstruction_metrics(region.observed, predicted).items():
                    metrics.append({**base, "metric": metric, **record, "n_targets": n,
                                    "metric_unit": region.unit if metric in {"rmse", "mae"} else "dimensionless"})
                fallback = allocator.startswith("IDR") and (not gate["g0_pass"] or not gate["g1_pass"])
                if fallback and not np.array_equal(assignment, region.assignment):
                    raise ValueError("C3 回退 assignment 不等于 canonical VD")
                if allocator == "IDR-matched" and gate["candidate_sha256"] != field.lineage["sha256"]:
                    raise ValueError("C3 matched 门引用了不同的候选场")
                gates.append({**base, "selected_mode": gate["selected_mode"], "fallback": fallback,
                    "g0_pass": gate.get("g0_pass"), "g1_pass": gate.get("g1_pass"), "gate_tv_mass": gate.get("tv_mass"),
                    "gate_mass_basis": gate.get("source_total_basis"), "transport_budget": gate.get("transport_budget"),
                    "candidate_tv_mass": float(np.abs(predicted - canonical).sum() / (2 * predicted.sum())) if predicted.sum() else None,
                    "changed_grid_count": int(np.count_nonzero(assignment != canonical_assignment)), "n_grid": len(assignment),
                    "changed_grid_basis": "canonical_vd_station_cluster" if allocator == "CIVD" else "canonical_vd_station_row",
                    "station_prediction_rule": "equal_split_cluster_demand" if allocator == "CIVD" else "aggregate_station_demand",
                    "max_station_mass_change": float(np.max(np.abs(predicted - canonical))), "fallback_reason": gate.get("fallback_reason", "")})
                if region.map_selected and field.label == "GNN" and field.seed == 42:
                    maps.append(pd.DataFrame({**base, "x": region.grid_xy[:, 0], "y": region.grid_xy[:, 1], "selected_assignment": assignment,
                                              "raw_assignment": raw, "assignment_kind": "cluster_ordinal" if allocator == "CIVD" else "station_row",
                                              "field_value": field.values}))
    return {"predictions": pd.concat(predictions, ignore_index=True), "metrics": pd.DataFrame(metrics),
            "gates": pd.DataFrame(gates), "maps": pd.concat(maps, ignore_index=True) if maps else pd.DataFrame()}
