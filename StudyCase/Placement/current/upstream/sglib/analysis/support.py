"""006 的机制、敏感性与覆盖支撑；不改变核心检验族。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .paired_inference import _bootstrap


def _interval(values, *, seed=20260906):
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        return None
    result = _bootstrap(values, np.zeros(len(values)), [[i] for i in range(len(values))],
                        seed=seed, repetitions=10000)
    return result["native_ci"], result["indices_sha256"]


def c2_support(correction, sweeps):
    """区域聚类汇总旧阈值、完整恒等式、六协议翻转和全扫描曲线。"""
    required = {"country", "region", "base", "corrected", "operator", "signal", "station_set",
                "epsilon_ratio", "identifiable", "identity_pass", "old_threshold_agrees_log",
                "old_threshold_agrees_rmse", "log_mse_help", "delta_rmse", "rho", "old_threshold",
                "mean_term", "variance_term", "deployment_status"}
    if not required.issubset(correction.columns):
        raise ValueError("C2 机制观测缺少必要列")
    cells = correction.copy()
    cells["protocol_id"] = cells.station_set.astype(str) + "@eps=" + cells.epsilon_ratio.astype(str)
    summary = []
    group_columns = ["country", "base", "operator", "signal", "station_set", "epsilon_ratio"]
    for keys, frame in cells.groupby(group_columns, sort=False, dropna=False):
        identifiable = frame.identifiable.astype(bool)
        valid = frame[identifiable]
        regional = valid.groupby("region", sort=False).agg(
            threshold_accuracy_log=("old_threshold_agrees_log", "mean"),
            threshold_accuracy_rmse=("old_threshold_agrees_rmse", "mean"),
            identity_pass_rate=("identity_pass", "mean"),
            positive_log_rate=("log_mse_help", "mean"),
        ).reset_index()
        interval = _interval(regional.threshold_accuracy_log.to_numpy(float)) if len(regional) else None
        positive = float(valid.log_mse_help.astype(bool).mean()) if len(valid) else None
        majority = max(positive, 1 - positive) if positive is not None else None
        summary.append({**dict(zip(group_columns, keys, strict=True)), "n_rows": len(frame),
            "n_regions": frame.region.nunique(), "n_identifiable": int(identifiable.sum()),
            "identifiable_share": float(identifiable.mean()),
            "threshold_accuracy_log_region_mean": float(regional.threshold_accuracy_log.mean()) if len(regional) else None,
            "threshold_accuracy_log_ci_low": interval[0][0] if interval else None,
            "threshold_accuracy_log_ci_high": interval[0][1] if interval else None,
            "threshold_accuracy_log_bootstrap_sha256": interval[1] if interval else None,
            "threshold_accuracy_rmse_region_mean": float(regional.threshold_accuracy_rmse.mean()) if len(regional) else None,
            "majority_class_baseline": majority,
            "accuracy_minus_majority": float(regional.threshold_accuracy_log.mean() - majority) if len(regional) and majority is not None else None,
            "identity_pass_rate": float(valid.identity_pass.astype(bool).mean()) if len(valid) else None,
            "mean_term_nonzero_share": float((valid.mean_term.abs() > valid.get("identity_tolerance", pd.Series(0., index=valid.index))).mean()) if len(valid) else None,
            "deployment_status": "not_assessable_retrospective_diagnostic"})
    protocol_keys = ["country", "region", "base", "corrected", "operator", "signal", "seed", "fold"]
    flips = []
    for keys, frame in cells.groupby(protocol_keys, sort=False, dropna=False):
        if len(frame) != 6 or frame.protocol_id.nunique() != 6:
            raise ValueError("C2 六协议观测缺失或重复")
        flips.append({**dict(zip(protocol_keys, keys, strict=True)), "n_protocols": 6,
            "log_direction_unique": frame.log_mse_help.astype(str).nunique(),
            "rmse_direction_unique": frame.linear_rmse_help.astype(str).nunique(),
            "threshold_direction_unique": frame.old_threshold_help.astype(str).nunique(),
            "log_direction_flipped": frame.log_mse_help.astype(str).nunique() > 1,
            "rmse_direction_flipped": frame.linear_rmse_help.astype(str).nunique() > 1,
            "threshold_direction_flipped": frame.old_threshold_help.astype(str).nunique() > 1,
            "identifiable_protocols": int(frame.identifiable.astype(bool).sum()),
            "all_identity_pass": bool(frame.identity_pass.astype(bool).all())})
    sweep_region = (sweeps.groupby(["country", "region", "parameter_name", "parameter_value", "signal", "metric", "evidence"],
                                    sort=False, dropna=False)
                    .agg(value=("value", "mean"), realization_count=("value", "size"),
                         valid_count=("status", lambda x: int(x.eq("VALID").sum())))
                    .reset_index())
    sweep_summary = (sweep_region.groupby(["country", "parameter_name", "parameter_value", "signal", "metric", "evidence"],
                                           sort=False, dropna=False)
                     .agg(mean=("value", "mean"), q25=("value", lambda x: x.quantile(.25, interpolation="linear")),
                          median=("value", "median"), q75=("value", lambda x: x.quantile(.75, interpolation="linear")),
                          n_regions=("region", "nunique"), realization_count=("realization_count", "sum"))
                     .reset_index())
    sweep_summary["parameter_selected"] = False
    sweep_summary["scope_note"] = np.where(sweep_summary.parameter_name.eq("lambda"), "seed42_fold1_test_subset",
        np.where(sweep_summary.parameter_name.eq("tau"), "seed42_four_folds", "registered_fixed_grid"))
    return {"C2_mechanism_cells": cells, "C2_mechanism_summary": pd.DataFrame(summary),
            "C2_protocol_flips": pd.DataFrame(flips), "C2_sweep_region_curves": sweep_region,
            "C2_sweep_curves": sweep_summary}


def c3_support(gates, t1_region, t1_station, region_metrics, *, correction_scope=None):
    gate_summary = (gates.groupby(["country", "candidate", "allocator"], sort=False, dropna=False)
                    .agg(produced=("assignment_sha256", "size"), regions=("region", "nunique"),
                         changed=("changed_grid_count", lambda x: int((x > 0).sum())),
                         fallback=("fallback", "sum"), g0_pass=("g0_pass", "sum"), g1_pass=("g1_pass", "sum"),
                         tv_mean=("candidate_tv_mass", "mean"), tv_max=("candidate_tv_mass", "max"),
                         max_station_mass_change=("max_station_mass_change", "max"))
                    .reset_index())
    gate_summary["fallback_share"] = gate_summary.fallback / gate_summary.produced
    gate_summary["evidence"] = "complete_program_includes_fallback"
    if len(t1_region):
        t1_summary = (t1_region.groupby(["country", "allocator", "candidate", "assignment_kind", "status", "reason"], sort=False, dropna=False)
                      .agg(regions=("region", "nunique"), valid_regions=("status", lambda x: int(x.eq("VALID").sum())),
                           mean_iou_loss_q80=("iou_loss_q80", "mean"), mean_matched_fraction=("matched_fraction", "mean"))
                      .reset_index())
        t1_summary["scope"] = "UK_eight_independent_non_VD_reference_regions"
    else:
        t1_summary = pd.DataFrame(columns=["country", "allocator", "candidate", "assignment_kind", "status", "reason", "regions", "valid_regions",
                                                "mean_iou_loss_q80", "mean_matched_fraction", "scope"])
    civd = region_metrics[region_metrics.allocator.eq("CIVD")].copy()
    if len(civd):
        civd_summary = (civd.groupby(["country", "candidate", "metric", "status"], sort=False, dropna=False)
                        .agg(n_regions=("region", "nunique"), mean=("value", "mean"), unit=("unit", "first"))
                        .reset_index())
        civd_summary["evidence"] = correction_scope or "historical_defense_not_four_country_core_axis"
    else:
        civd_summary = pd.DataFrame(columns=["country", "candidate", "metric", "status", "n_regions", "mean", "unit", "evidence"])
    duplicates = pd.DataFrame([{"task": task, "allocator": allocator, "status": "STRUCTURAL_DUPLICATE",
        "canonical_allocator": "VD", "reason": "PLANNING_OPERATOR_DOES_NOT_CONSUME_STATION_ALLOCATOR"}
        for task in ("siting", "sizing", "connection") for allocator in ("IDR-fixed", "IDR-matched")])
    return {"C3_gate_summary": gate_summary, "C3_gate_observations": gates,
            "C3_T1_region": t1_region, "C3_T1_summary": t1_summary,
            "C3_T1_station": t1_station, "C3_CIVD_summary": civd_summary,
            "C3_structural_duplicates": duplicates}


def _regional_levels(reconstruction, planning, connection):
    rows = []
    recon = reconstruction[reconstruction.metric.eq("rmse") & reconstruction.allocator.isin(["VD", "DIRECT"])]
    for (country, region, candidate), frame in recon.groupby(["country", "region", "candidate"], sort=False):
        valid = frame.status.eq("VALID").all() and np.isfinite(frame.value.to_numpy(float)).all()
        rows.append({"country": country, "region": region, "candidate": candidate, "task_coordinate": "reconstruction",
                     "metric": "rmse", "value": float(frame.value.mean()) if valid else None,
                     "status": "VALID" if valid else "METRIC_NOT_ASSESSABLE", "unit": frame.metric_unit.iloc[0]})
    for task, metric, matching in (("siting", "WSD", "not_applicable"), ("sizing", "RSD_median", "many_to_one")):
        selected = planning[planning.task.eq(task) & planning.metric.eq(metric) & planning.matching.eq(matching)]
        for (country, region, candidate), frame in selected.groupby(["country", "region", "candidate"], sort=False):
            valid = frame.status.eq("VALID").all() and np.isfinite(frame.value.to_numpy(float)).all()
            rows.append({"country": country, "region": region, "candidate": candidate, "task_coordinate": task,
                "metric": metric, "value": float(frame.value.mean()) if valid else None,
                "status": "VALID" if valid else "METRIC_NOT_ASSESSABLE", "unit": frame.unit.iloc[0]})
    for (country, region, candidate, radius, load), frame in connection.groupby(
            ["country", "region", "candidate", "radius_km", "lambda"], sort=False, dropna=False):
        eligible = frame.c4_eligible.astype(bool).all()
        for metric in ("L_E_over_X", "L_S_over_X"):
            valid = eligible and frame.status.eq("VALID").all() and np.isfinite(frame[metric].to_numpy(float)).all()
            rows.append({"country": country, "region": region, "candidate": candidate,
                "task_coordinate": f"connection_R{float(radius):g}_lambda{float(load):g}", "metric": metric,
                "value": float(frame[metric].mean()) if valid else None,
                "status": "VALID" if valid else "INELIGIBLE_BY_DESIGN" if not eligible else "METRIC_NOT_ASSESSABLE",
                "unit": "dimensionless"})
    return pd.DataFrame(rows)


def _paired_secondary(levels, left, right, task, metric, *, experiment_id):
    selected = levels[(levels.task_coordinate == task) & (levels.metric == metric) & levels.candidate.isin([left, right])]
    wide = selected.pivot(index="region", columns="candidate", values="value") if len(selected) else pd.DataFrame()
    if left not in wide or right not in wide:
        return {"experiment_id": experiment_id, "left": left, "right": right, "task_coordinate": task,
                "metric": metric, "n_valid": 0, "status": "MISSING_REQUIRED", "reason": "METHOD_LEVEL_NOT_PRODUCED"}
    valid = wide[[left, right]].notna().all(axis=1)
    a, b = wide.loc[valid, left].to_numpy(float), wide.loc[valid, right].to_numpy(float)
    if not len(a):
        return {"experiment_id": experiment_id, "left": left, "right": right, "task_coordinate": task,
                "metric": metric, "n_valid": 0, "status": "METRIC_NOT_ASSESSABLE", "reason": "NO_COMMON_VALID_REGIONS"}
    result = _bootstrap(a, b, [[i] for i in range(len(a))], seed=20260906, repetitions=10000)
    unit = selected.unit.dropna().iloc[0]
    return {"country": selected.country.iloc[0], "experiment_id": experiment_id, "left": left, "right": right,
        "task_coordinate": task, "metric": metric, "unit": unit, "n_valid": len(a),
        "effect": float((a-b).mean()), "ci_low": result["native_ci"][0], "ci_high": result["native_ci"][1],
        "relative_pct": float(100*(a-b).sum()/b.sum()) if b.sum() else None,
        "relative_ci_low": result["relative_ci"][0] if result["relative_ci"] else None,
        "relative_ci_high": result["relative_ci"][1] if result["relative_ci"] else None,
        "zero_denominator_draws": result["zero_denominator_draws"],
        "bootstrap_indices_sha256": result["indices_sha256"], "status": "VALID", "reason": "",
        "inference_mode": "secondary_effect_and_interval_without_test", "p": None, "holm_p": None}


def c4_support(reconstruction, planning, connection, fixed_load):
    levels = _regional_levels(reconstruction, planning, connection)
    tasks = [("reconstruction", "rmse"), ("siting", "WSD"), ("sizing", "RSD_median")]
    tasks += [(f"connection_R10_lambda{load:g}", metric) for load in (.25, .5, 1.) for metric in ("L_E_over_X", "L_S_over_X")]
    rows = []
    for left, right in (("MLP", "GPM"), ("MLPpostN", "GPMpostN"), ("MLPaddN", "GPMaddN"),
                        ("GNNpostN", "GPMpostN"), ("GNNaddN", "GPMaddN")):
        rows.extend(_paired_secondary(levels, left, right, task, metric, experiment_id="C4-E03") for task, metric in tasks)
    for load in (.25, .5, 1.):
        for metric in ("L_E_over_X", "L_S_over_X"):
            rows.append(_paired_secondary(levels, "GNN", "GPM", f"connection_R20_lambda{load:g}", metric,
                                          experiment_id="C4-E03"))
    fixed_summary = []
    if len(fixed_load) and fixed_load.status.eq("VALID").any():
        fixed_levels = (fixed_load[fixed_load.status.eq("VALID")]
                        .groupby(["country", "region", "candidate", "radius_km"], sort=False, dropna=False)
                        .agg(L_E_over_X=("L_E_over_X", "mean"), L_S_over_X=("L_S_over_X", "mean"), unit=("unit", "first"))
                        .reset_index())
        for radius in (10., 20.):
            for metric in ("L_E_over_X", "L_S_over_X"):
                temp = fixed_levels.rename(columns={metric: "value"})
                temp["task_coordinate"] = f"fixed300_R{radius:g}"
                temp["metric"] = metric
                temp["status"] = "VALID"
                fixed_summary.append(_paired_secondary(temp[temp.radius_km.eq(radius)], "GNN", "GPM",
                    f"fixed300_R{radius:g}", metric, experiment_id="C4-E03"))
    else:
        fixed_summary.append({"experiment_id": "C4-E03", "task_coordinate": "fixed300", "status": "INELIGIBLE_BY_DESIGN",
                              "reason": fixed_load.reason.iloc[0] if len(fixed_load) else "NOT_PRODUCED"})
    scenario = (connection.groupby(["country", "region", "radius_km", "lambda"], sort=False)
                .agg(X=("X", "first"), unit=("unit", "first"), ref_eligible=("ref_eligible", "all"),
                     c4_eligible=("c4_eligible", "all"), true_zero_q_fraction=("true_zero_q_fraction", "mean"),
                     estimated_zero_q_fraction=("estimated_zero_q_fraction", "mean"),
                     cutoff_tie_size_max=("cutoff_tie_size", "max"))
                .reset_index())
    matching = planning[planning.task.eq("sizing") & planning.matching.isin(["many_to_one", "one_to_one"])].copy()
    matching_summary = (matching.groupby(["country", "candidate", "metric", "matching", "unit"], sort=False, dropna=False)
                        .agg(mean=("value", "mean"), n_regions=("region", "nunique"), valid=("status", lambda x: int(x.eq("VALID").sum())))
                        .reset_index())
    statuses = pd.DataFrame([
        {"experiment_id": "C4-E03", "branch": "R20_and_enhanced_static_learning", "status": "VALID", "reason": ""},
        {"experiment_id": "C4-E03", "branch": "fixed_300_native_unit", "status": "VALID" if fixed_load.status.eq("VALID").any() else "INELIGIBLE_BY_DESIGN",
         "reason": "" if fixed_load.status.eq("VALID").any() else fixed_load.reason.iloc[0]},
        {"experiment_id": "C4-E03", "branch": "UK_GBP_TNUoS", "status": "MISSING_REQUIRED",
         "reason": "VERSIONED_TARIFF_AND_CONVERSION_CONTRACT_NOT_PROVIDED"},
        {"experiment_id": "C4-E04", "branch": "matching_many_to_one_vs_one_to_one", "status": "VALID", "reason": ""},
        {"experiment_id": "C4-E04", "branch": "legacy_station_operator", "status": "NOT_EVALUATED",
         "reason": "LEGACY_OPERATOR_NOT_REUSED_AS_CURRENT_PROTOCOL"},
        {"experiment_id": "C4-E05", "branch": "MILP_cross_check", "status": "NOT_AUTHORIZED",
         "reason": "MILP_EXPLICITLY_OUT_OF_SCOPE"}])
    return {"C4_secondary_effects": pd.DataFrame(rows + fixed_summary), "C4_regional_levels": levels,
            "C4_scenario_summary": scenario, "C4_matching_summary": matching_summary,
            "C4_defense_status": statuses}


def c5_support(panel_metrics, panel_associations, scale_metrics):
    scale = (scale_metrics.groupby(["country", "candidate", "seed", "radius_km", "centre_kind", "target_kind", "metric", "unit"],
                                  sort=False, dropna=False)
             .agg(mean=("value", "mean"), q25=("value", lambda x: x.quantile(.25, interpolation="linear")),
                  median=("value", "median"), q75=("value", lambda x: x.quantile(.75, interpolation="linear")),
                  n_regions=("region", "nunique"), valid=("status", lambda x: int(x.eq("VALID").sum())))
             .reset_index())
    support = (panel_metrics.groupby(["country", "radius_km", "lambda"], sort=False)
               .agg(regions=("region", "nunique"), ref_eligible_rows=("ref_eligible", "sum"),
                    scenario_defined_rows=("scenario_defined", "sum"),
                    true_constant_rows=("q_true_constant", "sum"), estimated_constant_rows=("q_estimated_constant", "sum"),
                    mean_true_zero_q_fraction=("true_zero_q_fraction", "mean"),
                    mean_estimated_zero_q_fraction=("estimated_zero_q_fraction", "mean"))
               .reset_index())
    sensitivity = panel_associations[~panel_associations.primary].copy()
    return {"C5_scale_curves": scale, "C5_support_summary": support,
            "C5_panel_associations": panel_associations, "C5_sensitivity": sensitivity}


def c6_support(bounds, bounds_audit, tolerance_curves):
    calibration = bounds[["country", "region", "candidate", "seed", "radius_km", "eta", "calibration_regions",
                          "n_calibration", "source_total", "budget_normalized", "budget", "realized_normalized_error",
                          "budget_realized", "maximum_normalized_error_ties", "unit"]].drop_duplicates()
    saturation = (bounds.groupby(["country", "candidate", "seed", "radius_km", "lambda", "eta"], sort=False, dropna=False)
                  .agg(n_total=("region", "size"), n_assessable=("assessable", "sum"),
                       n_budget_realized=("budget_realized", "sum"), mean_slack_E=("slack_E", "mean"),
                       mean_slack_S=("slack_S", "mean"), mean_true_zero_q=("true_zero_q_fraction", "mean"),
                       mean_estimated_zero_q=("estimated_zero_q_fraction", "mean"),
                       zero_actual_E=("L_E", lambda x: int((x == 0).sum())),
                       zero_actual_S=("L_S", lambda x: int((x == 0).sum())),
                       zero_bound_E=("B_E", lambda x: int((x == 0).sum())),
                       zero_bound_S=("B_S", lambda x: int((x == 0).sum())))
                  .reset_index())
    status = pd.DataFrame([
        {"branch": "normalized_tolerance_curve", "status": "VALID", "reason": ""},
        {"branch": "planner_tolerance", "status": "NOT_PROVIDED", "reason": "NO_EXTERNAL_PLANNER_TOLERANCE"},
        {"branch": "external_budget", "status": "NOT_PROVIDED", "reason": "RETROSPECTIVE_LOO_BUDGET_ONLY"},
        {"branch": "conditional_bound_violations", "status": "VALID" if not bounds.conditional_violation.any() else "FAILED_PRODUCTION",
         "reason": "" if not bounds.conditional_violation.any() else "CONDITIONAL_BOUND_VIOLATION"}])
    return {"C6_calibration": calibration, "C6_saturation": saturation,
            "C6_bounds_audit": bounds_audit, "C6_tolerance_curves": tolerance_curves,
            "C6_support_status": status}




def evidence_status(country, core, support_tables):
    """给每条主张生成同源 expected/produced/eligible/valid/fallback 状态。"""
    from .config import DIRECTORIES
    directory = DIRECTORIES[country]
    rows = []
    for claim in ("C1", "C2", "C3", "C4", "C5"):
        contrasts = core[claim]["contrasts"]
        expected = {"C1": 10, "C2": 13, "C3": 5, "C4": 9, "C5": 2}[claim]
        produced = len(contrasts)
        valid = int(contrasts.status.eq("VALID").sum()) if "status" in contrasts else produced
        rows.append({"country": country, "claim_id": claim, "expected": expected, "produced": produced,
            "eligible": valid, "valid": valid, "fallback": int(support_tables.get("C3_gate_observations", pd.DataFrame()).fallback.sum()) if claim == "C3" else 0,
            "status": "VALID" if produced == expected and valid == expected else "METRIC_NOT_ASSESSABLE",
            "reason": "" if produced == expected and valid == expected else "PARTIAL_OR_SCIENTIFICALLY_INELIGIBLE",
            "locator": f"4_Analysis/{directory}/{claim}/receipt.json"})
    c6 = core["C6"]["bounds_audit"]
    rows.append({"country": country, "claim_id": "C6", "expected": len(c6), "produced": len(c6),
        "eligible": int((c6.n_assessable > 0).sum()), "valid": int((c6.n_budget_realized > 0).sum()), "fallback": 0,
        "status": "VALID" if not c6.conditional_violations.any() else "FAILED_PRODUCTION",
        "reason": "NO_PLANNER_TOLERANCE_OR_EXTERNAL_BUDGET" if not c6.conditional_violations.any() else "CONDITIONAL_BOUND_VIOLATION",
        "locator": f"4_Analysis/{directory}/C6/receipt.json"})
    return pd.DataFrame(rows)
