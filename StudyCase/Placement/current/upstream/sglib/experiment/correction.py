"""C2 的回顾性因子—残差诊断；完整保留原始 log loss 的均值项。"""

import numpy as np
import pandas as pd


def correction_pairs(registry):
    definitions = {item["label"]: item for item in registry["candidates"] if item["materialize"] and not item["qa_only"]}
    pairs = []
    for label, item in definitions.items():
        if item["operator"] not in {"multiply", "add"}:
            continue
        base = "EqualGrid" if item["family"] == "Equal" else item["family"]
        if base not in definitions:
            raise ValueError("修正臂缺少正式基座")
        seeds = item["seed_policy"]["seeds"] or [None]
        if seeds != (definitions[base]["seed_policy"]["seeds"] or [None]):
            raise ValueError("修正与基座 seed 合同不同")
        pairs.append({"base": base, "corrected": label, "family": item["family"], "operator": item["operator"],
                      "signal": {"ntl": "N", "proximity": "P", "ntl_proximity": "NP"}[item["auxiliary"]], "seeds": seeds})
    return pairs


def coordinates(truth, baseline, corrected, epsilon):
    y, b, p = map(lambda value: np.asarray(value, float), (truth, baseline, corrected))
    if y.ndim != 1 or b.shape != y.shape or p.shape != y.shape or not all(np.isfinite(a).all() and (a >= 0).all() for a in (y, b, p)):
        raise ValueError("机制输入必须是对齐的非负有限向量")
    if not len(y) or epsilon <= 0 or not np.isfinite(epsilon):
        return {"status": "METRIC_NOT_ASSESSABLE", "reason": "EMPTY_STATION_SET_OR_ZERO_SOURCE_TOTAL"}
    r, c = np.log((y + epsilon) / (b + epsilon)), np.log((p + epsilon) / (b + epsilon))
    mr, mc, sr, sc = float(r.mean()), float(c.mean()), float(r.std()), float(c.std())
    covariance = float(np.mean((r - mr) * (c - mc)))
    identifiable = len(y) >= 2 and sr > 1e-12 and sc > 1e-12
    rho = float(np.clip(covariance / (sr * sc), -1., 1.)) if identifiable else None
    threshold = sc / (2 * sr) if identifiable else None
    variance_term, mean_term = sc**2 - 2 * covariance, mc**2 - 2 * mr * mc
    delta_log = float(np.mean(np.log((y + epsilon) / (p + epsilon))**2) - np.mean(r**2))
    residual = delta_log - variance_term - mean_term
    tolerance = 1e-10 * max(1., abs(delta_log), abs(variance_term + mean_term))
    if abs(residual) > tolerance:
        raise ValueError("log-MSE 精确恒等式审计失败")
    old_help = rho > threshold if identifiable else None
    delta_rmse = float(np.sqrt(np.mean((y - p)**2)) - np.sqrt(np.mean((y - b)**2)))
    return {"status": "VALID", "reason": "" if identifiable else "ZERO_OR_NEAR_ZERO_LOG_VARIANCE",
        "identifiable": identifiable, "mu_r": mr, "mu_c": mc, "sigma_r": sr, "sigma_c": sc, "rho": rho,
        "old_threshold": threshold, "old_threshold_help": old_help, "variance_term": variance_term, "mean_term": mean_term,
        "delta_log_mse": delta_log, "identity_residual": residual, "identity_tolerance": tolerance, "identity_pass": True,
        "delta_rmse": delta_rmse, "log_mse_help": delta_log < 0, "linear_rmse_help": delta_rmse < 0,
        "old_threshold_agrees_log": old_help == (delta_log < 0) if identifiable else None,
        "old_threshold_agrees_rmse": old_help == (delta_rmse < 0) if identifiable else None,
        "log_linear_direction_agree": (delta_log < 0) == (delta_rmse < 0),
        "evidence": "retrospective_alignment_diagnostic_not_independent_mechanism_test", "deployment_status": "not_assessable"}


def observe(predictions, pairs, source_totals):
    if predictions.country.nunique() != 1:
        raise ValueError("机制观测必须分国生产")
    lookup = {}
    for (region, candidate, seed), table in predictions.groupby(["region", "candidate", "seed"], dropna=False, sort=False):
        key = (region, candidate, None if pd.isna(seed) else int(seed))
        if table.target_id.duplicated().any() or table.fold.nunique() > 1:
            raise ValueError("机制输入存在重复 target 或混合 fold")
        lookup[key] = table.set_index("target_id", verify_integrity=True).sort_index()
    rows = []
    for region, total in source_totals.items():
        for pair in pairs:
            for seed in pair["seeds"]:
                keys = [(region, method, seed) for method in (pair["base"], pair["corrected"])]
                if any(key not in lookup for key in keys):
                    raise ValueError(f"机制必要修正对缺失：{keys}")
                a, b = [lookup[key] for key in keys]
                if not a.index.equals(b.index) or not np.array_equal(a.observed, b.observed) or not np.array_equal(a.source_id, b.source_id):
                    raise ValueError("机制修正对的目标 / 真值未对齐")
                for table in (a, b):
                    if not np.isclose(table.predicted.sum(), total, rtol=1e-8, atol=1e-8):
                        raise ValueError("机制预测总量不等于登记 source 总量")
                truth = a.observed.to_numpy(float)
                for station_set in ("all", "positive_truth"):
                    keep = np.ones(len(a), bool) if station_set == "all" else truth > 0
                    for ratio in (1e-8, 1e-6, 1e-4):
                        values = coordinates(truth[keep], a.predicted.to_numpy(float)[keep], b.predicted.to_numpy(float)[keep], total * ratio)
                        rows.append({"country": a.country.iloc[0], "region": region, **{k: v for k, v in pair.items() if k != "seeds"},
                            "seed": seed, "fold": a.fold.iloc[0], "station_set": station_set, "epsilon_ratio": ratio,
                            "source_total": total, "epsilon": total * ratio, "n_targets": int(keep.sum()), "n_excluded": int((~keep).sum()),
                            "unit": a.unit.iloc[0], "base_lineage": a.lineage_fingerprint.iloc[0], "corrected_lineage": b.lineage_fingerprint.iloc[0], **values})
    return pd.DataFrame(rows)
