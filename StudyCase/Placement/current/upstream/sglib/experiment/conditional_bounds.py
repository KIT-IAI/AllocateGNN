"""C6：固定 shortlist 的矩形同时集上界与留一地区回顾预算。"""

import numpy as np
import pandas as pd


def loss_bounds(q, qhat, lower, upper, k=20):
    truth, estimate, lo, hi = [np.asarray(v, float) for v in (q, qhat, lower, upper)]
    if not (truth.shape == estimate.shape == lo.shape == hi.shape) or truth.ndim != 1 or len(truth) < k or k <= 0:
        raise ValueError("上界向量与固定 shortlist 不对齐")
    if not all(np.isfinite(v).all() for v in (truth, estimate, lo, hi)) or (lo > hi).any():
        raise ValueError("上界输入非有限或区间反向")
    selected = np.argsort(estimate, kind="stable")[:k]
    oracle = np.argsort(truth, kind="stable")[:k]
    mask = np.zeros(len(truth), bool); mask[selected] = True
    exchanges = min(k, len(truth) - k)
    gain = np.r_[0., np.cumsum(np.sort(hi[mask])[::-1][:exchanges]) - np.cumsum(np.sort(lo[~mask])[:exchanges])]
    return {"L_E": float(np.mean(np.abs(estimate - truth))), "B_E": float(np.mean(np.maximum(estimate - lo, hi - estimate))),
            "L_S": max(0., float(np.mean(truth[selected]) - np.mean(truth[oracle]))), "B_S": float(np.max(gain) / k)}


def observe(country, regions, *, etas=(.5, .8, .9, 1.)):
    keys = [("Uni", 0), ("GPM", 0), ("GNN", 42), ("GNN", 123), ("GNN", 456)]
    normalized, ids = {}, list(regions)
    for region, data in regions.items():
        fields = list(zip(data["field_labels"].astype(str), data["field_seeds"].astype(int)))
        if float(data["source_total"]) <= 0:
            raise ValueError("C6 给定 source 总量为零，需显式登记校准域，不删除地区")
        for key in keys:
            if fields.count(key) != 1:
                raise ValueError("C6 必需方法 / seed 不唯一或缺失")
            index = fields.index(key)
            for band, radius in enumerate(data["radii_km"]):
                error = float(np.max(np.abs(data["Ghat"][index, band] - data["G"][band])))
                normalized[region, key, float(radius)] = error / float(data["source_total"])
    rows = []
    for region, data in regions.items():
        fields = list(zip(data["field_labels"].astype(str), data["field_seeds"].astype(int)))
        for key in keys:
            field = fields.index(key)
            for band, radius in enumerate(data["radii_km"]):
                calibration = [name for name in ids if name != region]
                others = np.array([normalized[name, key, float(radius)] for name in calibration])
                if len(others) != len(ids) - 1 or not len(others):
                    raise ValueError("C6 留一地区校准域不完整")
                current = normalized[region, key, float(radius)]
                for eta in etas:
                    budget_normalized = float(np.quantile(others, eta, method="linear"))
                    budget = budget_normalized * float(data["source_total"])
                    covered = current <= budget_normalized + 1e-12 * max(1., current, budget_normalized)
                    for load, x in zip(data["lambdas"], data["X"], strict=True):
                        truth, estimated, capacity = data["G"][band], data["Ghat"][field, band], data["F"][band]
                        q, qhat = np.maximum(x - capacity + truth, 0.), np.maximum(x - capacity + estimated, 0.)
                        values = loss_bounds(q, qhat,
                                             np.maximum(x - capacity + estimated - budget, 0.), np.maximum(x - capacity + estimated + budget, 0.))
                        tolerance = 1e-10 * max(1., *values.values())
                        violated = values["L_E"] > values["B_E"] + tolerance or values["L_S"] > values["B_S"] + tolerance
                        if covered and violated:
                            raise ValueError(f"C6 条件内上界违规：{country}/{region}/{key}/{radius}/{load}/{eta}")
                        maximum_error = max(current, float(others.max()))
                        rows.append({"country": country, "region": region, "candidate": key[0], "seed": key[1] or None, "unit": str(data["unit"]),
                            "radius_km": float(radius), "lambda": float(load), "X": float(x), "eta": eta,
                            "calibration_regions": "|".join(calibration), "n_calibration": len(calibration),
                            "source_total": float(data["source_total"]), "budget_normalized": budget_normalized, "budget": budget,
                            "realized_normalized_error": current, "budget_realized": covered, "conditional_violation": bool(covered and violated),
                            "numerical_tolerance": tolerance, "assessable": x > 0, "reason": "" if x > 0 else "ZERO_X",
                            "status": "METRIC_NOT_ASSESSABLE" if x <= 0 else "VALID" if covered else "BUDGET_EXCEEDED",
                            "observed_performance": "NOT_ASSESSED_NO_PLANNER_TOLERANCE",
                            "conditional_sufficiency": "NORMALIZED_CURVE_ONLY", "budget_realization": "REALIZED" if covered else "EXCEEDED",
                            **values, **{name + "_over_X": value / x if x > 0 else None for name, value in values.items()},
                            "slack_E": values["B_E"] - values["L_E"], "slack_S": values["B_S"] - values["L_S"],
                            "true_zero_q_fraction": float(np.mean(q == 0)), "estimated_zero_q_fraction": float(np.mean(qhat == 0)),
                            "maximum_normalized_error_ties": int(sum(abs(normalized[name, key, float(radius)] - maximum_error) <= 1e-12 * max(1., maximum_error) for name in ids))})
    return pd.DataFrame(rows)
