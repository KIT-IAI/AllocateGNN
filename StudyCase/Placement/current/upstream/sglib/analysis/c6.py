"""C6 的描述性上界审计与完整阶梯曲线，不生成假设检验或部署容忍度。"""

import numpy as np
import pandas as pd


def analyze(bounds, spec):
    expected = {(region, label, seed, radius, load, eta) for region in spec["regions"]
                for label, seed in (("Uni", None), ("GPM", None), ("GNN", 42), ("GNN", 123), ("GNN", 456))
                for radius in (10., 20.) for load in (.25, .5, 1.) for eta in (.5, .8, .9, 1.)}
    observed = [(r.region, r.candidate, None if pd.isna(r.seed) else int(r.seed), r.radius_km, getattr(r, "load"), r.eta)
                for r in bounds.rename(columns={"lambda": "load"}).itertuples()]
    if set(bounds.country) != {spec["country"]} or len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("C6 观测矩阵不完整或跨国家")
    if bounds.conditional_violation.eq(True).any():
        raise ValueError("C6 存在条件内上界违规，不能生成保证声明")
    groups = ["candidate", "seed", "radius_km", "lambda", "eta"]
    audit, curves = [], []
    for keys, frame in bounds.groupby(groups, sort=False, dropna=False):
        identity = {"country": spec["country"], **dict(zip(groups, keys, strict=True))}
        for loss in ("E", "S"):
            actual, bound = frame[f"L_{loss}_over_X"].to_numpy(float), frame[f"B_{loss}_over_X"].to_numpy(float)
            assessable = frame.assessable.to_numpy(bool) & np.isfinite(actual) & np.isfinite(bound)
            realized = frame.budget_realized.to_numpy(bool) & assessable
            conditional_count = int(realized.sum())
            audit.append({**identity, "loss": loss, "n_total": len(frame), "n_assessable": int(assessable.sum()),
                "n_not_assessable": int((~assessable).sum()), "n_budget_realized": conditional_count,
                "budget_coverage_all": conditional_count / len(frame), "conditional_violations": 0,
                "mean_actual": float(actual[assessable].mean()) if assessable.any() else None,
                "mean_bound": float(bound[assessable].mean()) if assessable.any() else None,
                "mean_slack": float((bound-actual)[assessable].mean()) if assessable.any() else None,
                "zero_actual_count": int(np.count_nonzero(assessable & (actual == 0))),
                "zero_bound_count": int(np.count_nonzero(assessable & (bound == 0))),
                "actual_planner_tolerance": None, "external_budget": None, "tolerance_mode": "normalized_curve",
                "budget_source": "country_leave_one_region_out_retrospective_type7"})
            thresholds = np.unique(np.r_[0., bound[assessable]])
            for threshold in thresholds:
                certified = realized & (bound <= threshold)
                observed_ok = assessable & (actual <= threshold)
                curves.append({**identity, "loss": loss, "tau_normalized": threshold,
                    "n_total": len(frame), "n_assessable": int(assessable.sum()), "n_budget_realized": conditional_count,
                    "n_certified": int(certified.sum()), "conditional_share": float(certified.sum()/conditional_count) if conditional_count else None,
                    "all_registered_share": float(certified.sum()/len(frame)),
                    "observed_performance_share_assessable": float(observed_ok.sum()/assessable.sum()) if assessable.any() else None,
                    "status": "VALID" if conditional_count else "METRIC_NOT_ASSESSABLE",
                    "reason": "" if conditional_count else "ZERO_CONDITIONAL_DENOMINATOR",
                    "interpretation": "retrospective_conditional_budget_not_deployment_approval"})
    return {"bounds_audit": pd.DataFrame(audit), "tolerance_curves": pd.DataFrame(curves), "bounds_observations": bounds.copy()}
