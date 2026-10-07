"""C3 的五个区域主对比，规划任务不进入 allocator 因子。"""

import pandas as pd

from .linear_contrasts import analyze_linear


def definitions(unit):
    coefficients = [
        {"MLP@IDR-fixed": 1., "MLP@VD": -1.},
        {"GNN@IDR-fixed": 1., "GNN@VD": -1.},
        {"GNN@IDR-fixed": 1., "GNN@VD": -1., "MLP@IDR-fixed": -1., "MLP@VD": 1.},
        {"MLP@IDR-matched": 1., "MLP@IDR-fixed": -1.},
        {"GNN@IDR-matched": 1., "GNN@IDR-fixed": -1.}]
    return [{"contrast_id": f"C3-{i:02d}", "order": i, "experiment_id": "C3-E01" if i <= 3 else "C3-E02",
             "coefficients": coefficient, "relative_baseline": next((key for key, value in coefficient.items() if value < 0), None) if i != 3 else None,
             "expression": " + ".join(f"{value:g}*{key}" for key, value in coefficient.items()), "unit": unit}
            for i, coefficient in enumerate(coefficients, 1)]


def analyze(tables, spec):
    metrics = tables["metrics"]
    identities = metrics[["region", "candidate", "seed", "allocator", "metric"]].copy()
    if identities.duplicated().any():
        raise ValueError("C3 观测存在重复")
    rows = []
    for (country, region, candidate, allocator, metric), group in metrics.groupby(["country", "region", "candidate", "allocator", "metric"], sort=False):
        seeds = spec["methods"][candidate]["seeds"]
        actual = [None if pd.isna(seed) else int(seed) for seed in group.seed]
        if set(actual) != set(seeds) or len(actual) != len(seeds):
            raise ValueError("C3 seed 坐标不完整")
        valid = group.status.eq("VALID").all()
        rows.append({"country": country, "region": region, "candidate": candidate, "allocator": allocator, "metric": metric,
                     "coordinate": candidate + "@" + allocator, "value": group.value.mean() if valid else None,
                     "status": "VALID" if valid else "METRIC_NOT_ASSESSABLE", "unit": group.metric_unit.iloc[0],
                     "realization_count": len(group), "n_targets": int(group.n_targets.iloc[0])})
    regional = pd.DataFrame(rows)
    main = regional[(regional.metric == "rmse") & regional.candidate.isin(["MLP", "GNN"]) & regional.allocator.ne("CIVD")]
    result = analyze_linear(main, spec, definitions(spec["unit"]), 5)
    result.update(region_metrics=regional, gates=tables["gates"], maps=tables["maps"])
    return result
