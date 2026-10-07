"""C1 国别统计与共享库存，消费已完成 Experiment 表。"""

import json
import numpy as np
import pandas as pd

from .paired_inference import compare, holm, _bootstrap


CONTRASTS = (("GPM", "Uni"), ("MLP", "GPM"), ("GNN", "GPM"), ("GNN", "MLP"),
             ("MLP", "EqualGrid"), ("GNN", "EqualGrid"), ("MLP", "EqualStation"), ("GNN", "EqualStation"),
             ("MLP", "EqualRegion"), ("GNN", "EqualRegion"))
METHODS = ("Uni", "GPM", "MLP", "GNN", "EqualGrid", "EqualStation", "EqualRegion")


def analyze(tables, prereg):
    metrics = tables["metrics"]
    region_names = prereg["regions"]
    definitions = prereg["methods"]
    expected = {(r, c, s, m) for r in region_names for c, d in definitions.items() for s in d["seeds"]
                for m in ("rmse", "mae", "wape", "predictive_r2", "corr")}
    observed = [(row.region, row.candidate, None if pd.isna(row.seed) else int(row.seed), row.metric) for row in metrics.itertuples()]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("C1 / 共享库存观测集合有缺失、额外或重复坐标")
    coordinate_rows = []
    for region in region_names:
        for candidate, policy in definitions.items():
            for seed in policy["seeds"]:
                selected = metrics[(metrics.region == region) & (metrics.candidate == candidate)]
                selected = selected[selected.seed.isna()] if seed is None else selected[selected.seed == seed]
                keys = list(selected.metric)
                valid = len(keys) == 5 and set(keys) == {"rmse", "mae", "wape", "predictive_r2", "corr"}
                coordinate_rows.append({"country": prereg["country"], "region": region, "candidate": candidate, "seed": seed,
                                        "status": "VALID" if valid else "MISSING_REQUIRED", "observed_metrics": len(keys)})
                if not valid:
                    raise ValueError(f"C1 观测缺失 / 重复：{region}/{candidate}/{seed}")
    rows = []
    for (country, region, candidate, allocator, metric), group in metrics.groupby(["country", "region", "candidate", "allocator", "metric"], sort=False):
        valid = bool(group.status.eq("VALID").all())
        rows.append({"country": country, "region": region, "candidate": candidate, "allocator": allocator,
            "metric": metric, "value": float(group.value.mean()) if valid else None,
            "status": "VALID" if valid else "METRIC_NOT_ASSESSABLE", "reason": "" if valid else "|".join(sorted(set(group.reason.dropna()))),
            "seed_count": int(group.seed.nunique()), "realization_count": len(group), "n_targets": int(group.n_targets.iloc[0]),
            "unit": group.metric_unit.iloc[0],
            "prediction_total_matches_observed_total": bool(group.prediction_total_matches_observed_total.all())})
    regional = pd.DataFrame(rows)
    inventory = []
    for (country, candidate, allocator, metric), group in regional.groupby(["country", "candidate", "allocator", "metric"], sort=False):
        valid = group.status.eq("VALID")
        inventory.append({"country": country, "candidate": candidate, "allocator": allocator, "metric": metric,
            "mean": float(group.loc[valid, "value"].mean()) if valid.any() else None, "n_valid": int(valid.sum()), "n_expected": len(region_names),
            "unit": group.unit.iloc[0], "status": "VALID" if valid.all() else "METRIC_NOT_ASSESSABLE"})
    contrasts, resolutions, spatial = [], [], []
    adjacency = np.asarray(prereg["queen_adjacency"], bool)
    blocks = prereg["blocks"]
    for ordinal, (left, right) in enumerate(CONTRASTS, 1):
        rmse = regional[regional.metric == "rmse"].pivot(index="region", columns="candidate", values="value").loc[region_names]
        result = compare(rmse[left].to_numpy(), rmse[right].to_numpy(), adjacency, blocks)
        native_ci, relative_ci = result["main"]["native_ci"], result["main"]["relative_ci"]
        block_ci = result["block"]["native_ci"]
        row = {"country": prereg["country"], "contrast_id": f"C1-{ordinal:02d}", "experiment_id": "C1-E01" if ordinal <= 4 else "C1-E04",
            "order": ordinal, "left": left, "right": right, "unit": prereg["unit"], "n_expected": len(region_names),
            "n_valid": result["n"], "improved": result["improved"], "effect": result["effect"], "ci_low": native_ci[0], "ci_high": native_ci[1],
            "relative_pct": result["relative_pct"], "relative_ci_low": relative_ci[0] if relative_ci else None,
            "relative_ci_high": relative_ci[1] if relative_ci else None, "median_region_pct": result["median_region_pct"],
            "n_pct_valid": result["n_pct_valid"], "zero_denominator_draws": result["main"]["zero_denominator_draws"],
            "relative_reason": result["relative_reason"],
            "p": result["sign_flip"]["p"], "p_min": result["sign_flip"]["p_min"],
            "bootstrap_indices_sha256": result["main"]["indices_sha256"],
            "block_ci_low": block_ci[0], "block_ci_high": block_ci[1], "block_p": result["block_sign_flip"]["p"],
            "block_p_min": result["block_sign_flip"]["p_min"], "block_count": len(blocks),
            "block_bootstrap_indices_sha256": result["block"]["indices_sha256"],
            "block_support": result["block_support"], "moran_p": result["moran"]["p"], "moran_I": result["moran"]["I"],
            "moran_alarm": result["moran"]["alarm"], "island_count": result["moran"]["islands"],
            "descriptive_only": result["descriptive_only"], "inference_mode": "paired_sign_flip_holm",
            "reason": "SPATIAL_DEPENDENCE_BLOCK_UNSUPPORTED" if result["descriptive_only"] else "MORAN_ALARM_BLOCK_SUPPORTED" if result["moran"]["alarm"] else result["moran"].get("reason", "")}
        contrasts.append(row)
        for level, source in (("region", result["sign_flip"]), ("block", result["block_sign_flip"])):
            resolutions.append({"country": prereg["country"], "contrast_id": row["contrast_id"], "level": level,
                "n_units": source["n"], "nonzero_units": source["nonzero"], "p_min": source["p_min"],
                "holm_family_size": 10 if level == "region" else None, "islands": result["moran"]["islands"]})
        spatial.append({"country": prereg["country"], "contrast_id": row["contrast_id"], **result["moran"],
                        "block_relative_ci": json.dumps(result["block"]["relative_ci"]),
                        "block_zero_denominator_draws": result["block"]["zero_denominator_draws"]})
    contrasts = pd.DataFrame(contrasts)
    contrasts["holm_p"] = holm(contrasts.p)
    contrasts["holm_reject"] = contrasts.holm_p <= .05
    contrasts["claim_supported"] = contrasts.holm_reject & ~contrasts.descriptive_only
    contrasts["inference_resolution_limited"] = bool(np.all(holm(contrasts.p_min) > .05))
    secondary = []
    for metric in ("mae", "wape", "predictive_r2", "corr"):
        wide = regional[regional.metric == metric].pivot(index="region", columns="candidate", values="value").loc[region_names]
        for ordinal, (left, right) in enumerate(CONTRASTS, 1):
            valid = wide[left].notna() & wide[right].notna()
            a, b = wide.loc[valid, left].to_numpy(float), wide.loc[valid, right].to_numpy(float)
            record = {"country": prereg["country"], "contrast_id": f"C1-{ordinal:02d}", "metric": metric,
                "left": left, "right": right, "n_valid": len(a), "n_expected": len(region_names),
                "effect": None, "ci_low": None, "ci_high": None,
                "status": "VALID" if len(a) else "METRIC_NOT_ASSESSABLE",
                "reason": "" if valid.all() else "CONSTANT_VECTOR_OR_ZERO_DENOMINATOR",
                "inference_mode": "secondary_effect_and_interval_without_test"}
            if len(a):
                interval = _bootstrap(a, b, [[i] for i in range(len(a))], seed=20260906, repetitions=10000)
                record.update(effect=float((a-b).mean()), ci_low=interval["native_ci"][0], ci_high=interval["native_ci"][1],
                              indices_sha256=interval["indices_sha256"])
            secondary.append(record)
    inventory = pd.DataFrame(inventory)
    method_order = prereg.get("method_order", list(definitions))
    inventory["method_order"] = inventory.candidate.map({c: i for i, c in enumerate(method_order)})
    inventory = inventory.sort_values(["method_order", "metric"], kind="stable")
    return {"contrasts": contrasts, "region_metrics": regional, "shared_inventory": inventory,
            "coordinate_audit": pd.DataFrame(coordinate_rows), "inference_resolution_audit": pd.DataFrame(resolutions),
            "spatial_diagnostics": pd.DataFrame(spatial), "equal_route_regions": tables["equal_route_regions"],
            "equal_route": tables["equal_route"], "map_grid": tables["map_grid"], "map_sources": tables["map_sources"],
            "secondary_effects": pd.DataFrame(secondary)}
