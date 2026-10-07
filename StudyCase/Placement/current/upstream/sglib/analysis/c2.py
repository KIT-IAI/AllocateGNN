"""C2 九个主对比与四个登记次级；只消费国别区域指标。"""

import json

import numpy as np
import pandas as pd

from .paired_inference import compare, holm


def contrast_definitions():
    definitions = []
    for signal in ("N", "P", "NP"):
        for operator in ("post", "add"):
            definitions.append({"experiment_id": "C2-E01", "kind": "interaction", "primary": True,
                "expression": f"[GNN{operator}{signal}-GNN]-[GPM{operator}{signal}-GPM]",
                "coefficients": {f"GNN{operator}{signal}": 1., "GNN": -1., f"GPM{operator}{signal}": -1., "GPM": 1.},
                "relative_baseline": None})
    for signal in ("N", "P", "NP"):
        definitions.append({"experiment_id": "C2-E02", "kind": "program_difference", "primary": True,
            "expression": f"GNNadd{signal}-GNNpost{signal}",
            "coefficients": {f"GNNadd{signal}": 1., f"GNNpost{signal}": -1.}, "relative_baseline": None})
    for operator in ("post", "add"):
        for learner in ("MLP", "GNN"):
            left, right = f"{learner}{operator}N", f"GPM{operator}N"
            definitions.append({"experiment_id": "C2-E08", "kind": "level_difference", "primary": False,
                "expression": f"{left}-{right}", "coefficients": {left: 1., right: -1.}, "relative_baseline": right})
    return [{"contrast_id": f"C2-{i:02d}", "order": i, **item} for i, item in enumerate(definitions, 1)]


def analyze(regional, spec, definitions):
    if definitions != contrast_definitions():
        raise ValueError("C2 对比定义与登记版本不符")
    required = {method for item in definitions for method in item["coefficients"]}
    selected = regional[(regional.metric == "rmse") & regional.candidate.isin(required)].copy()
    expected = {(region, method) for region in spec["regions"] for method in required}
    actual = list(zip(selected.region, selected.candidate))
    if selected.country.nunique() != 1 or set(selected.country) != {spec["country"]}:
        raise ValueError("C2 必须只消费一个登记国家")
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("C2 必需区域指标存在缺失、额外或重复坐标")
    if not selected.allocator.eq("VD").all():
        raise ValueError("C2 必须固定 VD")
    if not selected.status.eq("VALID").all() or not np.isfinite(selected.value.to_numpy(float)).all():
        raise ValueError("C2 必需 RMSE 不可评价，必须显式处理缺格，不能完整案例删除")
    if not selected.unit.eq(spec["unit"]).all():
        raise ValueError("C2 国别单位不符")
    for row in selected.itertuples():
        if row.realization_count != len(spec["methods"][row.candidate]["seeds"]):
            raise ValueError("C2 区域指标的 realization 数不符")
    wide = selected.pivot(index="region", columns="candidate", values="value").loc[spec["regions"]]
    rows, resolutions, differences = [], [], []
    for definition in definitions:
        diff = sum(wide[method].to_numpy(float) * coefficient for method, coefficient in definition["coefficients"].items())
        baseline = wide[definition["relative_baseline"]].to_numpy(float) if definition["relative_baseline"] else np.zeros(len(diff))
        result = compare(diff + baseline, baseline, spec["queen_adjacency"], spec["blocks"])
        primary, relative = definition["primary"], definition["relative_baseline"] is not None
        ci, bci = result["main"]["native_ci"], result["block"]["native_ci"]
        rci = result["main"]["relative_ci"] if relative else None
        row = {"country": spec["country"], **{key: definition[key] for key in ("contrast_id", "order", "experiment_id", "kind", "expression", "primary")},
            "estimand": "mean_region_linear_contrast_of_seed_mean_RMSE", "unit": spec["unit"],
            "expected": len(diff), "produced": len(diff), "eligible": len(diff), "valid": len(diff), "fallback": 0,
            "n_expected": len(diff), "n_valid": len(diff), "negative_count": int((diff < 0.).sum()),
            "effect": result["effect"], "ci_low": ci[0], "ci_high": ci[1],
            "relative_pct": result["relative_pct"] if relative else None,
            "relative_ci_low": rci[0] if rci else None, "relative_ci_high": rci[1] if rci else None,
            "relative_reason": result["relative_reason"] if relative else "NOT_REGISTERED_FOR_THIS_ESTIMAND",
            "median_region_pct": result["median_region_pct"] if relative else None,
            "n_pct_valid": result["n_pct_valid"] if relative else None,
            "zero_denominator_draws": result["main"]["zero_denominator_draws"] if relative else None,
            "bootstrap_indices_sha256": result["main"]["indices_sha256"],
            "p": result["sign_flip"]["p"] if primary else None,
            "p_min": result["sign_flip"]["p_min"] if primary else None,
            "block_ci_low": bci[0], "block_ci_high": bci[1],
            "block_bootstrap_indices_sha256": result["block"]["indices_sha256"],
            "block_relative_ci": json.dumps(result["block"]["relative_ci"]) if relative else None,
            "block_zero_denominator_draws": result["block"]["zero_denominator_draws"] if relative else None,
            "block_count": len(spec["blocks"]), "block_support": result["block_support"],
            "block_p": result["block_sign_flip"]["p"] if primary else None,
            "block_p_min": result["block_sign_flip"]["p_min"] if primary else None,
            "moran_p": result["moran"]["p"] if primary else None,
            "moran_I": result["moran"]["I"] if primary else None,
            "moran_alarm": result["moran"]["alarm"] if primary else None,
            "descriptive_only": result["descriptive_only"] if primary else True,
            "inference_mode": "paired_sign_flip_holm" if primary else "secondary_effect_and_interval",
            "status": "VALID", "reason": "SPATIAL_DEPENDENCE_BLOCK_UNSUPPORTED" if primary and result["descriptive_only"] else
                      "MORAN_ALARM_BLOCK_SUPPORTED" if primary and result["moran"]["alarm"] else ""}
        rows.append(row)
        differences.extend({"country": spec["country"], "region": region, "contrast_id": definition["contrast_id"], "value": value,
                            "unit": spec["unit"]} for region, value in zip(spec["regions"], diff, strict=True))
        if primary:
            for level, source in (("region", result["sign_flip"]), ("block", result["block_sign_flip"])):
                resolutions.append({"country": spec["country"], "contrast_id": definition["contrast_id"], "level": level,
                    "n_units": source["n"], "nonzero_units": source["nonzero"], "p_min": source["p_min"],
                    "holm_family_size": 9 if level == "region" else None, "islands": result["moran"]["islands"]})
    table = pd.DataFrame(rows)
    primary = table.primary
    table["holm_p"] = np.nan
    table.loc[primary, "holm_p"] = holm(table.loc[primary, "p"])
    table["holm_reject"] = False
    table.loc[primary, "holm_reject"] = table.loc[primary, "holm_p"] <= .05
    table["claim_supported"] = table.holm_reject & ~table.descriptive_only
    table["inference_resolution_limited"] = False
    table.loc[primary, "inference_resolution_limited"] = bool(np.all(holm(table.loc[primary, "p_min"]) > .05))
    audit = selected[["country", "region", "candidate", "allocator", "metric", "status", "realization_count"]].copy()
    audit["claim_id"] = "C2"
    return {"contrasts": table, "region_differences": pd.DataFrame(differences),
            "coordinate_audit": audit, "inference_resolution_audit": pd.DataFrame(resolutions), "region_metrics": selected}
