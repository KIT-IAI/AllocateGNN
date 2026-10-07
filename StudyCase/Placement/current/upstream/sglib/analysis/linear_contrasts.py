"""登记线性对比的国别推断；对科学不可评价子集保留固定 block 成员关系。"""

import json

import numpy as np
import pandas as pd

from .paired_inference import compare, holm


def analyze_linear(table, spec, definitions, family_size):
    if len(definitions) != family_size or len({d["contrast_id"] for d in definitions}) != family_size:
        raise ValueError("主 family 的成员数或身份不符")
    if set(table.country) != {spec["country"]} or table.duplicated(["region", "coordinate"]).any():
        raise ValueError("线性对比输入必须分国且坐标唯一")
    lookup = table.set_index(["region", "coordinate"])
    rows, differences, resolutions = [], [], []
    for definition in definitions:
        keys = list(definition["coefficients"])
        observed, right, valid_indices = [], [], []
        invalid = []
        for index, region in enumerate(spec["regions"]):
            if any((region, key) not in lookup.index for key in keys):
                raise ValueError(f"缺少必要观测：{region}/{keys}")
            selected = [lookup.loc[(region, key)] for key in keys]
            if any(row.status in {"MISSING_REQUIRED", "FAILED_PRODUCTION", "LINEAGE_MISMATCH"} for row in selected):
                raise ValueError("线性对比存在工程缺件，不能静默删除")
            if not all(row.status == "VALID" and pd.notna(row.value) for row in selected):
                invalid.append(region)
                continue
            if any(row.unit != definition["unit"] for row in selected):
                raise ValueError("线性对比量纲混用")
            values = {key: float(row.value) for key, row in zip(keys, selected, strict=True)}
            delta = sum(values[key] * coefficient for key, coefficient in definition["coefficients"].items())
            observed.append(delta)
            right.append(values[definition["relative_baseline"]] if definition.get("relative_baseline") else 0.)
            valid_indices.append(index)
            differences.append({"country": spec["country"], "region": region, "contrast_id": definition["contrast_id"], "value": delta, "unit": definition["unit"]})
        base = {"country": spec["country"], **{k: v for k, v in definition.items() if k not in {"coefficients"}},
                "coefficients": json.dumps(definition["coefficients"], sort_keys=True), "n_expected": len(spec["regions"]),
                "n_produced": len(spec["regions"]), "n_valid": len(observed), "excluded_regions": json.dumps(invalid),
                "inference_mode": "paired_sign_flip_holm", "family_size": family_size}
        if not observed:
            rows.append({**base, "status": "METRIC_NOT_ASSESSABLE", "reason": "NO_SCIENTIFICALLY_ASSESSABLE_REGIONS", "p": None, "p_min": None})
            continue
        mapping = {old: new for new, old in enumerate(valid_indices)}
        blocks = [[mapping[i] for i in block if i in mapping] for block in spec["blocks"]]
        blocks = [block for block in blocks if block]
        adjacency = np.asarray(spec["queen_adjacency"], bool)[np.ix_(valid_indices, valid_indices)]
        d, b = np.asarray(observed), np.asarray(right)
        result = compare(d + b, b, adjacency, blocks)
        relative = bool(definition.get("relative_baseline"))
        ci, bci = result["main"]["native_ci"], result["block"]["native_ci"]
        rci = result["main"]["relative_ci"] if relative else None
        rows.append({**base, "status": "VALID", "effect": result["effect"], "ci_low": ci[0], "ci_high": ci[1],
            "negative_count": int((d < 0).sum()), "p": result["sign_flip"]["p"], "p_min": result["sign_flip"]["p_min"],
            "relative_pct": result["relative_pct"] if relative else None, "relative_ci_low": rci[0] if rci else None,
            "relative_ci_high": rci[1] if rci else None, "relative_reason": result["relative_reason"] if relative else "NOT_APPLICABLE",
            "median_region_pct": result["median_region_pct"] if relative else None,
            "bootstrap_indices_sha256": result["main"]["indices_sha256"], "block_bootstrap_indices_sha256": result["block"]["indices_sha256"],
            "block_ci_low": bci[0], "block_ci_high": bci[1], "block_count": len(blocks), "block_support": result["block_support"],
            "block_p": result["block_sign_flip"]["p"], "block_p_min": result["block_sign_flip"]["p_min"],
            "block_relative_ci": json.dumps(result["block"]["relative_ci"]) if relative else None,
            "moran_p": result["moran"]["p"], "moran_I": result["moran"]["I"], "moran_alarm": result["moran"]["alarm"],
            "descriptive_only": result["descriptive_only"], "reason": "SPATIAL_DEPENDENCE_BLOCK_UNSUPPORTED" if result["descriptive_only"] else
            "MORAN_ALARM_BLOCK_SUPPORTED" if result["moran"]["alarm"] else ""})
        for level, source in (("region", result["sign_flip"]), ("block", result["block_sign_flip"])):
            resolutions.append({"country": spec["country"], "contrast_id": definition["contrast_id"], "level": level,
                "n_units": source["n"], "nonzero_units": source["nonzero"], "p_min": source["p_min"],
                "holm_family_size": family_size if level == "region" else None, "islands": result["moran"]["islands"]})
    contrasts = pd.DataFrame(rows)
    complete = contrasts.p.notna().all()
    contrasts["holm_p"] = holm(contrasts.p) if complete else np.nan
    contrasts["inference_resolution_limited"] = bool(np.all(holm(contrasts.p_min) > .05)) if complete else True
    contrasts["claim_supported"] = contrasts.holm_p.le(.05) & ~contrasts.get("descriptive_only", pd.Series(True, index=contrasts.index)).fillna(True)
    contrasts["family_evaluable"] = complete
    return {"contrasts": contrasts, "region_differences": pd.DataFrame(differences), "inference_resolution_audit": pd.DataFrame(resolutions)}
