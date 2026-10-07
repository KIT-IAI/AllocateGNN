"""C5 固定九分布主面板与登记敏感性；推断单位始终为区域。"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .linear_contrasts import analyze_linear


PANEL = ("Ref", "PERM-R/3", "PERM-R", "PERM-3R", "SMOOTH-0.5", "SMOOTH-1", "SMOOTH-2", "GPM", "GNN")


def definitions():
    return [{"contrast_id": f"C5-{i:02d}", "order": i, "experiment_id": "C5-E01", "radius_km": radius,
             "unit": "dimensionless", "expression": "aggregate-regret rho minus cell-regret rho",
             "coefficients": {f"R{radius:g}:aggregate": 1., f"R{radius:g}:cell": -1.}, "relative_baseline": None}
            for i, radius in enumerate((10., 20.), 1)]


def panel_associations(panel, spec):
    if set(panel.country) != {spec["country"]}:
        raise ValueError("C5 只能消费本国面板")
    if set(panel.region) != set(spec["regions"]) or len(panel) != len(spec["regions"])*2*3*3*9 or panel.duplicated(["region", "radius_km", "lambda", "gnn_seed", "distribution"]).any():
        raise ValueError("C5 面板完整坐标存在缺件、重复或额外区域")
    rows = []
    for region in spec["regions"]:
        for radius in (10., 20.):
            for load in (.25, .5, 1.):
                for seed in (42, 123, 456):
                    frame = panel[panel.region.eq(region) & panel.radius_km.eq(radius) & panel["lambda"].eq(load) & panel.gnn_seed.eq(seed)]
                    if len(frame) != 9 or frame.distribution.duplicated().any() or set(frame.distribution) != set(PANEL):
                        raise ValueError("C5 九分布存在缺件或重复，不能自动缩为八分布")
                    for retain_ref in (True, False):
                        chosen = frame.set_index("distribution").loc[list(PANEL if retain_ref else PANEL[1:])]
                        eligible = bool(chosen.ref_eligible.eq(True).all() and chosen.scenario_defined.eq(True).all())
                        for loss in ("L_S_over_X", "L_E_over_X"):
                            columns = ["cell_spearman", "aggregate_spearman", loss]
                            values = chosen[columns].to_numpy(float)
                            valid = eligible and np.isfinite(values).all() and all(np.ptp(values[:, i]) > 0 for i in range(3))
                            cell = float(spearmanr(values[:, 0], values[:, 2]).statistic) if valid else None
                            aggregate = float(spearmanr(values[:, 1], values[:, 2]).statistic) if valid else None
                            rows.append({"country": spec["country"], "region": region, "radius_km": radius, "lambda": load,
                                "gnn_seed": seed, "retain_ref": retain_ref, "distribution_count": len(chosen), "loss": loss,
                                "cell_rho": cell, "aggregate_rho": aggregate, "difference": aggregate-cell if valid else None,
                                "status": "VALID" if valid else "INELIGIBLE_BY_DESIGN" if not eligible else "METRIC_NOT_ASSESSABLE",
                                "reason": "" if valid else "REF_OR_ZERO_X" if not eligible else "CONSTANT_OR_UNDEFINED_PANEL_METRIC",
                                "primary": bool(retain_ref and load == .5 and seed == 42 and loss == "L_S_over_X")})
    return pd.DataFrame(rows)


def analyze(panel, spec):
    associations = panel_associations(panel, spec)
    selected = associations[associations.primary]
    rows = []
    for row in selected.to_dict("records"):
        for side in ("aggregate", "cell"):
            rows.append({"country": row["country"], "region": row["region"], "coordinate": f"R{row['radius_km']:g}:{side}",
                         "unit": "dimensionless", "value": row[side + "_rho"], "status": row["status"], "reason": row["reason"]})
    output = analyze_linear(pd.DataFrame(rows), spec, definitions(), 2)
    output["panel_associations"] = associations
    return output
