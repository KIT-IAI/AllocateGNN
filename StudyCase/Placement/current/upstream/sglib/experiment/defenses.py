"""006 预登记 defense 观测；只消费冻结预测、曲面和版本化数据台账。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from sglib.core.algorithms.reconstruction_metrics import reconstruction_metrics
from .connection_observations import selection_losses


METRICS = ("rmse", "mae", "wape", "predictive_r2", "corr")


def _empty(columns):
    return pd.DataFrame(columns=columns)


def fixed_load_observations(country, surfaces, *, x=300.0):
    """在既有候选、邻域和场上评价固定大负荷；不重建任何上游场。"""
    columns = ["country", "region", "candidate", "seed", "radius_km", "X", "unit", "status", "reason",
               "L_E", "L_S", "L_E_over_X", "L_S_over_X", "true_zero_q_fraction",
               "estimated_zero_q_fraction", "shortlist_k", "cutoff_tie_size",
               "selected_cutoff_tie_count", "selected_cutoff_tie_fraction", "q_true_constant",
               "q_estimated_constant", "scenario_id", "scope"]
    if country == "nl":
        return pd.DataFrame([{name: value for name, value in {
            "country": country, "region": "ALL", "candidate": "NOT_APPLICABLE", "seed": None,
            "radius_km": None, "X": None, "unit": "kW", "status": "INELIGIBLE_BY_DESIGN",
            "reason": "FIXED_300_MW_NOT_SAME_UNIT_AS_NL_KW_WITHOUT_AUTHORIZED_CONVERSION",
            "scenario_id": "C4-E03.fixed_300", "scope": "country_specific_secondary_no_test"}.items()}
            ]).reindex(columns=columns)
    rows = []
    for region, data in surfaces.items():
        labels = data["field_labels"].astype(str)
        seeds = data["field_seeds"].astype(int)
        if len(set(zip(labels, seeds, strict=True))) != len(labels):
            raise ValueError("固定大负荷场身份重复")
        for field, (candidate, seed) in enumerate(zip(labels, seeds, strict=True)):
            for band, radius in enumerate(data["radii_km"]):
                observed, _ = selection_losses(data["G"][band], data["Ghat"][field, band], data["F"][band], x, k=20)
                rows.append({"country": country, "region": region, "candidate": candidate,
                    "seed": None if seed == 0 else int(seed), "radius_km": float(radius), "X": float(x),
                    "unit": str(data["unit"]), "status": "VALID", "reason": "", **observed,
                    "scenario_id": "C4-E03.fixed_300", "scope": "country_specific_secondary_no_test"})
    return pd.DataFrame(rows).reindex(columns=columns)


def connection_map_observations(country, region, data, grid_xy, *, radius_km=10.0, load=0.5):
    """为预登记代表区保存同一候选池的接入地图源表。"""
    candidates = data["candidate_id"].astype(int)
    grid_xy = np.asarray(grid_xy, float)
    if candidates.ndim != 1 or np.any((candidates < 0) | (candidates >= len(grid_xy))):
        raise ValueError("Connection 地图候选与原始 grid 行序不符")
    band = list(map(float, data["radii_km"])).index(float(radius_km))
    load_index = list(map(float, data["lambdas"])).index(float(load))
    x = float(data["X"][load_index])
    labels = list(zip(data["field_labels"].astype(str), data["field_seeds"].astype(int), strict=True))
    chosen = []
    for identity in (("GPM", 0), ("GNN", 42)):
        if labels.count(identity) != 1:
            raise ValueError(f"Connection 地图必要场不唯一: {identity}")
        chosen.append((identity, labels.index(identity)))
    truth, capacity = data["G"][band], data["F"][band]
    q_true = np.maximum(x-capacity+truth, 0.)
    oracle = np.argsort(q_true, kind="stable")[:20]
    rows = []
    for (candidate, seed), field in chosen:
        estimate = data["Ghat"][field, band]
        q_estimated = np.maximum(x-capacity+estimate, 0.)
        selected = np.argsort(q_estimated, kind="stable")[:20]
        selected_mask = np.zeros(len(candidates), bool); selected_mask[selected] = True
        oracle_mask = np.zeros(len(candidates), bool); oracle_mask[oracle] = True
        for i, candidate_id in enumerate(candidates):
            rows.append({"country": country, "region": region, "candidate": candidate, "seed": seed or None,
                "radius_km": radius_km, "lambda": load, "X": x, "unit": str(data["unit"]),
                "candidate_id": int(candidate_id), "x": float(grid_xy[candidate_id, 0]), "y": float(grid_xy[candidate_id, 1]),
                "G": float(truth[i]), "Ghat": float(estimate[i]), "F": float(capacity[i]),
                "q_true": float(q_true[i]), "q_estimated": float(q_estimated[i]),
                "selected": bool(selected_mask[i]), "oracle": bool(oracle_mask[i]),
                "candidate_pool_fixed": True})
    return pd.DataFrame(rows)








