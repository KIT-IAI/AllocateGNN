"""006 共用接入、控制面板与尺度观测；不执行国别显著性推断。"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sglib.core.algorithms.reconstruction_metrics import reconstruction_metrics
from .connection import stride_candidates, neighbourhood_members, neighbourhood_sum, reference_field
from .control_fields import build_control_arms


PANEL_ORDER = ("Ref", "PERM-R/3", "PERM-R", "PERM-3R", "SMOOTH-0.5", "SMOOTH-1", "SMOOTH-2", "GPM", "GNN")
SCALE_RADII = (1., 2., 3., 5., 7.5, 10., 15., 20.)


def compiled_members(points, queries, radius_m):
    """将求和函数原本逐次进行的同 dtype 转换外提，成员及顺序完全不变。"""
    return [np.asarray(indices, dtype=int) for indices in neighbourhood_members(points, queries, radius_m)]


def rank_agreement(left, right):
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("秩指标输入未对齐或不有限")
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(rankdata(a, method="average"), rankdata(b, method="average"))[0, 1])


def selection_losses(truth, estimate, capacity, x, *, k=20):
    g, a, f = map(lambda v: np.asarray(v, float), (truth, estimate, capacity))
    if g.ndim != 1 or g.shape != a.shape or g.shape != f.shape or len(g) < k or k <= 0:
        raise ValueError("接入向量或固定 shortlist 不符合合同")
    if not all(np.isfinite(v).all() for v in (g, a, f)) or not np.isfinite(x) or x < 0:
        raise ValueError("接入输入必须有限且负荷非负")
    q, qhat = np.maximum(x - f + g, 0.), np.maximum(x - f + a, 0.)
    selected, oracle = np.argsort(qhat, kind="stable")[:k], np.argsort(q, kind="stable")[:k]
    fixed = float(np.mean(np.abs(qhat - q)))
    regret = float(np.mean(q[selected]) - np.mean(q[oracle]))
    if regret < -1e-10 * max(1., float(np.max(q))):
        raise ValueError("选择 regret 出现超容差负值")
    regret = max(0., regret)
    tie = qhat == qhat[selected[-1]]
    return {"L_E": fixed, "L_S": regret, "L_E_over_X": fixed / x if x > 0 else None,
        "L_S_over_X": regret / x if x > 0 else None, "true_zero_q_fraction": float(np.mean(q == 0)),
        "estimated_zero_q_fraction": float(np.mean(qhat == 0)), "shortlist_k": k,
        "cutoff_tie_size": int(tie.sum()), "selected_cutoff_tie_count": int(tie[selected].sum()),
        "selected_cutoff_tie_fraction": float(tie[selected].sum() / tie.sum()),
        "q_true_constant": bool(np.ptp(q) == 0), "q_estimated_constant": bool(np.ptp(qhat) == 0)}, selected


def observe_region(region, fields, source_total, w0_surface, w0_scenarios, planning_labels):
    rows = stride_candidates(len(region.grid_xy), 2000)
    queries = region.grid_xy[rows]
    reference = reference_field(region.assignment, region.demand)
    lookup = {(field.label, field.seed): field for field in fields if not field.qa_only}
    chosen = [field for field in fields if field.label in planning_labels and not field.qa_only]
    labels, seeds = [f.label for f in chosen], [f.seed or 0 for f in chosen]
    if len(set(zip(labels, seeds))) != len(chosen):
        raise ValueError("接入场坐标重复")
    for label, seed in (("GPM", None), ("GNN", 42), ("GNN", 123), ("GNN", 456)):
        if (label, seed) not in lookup:
            raise ValueError("尺度 / 面板必要场缺失")
    core, panel, scale, selections = [], [], [], []
    packed = {"candidate_id": rows, "field_labels": np.asarray(labels, dtype=str), "field_seeds": np.asarray(seeds, dtype=int),
              "unit": np.asarray(region.unit),
              "field_sha256": np.asarray([f.lineage["sha256"] for f in chosen], dtype=str), "source_total": np.asarray(source_total),
              "radii_km": np.asarray([10., 20.]), "lambdas": np.asarray([.25, .5, 1.])}
    aggregates, truth_bands, capacity_bands, ref_bands, controls_raw = [], [], [], [], []
    base = {"country": region.country, "region": region.region, "unit": region.unit, "source_total": source_total}
    for radius in (10., 20.):
        context = w0_surface[w0_surface.radius_km.eq(radius)].sort_values("candidate_id", kind="stable")
        if not np.array_equal(context.candidate_id.to_numpy(int), rows):
            raise ValueError("共用接入候选池与 W0 行序不符")
        g, f, ref = context.G.to_numpy(float), context.F.to_numpy(float), context.A_ref.to_numpy(float)
        members = compiled_members(region.grid_xy, queries, radius * 1000.)
        if not np.array_equal(neighbourhood_sum(reference, members), ref):
            raise ValueError("当前 VD-Ref 与 W0 预审不等价")
        field_aggregates = np.stack([neighbourhood_sum(field.values, members) for field in chosen])
        aggregates.append(field_aggregates)
        truth_bands.append(g); capacity_bands.append(f); ref_bands.append(ref)
        scenarios = w0_scenarios[w0_scenarios.radius_km.eq(radius)].sort_values("lambda")
        for field, estimated in zip(chosen, field_aggregates, strict=True):
            for scenario in scenarios.to_dict("records"):
                observation, selected = selection_losses(g, estimated, f, float(scenario["X"]))
                identity = {**base, "candidate": field.label, "seed": field.seed, "fold": field.fold, "radius_km": radius,
                    "lambda": scenario["lambda"], "X": scenario["X"], "field_sha256": field.lineage["sha256"],
                    "candidate_pool_hash": context.candidate_pool_hash.iloc[0], "ref_eligible": scenario["ref_eligible"],
                    "scenario_defined": scenario["scenario_defined"], "c4_eligible": scenario["c4_c5_expected_eligible"],
                    "status": "VALID" if scenario["scenario_defined"] else "METRIC_NOT_ASSESSABLE",
                    "reason": "" if scenario["scenario_defined"] else "ZERO_X", "max_candidate_demand_error": float(np.max(np.abs(estimated - g)))}
                core.append({**identity, **observation})
                selections.append({**identity, "selected_candidate_ids": "|".join(map(str, rows[selected]))})
        controls = build_control_arms(reference, region.grid_xy / 1000., radius, seed=42)
        controls_raw.append(np.stack([controls[name] for name in PANEL_ORDER[1:7]]))
        panel_values = {"Ref": reference, **controls, "GPM": lookup[("GPM", None)].values}
        cached = {name: (neighbourhood_sum(value, members), rank_agreement(value, reference)) for name, value in panel_values.items()}
        for gnn_seed in (42, 123, 456):
            field = lookup[("GNN", gnn_seed)]
            values = {**cached, "GNN": (neighbourhood_sum(field.values, members), rank_agreement(field.values, reference))}
            for name in PANEL_ORDER:
                estimated, cell_rho = values[name]
                aggregate_rho = rank_agreement(estimated, ref)
                for scenario in scenarios.to_dict("records"):
                    observation, _ = selection_losses(g, estimated, f, float(scenario["X"]))
                    panel.append({**base, "distribution": name, "gnn_seed": gnn_seed, "radius_km": radius,
                        "lambda": scenario["lambda"], "X": scenario["X"], "cell_spearman": cell_rho,
                        "aggregate_spearman": aggregate_rho, "ref_eligible": scenario["ref_eligible"],
                        "scenario_defined": scenario["scenario_defined"], **observation,
                        "rank_status": "VALID" if cell_rho is not None and aggregate_rho is not None else "METRIC_NOT_ASSESSABLE"})
        packed["X"] = scenarios.X.to_numpy(float)
    for radius in SCALE_RADII:
        for centre_kind, centres in (("grid", queries), ("station", region.station_xy)):
            grid_members = compiled_members(region.grid_xy, centres, radius * 1000.)
            station_members = compiled_members(region.station_xy, centres, radius * 1000.)
            targets = {"ledger": neighbourhood_sum(region.demand, station_members), "VD-Ref": neighbourhood_sum(reference, grid_members)}
            for label, seed in (("GPM", None), ("GNN", 42), ("GNN", 123), ("GNN", 456)):
                estimate = neighbourhood_sum(lookup[(label, seed)].values, grid_members)
                for target_kind, truth in targets.items():
                    metrics = reconstruction_metrics(truth, estimate)
                    rho = rank_agreement(truth, estimate)
                    metrics["spearman"] = {"value": rho, "status": "VALID" if rho is not None else "METRIC_NOT_ASSESSABLE", "reason": "" if rho is not None else "CONSTANT_VECTOR"}
                    scale.extend({**base, "candidate": label, "seed": seed, "radius_km": radius, "centre_kind": centre_kind,
                        "target_kind": target_kind, "n_centres": len(centres), "metric": metric, **value} for metric, value in metrics.items())
    packed.update(G=np.stack(truth_bands), F=np.stack(capacity_bands), A_ref=np.stack(ref_bands),
                  Ghat=np.stack(aggregates, axis=1), control_fields=np.stack(controls_raw),
                  control_labels=np.asarray(PANEL_ORDER[1:7], dtype=str))
    return packed, {"connection_metrics": pd.DataFrame(core), "panel_metrics": pd.DataFrame(panel), "scale_metrics": pd.DataFrame(scale), "selections": pd.DataFrame(selections)}
