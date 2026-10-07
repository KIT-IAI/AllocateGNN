"""重建五指标与显式不可定义原因。"""

import numpy as np


def reconstruction_metrics(actual, predicted):
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if y.ndim != 1 or not len(y) or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("预测与真值必须为对齐的有限非空向量")
    if (y < 0).any() or (p < 0).any():
        raise ValueError("重建需求必须非负")
    error = p - y
    ss = float(np.square(y - y.mean()).sum())
    total = float(y.sum())
    metrics = {"rmse": float(np.sqrt(np.square(error).mean())),
               "mae": float(np.abs(error).mean()),
               "wape": float(np.abs(error).sum()/total) if total else None,
               "predictive_r2": float(1-np.square(error).sum()/ss) if ss else None,
               "corr": float(np.corrcoef(y, p)[0, 1]) if ss and np.var(p) > 0 and len(y) > 1 else None}
    reasons = {"wape": "ZERO_OBSERVED_TOTAL", "predictive_r2": "CONSTANT_TRUTH", "corr": "CONSTANT_TRUTH_OR_PREDICTION"}
    return {k: {"value": v, "status": "VALID" if v is not None else "METRIC_NOT_ASSESSABLE",
                "reason": "" if v is not None else reasons[k]} for k, v in metrics.items()}


def equal_region(station_sources, source_totals):
    sources = np.asarray(station_sources).astype(str)
    missing_totals = set(sources) - set(source_totals)
    unrepresented = set(source_totals) - set(sources)
    if missing_totals or any(source_totals[s] != 0 for s in unrepresented):
        raise ValueError("EqualRegion 的站点缺少 source 总量，或有正需求 source 无站点")
    result = np.empty(len(sources), dtype=float)
    for key in dict.fromkeys(sources.tolist()):
        mass = source_totals[key]
        selected = sources == key
        result[selected] = mass/selected.sum()
    return result
