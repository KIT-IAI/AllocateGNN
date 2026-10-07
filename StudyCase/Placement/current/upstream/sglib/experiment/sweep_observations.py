"""六类冻结扫描场的站点观测；保持实际 seed/fold 支持，不选择最优参数。"""

import numpy as np
import pandas as pd

from sglib.core.algorithms.reconstruction_metrics import reconstruction_metrics


def observe(region, sweeps):
    metrics, predictions, seen = [], [], set()
    for field in sweeps:
        identity = (field.parameter, field.signal, field.value, field.seed, field.fold, field.region)
        if identity in seen or field.region != region.region:
            raise ValueError("扫描坐标重复或跨区域")
        seen.add(identity)
        values = np.asarray(field.values, float)
        if values.shape != region.assignment.shape or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("扫描场与规范格点支持不符")
        predicted = np.bincount(region.assignment, weights=values, minlength=len(region.observed))
        base = {"country": region.country, "region": region.region, "parameter_name": field.parameter,
                "parameter_value": field.value, "signal": field.signal, "seed": field.seed, "fold": field.fold,
                "allocator": "VD", "field_sha256": field.lineage["sha256"], "unit": region.unit,
                "evidence": "fixed_grid_sensitivity_no_parameter_selection"}
        for metric, record in reconstruction_metrics(region.observed, predicted).items():
            metrics.append({**base, "metric": metric, **record, "n_targets": len(region.observed)})
        predictions.append(pd.DataFrame({**base, "target_id": region.station_ids.astype(str),
                                          "observed": region.observed, "predicted": predicted}))
    return {"metrics": pd.DataFrame(metrics), "predictions": pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()}
