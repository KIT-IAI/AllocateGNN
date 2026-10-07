"""C4 的每国九成员主族；只消费重建、规划和接入观测表。"""

import numpy as np
import pandas as pd

from .linear_contrasts import analyze_linear


def definitions(unit):
    coordinates = [("reconstruction", "rmse", unit), ("siting", "WSD", "km"), ("sizing", "RSD_median", "percent")]
    coordinates += [(f"connection_R10_lambda{load:g}", metric, "dimensionless") for load in (.25, .5, 1.) for metric in ("L_E_over_X", "L_S_over_X")]
    return [{"contrast_id": f"C4-{i:02d}", "order": i, "experiment_id": "C4-E01" if i <= 3 else "C4-E02",
             "task_coordinate": task, "metric": metric, "unit": metric_unit, "expression": "GNN - GPM",
             "coefficients": {f"{task}:{metric}:GNN": 1., f"{task}:{metric}:GPM": -1.},
             "relative_baseline": f"{task}:{metric}:GPM"}
            for i, (task, metric, metric_unit) in enumerate(coordinates, 1)]


def summarize_realizations(table, spec, *, value_column, unit, eligibility=None):
    result = []
    for region in spec["regions"]:
        for candidate in ("GPM", "GNN"):
            frame = table[table.region.eq(region) & table.candidate.eq(candidate)]
            expected = spec["methods"][candidate]["seeds"]
            observed = [None if pd.isna(v) else int(v) for v in frame.seed]
            if len(observed) != len(expected) or set(observed) != set(expected):
                raise ValueError("C4 必需方法的区域/seed 坐标缺失或重复")
            eligible = eligibility is None or frame[eligibility].eq(True).all()
            valid = eligible and frame.status.eq("VALID").all() and np.isfinite(frame[value_column].to_numpy(float)).all()
            result.append({"country": spec["country"], "region": region, "candidate": candidate,
                "value": float(frame[value_column].mean()) if valid else None, "unit": unit,
                "status": "VALID" if valid else "INELIGIBLE_BY_DESIGN" if not eligible else "METRIC_NOT_ASSESSABLE",
                "reason": "" if valid else "REF_OR_SCENARIO_NOT_ELIGIBLE" if not eligible else "INVALID_METRIC_REALIZATION",
                "n_realizations": len(frame), "n_valid_realizations": int(frame.status.eq("VALID").sum())})
    return pd.DataFrame(result)


def analyze(reconstruction, planning, connection, spec):
    for table in (reconstruction, planning, connection):
        if set(table.country) != {spec["country"]}:
            raise ValueError("C4 禁止跨国混用观测")
    parts = []
    for definition in definitions(spec["unit"]):
        task, metric, unit = definition["task_coordinate"], definition["metric"], definition["unit"]
        if task == "reconstruction":
            selected = reconstruction[reconstruction.metric.eq(metric) & reconstruction.allocator.eq("VD")]
            summarized = summarize_realizations(selected, spec, value_column="value", unit=unit)
        elif task in {"siting", "sizing"}:
            matching = "not_applicable" if task == "siting" else "many_to_one"
            selected = planning[planning.task.eq(task) & planning.metric.eq(metric) & planning.matching.eq(matching)]
            if not selected.unit.eq(unit).all():
                raise ValueError("C4 规划指标单位不符")
            summarized = summarize_realizations(selected, spec, value_column="value", unit=unit)
        else:
            load = float(task.split("lambda")[1])
            selected = connection[connection.radius_km.eq(10.) & connection["lambda"].eq(load)]
            summarized = summarize_realizations(selected, spec, value_column=metric, unit=unit, eligibility="c4_eligible")
        summarized["task_coordinate"] = task
        summarized["metric"] = metric
        summarized["coordinate"] = task + ":" + metric + ":" + summarized.candidate
        parts.append(summarized)
    regional = pd.concat(parts, ignore_index=True)
    output = analyze_linear(regional, spec, definitions(spec["unit"]), 9)
    output["region_metrics"] = regional
    return output
