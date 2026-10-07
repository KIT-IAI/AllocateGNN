"""只消费类型化场、assignment 与台账的重建观测生产。"""

from dataclasses import dataclass
from typing import Mapping, Any

import numpy as np
import pandas as pd

from sglib.core.algorithms.reconstruction_metrics import reconstruction_metrics, equal_region


@dataclass(frozen=True)
class ReconstructionRegion:
    country: str
    region: str
    unit: str
    station_ids: np.ndarray
    station_sources: np.ndarray
    observed: np.ndarray
    source_totals: Mapping[str, float]
    grid_sources: np.ndarray
    assignment: np.ndarray
    fields: tuple[Any, ...]
    input_fingerprint: str
    station_xy: np.ndarray
    grid_xy: np.ndarray
    sources_wkt: tuple[str, ...]
    map_selected: bool


def observe(regions):
    predictions, metrics, routes, route_regions = [], [], [], []
    map_grid, map_sources = [], []
    for region in regions:
        y = np.asarray(region.observed, float)
        assignment = np.asarray(region.assignment, int)
        if assignment.shape != region.grid_sources.shape or np.any((assignment < 0) | (assignment >= len(y))):
            raise ValueError(f"{region.region}: assignment 与台账不匹配")
        total = float(sum(region.source_totals.values()))
        values = []
        for field in region.fields:
            if field.qa_only:
                continue
            p = np.bincount(assignment, weights=field.values, minlength=len(y))
            values.append((field.label, field.seed, field.fold, "VD", p, field.lineage["sha256"]))
        values.extend([
            ("EqualStation", None, None, "DIRECT", np.full(len(y), total/len(y)), region.input_fingerprint),
            ("EqualRegion", None, None, "DIRECT", equal_region(region.station_sources, region.source_totals), region.input_fingerprint),
        ])
        for candidate, seed, fold, allocator, p, lineage in values:
            base = {"country": region.country, "region": region.region, "task": "reconstruction", "candidate": candidate,
                    "seed": seed, "fold": fold, "allocator": allocator, "unit": region.unit,
                    "lineage_fingerprint": lineage}
            for index, (truth, estimate) in enumerate(zip(y, p, strict=True)):
                predictions.append({**base, "target_id": str(region.station_ids[index]), "source_id": str(region.station_sources[index]),
                                    "observed": float(truth), "predicted": float(estimate)})
            aligned = bool(np.isclose(p.sum(), y.sum(), rtol=1e-8, atol=1e-8))
            for metric, record in reconstruction_metrics(y, p).items():
                metrics.append({**base, "metric": metric, **record, "n_targets": len(y),
                    "metric_unit": region.unit if metric in {"rmse", "mae"} else "dimensionless",
                    "prediction_total_matches_observed_total": aligned,
                    "mass_alignment_basis": "source_total_equals_observed_total" if aligned else "source_and_observed_totals_differ"})
        direct = next(p for label, _, _, _, p, _ in values if label == "EqualRegion")
        equal = next(p for label, _, _, _, p, _ in values if label == "EqualGrid")
        equal_field = next(f.values for f in region.fields if f.label == "EqualGrid")
        cross = region.grid_sources.astype(str) != region.station_sources.astype(str)[assignment]
        affected = np.bincount(assignment, weights=cross.astype(int), minlength=len(y)) > 0
        for index in range(len(y)):
            routes.append({"country": region.country, "region": region.region, "target_id": str(region.station_ids[index]),
                "equal_region": float(direct[index]), "equal_grid": float(equal[index]), "difference": float(equal[index]-direct[index]),
                "cross_source_affected": bool(affected[index]), "unit": region.unit,
                "x": float(region.station_xy[index, 0]), "y": float(region.station_xy[index, 1]),
                "map_selected": region.map_selected})
        route_regions.append({"country": region.country, "region": region.region,
            "rmse_direct": reconstruction_metrics(y, direct)["rmse"]["value"],
            "rmse_grid": reconstruction_metrics(y, equal)["rmse"]["value"],
            "affected_target_share": float(affected.mean()),
            "cross_source_mass_share": float(equal_field[cross].sum()/total) if total else None,
            "unit": region.unit})
        if region.map_selected:
            map_grid.extend({"country": region.country, "region": region.region, "x": float(x), "y": float(y), "assignment": int(a)}
                            for (x, y), a in zip(region.grid_xy, assignment, strict=True))
            map_sources.extend({"country": region.country, "region": region.region, "geometry_wkt": wkt} for wkt in region.sources_wkt)
    prediction_table = pd.DataFrame(predictions)
    return {"predictions": prediction_table, "metrics": pd.DataFrame(metrics),
            "equal_references": prediction_table[prediction_table.candidate.isin(["EqualGrid", "EqualStation", "EqualRegion"])].copy(),
            "equal_route": pd.DataFrame(routes), "equal_route_regions": pd.DataFrame(route_regions),
            "map_grid": pd.DataFrame(map_grid), "map_sources": pd.DataFrame(map_sources)}
