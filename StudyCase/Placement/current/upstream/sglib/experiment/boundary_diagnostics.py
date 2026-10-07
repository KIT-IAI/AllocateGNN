"""独立参照的名称 / 点位匹配与离散 C/U 栅格足迹 IoU 诊断。"""

import re
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import box


def normalize_name(value):
    text = unicodedata.normalize("NFKC", str(value)).upper()
    text = re.sub(r"\b(?:PRIMARY|GRID|SUBSTATION|SSTN)\b", " ", text)
    text = re.sub(r"\b(?:11|33|66|132)[ /_-]*KV\b", " ", text)
    text = re.sub(r"\bT\d+\b", " ", text)
    return re.sub(r"[^A-Z0-9]", "", text)


def indel_ratio(left, right):
    if not left and not right:
        return 100.
    previous = [0] * (len(right) + 1)
    for char in left:
        current = [0]
        for index, other in enumerate(right, 1):
            current.append(previous[index-1] + 1 if char == other else max(previous[index], current[-1]))
        previous = current
    return 200. * previous[-1] / max(len(left) + len(right), 1)


def crosswalk(stations, reference):
    source = stations[["station_name", "geometry"]].reset_index(drop=True).copy()
    source["station_ordinal"] = np.arange(len(source))
    source["station_norm"] = source.station_name.map(normalize_name)
    joined = gpd.sjoin(source, reference[["reference_id", "reference_name", "provenance", "geometry"]], how="inner", predicate="within")
    joined["reference_norm"] = joined.reference_name.map(normalize_name)
    joined["similarity"] = [indel_ratio(a, b) for a, b in zip(joined.station_norm, joined.reference_norm, strict=True)]
    records = []
    for ordinal in range(len(stations)):
        group = joined[joined.station_ordinal.eq(ordinal)].sort_values(["similarity", "reference_id"], ascending=[False, True], kind="stable")
        if group.empty:
            records.append({"station_ordinal": ordinal, "station_name": source.station_name.iloc[ordinal], "accepted": False, "reason": "NO_CONTAINING_INDEPENDENT_REFERENCE"})
            continue
        best = group.iloc[0]
        runner = float(group.similarity.iloc[1]) if len(group) > 1 else -np.inf
        exact = best.station_norm == best.reference_norm
        accepted = exact or (best.similarity >= 90 and best.similarity - runner >= 10)
        records.append({"station_ordinal": ordinal, "station_name": best.station_name, "reference_id": best.reference_id,
                        "reference_name": best.reference_name, "provenance": best.provenance, "similarity": best.similarity,
                        "runner_up_similarity": runner if np.isfinite(runner) else None, "accepted": bool(accepted),
                        "reason": "exact" if exact else "point_unique_fuzzy90_gap10" if accepted else "ambiguous"})
    return pd.DataFrame(records)


def _validated_view(view, n_grid, n_stations):
    kind = view.get("assignment_kind")
    if kind not in ("station", "cluster"):
        raise ValueError("T1 requires explicit assignment_kind station or cluster")
    if view.get("allocator") == "CIVD" and kind != "cluster":
        raise ValueError("CIVD assignment_kind must be cluster, not station")
    labels = np.asarray(view["assignment"])
    if labels.shape != (n_grid,) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("几何 assignment 行序或整数 target 无效")
    if kind == "cluster":
        station_cluster = np.asarray(view.get("station_cluster"))
        if (station_cluster.shape != (n_stations,) or not np.issubdtype(station_cluster.dtype, np.integer)
                or np.any(station_cluster < 0) or np.any(labels >= len(np.unique(station_cluster)))):
            raise ValueError("CIVD station_cluster must identify a station member for every assigned cluster")
    elif np.any(labels >= n_stations):
        raise ValueError("几何 assignment target 越界")
    return labels, {key: value for key, value in view.items() if key not in ("assignment", "station_cluster")}


def evaluate(grid_xy, support, stations, reference, views):
    """Station footprints require station assignments; CIVD Step 4 only splits demand."""
    validated_views = [_validated_view(view, len(grid_xy), len(stations)) for view in views]
    mapping = crosswalk(stations, reference)
    accepted = mapping[mapping.accepted].copy()
    eligible = len(accepted) >= 5 and len(accepted) / len(stations) >= .3
    mapping["region_eligible"] = eligible
    if not eligible:
        return mapping, pd.DataFrame(), pd.DataFrame()
    xy = np.asarray(grid_xy, float)
    step = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    cells = gpd.GeoDataFrame({"grid_row": np.arange(len(xy))}, geometry=[box(x-step/2, y-step/2, x+step/2, y+step/2) for x, y in xy], crs=stations.crs)
    cells = cells[np.asarray(support, bool)]
    index = cells.sindex
    references = reference.set_index("reference_id", verify_integrity=True)
    overlaps = {}
    for match in accepted.itertuples():
        polygon = references.loc[match.reference_id].geometry
        subset = cells.iloc[index.query(polygon, predicate="intersects")]
        areas = subset.geometry.intersection(polygon).area.to_numpy(float)
        overlaps[match.station_ordinal] = (subset.grid_row.to_numpy(int), areas, float(areas.sum()), match)
    station_rows, region_rows = [], []
    for labels, metadata in validated_views:
        if metadata["assignment_kind"] == "cluster":
            reason = "CIVD_CLUSTER_ALLOCATION_HAS_NO_UNIQUE_STATION_FOOTPRINT"
            for ordinal, (_, _, reference_area, match) in overlaps.items():
                station_rows.append({**metadata, "station_ordinal": ordinal, "station_name": match.station_name,
                    "reference_id": match.reference_id, "provenance": match.provenance, "iou_loss": None,
                    "reference_area_m2": reference_area, "estimated_area_m2": None, "intersection_m2": None,
                    "step_m": step, "status": "METRIC_NOT_ASSESSABLE", "reason": reason})
            region_rows.append({**metadata, "n_matched": len(accepted), "n_total_targets": len(stations), "n_valid": 0,
                "matched_fraction": len(accepted) / len(stations), "iou_loss_q80": None,
                "status": "METRIC_NOT_ASSESSABLE", "reason": reason, "metric": "raster_footprint_1_minus_IoU_Q80"})
            continue
        counts = np.bincount(labels[np.asarray(support, bool)], minlength=len(stations))
        values = []
        for ordinal, (indices, areas, reference_area, match) in overlaps.items():
            estimated_area = float(counts[ordinal] * step * step)
            intersection = float(areas[labels[indices] == ordinal].sum())
            union = estimated_area + reference_area - intersection
            value = 1 - intersection / union if union > 0 else None
            if value is not None:
                values.append(value)
            station_rows.append({**metadata, "station_ordinal": ordinal, "station_name": match.station_name,
                "reference_id": match.reference_id, "provenance": match.provenance, "iou_loss": value,
                "reference_area_m2": reference_area, "estimated_area_m2": estimated_area, "intersection_m2": intersection,
                "step_m": step, "status": "VALID" if value is not None else "METRIC_NOT_ASSESSABLE",
                "reason": "" if value is not None else "ZERO_RASTER_FOOTPRINT_UNION"})
        region_rows.append({**metadata, "n_matched": len(accepted), "n_total_targets": len(stations), "n_valid": len(values),
            "matched_fraction": len(accepted) / len(stations), "iou_loss_q80": float(np.quantile(values, .8, method="linear")) if values else None,
            "status": "VALID" if values else "METRIC_NOT_ASSESSABLE", "reason": "" if values else "ZERO_RASTER_FOOTPRINT_UNION",
            "metric": "raster_footprint_1_minus_IoU_Q80"})
    return mapping, pd.DataFrame(station_rows), pd.DataFrame(region_rows)
