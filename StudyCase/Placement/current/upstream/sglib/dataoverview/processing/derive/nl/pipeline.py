"""Liander/CBS/PDOK derivation and pre-training admission gates.

Only source-specific parsing and deterministic analysis-region construction live in
this module.  Grid generation, Generator assignment and learned models remain shared.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shutil
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, distance

from sglib.core.algorithms.grid_adjacency import build_grid_adjacency_indices
from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.hashing import sha256_file, sha256_json
from sglib.dataoverview.engineering_admission import (
    EngineeringAdmission,
    chunked_nearest_assignment,
    evaluate_region,
    load_engineering_admission,
)
from sglib.dataoverview.processing.config import CountryPipelineContext
from sglib.dataoverview.processing.derive.common import atomic_geofile
from sglib.dataoverview.processing.features.grid_bundle import write_grid_bundle
from sglib.dataoverview.processing.features.grid_generator import regenerate_grid_reference


LIANDER_REQUIRED = {
    "SUBSTATION_NAME",
    "CONDUCTINGEQUIPMENT_NAME",
    "STREETDETAIL_CODE",
    "TOWNDETAIL_STATEORPROVINCE",
    "POSITIONPOINT_YPOSITION",
    "POSITIONPOINT_XPOSITION",
    "ANALOGVALUE_VALUE",
    "ANALOGVALUE_TIMESTAMP",
    "ACTIVEPOWERLIMIT_VALUE",
    "ACTIVEPOWERLIMIT_VALUE_NBL",
}
CBS_SECTOR_COLUMNS = (
    "a_bed_a",
    "a_bed_bf",
    "a_bed_gi",
    "a_bed_hj",
    "a_bed_kl",
    "a_bed_mn",
    "a_bed_oq",
    "a_bed_ru",
)
CROSSWALK_CLASSES = (
    "exact_code",
    "pip_fallback",
    "conflict",
    "outside_all_polygons",
)


@dataclass(frozen=True)
class NLPaths:
    raw_liander: Path
    raw_cbs: Path
    raw_polygons: Path
    pagination_audit: Path
    source_regions: Path
    stations: Path
    equipment_lineage: Path
    analysis_regions: Path
    crosswalk: Path
    buurt_wijk_map: Path
    region_inventory: Path
    gate_a: Path
    engineering_admission: Path


def _paths(context: CountryPipelineContext) -> NLPaths:
    raw = context.raw_root
    derived = context.derived_root
    return NLPaths(
        raw_liander=raw / "liander_knelpunt_2026_01.csv",
        raw_cbs=raw / "kwb2025.xlsx",
        raw_polygons=raw / "pdok_buurten_2025.geojson",
        pagination_audit=raw / "pdok_buurten_2025.geojson.pagination.json",
        source_regions=derived / "bplus/source_regions.gpkg",
        stations=derived / "bplus/buurt_pseudo_stations.gpkg",
        equipment_lineage=derived / "lineage/equipment_register.gpkg",
        analysis_regions=derived / "bplus/analysis_regions.gpkg",
        crosswalk=derived / "audit/crosswalk.csv",
        buurt_wijk_map=derived / "authority/buurt_wijk_map.csv",
        region_inventory=derived / "authority/analysis_region_inventory.json",
        gate_a=derived / "audit/gate_a.json",
        engineering_admission=derived / "audit/engineering_admission.json",
    )


def _admission(context: CountryPipelineContext) -> EngineeringAdmission:
    return load_engineering_admission(
        context.repo_root,
        "casestudy/1_DataOverview/4_NL/admission.toml",
    )


def _atomic_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.part")
    frame.to_csv(partial, index=False)
    partial.replace(path)
    return path


def _read_liander(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep=";", dtype="string", keep_default_na=False)
    missing = LIANDER_REQUIRED - set(frame)
    if missing:
        raise ValueError(f"Liander columns missing: {sorted(missing)}")
    for column in ("POSITIONPOINT_YPOSITION", "POSITIONPOINT_XPOSITION"):
        frame[column] = pd.to_numeric(frame[column].str.replace(",", ".", regex=False), errors="raise")
    for column in ("ANALOGVALUE_VALUE", "ACTIVEPOWERLIMIT_VALUE", "ACTIVEPOWERLIMIT_VALUE_NBL"):
        frame[column] = pd.to_numeric(frame[column].str.replace(",", ".", regex=False), errors="raise")
    frame["ANALOGVALUE_TIMESTAMP"] = pd.to_datetime(
        frame["ANALOGVALUE_TIMESTAMP"], dayfirst=True, errors="raise"
    )
    if frame["CONDUCTINGEQUIPMENT_NAME"].duplicated().any():
        raise ValueError("Liander equipment id is not unique")
    required_numeric = [
        "POSITIONPOINT_YPOSITION",
        "POSITIONPOINT_XPOSITION",
        "ANALOGVALUE_VALUE",
        "ACTIVEPOWERLIMIT_VALUE",
    ]
    if frame[required_numeric].isna().any().any():
        raise ValueError("Liander required numeric values contain nulls")
    return frame


def _read_cbs(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_excel(path, sheet_name="KWB2025", dtype={"gwb_code_10": "string"})
    missing = {"gwb_code_10", "recs", "a_inw", *CBS_SECTOR_COLUMNS} - set(frame)
    if missing:
        raise ValueError(f"CBS KWB columns missing: {sorted(missing)}")
    frame = frame.loc[frame["recs"].astype(str).isin({"Buurt", "Wijk"})].copy()
    frame["gwb_code_10"] = frame["gwb_code_10"].astype("string").str.strip()
    if frame["gwb_code_10"].duplicated().any():
        raise ValueError("CBS buurt/wijk code is not unique")
    return frame


def _read_polygons(path: Path) -> gpd.GeoDataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    polygons = gpd.read_file(path)
    code = "buurtcode" if "buurtcode" in polygons else "gwb_code_10"
    wijk = "wijkcode" if "wijkcode" in polygons else None
    if code not in polygons or polygons.crs is None:
        raise ValueError("PDOK polygons lack buurt code or CRS")
    polygons = polygons.rename(columns={code: "buurt_code"})
    polygons["buurt_code"] = polygons["buurt_code"].astype("string").str.strip()
    if wijk is not None:
        polygons = polygons.rename(columns={wijk: "wijk_code"})
        polygons["wijk_code"] = polygons["wijk_code"].astype("string").str.strip()
    else:
        polygons["wijk_code"] = "WK" + polygons["buurt_code"].str.slice(2, 8)
    derived_wijk = "WK" + polygons["buurt_code"].str.slice(2, 8)
    if not polygons["wijk_code"].eq(derived_wijk).all():
        raise ValueError("PDOK BU->WK hierarchy differs from the official code prefix")
    polygons = polygons.loc[
        polygons["buurt_code"].str.startswith("BU", na=False),
        ["buurt_code", "wijk_code", "geometry"],
    ]
    polygons = polygons.loc[polygons.geometry.notna() & ~polygons.geometry.is_empty].copy()
    polygons.geometry = polygons.geometry.make_valid()
    if polygons["buurt_code"].duplicated().any():
        raise ValueError("PDOK buurt code is not unique")
    return polygons.to_crs("EPSG:4326").reset_index(drop=True)


def classify_crosswalk(
    equipment: pd.DataFrame,
    cbs: pd.DataFrame,
    polygons: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Return mutually exclusive exact/PiP/conflict/outside classifications."""

    points = gpd.GeoDataFrame(
        equipment.copy(),
        geometry=gpd.points_from_xy(
            equipment["POSITIONPOINT_XPOSITION"], equipment["POSITIONPOINT_YPOSITION"]
        ),
        crs="EPSG:4326",
    ).reset_index(drop=True)
    points["_row"] = np.arange(len(points), dtype=np.int64)
    polygon_codes = set(polygons["buurt_code"].astype(str))
    cbs_codes = set(cbs.loc[cbs["recs"].astype(str).eq("Buurt"), "gwb_code_10"].astype(str))
    valid_codes = polygon_codes & cbs_codes
    spatial = gpd.sjoin(
        points[["_row", "geometry"]],
        polygons.rename(columns={"buurt_code": "_pip_code"})[["_pip_code", "geometry"]],
        how="left",
        predicate="within",
    )
    candidates: dict[int, tuple[str, ...]] = {}
    for row, values in spatial.dropna(subset=["_pip_code"]).groupby("_row", sort=False)["_pip_code"]:
        candidates[int(row)] = tuple(sorted(set(map(str, values))))
    original = points["STREETDETAIL_CODE"].astype(str).str.strip()
    rows: list[dict] = []
    assigned: list[str | None] = []
    classes: list[str] = []
    for index, exact_code in enumerate(original):
        hits = candidates.get(index, ())
        pip_code = hits[0] if hits else None
        exact_valid = exact_code in valid_codes
        conflict = len(hits) > 1 or (exact_valid and pip_code is not None and pip_code != exact_code)
        if conflict:
            category = "conflict"
            # Geometry owns formal spatial support.  The declared identifier is
            # retained below as lineage only; it must never move a real point
            # into a polygon that does not contain it.
            selected = pip_code
        elif exact_valid:
            category = "exact_code"
            selected = exact_code
        elif pip_code is not None and pip_code in cbs_codes:
            category = "pip_fallback"
            selected = pip_code
        else:
            category = "outside_all_polygons"
            selected = None
        classes.append(category)
        assigned.append(selected)
        rows.append(
            {
                "equipment_id": str(points.at[index, "CONDUCTINGEQUIPMENT_NAME"]),
                "declared_buurt_code": exact_code,
                "pip_buurt_codes": "|".join(hits),
                "assigned_buurt_code": selected or "",
                "crosswalk_class": category,
                "assignment_rule": "pip_spatial_support__declared_code_lineage_only",
            }
        )
    points["buurt_code"] = pd.Series(assigned, dtype="string")
    if "wijk_code" in polygons:
        wijk_by_buurt = polygons.set_index("buurt_code")["wijk_code"]
    else:
        wijk_by_buurt = pd.Series(
            "WK" + polygons["buurt_code"].astype(str).str.slice(2, 8).to_numpy(),
            index=polygons["buurt_code"].astype(str),
        )
    points["wijk_code"] = points["buurt_code"].astype(str).map(wijk_by_buurt)
    points["crosswalk_class"] = classes
    return pd.DataFrame(rows), points.drop(columns="_row")


def _source_features(frame: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    result = frame.copy()
    for column in ("a_inw", *CBS_SECTOR_COLUMNS):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result.loc[result[column] < 0, column] = np.nan
    missing = result[list(CBS_SECTOR_COLUMNS)].isna().any(axis=1)
    weights: dict[str, np.ndarray] = {}
    for name in ("residential", "commercial", "industrial", "agricultural", "others"):
        columns = list(mapping[name])
        values = result[columns].fillna(0.0).sum(axis=1).to_numpy(float)
        if name != "residential":
            values[missing.to_numpy()] = 0.0
        weights[name] = values
    matrix = np.column_stack([weights[name] for name in weights])
    totals = matrix.sum(axis=1)
    empty = totals <= 0
    matrix[empty, -1] = 1.0
    totals[empty] = 1.0
    matrix /= totals[:, None]
    for index, name in enumerate(weights):
        result[f"{name}_percent"] = matrix[:, index]
    result["industry_fields_missing"] = missing.to_numpy(bool)
    fallback_rule = str(mapping["missing_policy"])
    result["industry_fallback_rule"] = np.where(
        missing,
        fallback_rule,
        "none",
    )
    return result


def _slug(value: str) -> str:
    plain = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Z0-9]+", "_", plain.upper()).strip("_") or "UNKNOWN"


def _dominant_stratum(
    frame: pd.DataFrame,
    group_key: str,
    stratum_column: str = "operational_stratum_original",
) -> pd.DataFrame:
    """Choose a deterministic operational stratum for each formal support unit."""

    summary = (
        frame.groupby([group_key, stratum_column], observed=True)
        .agg(n_equipment=("equipment_id", "size"), peak_kw=("peak_kw", "sum"))
        .reset_index()
        .sort_values(
            [group_key, "n_equipment", "peak_kw", stratum_column],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .drop_duplicates(group_key)
    )
    return summary[[group_key, stratum_column]].rename(
        columns={stratum_column: "dominant_operational_stratum"}
    )


def merge_low_equipment_strata(
    equipment: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    *,
    minimum_equipment: int,
    working_crs: str,
) -> tuple[dict[str, str], list[dict]]:
    """Merge tiny Liander strata into the neighbour with longest shared border.

    Borders are measured on real PDOK buurt polygons in RD New.  Each buurt is
    attributed to the equipment-count dominant original stratum before dissolving;
    ties use peak demand and then the stratum string.  The procedure reads no model,
    grid allocation, or evaluation artifact.
    """

    if minimum_equipment <= 0:
        raise ValueError("minimum_equipment must be positive")
    eligible = equipment.loc[equipment["generator_eligible"].astype(bool)].copy()
    counts = eligible["operational_stratum_original"].astype(str).value_counts()
    high = sorted(counts[counts >= minimum_equipment].index.astype(str), key=str.casefold)
    low = sorted(counts[counts < minimum_equipment].index.astype(str), key=str.casefold)
    if low and not high:
        raise RuntimeError("all NL operational strata are below the merge threshold")
    dominant = _dominant_stratum(eligible, "buurt_code")
    support = polygons.merge(
        dominant,
        on="buurt_code",
        how="inner",
        validate="one_to_one",
    ).to_crs(working_crs)
    dissolved = support[["dominant_operational_stratum", "geometry"]].dissolve(
        by="dominant_operational_stratum"
    )
    mapping = {str(name): str(name) for name in counts.index.astype(str)}
    evidence: list[dict] = []
    for name in low:
        if name not in dissolved.index:
            raise RuntimeError(f"low NL stratum has no polygon support: {name}")
        boundary = dissolved.loc[name].geometry.boundary
        candidates: list[tuple[float, str]] = []
        for neighbour in high:
            if neighbour not in dissolved.index:
                continue
            shared_m = float(
                boundary.intersection(dissolved.loc[neighbour].geometry.boundary).length
            )
            candidates.append((shared_m, neighbour))
        if not candidates or max(length for length, _ in candidates) <= 0:
            raise RuntimeError(f"low NL stratum has no adjacent high stratum: {name}")
        longest = max(length for length, _ in candidates)
        selected = min(
            neighbour
            for length, neighbour in candidates
            if math.isclose(length, longest, rel_tol=0.0, abs_tol=1e-9)
        )
        mapping[name] = selected
        evidence.append(
            {
                "stratum": name,
                "n_equipment": int(counts[name]),
                "merged_into": selected,
                "shared_boundary_m": longest,
                "candidate_shared_boundaries_m": {
                    neighbour: length
                    for length, neighbour in sorted(candidates, key=lambda item: item[1].casefold())
                },
                "tie_break": "shared_boundary_length_desc_then_stratum_lexicographic",
            }
        )
    return mapping, evidence


def aggregate_buurt_targets(
    equipment: gpd.GeoDataFrame,
    *,
    working_crs: str,
    capacity_basis: str,
) -> gpd.GeoDataFrame:
    """Aggregate eligible equipment into one formal pseudo-station per buurt."""

    eligible = equipment.loc[equipment["generator_eligible"].astype(bool)].copy()
    if eligible.empty:
        raise RuntimeError("NL target aggregation has no eligible equipment")
    projected = eligible.to_crs(working_crs)
    rows: list[dict] = []
    for buurt_code, group in projected.groupby("buurt_code", sort=True, observed=True):
        group = group.sort_values("equipment_id", kind="stable")
        capacity = group["capacity_kw"].to_numpy(dtype=float)
        peak = group["peak_kw"].to_numpy(dtype=float)
        if (capacity < 0).any() or (peak < 0).any():
            raise ValueError(f"{buurt_code}: negative capacity or peak is not admissible")
        x = group.geometry.x.to_numpy(float)
        y = group.geometry.y.to_numpy(float)
        capacity_sum = float(capacity.sum())
        if capacity_sum > 0:
            centroid_x = float(np.average(x, weights=capacity))
            centroid_y = float(np.average(y, weights=capacity))
            centroid_method = "capacity_weighted_rd_new"
        else:
            centroid_x = float(x.mean())
            centroid_y = float(y.mean())
            centroid_method = "unweighted_rd_new_zero_capacity_fallback"
        wijk_codes = sorted(set(group["wijk_code"].astype(str)))
        if len(wijk_codes) != 1:
            raise RuntimeError(f"{buurt_code}: formal equipment spans multiple official wijken")
        strata = sorted(set(group["operational_stratum"].astype(str)))
        if len(strata) != 1:
            raise RuntimeError(f"{buurt_code}: effective stratum is not unique")
        peak_sum = float(peak.sum())
        equipment_ids = group["equipment_id"].astype(str).tolist()
        rows.append(
            {
                "station_id": f"nl:buurt:{buurt_code}",
                "buurt_code": str(buurt_code),
                "wijk_code": wijk_codes[0],
                "operational_stratum": strata[0],
                "peak_kw": peak_sum,
                "capacity_kw": capacity_sum,
                "nbl_limit_kw": float(group["nbl_limit_kw"].astype(float).sum()),
                "capacity_basis": capacity_basis,
                "capacity_semantics": "sum_of_equipment_nominal_normal_state_limits_not_firm_or_n_minus_1",
                "noncoincident_conservative": True,
                "noncoincident_semantics": "sum_of_noncoincident_equipment_annual_peaks_within_buurt",
                "firm_capacity_eligible": False,
                "zero_peak": peak_sum == 0.0,
                "zero_capacity": capacity_sum == 0.0,
                "over_capacity": peak_sum > capacity_sum,
                "robustness_main_inclusion": True,
                "generator_eligible": True,
                "status": "generator_eligible",
                "n_equipment": len(group),
                "equipment_ids_sha256": sha256_json(equipment_ids),
                "centroid_method": centroid_method,
                "geometry": gpd.points_from_xy([centroid_x], [centroid_y], crs=working_crs)[0],
            }
        )
    targets = gpd.GeoDataFrame(rows, geometry="geometry", crs=working_crs)
    if targets["station_id"].duplicated().any() or targets["buurt_code"].duplicated().any():
        raise RuntimeError("NL buurt pseudo-station identity is not unique")
    if int(targets["n_equipment"].sum()) != len(eligible):
        raise RuntimeError("NL target aggregation dropped equipment lineage")
    if not np.isclose(targets["peak_kw"].sum(), eligible["peak_kw"].sum()):
        raise RuntimeError("NL target aggregation changed peak demand")
    if not np.isclose(targets["capacity_kw"].sum(), eligible["capacity_kw"].sum()):
        raise RuntimeError("NL target aggregation changed capacity")
    return targets.to_crs(equipment.crs).sort_values("buurt_code", kind="stable").reset_index(drop=True)


def _split_indices(xy: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spans = np.ptp(xy, axis=0)
    axis = 0 if spans[0] >= spans[1] else 1
    other = 1 - axis
    order = np.lexsort((ids.astype(str), xy[:, other], xy[:, axis]))
    middle = len(order) // 2
    if middle == 0 or middle == len(order):
        raise RuntimeError("deterministic median split produced an empty child")
    return order[:middle], order[middle:]


def deterministic_analysis_regions(
    sources: gpd.GeoDataFrame,
    equipment: gpd.GeoDataFrame,
    *,
    working_crs: str,
    limits: dict,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[dict]]:
    """Apply the frozen stratum/longest-axis/median/nearest-centroid policy.

    The split tree is driven by equipment coordinates for historical continuity,
    while source and formal target admission use wijk and buurt identities.  A
    country-level minimum leaf count is applied by repeatedly splitting the leaf
    with most equipment; the tie break is its final region identifier.
    """

    source_key = "wijk_code" if "wijk_code" in sources else "buurt_code"
    if source_key not in equipment:
        raise ValueError(f"equipment lacks source key {source_key}")
    projected_sources = sources.to_crs(working_crs).copy()
    source_xy = np.column_stack(
        [projected_sources.geometry.centroid.x, projected_sources.geometry.centroid.y]
    )
    projected_equipment = equipment.to_crs(working_crs).copy()
    equipment_xy = np.column_stack(
        [projected_equipment.geometry.x, projected_equipment.geometry.y]
    )
    max_sources = int(limits.get("max_sources", 10**9))
    max_targets = int(limits.get("max_targets", limits.get("planning_k_cap", 10**9)))
    max_lineage = int(
        limits.get(
            "max_lineage_equipment_per_region",
            limits.get("provisional_max_targets", max_targets),
        )
    )
    minimum_regions = int(limits.get("minimum_analysis_regions", 0))
    assignments: dict[str, str] = {}
    strata_state: dict[str, dict] = {}

    for stratum in sorted(sources["operational_stratum"].astype(str).unique(), key=str.casefold):
        source_indexes = np.flatnonzero(
            sources["operational_stratum"].astype(str).to_numpy() == stratum
        )
        source_codes = sources.iloc[source_indexes][source_key].astype(str).to_numpy()
        lineage_indexes = np.flatnonzero(
            equipment[source_key].astype(str).isin(set(source_codes)).to_numpy()
        )
        if not len(lineage_indexes):
            raise RuntimeError(f"NL stratum has no equipment lineage: {stratum}")
        strata_state[stratum] = {
            "source_indexes": source_indexes,
            "source_codes": source_codes,
            "leaves": {"": lineage_indexes},
        }

    def assign_state(stratum: str) -> tuple[dict[str, str], dict[str, np.ndarray], np.ndarray]:
        state = strata_state[stratum]
        leaves = state["leaves"]
        centers = {
            path: equipment_xy[indexes].mean(axis=0)
            for path, indexes in leaves.items()
            if len(indexes)
        }
        ordered_paths = sorted(centers)
        nearest = np.argmin(
            distance.cdist(source_xy[state["source_indexes"]], np.vstack([centers[p] for p in ordered_paths])),
            axis=1,
        )
        source_to_leaf = {
            str(code): ordered_paths[int(leaf)]
            for code, leaf in zip(state["source_codes"], nearest)
        }
        lineage_to_leaf = np.asarray(
            [source_to_leaf[str(code)] for code in equipment.iloc[np.concatenate(list(leaves.values()))][source_key]],
            dtype=object,
        )
        return source_to_leaf, centers, lineage_to_leaf

    def split_leaf(stratum: str, path: str, source_to_leaf: dict[str, str]) -> None:
        state = strata_state[stratum]
        # Re-anchor the geometric tree to the current source ownership before
        # splitting so no equipment row can remain duplicated in a stale leaf.
        state["leaves"] = {
            leaf: np.flatnonzero(
                equipment[source_key]
                .astype(str)
                .isin({code for code, owner in source_to_leaf.items() if owner == leaf})
                .to_numpy()
            )
            for leaf in sorted(set(source_to_leaf.values()))
        }
        member_sources = {code for code, leaf in source_to_leaf.items() if leaf == path}
        members = np.flatnonzero(
            equipment[source_key].astype(str).isin(member_sources).to_numpy()
        )
        if len(members) < 2 or len(member_sources) < 2:
            raise RuntimeError(f"cannot split NL leaf {stratum}/{path or 'root'}")
        left_local, right_local = _split_indices(
            equipment_xy[members],
            equipment.iloc[members]["equipment_id"].astype(str).to_numpy(),
        )
        state["leaves"].pop(path, None)
        state["leaves"][path + "0"] = members[left_local]
        state["leaves"][path + "1"] = members[right_local]

    # Split only leaves that violate country-local lineage or shared node caps.
    for _ in range(64):
        failing: list[tuple[str, str, int]] = []
        for stratum in sorted(strata_state, key=str.casefold):
            source_to_leaf, _centers, _ = assign_state(stratum)
            for path in sorted(set(source_to_leaf.values())):
                codes = {code for code, leaf in source_to_leaf.items() if leaf == path}
                n_lineage = int(equipment[source_key].astype(str).isin(codes).sum())
                n_targets = int(
                    equipment.loc[equipment[source_key].astype(str).isin(codes), "buurt_code"]
                    .astype(str)
                    .nunique()
                )
                if len(codes) > max_sources or n_targets > max_targets or n_lineage > max_lineage:
                    failing.append((stratum, path, n_lineage))
        if not failing:
            break
        for stratum, path, _ in sorted(failing, key=lambda item: (item[0].casefold(), item[1])):
            source_to_leaf, _centers, _ = assign_state(stratum)
            split_leaf(stratum, path, source_to_leaf)
    else:
        raise RuntimeError("NL partition did not converge")

    # Preserve the preregistered 14-region spatial resolution floor.  Starting
    # from the admissible pruned tree, split the largest leaf until that minimum
    # is reached; no model or evaluation quantity participates.
    while sum(len(state["leaves"]) for state in strata_state.values()) < minimum_regions:
        candidates: list[tuple[int, str, str]] = []
        maps: dict[str, dict[str, str]] = {}
        for stratum in sorted(strata_state, key=str.casefold):
            source_to_leaf, _centers, _ = assign_state(stratum)
            maps[stratum] = source_to_leaf
            for path in sorted(set(source_to_leaf.values())):
                codes = {code for code, leaf in source_to_leaf.items() if leaf == path}
                if len(codes) >= 2:
                    n_lineage = int(equipment[source_key].astype(str).isin(codes).sum())
                    candidates.append((n_lineage, stratum, path))
        if not candidates:
            raise RuntimeError("cannot reach minimum NL analysis-region count")
        largest = max(item[0] for item in candidates)
        _, stratum, path = min(
            (item for item in candidates if item[0] == largest),
            key=lambda item: (item[1].casefold(), item[2]),
        )
        split_leaf(stratum, path, maps[stratum])

    leaf_records: list[dict] = []
    for stratum in sorted(strata_state, key=str.casefold):
        source_to_leaf, centers, _ = assign_state(stratum)
        for code, path in source_to_leaf.items():
            assignments[code] = f"NL_{_slug(stratum)}_{path or 'R'}"
        for path in sorted(set(source_to_leaf.values())):
            region = f"NL_{_slug(stratum)}_{path or 'R'}"
            codes = sorted(code for code, leaf in source_to_leaf.items() if leaf == path)
            member = equipment[source_key].astype(str).isin(codes)
            leaf_records.append(
                {
                    "id": region,
                    "operational_stratum": stratum,
                    "leaf_path": path or "R",
                    "centroid_x": float(centers[path][0]),
                    "centroid_y": float(centers[path][1]),
                    "n_sources": len(codes),
                    "n_targets": int(equipment.loc[member, "buurt_code"].astype(str).nunique()),
                    "n_lineage_equipment": int(member.sum()),
                    "source_key_order": codes,
                }
            )
    if set(assignments) != set(sources[source_key].astype(str)):
        raise RuntimeError("not every NL source was assigned to an analysis region")
    output_sources = sources.copy()
    output_sources["analysis_region"] = output_sources[source_key].astype(str).map(assignments)
    output_equipment = equipment.copy()
    output_equipment["analysis_region"] = output_equipment[source_key].astype(str).map(assignments)
    return output_sources, output_equipment, sorted(leaf_records, key=lambda row: row["id"])


def _build_tables(
    context: CountryPipelineContext,
) -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict],
    list[dict],
]:
    paths = _paths(context)
    equipment_raw = _read_liander(paths.raw_liander)
    cbs = _read_cbs(paths.raw_cbs)
    cbs_buurt = cbs.loc[cbs["recs"].astype(str).eq("Buurt")].copy()
    cbs_wijk = cbs.loc[cbs["recs"].astype(str).eq("Wijk")].copy()
    polygons = _read_polygons(paths.raw_polygons)
    crosswalk, points = classify_crosswalk(equipment_raw, cbs_buurt, polygons)
    points = points.rename(
        columns={
            "CONDUCTINGEQUIPMENT_NAME": "equipment_id",
            "SUBSTATION_NAME": "substation_id",
            "ANALOGVALUE_VALUE": "peak_kw",
            "ANALOGVALUE_TIMESTAMP": "peak_date",
            "ACTIVEPOWERLIMIT_VALUE": "capacity_kw",
            "ACTIVEPOWERLIMIT_VALUE_NBL": "nbl_limit_kw",
            "TOWNDETAIL_STATEORPROVINCE": "operational_stratum",
        }
    )
    points = points.merge(
        crosswalk[
            [
                "equipment_id",
                "declared_buurt_code",
                "pip_buurt_codes",
                "assigned_buurt_code",
                "assignment_rule",
            ]
        ],
        on="equipment_id",
        how="left",
        validate="one_to_one",
    )
    points["lineage_id"] = "nl:equipment:" + points["equipment_id"].astype(str)
    capacity_basis = str(context.merged["station_contract"]["capacity_basis"])
    points["capacity_basis"] = capacity_basis
    points["capacity_semantics"] = "equipment nameplate/normal-state limit; not firm or N-1"
    points["noncoincident_equipment_peak"] = True
    points["firm_capacity_eligible"] = False
    points["zero_peak"] = points["peak_kw"].eq(0)
    points["zero_capacity"] = points["capacity_kw"].eq(0)
    points["over_capacity"] = points["peak_kw"].gt(points["capacity_kw"])
    points["robustness_main_inclusion"] = True
    points["generator_eligible"] = points["buurt_code"].notna()
    points["formal_evaluation_eligible"] = False
    points["operational_stratum_original"] = points["operational_stratum"].astype(str)
    points["status"] = np.where(
        points["generator_eligible"],
        "lineage_only_formal_target_is_buurt_pseudo_station",
        "outside_all_polygons",
    )
    eligible = points.loc[points["generator_eligible"]].copy()
    if eligible["wijk_code"].isna().any():
        raise RuntimeError("formal NL support lacks official BU->WK lineage")
    contract = _admission(context)
    partition = dict(contract.country["partition"])
    stratum_map, stratum_merges = merge_low_equipment_strata(
        eligible,
        polygons,
        minimum_equipment=int(partition["low_stratum_equipment_threshold"]),
        working_crs=str(context.merged["crs"]["working"]),
    )
    eligible["operational_stratum"] = (
        eligible["operational_stratum_original"].astype(str).map(stratum_map)
    )

    # A formal buurt and its parent wijk each receive one effective stratum.
    # This keeps every pseudo-station inside the same analysis region as its
    # source even for the twelve mixed-stratum neighbourhoods.
    wijk_strata = _dominant_stratum(eligible, "wijk_code", "operational_stratum")
    eligible = eligible.drop(columns="operational_stratum").merge(
        wijk_strata.rename(columns={"dominant_operational_stratum": "operational_stratum"}),
        on="wijk_code",
        how="left",
        validate="many_to_one",
    )
    points = points.drop(columns="operational_stratum").merge(
        eligible[["equipment_id", "operational_stratum"]],
        on="equipment_id",
        how="left",
        validate="one_to_one",
    )

    targets = aggregate_buurt_targets(
        eligible,
        working_crs=str(context.merged["crs"]["working"]),
        capacity_basis=capacity_basis,
    )
    source_codes = sorted(targets["wijk_code"].astype(str).unique())
    source_geometry = (
        polygons.loc[polygons["wijk_code"].astype(str).isin(source_codes), ["wijk_code", "geometry"]]
        .dissolve(by="wijk_code")
        .reset_index()
    )
    source = source_geometry.merge(
        cbs_wijk,
        left_on="wijk_code",
        right_on="gwb_code_10",
        how="left",
        validate="one_to_one",
    )
    if len(source) != len(source_codes):
        raise RuntimeError("NL wijk polygons do not cover formal support")
    source["source_attribute_origin"] = "cbs_wijk_row"
    missing_wijk = source["gwb_code_10"].isna()
    if missing_wijk.any():
        # This vintage contains an official BU->WK code and polygons for
        # WK036399 without a separate Wijk row. Aggregate the same-source CBS
        # buurt rows instead of dropping demand or borrowing another wijk.
        buurt_with_wijk = cbs_buurt.merge(
            polygons[["buurt_code", "wijk_code"]],
            left_on="gwb_code_10",
            right_on="buurt_code",
            how="inner",
            validate="one_to_one",
        )
        for index in source.index[missing_wijk]:
            wijk_code = str(source.at[index, "wijk_code"])
            children = buurt_with_wijk.loc[buurt_with_wijk["wijk_code"].eq(wijk_code)]
            if children.empty:
                # Official PDOK water-only wijken may be omitted from the KWB
                # workbook. They remain real spatial support with zero public
                # activity features; equipment demand is still conserved.
                for column in ("a_inw", *CBS_SECTOR_COLUMNS):
                    source.at[index, column] = 0.0
                source.at[index, "gwb_code_10"] = wijk_code
                source.at[index, "recs"] = "Wijk_PDOK_geometry_without_CBS_activity_row"
                source.at[index, "regio"] = wijk_code
                source.at[index, "source_attribute_origin"] = "pdok_water_wijk_zero_activity"
                continue
            for column in ("a_inw", *CBS_SECTOR_COLUMNS):
                values = pd.to_numeric(children[column], errors="coerce").where(lambda x: x >= 0)
                source.at[index, column] = values.sum(min_count=1)
            source.at[index, "gwb_code_10"] = wijk_code
            source.at[index, "recs"] = "Wijk_aggregated_from_official_buurt_rows"
            source.at[index, "regio"] = wijk_code
            source.at[index, "source_attribute_origin"] = "cbs_buurt_rows_same_official_wijk"
    source = _source_features(source, dict(context.merged["industry_sectors"]))
    demand = eligible.groupby("wijk_code", observed=True)["peak_kw"].sum()
    source["demand_peak_kw"] = source["wijk_code"].astype(str).map(demand).astype(float)
    source = source.merge(
        wijk_strata.rename(columns={"dominant_operational_stratum": "operational_stratum"}),
        on="wijk_code",
        how="left",
        validate="one_to_one",
    )
    source["demand_semantics"] = "sum_of_noncoincident_equipment_annual_peaks_kw"
    source, eligible, region_records = deterministic_analysis_regions(
        source,
        eligible,
        working_crs=str(context.merged["crs"]["working"]),
        limits={
            **partition,
            # Engineering admission owns the final minimum.  Start from the
            # pruned count-only tree and split only regions that fail a shared
            # hard/static gate, which proves the final count is minimal.
            "minimum_analysis_regions": 0,
            "max_sources": int(contract.graph["max_source_nodes_per_region"]),
            "max_targets": int(contract.graph["max_target_nodes_per_region"]),
            "planning_k_cap": int(contract.graph["planning_k_max"]),
        },
    )
    source_region = source.set_index("wijk_code")["analysis_region"]
    targets["analysis_region"] = targets["wijk_code"].astype(str).map(source_region)
    points["analysis_region"] = points["wijk_code"].astype(str).map(source_region)
    points["formal_target_id"] = points["buurt_code"].map(
        targets.set_index("buurt_code")["station_id"]
    )
    for record in region_records:
        record["n_targets"] = int(targets["analysis_region"].eq(record["id"]).sum())
    analysis = source[["analysis_region", "geometry"]].dissolve(by="analysis_region").reset_index()
    summary = pd.DataFrame(region_records).set_index("id")
    analysis["operational_stratum"] = analysis["analysis_region"].map(summary["operational_stratum"])
    analysis["n_sources"] = analysis["analysis_region"].map(summary["n_sources"]).astype(int)
    analysis["n_targets"] = analysis["analysis_region"].map(summary["n_targets"]).astype(int)
    buurt_wijk = (
        polygons.loc[polygons["buurt_code"].isin(set(targets["buurt_code"])), ["buurt_code", "wijk_code"]]
        .sort_values("buurt_code", kind="stable")
        .reset_index(drop=True)
    )
    return source, targets, points, analysis, crosswalk, buurt_wijk, region_records, stratum_merges


def derive_nl(context: CountryPipelineContext, *, force: bool = False) -> list[Path]:
    paths = _paths(context)
    outputs = [
        paths.source_regions,
        paths.stations,
        paths.equipment_lineage,
        paths.analysis_regions,
        paths.crosswalk,
        paths.buurt_wijk_map,
        paths.region_inventory,
    ]
    if not force and all(path.is_file() and path.stat().st_size for path in outputs):
        return outputs
    source, targets, equipment, analysis, crosswalk, buurt_wijk, records, stratum_merges = _build_tables(context)
    atomic_geofile(source, paths.source_regions, layer="source_regions")
    atomic_geofile(targets, paths.stations, layer="buurt_pseudo_stations")
    atomic_geofile(equipment, paths.equipment_lineage, layer="equipment_lineage")
    atomic_geofile(analysis, paths.analysis_regions, layer="analysis_regions")
    _atomic_csv(crosswalk, paths.crosswalk)
    _atomic_csv(buurt_wijk, paths.buurt_wijk_map)
    admission = _admission(context)
    atomic_json(
        {
            "schema_version": "sg_nl_analysis_region_inventory_v2",
            "status": "PROVISIONAL_PENDING_ENGINEERING_ADMISSION",
            "country": "nl",
            "working_crs": context.merged["crs"]["working"],
            "partition_contract": "casestudy/1_DataOverview/4_NL/admission.toml",
            "partition_contract_sha256": admission.country_sha256,
            "shared_engineering_authority": admission.authority_path.relative_to(context.repo_root).as_posix(),
            "shared_engineering_authority_sha256": admission.authority_sha256,
            "source_level": "wijk",
            "target_level": "buurt_pseudo_station",
            "lineage_level": "equipment",
            "stratum_merges": stratum_merges,
            "n_regions": len(records),
            "minimum_region_count": int(admission.country["partition"]["minimum_analysis_regions"]),
            "maximum_region_count": int(admission.country["partition"]["maximum_analysis_regions"]),
            "regions": records,
        },
        paths.region_inventory,
    )
    return outputs


def write_gate_a(context: CountryPipelineContext, *, force: bool = False) -> Path:
    paths = _paths(context)
    if paths.gate_a.is_file() and not force:
        return paths.gate_a
    derive_nl(context, force=False)
    equipment_raw = _read_liander(paths.raw_liander)
    cbs = _read_cbs(paths.raw_cbs)
    polygons = _read_polygons(paths.raw_polygons)
    crosswalk = pd.read_csv(paths.crosswalk, dtype="string", keep_default_na=False)
    sources = gpd.read_file(paths.source_regions, layer="source_regions")
    targets = gpd.read_file(paths.stations, layer="buurt_pseudo_stations")
    equipment = gpd.read_file(paths.equipment_lineage, layer="equipment_lineage")
    inventory = json.loads(paths.region_inventory.read_text(encoding="utf-8"))
    if paths.pagination_audit.is_file():
        pagination = json.loads(paths.pagination_audit.read_text(encoding="utf-8"))
    else:
        pagination = {"schema_version": "missing", "pages": [], "total": 0}
    counts = {name: int(crosswalk["crosswalk_class"].eq(name).sum()) for name in CROSSWALK_CLASSES}
    eligible = equipment["generator_eligible"].astype(bool)
    raw_total = float(equipment["peak_kw"].sum())
    admitted_total = float(equipment.loc[eligible, "peak_kw"].sum())
    excluded_total = float(equipment.loc[~eligible, "peak_kw"].sum())
    source_total = float(sources["demand_peak_kw"].sum())
    cbs_buurt_codes = set(
        cbs.loc[cbs["recs"].astype(str).eq("Buurt"), "gwb_code_10"].astype(str)
    )
    cbs_wijk_codes = set(
        cbs.loc[cbs["recs"].astype(str).eq("Wijk"), "gwb_code_10"].astype(str)
    )
    polygon_codes = set(polygons["buurt_code"].astype(str))
    conflict = crosswalk["crosswalk_class"].eq("conflict")
    expected_conflict_support = crosswalk.loc[conflict, "pip_buurt_codes"].str.split("|").str[0]
    target_over = int(targets["over_capacity"].astype(bool).sum())
    equipment_over = int(equipment.loc[eligible, "over_capacity"].astype(bool).sum())
    checks = {
        "pdok_pagination_audit_present": paths.pagination_audit.is_file(),
        "pdok_pagination_total_matches_file": int(pagination.get("total", -1)) == len(polygons),
        "pdok_covers_every_cbs_buurt": cbs_buurt_codes <= polygon_codes,
        "crosswalk_classes_exhaustive": sum(counts.values()) == len(equipment),
        "all_crosswalk_rows_accounted": len(crosswalk) == len(equipment),
        "conflict_spatial_support_is_pip": crosswalk.loc[conflict, "assigned_buurt_code"].reset_index(drop=True).eq(expected_conflict_support.reset_index(drop=True)).all(),
        "declared_code_retained_as_lineage": crosswalk["declared_buurt_code"].notna().all(),
        "required_equipment_values_complete": not equipment[["peak_kw", "capacity_kw", "geometry"]].isna().any().any(),
        "equipment_is_lineage_only": not equipment["formal_evaluation_eligible"].astype(bool).any(),
        "capacity_basis_discriminated": targets["capacity_basis"].eq(context.merged["station_contract"]["capacity_basis"]).all(),
        "one_pseudo_station_per_buurt": len(targets) == targets["buurt_code"].nunique(),
        "pseudo_station_lineage_complete": int(targets["n_equipment"].sum()) == int(eligible.sum()),
        "capacity_weighted_centroids_declared": targets["centroid_method"].isin({"capacity_weighted_rd_new", "unweighted_rd_new_zero_capacity_fallback"}).all(),
        "official_bu_to_wk_hierarchy": targets["wijk_code"].astype(str).eq("WK" + targets["buurt_code"].astype(str).str.slice(2, 8)).all(),
        "every_wijk_has_official_same_source_attributes": sources["source_attribute_origin"].isin({"cbs_wijk_row", "cbs_buurt_rows_same_official_wijk", "pdok_water_wijk_zero_activity"}).all(),
        "industry_missing_rule_total": sources["industry_fields_missing"].notna().all(),
        "demand_conservation": np.isclose(raw_total, admitted_total + excluded_total) and np.isclose(admitted_total, source_total) and np.isclose(admitted_total, targets["peak_kw"].sum()),
        "capacity_conservation": np.isclose(equipment.loc[eligible, "capacity_kw"].sum(), targets["capacity_kw"].sum()),
        "provisional_analysis_region_count_within_upper_bound": 0 < int(inventory["n_regions"]) <= int(inventory["maximum_region_count"]),
        "every_source_has_analysis_region": sources["analysis_region"].notna().all(),
        "every_target_has_analysis_region": targets["analysis_region"].notna().all(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    document = {
        "schema_version": "sg_nl_gate_a_v2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "country": "nl",
        "scope": "Liander service territory; not nationally representative",
        "checks": checks,
        "pdok": {
            "features": len(polygons),
            "unique_buurt_codes": len(polygon_codes),
            "cbs_buurt_rows": len(cbs_buurt_codes),
            "cbs_wijk_rows": len(cbs_wijk_codes),
            "pages": len(pagination.get("pages", [])),
            "pagination_total": int(pagination.get("total", 0)),
            "working_crs": "EPSG:4326 source; EPSG:28992 spatial operations",
        },
        "crosswalk": {
            "rows": len(crosswalk),
            "counts": counts,
            "conflict_policy": "unique_or_lexicographic_PiP_owns_formal_spatial_support; declared code is lineage only",
            "conflict_spatial_support_rows": int(conflict.sum()),
        },
        "levels": {
            "source": "wijk",
            "n_sources": len(sources),
            "target": "buurt_pseudo_station",
            "n_targets": len(targets),
            "lineage": "equipment",
            "n_equipment_rows": len(equipment),
            "bu_to_wk_rule": "official PDOK wijkcode verified equal to WK + BU digits[0:6]",
            "wijk_attribute_origins": {
                str(name): int(value)
                for name, value in sources["source_attribute_origin"].value_counts().items()
            },
        },
        "industry": {
            "cbs_rows": len(cbs),
            "cbs_rows_with_all_eight_sector_groups": int(cbs[list(CBS_SECTOR_COLUMNS)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).sum()),
            "admitted_sources_with_missing_sector_fields": int(sources["industry_fields_missing"].astype(bool).sum()),
            "fallback_rule": "population_only_zero_industry",
        },
        "robustness": {
            "aggregation_level": "buurt_pseudo_station",
            "zero_peak": int(targets["zero_peak"].astype(bool).sum()),
            "zero_capacity": int(targets["zero_capacity"].astype(bool).sum()),
            "over_capacity": target_over,
            "equipment_zero_peak_diagnostic": int(equipment.loc[eligible, "zero_peak"].astype(bool).sum()),
            "equipment_zero_capacity_diagnostic": int(equipment.loc[eligible, "zero_capacity"].astype(bool).sum()),
            "equipment_over_capacity_diagnostic": equipment_over,
            "within_buurt_overload_purification_count": equipment_over - target_over,
            "disclosure": "within-buurt aggregation can net equipment overload against spare capacity and make reinforcement_error optimistic",
            "main_rule": "retain; publish leave-out robustness views without clipping",
        },
        "temporal": {
            "protocol": context.merged["temporal"]["protocol"],
            "peak_date_min": str(pd.to_datetime(equipment["peak_date"]).min().date()),
            "peak_date_max": str(pd.to_datetime(equipment["peak_date"]).max().date()),
            "declaration": "equipment-specific peaks span 2024-09..2025-07 and are not a calendar-year coincident peak; CBS attributes are 2025",
        },
        "demand_conservation": {
            "raw_equipment_peak_sum_kw": raw_total,
            "admitted_equipment_peak_sum_kw": admitted_total,
            "outside_excluded_peak_sum_kw": excluded_total,
            "buurt_pseudo_station_peak_sum_kw": float(targets["peak_kw"].sum()),
            "source_region_peak_sum_kw": source_total,
        },
        "analysis_regions": {"count": int(inventory["n_regions"]), "inventory_sha256": sha256_file(paths.region_inventory)},
        "inputs": {
            "liander_sha256": sha256_file(paths.raw_liander),
            "cbs_sha256": sha256_file(paths.raw_cbs),
            "pdok_sha256": sha256_file(paths.raw_polygons),
            "pagination_audit_sha256": sha256_file(paths.pagination_audit) if paths.pagination_audit.is_file() else None,
        },
    }
    result = atomic_json(document, paths.gate_a)
    if document["status"] != "PASS":
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"NL Gate A failed: {failed}")
    return result


def _chunked_proximity(grid_xy: np.ndarray, target_xy: np.ndarray, *, gamma: float, clamp_m: float) -> tuple[bool, int]:
    peak = 0
    finite_positive = True
    for start in range(0, len(grid_xy), 2048):
        values = distance.cdist(grid_xy[start : start + 2048], target_xy)
        peak = max(peak, int(values.nbytes))
        values = np.maximum(values, clamp_m)
        scores = np.sum(np.power(values / 1000.0, -gamma), axis=1)
        finite_positive &= bool(np.isfinite(scores).all() and (scores > 0).all())
    return finite_positive, peak


def _split_failed_analysis_regions(
    context: CountryPipelineContext,
    region_ids: list[str],
    *,
    failure_evidence: list[dict] | None = None,
) -> None:
    """Advance only failing leaves one deterministic level and rewrite NL tables."""

    paths = _paths(context)
    source = gpd.read_file(paths.source_regions, layer="source_regions")
    targets = gpd.read_file(paths.stations, layer="buurt_pseudo_stations")
    equipment = gpd.read_file(paths.equipment_lineage, layer="equipment_lineage")
    inventory = json.loads(paths.region_inventory.read_text(encoding="utf-8"))
    records = {row["id"]: dict(row) for row in inventory["regions"]}
    work_source = source.to_crs(context.merged["crs"]["working"])
    work_equipment = equipment.to_crs(context.merged["crs"]["working"])
    for region in sorted(region_ids):
        try:
            parent = records.pop(region)
        except KeyError as exc:
            raise RuntimeError(f"unknown NL failing region {region}") from exc
        codes = set(source.loc[source["analysis_region"].eq(region), "wijk_code"].astype(str))
        member_mask = (
            equipment["generator_eligible"].astype(bool)
            & equipment["wijk_code"].astype(str).isin(codes)
        )
        member_indexes = np.flatnonzero(member_mask.to_numpy())
        xy = np.column_stack(
            [work_equipment.iloc[member_indexes].geometry.x, work_equipment.iloc[member_indexes].geometry.y]
        )
        ids = equipment.iloc[member_indexes]["equipment_id"].astype(str).to_numpy()
        left, right = _split_indices(xy, ids)
        centers = np.vstack([xy[left].mean(axis=0), xy[right].mean(axis=0)])
        source_indexes = np.flatnonzero(source["wijk_code"].astype(str).isin(codes).to_numpy())
        source_xy = np.column_stack(
            [work_source.iloc[source_indexes].geometry.centroid.x, work_source.iloc[source_indexes].geometry.centroid.y]
        )
        owner = np.argmin(distance.cdist(source_xy, centers), axis=1)
        source_to_child = {
            str(code): int(child)
            for code, child in zip(source.iloc[source_indexes]["wijk_code"], owner)
        }
        if set(source_to_child.values()) != {0, 1}:
            raise RuntimeError(f"{region}: deterministic split leaves an empty source child")
        base_path = "" if parent["leaf_path"] == "R" else str(parent["leaf_path"])
        prefix = f"NL_{_slug(str(parent['operational_stratum']))}_"
        for child in (0, 1):
            child_path = f"{base_path}{child}"
            child_id = prefix + child_path
            child_codes = sorted(code for code, value in source_to_child.items() if value == child)
            child_equipment = equipment["wijk_code"].astype(str).isin(child_codes) & equipment[
                "generator_eligible"
            ].astype(bool)
            records[child_id] = {
                "id": child_id,
                "operational_stratum": parent["operational_stratum"],
                "leaf_path": child_path,
                "centroid_x": float(centers[child, 0]),
                "centroid_y": float(centers[child, 1]),
                "n_sources": len(child_codes),
                "n_targets": int(targets["wijk_code"].astype(str).isin(child_codes).sum()),
                "n_lineage_equipment": int(child_equipment.sum()),
                "source_key_order": child_codes,
            }
            source.loc[source["wijk_code"].astype(str).isin(child_codes), "analysis_region"] = child_id
            targets.loc[targets["wijk_code"].astype(str).isin(child_codes), "analysis_region"] = child_id
            equipment.loc[equipment["wijk_code"].astype(str).isin(child_codes), "analysis_region"] = child_id
    ordered = [records[name] for name in sorted(records)]
    inventory["n_regions"] = len(ordered)
    inventory["regions"] = ordered
    inventory["status"] = "PROVISIONAL_ENGINEERING_REPARTITION"
    inventory.setdefault("repartition_history", []).append(
        {
            "split_parents": sorted(region_ids),
            "resulting_region_count": len(ordered),
            "reason": "shared_hard_or_static_occupancy_gate_failure",
            "inputs": "geometry_and_equipment_lineage_only",
            "parent_failure_evidence": list(failure_evidence or []),
        }
    )
    analysis = source[["analysis_region", "geometry"]].dissolve(by="analysis_region").reset_index()
    summary = pd.DataFrame(ordered).set_index("id")
    analysis["operational_stratum"] = analysis["analysis_region"].map(summary["operational_stratum"])
    analysis["n_sources"] = analysis["analysis_region"].map(summary["n_sources"]).astype(int)
    analysis["n_targets"] = analysis["analysis_region"].map(summary["n_targets"]).astype(int)
    atomic_geofile(source, paths.source_regions, layer="source_regions")
    atomic_geofile(targets, paths.stations, layer="buurt_pseudo_stations")
    atomic_geofile(equipment, paths.equipment_lineage, layer="equipment_lineage")
    atomic_geofile(analysis, paths.analysis_regions, layer="analysis_regions")
    atomic_json(inventory, paths.region_inventory)


def run_grid_skeleton(context: CountryPipelineContext, *, force: bool = False) -> Path:
    paths = _paths(context)
    if paths.engineering_admission.is_file() and not force:
        return paths.engineering_admission
    if force and context.grid_root.is_dir():
        # Scoped to NL derived artifacts; legacy evidence was moved to the
        # diagnostic tree before the v2 rebuild.
        shutil.rmtree(context.grid_root)
    write_gate_a(context, force=False)
    contract = _admission(context)
    grid_limits = contract.grid
    memory_limits = contract.memory
    sources = gpd.read_file(paths.source_regions, layer="source_regions")
    all_targets = gpd.read_file(paths.stations, layer="buurt_pseudo_stations")
    inventory = json.loads(paths.region_inventory.read_text(encoding="utf-8"))
    records_by_id = {item["id"]: item for item in inventory["regions"]}
    reports: list[dict] = []
    failures: list[str] = []
    for region in sorted(records_by_id):
        authority = records_by_id[region]
        source = sources.loc[sources["analysis_region"].eq(region)].copy()
        targets = all_targets.loc[all_targets["analysis_region"].eq(region)].copy()
        grid, design = regenerate_grid_reference(
            source,
            target_points=int(grid_limits["target_points"]),
            min_ground_step_m=float(grid_limits["min_ground_step_m"]),
            max_ground_step_m=float(grid_limits["max_ground_step_m"]),
            area_crs=str(context.merged["crs"]["area"]),
            generation_crs=str(context.merged["grid"]["generation_crs"]),
        )
        order = {code: index for index, code in enumerate(authority["source_key_order"])}
        grid["_order"] = grid["wijk_code"].astype(str).map(order)
        if grid["_order"].isna().any():
            raise RuntimeError(f"{region}: grid contains an unregistered source")
        grid = grid.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)
        grid["analysis_region"] = region
        grid_work = grid.to_crs(context.merged["crs"]["working"])
        target_work = targets.to_crs(context.merged["crs"]["working"])
        grid_xy = np.column_stack([grid_work.geometry.x, grid_work.geometry.y])
        target_xy = np.column_stack([target_work.geometry.x, target_work.geometry.y])
        nearest, proximity_peak, proximity_chunk_rows = chunked_nearest_assignment(
            grid_xy,
            target_xy,
            max_workspace_mib=float(memory_limits["max_cdist_workspace_mib"]),
        )
        nearest_distance = np.linalg.norm(grid_xy - target_xy[nearest], axis=1)
        tree = cKDTree(target_xy)
        two_distance, _ = tree.query(grid_xy, k=2)
        tie_count = int(np.isclose(two_distance[:, 0], two_distance[:, 1], rtol=0.0, atol=1e-8).sum())
        counts = np.bincount(nearest.astype(np.int64), minlength=len(targets))
        adjacency = build_grid_adjacency_indices(
            grid_xy,
            mode="neumann",
            step_size=float(design.projected_step_m),
        )
        agent_edges = int(adjacency.shape[1])
        source_agent_edges = 2 * len(grid)
        total_edges = agent_edges + source_agent_edges
        proximity_ok, proximity_score_peak = _chunked_proximity(
            grid_xy,
            target_xy,
            gamma=2.0,
            clamp_m=10.0,
        )
        source_counts = (
            grid["wijk_code"]
            .astype(str)
            .value_counts()
            .reindex(list(map(str, authority["source_key_order"])), fill_value=0)
        )
        metrics = {
            "n_sources": len(source),
            "n_targets": len(targets),
            "n_cells": len(grid),
            "min_cells_per_source": int(source_counts.min()),
            "min_cells_per_target": int(counts.min()),
            "total_edges_directed": total_edges,
            "active_agent_nodes_upper_bound": len(grid),
            "cdist_peak_workspace_bytes": proximity_peak,
        }
        shared = evaluate_region(contract, metrics, formal_pre_submission=False)
        checks = {
            **shared["hard_checks"],
            "canonical_target_occupancy": bool((counts > 0).all()),
            "canonical_nearest_tie_free": tie_count == 0,
            "proximity_finite_positive": proximity_ok,
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed:
            failures.append(f"{region}: {','.join(failed)}")
        write_grid_bundle(
            region,
            context.grid_root,
            grid,
            design,
            source_key="wijk_code",
            storage_columns=["index_region", "wijk_code", "analysis_region", "geometry"],
            metadata_extra={
                "analysis_region_authority": "data/datasets/2_derived/nl/authority/analysis_region_inventory.json",
                "engineering_admission_status": "ADMITTED" if not failed else "REPARTITION_REQUIRED",
                "engineering_authority_sha256": contract.authority_sha256,
            },
        )
        reports.append(
            {
                "region": region,
                "status": "PASS" if not failed else "FAIL",
                "checks": checks,
                "n_cells": len(grid),
                "n_sources": len(source),
                "n_targets": len(targets),
                "cells_per_source": len(grid) / len(source),
                "cells_per_target": len(grid) / len(targets),
                "min_cells_per_source": int(source_counts.min()),
                "min_cells_per_target": int(counts.min()),
                "canonical_empty_targets": int((counts == 0).sum()),
                "canonical_assignment_kernel": "Euclidean cKDTree; tie-free equivalence to shared canonical nearest VD asserted",
                "canonical_nearest_distance_max_m": float(nearest_distance.max()),
                "directed_agent_edges": agent_edges,
                "directed_source_agent_edges": source_agent_edges,
                "directed_graph_edges": total_edges,
                "oversampling_disclosure": shared["oversampling_disclosure"],
                "memory_advisory": shared["memory"],
                "estimated_full_cdist_bytes": len(grid) * len(targets) * np.dtype(np.float64).itemsize,
                "proximity_chunk_peak_bytes": proximity_peak,
                "proximity_score_chunk_peak_bytes": proximity_score_peak,
                "proximity_chunk_rows": proximity_chunk_rows,
                "working_crs": context.merged["crs"]["working"],
            }
        )
    failed_region_ids = [row["region"] for row in reports if row["status"] == "FAIL"]
    minimum_regions = int(inventory["minimum_region_count"])
    maximum_regions = int(inventory["maximum_region_count"])
    split_ids = list(failed_region_ids)
    if not split_ids and len(reports) < minimum_regions:
        largest = max(
            inventory["regions"],
            key=lambda row: (int(row["n_lineage_equipment"]), str(row["id"])),
        )
        split_ids = [str(largest["id"])]
    if split_ids and len(reports) + len(split_ids) <= maximum_regions:
        evidence = [
            {
                "region": row["region"],
                "n_cells": row["n_cells"],
                "n_sources": row["n_sources"],
                "n_targets": row["n_targets"],
                "min_cells_per_source": row["min_cells_per_source"],
                "min_cells_per_target": row["min_cells_per_target"],
                "canonical_empty_targets": row["canonical_empty_targets"],
                "failed_checks": sorted(name for name, passed in row["checks"].items() if not passed),
            }
            for row in reports
            if row["region"] in set(split_ids)
        ]
        _split_failed_analysis_regions(context, split_ids, failure_evidence=evidence)
        write_gate_a(context, force=True)
        return run_grid_skeleton(context, force=True)
    document = {
        "schema_version": "sg_nl_engineering_admission_report_v2",
        "status": "ADMITTED" if not failures else "REPARTITION_REQUIRED",
        "country": "nl",
        "contract": "casestudy/1_DataOverview/4_NL/admission.toml",
        "contract_sha256": contract.country_sha256,
        "shared_authority": contract.authority_path.relative_to(context.repo_root).as_posix(),
        "shared_authority_sha256": contract.authority_sha256,
        "feature_free": True,
        "forbidden_feature_inputs": ["OSM", "GHSL", "NTL", "model/evaluation outputs"],
        "regions": reports,
        "idr_matched": {
            "realizations": int(contract.country["idr_matched"]["formal_candidates"]) * int(contract.country["idr_matched"]["seeds"]) * len(reports),
            "maximum": int(contract.country["idr_matched"]["max_realizations_all_regions"]),
            "pass": int(contract.country["idr_matched"]["formal_candidates"]) * int(contract.country["idr_matched"]["seeds"]) * len(reports) <= int(contract.country["idr_matched"]["max_realizations_all_regions"]),
        },
        "failures": failures,
    }
    result = atomic_json(document, paths.engineering_admission)
    idr_pass = bool(document["idr_matched"]["pass"])
    region_range = int(inventory["minimum_region_count"]) <= len(reports) <= int(inventory["maximum_region_count"])
    if not idr_pass:
        document["failures"].append("global: idr_matched_realization_limit")
    if not region_range:
        document["failures"].append("global: analysis_region_count_range")
    if document["failures"]:
        document["status"] = "REPARTITION_REQUIRED"
        result = atomic_json(document, paths.engineering_admission)
    inventory["status"] = "FROZEN_PASS" if document["status"] == "ADMITTED" else "PROVISIONAL_REPARTITION_REQUIRED"
    inventory["engineering_admission_sha256"] = sha256_file(result)
    atomic_json(inventory, paths.region_inventory)
    # Gate A binds the final frozen inventory rather than the provisional tree.
    write_gate_a(context, force=True)
    return result


__all__ = [
    "CROSSWALK_CLASSES",
    "aggregate_buurt_targets",
    "classify_crosswalk",
    "derive_nl",
    "deterministic_analysis_regions",
    "merge_low_equipment_strata",
    "run_grid_skeleton",
    "write_gate_a",
]
