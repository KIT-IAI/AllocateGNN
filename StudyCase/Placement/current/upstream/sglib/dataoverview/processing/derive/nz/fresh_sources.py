"""Auditable fresh-source acquisition and parsing for NZ core-9.

Only official files and ArcGIS records are landed, below
``data/datasets/1_raw/nz``; a manifest proves their URL, byte identity and
pagination completeness.  No research materialisation is read, copied or used
as a comparison target.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import tomllib
from typing import Any, Iterable, Mapping, Protocol
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.hashing import sha256_file, sha256_json


CORE_EDBS = ("Vector Lines", "Orion NZ", "Wellington Electricity")
FOCUS_EDBS = (*CORE_EDBS, "WEL Networks")
SECURE_CLASSES = ("N-1", "N-1 switched")
POP_FIELD = "VAR_1_3"
INDUSTRY_FIELDS = tuple(f"VAR_2_{number}" for number in range(591, 611))
WORKPLACE_INDUSTRY_FIELDS = tuple(
    f"VAR_2_{number}" for number in range(657, 677)
)
INDUSTRY_LABELS = {
    591: "agriculture_forestry_fishing",
    592: "mining",
    593: "manufacturing",
    594: "utilities_waste",
    595: "construction",
    596: "wholesale",
    597: "retail",
    598: "accommodation_food",
    599: "transport_postal_warehousing",
    600: "information_media_telecommunications",
    601: "finance_insurance",
    602: "rental_real_estate",
    603: "professional_scientific_technical",
    604: "administrative_support",
    605: "public_administration_safety",
    606: "education_training",
    607: "health_social_assistance",
    608: "arts_recreation",
    609: "other_services",
    610: "not_elsewhere_included",
}
TRUTH_DESCRIPTIONS = (
    "Current Peak Load (MVA)",
    "Installed Firm Capacity (MVA)",
    "Security of Supply Classification (type)",
)
EXPLICIT_ALIASES = {
    ("Vector Lines", "Balmain"): "Balmain Rd",
    ("Vector Lines", "Waikaukau"): "Waikaukau Rd",
    ("Vector Lines", "Woodford"): "Woodford Ave",
    ("Wellington Electricity", "Waikowhai Street"): "Waikowhai",
    ("WEL Networks", "HAMPTON DOWNS"): "HAPTON DOWNS",
}


class NZFreshSourceError(RuntimeError):
    """Raised when fresh-source provenance or completeness is not provable."""


class ResponseLike(Protocol):
    headers: Mapping[str, str]

    def raise_for_status(self) -> None: ...

    def json(self) -> dict[str, Any]: ...

    def iter_content(self, chunk_size: int) -> Iterable[bytes]: ...


class RequesterLike(Protocol):
    def get(self, url: str, **kwargs: Any) -> ResponseLike: ...


@dataclass(frozen=True)
class FreshSourcePaths:
    raw_root: Path
    d5: Path
    d6: Path
    sa2: Path
    sa3: Path
    census_population: Path
    census_industry: Path
    determination_2023: Path
    determination_2026: Path
    manifest: Path

    @classmethod
    def from_repo(cls, repo_root: Path | str) -> "FreshSourcePaths":
        root = Path(repo_root).resolve() / "data" / "datasets" / "1_raw" / "nz"
        return cls(
            raw_root=root,
            d5=root / "comcom" / "zone_substations_geospatial_2025.parquet",
            d6=root / "comcom" / "edb_id_full_2026.parquet",
            sa2=root / "statsnz" / "sa2_2023_generalised.geojson",
            sa3=root / "statsnz" / "sa3_2023.geojson",
            census_population=root / "statsnz" / "census_2023_sa2_population.json",
            census_industry=root / "statsnz" / "census_2023_sa2_industry.json",
            determination_2023=(
                root
                / "determinations"
                / "id_determination_consolidated_2023-07-06.pdf"
            ),
            determination_2026=(
                root
                / "determinations"
                / "id_determination_consolidated_2026-08-13.pdf"
            ),
            manifest=root / "fresh_sources_manifest.json",
        )


@dataclass(frozen=True)
class FreshCore9Frames:
    ledger: gpd.GeoDataFrame
    sites: gpd.GeoDataFrame
    sources: gpd.GeoDataFrame
    analysis_regions: gpd.GeoDataFrame
    dropped_suppressed: pd.DataFrame
    truth_audit: dict[str, Any]
    section_reconciliation: pd.DataFrame
    source_reconciliation: pd.DataFrame
    candidate_sa3_coverage: pd.DataFrame


def _overlay(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "casestudy" / "1_DataOverview" / "5_NZ" / "nz.toml"
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _configured_sources(repo_root: Path) -> dict[str, dict[str, Any]]:
    datasets = _overlay(repo_root)["datasets"]
    required = {
        "comcom_geospatial",
        "comcom_disclosure",
        "statsnz_sa2",
        "statsnz_sa3",
        "census_population",
        "census_industry",
        "determination_2023",
        "determination_2026",
    }
    missing = required - set(datasets)
    if missing:
        raise NZFreshSourceError(f"NZ overlay is missing sources: {sorted(missing)}")
    return {name: dict(datasets[name]) for name in sorted(required)}


def _url_and_query(configured_url: str) -> tuple[str, dict[str, str]]:
    parsed = urlsplit(configured_url)
    endpoint = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return endpoint, dict(parse_qsl(parsed.query, keep_blank_values=True))


def _existing_manifest(paths: FreshSourcePaths) -> dict[str, Any]:
    if not paths.manifest.is_file():
        return {}
    try:
        document = json.loads(paths.manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return document if document.get("schema_version") == "sg_nz_fresh_sources_v1" else {}


def _can_reuse(
    path: Path, url: str, previous: Mapping[str, Any], dataset: str
) -> bool:
    item = previous.get("sources", {}).get(dataset, {})
    return bool(
        path.is_file()
        and path.stat().st_size > 0
        and item.get("url") == url
        and item.get("sha256") == sha256_file(path)
        and item.get("fresh_official_download") is True
    )


def _stream_download(
    *,
    url: str,
    target: Path,
    requester: RequesterLike,
    reuse: bool,
    chunk_size: int = 1024 * 1024,
) -> dict[str, Any]:
    if reuse:
        return {
            "status": "reused_verified_landing",
            "url": url,
            "path": target.as_posix(),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "fresh_official_download": True,
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.part")
    partial.unlink(missing_ok=True)
    response = requester.get(
        url,
        stream=True,
        timeout=(30, 300),
        headers={"User-Agent": "SpatialGranularity-NZ-fresh/1.0"},
    )
    response.raise_for_status()
    size = 0
    try:
        with partial.open("wb") as stream:
            for block in response.iter_content(chunk_size=chunk_size):
                if not block:
                    continue
                stream.write(block)
                size += len(block)
        expected = response.headers.get("Content-Length")
        if expected is not None and int(expected) != size:
            raise NZFreshSourceError(
                f"download length mismatch for {url}: expected {expected}, got {size}"
            )
        if size == 0:
            raise NZFreshSourceError(f"download returned an empty file: {url}")
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {
        "status": "downloaded",
        "url": url,
        "path": target.as_posix(),
        "bytes": size,
        "sha256": sha256_file(target),
        "fresh_official_download": True,
    }


def _arcgis_ids(
    *,
    endpoint: str,
    base_params: Mapping[str, Any],
    envelopes: Iterable[tuple[float, float, float, float] | None],
    requester: RequesterLike,
) -> tuple[list[int], list[dict[str, Any]]]:
    identities: set[int] = set()
    probes: list[dict[str, Any]] = []
    for envelope in envelopes:
        params: dict[str, Any] = {
            "where": str(base_params.get("where", "1=1")),
            "returnIdsOnly": "true",
            "f": "json",
        }
        if envelope is not None:
            params.update(
                {
                    "geometry": ",".join(format(value, ".12g") for value in envelope),
                    "geometryType": "esriGeometryEnvelope",
                    "inSR": "4326",
                    "spatialRel": "esriSpatialRelIntersects",
                }
            )
        response = requester.get(
            endpoint,
            params=params,
            timeout=(30, 180),
            headers={"User-Agent": "SpatialGranularity-NZ-fresh/1.0"},
        )
        response.raise_for_status()
        document = response.json()
        if document.get("error"):
            raise NZFreshSourceError(f"ArcGIS id query failed: {document['error']}")
        ids = document.get("objectIds")
        if not isinstance(ids, list):
            raise NZFreshSourceError("ArcGIS id query did not return objectIds")
        integer_ids = sorted({int(value) for value in ids})
        identities.update(integer_ids)
        probes.append(
            {
                "envelope": list(envelope) if envelope is not None else None,
                "returned_ids": len(integer_ids),
                "identity_sha256": sha256_json(integer_ids),
            }
        )
    return sorted(identities), probes


def _arcgis_fetch(
    *,
    endpoint: str,
    base_params: Mapping[str, Any],
    object_ids: list[int],
    identity_field: str,
    requester: RequesterLike,
    geometry: bool,
    page_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: dict[int, dict[str, Any]] = {}
    pages: list[dict[str, Any]] = []
    for page_number, start in enumerate(range(0, len(object_ids), page_size), start=1):
        requested = object_ids[start : start + page_size]
        params: dict[str, Any] = {
            "objectIds": ",".join(map(str, requested)),
            "outFields": str(base_params.get("outFields", "*")),
            "returnGeometry": "true" if geometry else "false",
            "orderByFields": f"{identity_field} ASC",
            "outSR": str(base_params.get("outSR", "4326")),
            "geometryPrecision": str(base_params.get("geometryPrecision", "6")),
            "f": "geojson" if geometry else "json",
        }
        response = requester.get(
            endpoint,
            params=params,
            timeout=(30, 180),
            headers={"User-Agent": "SpatialGranularity-NZ-fresh/1.0"},
        )
        response.raise_for_status()
        document = response.json()
        if document.get("error"):
            raise NZFreshSourceError(f"ArcGIS page query failed: {document['error']}")
        features = document.get("features")
        if not isinstance(features, list):
            raise NZFreshSourceError("ArcGIS page has no features array")
        returned_ids: list[int] = []
        for feature in features:
            attributes = feature.get("properties") if geometry else feature.get("attributes")
            if not isinstance(attributes, dict) or identity_field not in attributes:
                raise NZFreshSourceError(
                    f"ArcGIS feature is missing identity field {identity_field}"
                )
            identity = int(attributes[identity_field])
            if identity in records:
                raise NZFreshSourceError(f"duplicate ArcGIS identity {identity}")
            records[identity] = feature
            returned_ids.append(identity)
        pages.append(
            {
                "page": page_number,
                "requested": len(requested),
                "returned": len(features),
                "requested_identity_sha256": sha256_json(requested),
                "returned_identity_sha256": sha256_json(sorted(returned_ids)),
            }
        )
    missing = sorted(set(object_ids) - set(records))
    unexpected = sorted(set(records) - set(object_ids))
    if missing or unexpected:
        raise NZFreshSourceError(
            f"ArcGIS pagination identity mismatch: missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )
    return [records[identity] for identity in object_ids], pages


def fetch_arcgis_complete(
    *,
    configured_url: str,
    target: Path,
    identity_field: str,
    requester: RequesterLike,
    geometry: bool,
    envelopes: Iterable[tuple[float, float, float, float] | None] = (None,),
    page_size: int = 1000,
) -> dict[str, Any]:
    """Land an ArcGIS query using object-id discovery and audited chunking."""

    endpoint, configured = _url_and_query(configured_url)
    object_ids, probes = _arcgis_ids(
        endpoint=endpoint,
        base_params=configured,
        envelopes=envelopes,
        requester=requester,
    )
    if not object_ids:
        raise NZFreshSourceError(f"ArcGIS query returned no identities: {endpoint}")
    features, pages = _arcgis_fetch(
        endpoint=endpoint,
        base_params=configured,
        object_ids=object_ids,
        identity_field=identity_field,
        requester=requester,
        geometry=geometry,
        page_size=page_size,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if geometry:
        payload: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    else:
        payload = {"features": features}
    atomic_json(payload, target)
    audit_path = target.with_suffix(target.suffix + ".pagination.json")
    audit = {
        "schema_version": "sg_arcgis_pagination_audit_v1",
        "complete": True,
        "endpoint": endpoint,
        "configured_query": configured,
        "identity_field": identity_field,
        "identity_sha256": sha256_json(object_ids),
        "identity_count": len(object_ids),
        "bbox_probes": probes,
        "pages": pages,
        "output_sha256": sha256_file(target),
    }
    atomic_json(audit, audit_path)
    return {
        "status": "downloaded",
        "url": configured_url,
        "path": target.as_posix(),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "fresh_official_download": True,
        "pagination_audit": audit_path.as_posix(),
        "pagination_audit_sha256": sha256_file(audit_path),
        "identity_count": len(object_ids),
        "identity_sha256": sha256_json(object_ids),
    }


def parse_d5(path: Path | str) -> gpd.GeoDataFrame:
    frame = pd.read_parquet(path)
    required = {"edb", "name", "geometry"}
    if missing := required - set(frame):
        raise NZFreshSourceError(f"D5 is missing columns: {sorted(missing)}")
    # D5 contains both zone-substation points and service-area polygons.  The
    # station crosswalk is explicitly point-granular, matching the published
    # D5 point slice used during research.
    frame = frame.loc[frame["geom_kind"].astype(str).str.upper().eq("POINT")].copy()
    frame = frame.loc[frame["name"].notna()].copy()
    geometry = shapely.from_wkb(frame.pop("geometry").to_numpy())
    result = gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:2193").to_crs(4326)
    if result.geometry.isna().any() or result.geometry.is_empty.any():
        raise NZFreshSourceError("D5 contains empty station geometry")
    result["longitude"] = result.geometry.x
    result["latitude"] = result.geometry.y
    result["source_row_identity"] = [
        f"{edb}|{name}|{longitude:.10f}|{latitude:.10f}"
        for edb, name, longitude, latitude in zip(
            result["edb"],
            result["name"],
            result["longitude"],
            result["latitude"],
            strict=True,
        )
    ]
    if result["source_row_identity"].duplicated().any():
        raise NZFreshSourceError("D5 EDB/name/coordinate identities are not unique")
    return result


def select_d6_truth_2024(path: Path | str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only the formal 2024 disclosure truth from the full D6 archive."""

    columns = [
        "edb",
        "network",
        "disc_yr",
        "section",
        "category",
        "sub_category",
        "description",
        "value",
        "text_input",
        "source",
        "version",
    ]
    try:
        frame = pd.read_parquet(
            path,
            columns=columns,
            filters=[[('disc_yr', '==', 2024.0), ('category', '==', 'Existing Zone Substation')]],
        )
    except (TypeError, ValueError):
        frame = pd.read_parquet(path, columns=columns)
    frame = frame.loc[
        frame["disc_yr"].eq(2024.0)
        & frame["category"].eq("Existing Zone Substation")
        & frame["description"].isin(TRUTH_DESCRIPTIONS)
    ].copy()
    if frame.empty or set(frame["description"]) != set(TRUTH_DESCRIPTIONS):
        raise NZFreshSourceError("D6 does not contain the three 2024 truth fields")
    identity_columns = ["edb", "network", "disc_yr", "sub_category", "description"]
    if frame.duplicated(identity_columns).any():
        raise NZFreshSourceError("D6 2024 truth has duplicate semantic identities")
    frame["measure_value"] = frame["value"].where(
        frame["value"].notna(), frame["text_input"]
    )
    wide = frame.pivot(
        index=["edb", "network", "disc_yr", "sub_category"],
        columns="description",
        values="measure_value",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={
            "Current Peak Load (MVA)": "actual_peak_mva",
            "Installed Firm Capacity (MVA)": "firm_capacity_mva",
            "Security of Supply Classification (type)": "security_class",
        }
    )
    wide["actual_peak_mva"] = pd.to_numeric(wide["actual_peak_mva"], errors="coerce")
    wide["firm_capacity_mva"] = pd.to_numeric(wide["firm_capacity_mva"], errors="coerce")
    ordered_identity = (
        frame[identity_columns]
        .astype(str)
        .sort_values(identity_columns, kind="stable")
        .to_dict(orient="records")
    )
    audit = {
        "schema_version": "sg_nz_d6_truth_selection_v1",
        "source_sha256": sha256_file(Path(path)),
        "disclosure_year": 2024,
        "category": "Existing Zone Substation",
        "descriptions": list(TRUTH_DESCRIPTIONS),
        "long_rows": len(frame),
        "wide_rows": len(wide),
        "semantic_identity_sha256": sha256_json(ordered_identity),
        "forecast_2026_excluded": bool(wide["disc_yr"].eq(2024).all()),
    }
    return wide, audit


def _normalise_name(value: object) -> str:
    text = str(value).upper().strip()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"#\s*\d+\b", " ", text)
    text = re.sub(r"\b\d+(?:\s*/\s*\d+)*\s*KV\b", " ", text)
    text = re.sub(r"\bZONE\s+SUBSTATION\b|\bSUBSTATION\b", " ", text)
    text = re.sub(r"\bROAD\b", "RD", text)
    text = re.sub(r"\bSTREET\b", "ST", text)
    text = re.sub(r"\bAVENUE\b", "AVE", text)
    return re.sub(r"[^A-Z0-9]+", "", text)


def _build_crosswalk(points: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    points = points.copy()
    stations = stations.copy()
    points["normalised_name"] = points["name"].map(_normalise_name)
    stations["normalised_name"] = stations["sub_category"].map(_normalise_name)
    output: list[dict[str, Any]] = []
    for station in stations.itertuples(index=False):
        station_values = station._asdict()
        edb = station_values["edb"]
        candidates = points.loc[points["edb"].eq(edb)].copy()
        exact = candidates.loc[
            candidates["normalised_name"].eq(station_values["normalised_name"])
        ]
        method = "unmatched"
        confidence = 0.0
        chosen: pd.Series | None = None
        if len(exact) == 1:
            chosen = exact.iloc[0]
            method = "exact_normalised"
            confidence = 1.0
        elif len(exact) > 1:
            method = "ambiguous_geometry"
        else:
            alias = EXPLICIT_ALIASES.get((edb, station_values["sub_category"]))
            if alias is not None:
                alias_rows = candidates.loc[candidates["name"].eq(alias)]
                if len(alias_rows) == 1:
                    chosen = alias_rows.iloc[0]
                    method = "explicit_alias"
                    confidence = 1.0
            if chosen is None and len(candidates):
                scored = sorted(
                    ((
                        SequenceMatcher(
                            None,
                            station_values["normalised_name"],
                            row["normalised_name"],
                        ).ratio(),
                        row,
                    ) for _, row in candidates.iterrows()),
                    key=lambda item: item[0],
                    reverse=True,
                )
                best_score, best = scored[0]
                second_score = scored[1][0] if len(scored) > 1 else 0.0
                if best_score >= 0.92 and best_score - second_score >= 0.08:
                    chosen = best
                    method = "high_confidence_fuzzy"
                    confidence = best_score
        result = {
            **station_values,
            "match_method": method,
            "match_confidence": confidence,
            "matched_d5_name": None,
            "longitude": np.nan,
            "latitude": np.nan,
        }
        if chosen is not None:
            result.update(
                {
                    "matched_d5_name": chosen["name"],
                    "longitude": float(chosen["longitude"]),
                    "latitude": float(chosen["latitude"]),
                }
            )
        output.append(result)
    return pd.DataFrame(output)


def _read_arcgis_attributes(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() == ".csv":
        result = pd.read_csv(path)
    else:
        document = json.loads(path.read_text(encoding="utf-8"))
        features = document.get("features")
        if not isinstance(features, list):
            raise NZFreshSourceError(f"ArcGIS attribute file has no features: {path}")
        result = pd.DataFrame([feature.get("attributes", {}) for feature in features])
    if "SA22023_V1_00" in result:
        result["SA22023_V1_00"] = result["SA22023_V1_00"].astype(str)
    return result


def _group_sum(frame: pd.DataFrame, fields: Iterable[str]) -> pd.Series:
    columns = list(fields)
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    values = values.mask(values < 0)
    return values.sum(axis=1, min_count=len(columns))


def _share_within_region(frame: pd.DataFrame, value: str) -> pd.Series:
    totals = frame.groupby("analysis_region")[value].transform("sum")
    result = frame[value] / totals.where(totals > 0)
    if result.isna().any():
        raise NZFreshSourceError(f"cannot form complete {value} shares")
    return result


def _balanced_spatial_groups(
    coordinates: np.ndarray, n_groups: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=float)
    if len(coordinates) < n_groups:
        raise NZFreshSourceError("an EDB has fewer sites than required spatial groups")
    if n_groups != 3:
        raise NZFreshSourceError("NZ core-9 requires exactly three groups per EDB")
    spread = np.ptp(coordinates, axis=0)
    axis = int(np.argmax(spread))
    other = 1 - axis
    ordered = np.lexsort((coordinates[:, other], coordinates[:, axis]))
    groups = [np.asarray(group) for group in np.array_split(ordered, 3)]
    centres = np.asarray([coordinates[group].mean(axis=0) for group in groups])
    order = np.lexsort((centres[:, 0], -centres[:, 1]))
    labels = np.empty(len(coordinates), dtype=int)
    ordered_centres: list[np.ndarray] = []
    for label, group_index in enumerate(order, start=1):
        labels[groups[group_index]] = label
        ordered_centres.append(centres[group_index])
    return labels, np.asarray(ordered_centres)


def _prepare_source_features(
    sa2: gpd.GeoDataFrame, population: pd.DataFrame, industry: pd.DataFrame
) -> gpd.GeoDataFrame:
    for frame in (sa2, population, industry):
        frame["SA22023_V1_00"] = frame["SA22023_V1_00"].astype(str)
    source = sa2.merge(
        population[["SA22023_V1_00", POP_FIELD]],
        on="SA22023_V1_00",
        how="left",
        validate="one_to_one",
    ).merge(
        industry[["SA22023_V1_00", *INDUSTRY_FIELDS, *WORKPLACE_INDUSTRY_FIELDS]],
        on="SA22023_V1_00",
        how="left",
        validate="one_to_one",
    )
    source = gpd.GeoDataFrame(source, geometry="geometry", crs=sa2.crs).rename(
        columns={POP_FIELD: "population_2023"}
    )
    source["population_2023"] = pd.to_numeric(source["population_2023"], errors="coerce")
    source["population_2023"] = source["population_2023"].mask(
        source["population_2023"] < 0
    )
    for number, label in INDUSTRY_LABELS.items():
        source[label] = pd.to_numeric(source[f"VAR_2_{number}"], errors="coerce").mask(
            lambda values: values < 0
        )
    source["agricultural_count"] = _group_sum(source, ["VAR_2_591"])
    source["industrial_count"] = _group_sum(
        source, [f"VAR_2_{number}" for number in range(592, 596)]
    )
    source["commercial_count"] = _group_sum(
        source, [f"VAR_2_{number}" for number in range(596, 605)]
    )
    source["others_count"] = _group_sum(
        source, [f"VAR_2_{number}" for number in range(605, 611)]
    )
    source["agricultural_count_workplace"] = _group_sum(source, ["VAR_2_657"])
    source["industrial_count_workplace"] = _group_sum(
        source, [f"VAR_2_{number}" for number in range(658, 662)]
    )
    source["commercial_count_workplace"] = _group_sum(
        source, [f"VAR_2_{number}" for number in range(662, 671)]
    )
    source["others_count_workplace"] = _group_sum(
        source, [f"VAR_2_{number}" for number in range(671, 677)]
    )
    for category in ("agricultural", "industrial", "commercial", "others"):
        residence = f"{category}_count"
        workplace = f"{category}_count_workplace"
        source[f"{category}_count_residence"] = source[residence]
        source[residence] = source[workplace].combine_first(source[residence])
    return source


SECTION_KEY = ("edb", "network", "sub_category")


def _section_reconciliation(d6: pd.DataFrame, focus: pd.DataFrame) -> pd.DataFrame:
    """Every 2024 truth row of the core EDBs with its decision and all failing rules.

    Completeness is shown row by row against the landed disclosure, not by
    comparing totals with the counts of an earlier materialisation.
    """

    raw = focus.loc[focus["edb"].isin(CORE_EDBS)].copy()
    rules = {
        "coordinate_not_matched": ~raw["coordinate_matched"],
        "peak_missing": raw["actual_peak_mva"].isna(),
        "peak_nonpositive": raw["actual_peak_mva"].le(0),
        "firm_capacity_missing": raw["firm_capacity_mva"].isna(),
        "firm_capacity_nonpositive": raw["firm_capacity_mva"].le(0),
        "security_class_not_declared_secure": ~raw["security_class"].isin(SECURE_CLASSES),
    }
    failed = pd.DataFrame(rules, index=raw.index)
    raw["exclusion_reasons"] = [
        ";".join(name for name, hit in row.items() if hit) for _, row in failed.iterrows()
    ]
    raw["status"] = np.where(failed.any(axis=1), "excluded", "included")
    if not raw["status"].eq("included").equals(raw["four_task_immediate"].astype(bool)):
        raise NZFreshSourceError("section reconciliation rules differ from the truth filter")
    if raw.duplicated(list(SECTION_KEY)).any():
        raise NZFreshSourceError("D6 2024 core sections have duplicate identities")
    landed = d6.loc[d6["edb"].isin(CORE_EDBS), list(SECTION_KEY)].astype(str)
    if len(landed) != len(raw) or set(map(tuple, landed.to_numpy())) != set(
        map(tuple, raw[list(SECTION_KEY)].astype(str).to_numpy())
    ):
        raise NZFreshSourceError("section reconciliation does not cover the landed D6 rows")
    columns = [
        *SECTION_KEY,
        "disc_yr",
        "match_method",
        "matched_d5_name",
        "actual_peak_mva",
        "firm_capacity_mva",
        "security_class",
        "status",
        "exclusion_reasons",
    ]
    return raw[columns].sort_values(list(SECTION_KEY), kind="stable").reset_index(drop=True)


def _source_reconciliation(
    sa2_to_sa3: gpd.GeoDataFrame,
    sa3: gpd.GeoDataFrame,
    station_sa3: Mapping[str, set[str]],
    dropped: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Account for every landed SA2 and prove candidate SA3s are tiled by landed SA2s."""

    if sa2_to_sa3["SA22023_V1_00"].duplicated().any():
        raise NZFreshSourceError("an SA2 lies within more than one SA3")
    owner: dict[str, str] = {}
    for edb, codes in station_sa3.items():
        for code in codes:
            if owner.setdefault(code, edb) != edb:
                raise NZFreshSourceError(f"SA3 {code} holds stations of more than one core EDB")
    frame = pd.DataFrame(sa2_to_sa3.drop(columns="geometry"))[
        ["SA22023_V1_00", "SA22023_V1_00_NAME", "SA32023_V1_00"]
    ].copy()
    frame["SA32023_V1_00"] = frame["SA32023_V1_00"].astype("string")
    frame["edb"] = frame["SA32023_V1_00"].map(owner)
    dropped_keys = set(zip(dropped["edb"], dropped["SA22023_V1_00"].astype(str)))
    reasons = []
    for row in frame.itertuples(index=False):
        if pd.isna(row.SA32023_V1_00):
            reasons.append("no_sa3_parent")
        elif pd.isna(row.edb):
            reasons.append("sa3_without_core_station")
        elif (row.edb, str(row.SA22023_V1_00)) in dropped_keys:
            reasons.append("suppressed_features_zero_demand")
        else:
            reasons.append("")
    frame["exclusion_reason"] = reasons
    frame["status"] = np.where(frame["exclusion_reason"].eq(""), "included", "excluded")

    # An SA2 outside every SA3 (generalised inlets and oceanic areas) is kept out
    # only while its dominant overlap lies outside the candidate SA3s.
    working_sa3 = sa3[["SA32023_V1_00", "geometry"]].to_crs(2193)
    working_sa3["geometry"] = shapely.make_valid(working_sa3.geometry.to_numpy())
    orphan_mask = sa2_to_sa3["SA32023_V1_00"].isna().to_numpy()
    orphans = sa2_to_sa3.loc[orphan_mask, ["SA22023_V1_00", "geometry"]].to_crs(2193)
    orphans["geometry"] = shapely.make_valid(orphans.geometry.to_numpy())
    frame["dominant_sa3"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["dominant_sa3_share"] = np.nan
    for index, row in orphans.iterrows():
        area = row.geometry.area
        overlaps = working_sa3.geometry.intersection(row.geometry).area
        if area > 0 and overlaps.max() > 0:
            best = int(np.argmax(overlaps.to_numpy()))
            frame.loc[index, "dominant_sa3"] = str(working_sa3.iloc[best]["SA32023_V1_00"])
            frame.loc[index, "dominant_sa3_share"] = float(overlaps.iloc[best] / area)
    frame["dominant_sa3_is_candidate"] = frame["dominant_sa3"].isin(set(owner))

    candidates = working_sa3.loc[working_sa3["SA32023_V1_00"].isin(set(owner))].copy()
    child_area = (
        sa2_to_sa3.to_crs(2193)
        .assign(child_area=lambda frame_: frame_.area)
        .groupby("SA32023_V1_00")["child_area"]
        .sum()
    )
    coverage = pd.DataFrame(
        {
            "edb": candidates["SA32023_V1_00"].map(owner).to_numpy(),
            "SA32023_V1_00": candidates["SA32023_V1_00"].to_numpy(),
            "sa3_area_m2": candidates.area.to_numpy(float),
            "sa2_child_area_m2": candidates["SA32023_V1_00"].map(child_area).fillna(0.0).to_numpy(float),
        }
    )
    coverage["coverage"] = coverage["sa2_child_area_m2"] / coverage["sa3_area_m2"]
    frame = frame.sort_values(["SA22023_V1_00"], kind="stable").reset_index(drop=True)
    coverage = coverage.sort_values(["edb", "SA32023_V1_00"], kind="stable").reset_index(drop=True)
    return frame, coverage


def derive_core9_frames(paths: FreshSourcePaths) -> FreshCore9Frames:
    """Derive the complete core-9 handoff from landed official sources."""

    d5 = parse_d5(paths.d5)
    d6, truth_audit = select_d6_truth_2024(paths.d6)
    crosswalk = _build_crosswalk(d5, d6)
    focus = crosswalk.loc[crosswalk["edb"].isin(FOCUS_EDBS)].copy()
    focus["coordinate_matched"] = focus[["longitude", "latitude"]].notna().all(axis=1)
    focus["task123_data_eligible"] = (
        focus["coordinate_matched"] & focus["actual_peak_mva"].gt(0)
    )
    focus["four_task_value_eligible"] = (
        focus["actual_peak_mva"].gt(0)
        & focus["firm_capacity_mva"].gt(0)
        & focus["security_class"].isin(SECURE_CLASSES)
    )
    focus["four_task_immediate"] = (
        focus["coordinate_matched"] & focus["four_task_value_eligible"]
    )
    section_reconciliation = _section_reconciliation(d6, focus)
    stations = focus.loc[focus["four_task_immediate"]].copy()
    stations = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["longitude"], stations["latitude"]),
        crs=4326,
    )

    sa2 = gpd.read_file(paths.sa2).to_crs(4326)
    sa3 = gpd.read_file(paths.sa3).to_crs(4326)
    for code in ("SA22023_V1_00",):
        sa2[code] = sa2[code].astype(str)
    sa3["SA32023_V1_00"] = sa3["SA32023_V1_00"].astype(str)
    population = _read_arcgis_attributes(paths.census_population)
    industry = _read_arcgis_attributes(paths.census_industry)
    source = _prepare_source_features(sa2, population, industry)
    stations = gpd.sjoin(
        stations,
        source[["SA22023_V1_00", "SA22023_V1_00_NAME", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])
    stations = gpd.sjoin(
        stations,
        sa3[["SA32023_V1_00", "SA32023_V1_00_NAME", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])
    core = stations.loc[stations["edb"].isin(CORE_EDBS)].copy()
    if core[["SA22023_V1_00", "SA32023_V1_00"]].isna().any().any():
        raise NZFreshSourceError("a core station is outside downloaded SA2/SA3 geography")

    sa2_to_sa3 = gpd.sjoin(
        source,
        sa3[["SA32023_V1_00", "SA32023_V1_00_NAME", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])
    projected = core.to_crs(2193)
    station_parts: list[gpd.GeoDataFrame] = []
    centre_rows: list[dict[str, Any]] = []
    for edb, group in projected.groupby("edb", sort=True):
        sites = group.drop_duplicates("matched_d5_name").copy()
        coordinates = np.column_stack((sites.geometry.x, sites.geometry.y))
        labels, centres = _balanced_spatial_groups(coordinates, n_groups=3)
        sites["cluster_number"] = labels
        site_map = sites.set_index("matched_d5_name")["cluster_number"].to_dict()
        assigned = group.copy()
        assigned["cluster_number"] = assigned["matched_d5_name"].map(site_map)
        assigned["analysis_region"] = (
            edb.replace(" ", "_")
            + "__historical_spatial_"
            + assigned["cluster_number"].astype(int).astype(str)
        )
        station_parts.append(assigned)
        for number, centre in enumerate(centres, start=1):
            centre_rows.append(
                {
                    "edb": edb,
                    "cluster_number": number,
                    "analysis_region": (
                        f"{edb.replace(' ', '_')}__historical_spatial_{number}"
                    ),
                    "centre_x": float(centre[0]),
                    "centre_y": float(centre[1]),
                }
            )
    ledger = gpd.GeoDataFrame(
        pd.concat(station_parts, ignore_index=True), geometry="geometry", crs=2193
    ).to_crs(4326)
    centres = pd.DataFrame(centre_rows)

    source_parts: list[gpd.GeoDataFrame] = []
    for edb, station_group in ledger.groupby("edb", sort=True):
        edb_sa3 = set(station_group["SA32023_V1_00"].dropna().astype(str))
        candidates = sa2_to_sa3.loc[
            sa2_to_sa3["SA32023_V1_00"].astype(str).isin(edb_sa3)
        ].copy()
        representative = candidates.to_crs(2193).representative_point()
        edb_centres = centres.loc[centres["edb"].eq(edb)].copy()
        centre_xy = edb_centres[["centre_x", "centre_y"]].to_numpy(float)
        point_xy = np.column_stack((representative.x, representative.y))
        nearest = np.argmin(
            ((point_xy[:, None, :] - centre_xy[None, :, :]) ** 2).sum(axis=2),
            axis=1,
        )
        candidates["edb"] = edb
        candidates["analysis_region"] = edb_centres.iloc[nearest][
            "analysis_region"
        ].to_numpy()
        source_parts.append(candidates)
    sources = gpd.GeoDataFrame(
        pd.concat(source_parts, ignore_index=True), geometry="geometry", crs=4326
    )
    sources["source_id"] = sources["edb"] + ":" + sources["SA22023_V1_00"].astype(str)
    demand = ledger.groupby(["edb", "SA22023_V1_00"])["actual_peak_mva"].sum()
    sources["demand_peak_mva"] = [
        demand.get((row.edb, row.SA22023_V1_00), 0.0)
        for row in sources.itertuples(index=False)
    ]
    feature_columns = [
        "agricultural_count",
        "industrial_count",
        "commercial_count",
        "others_count",
    ]
    suppressed = sources[feature_columns].isna().any(axis=1)
    drop = suppressed & sources["demand_peak_mva"].eq(0)
    dropped = sources.loc[
        drop,
        [
            "edb",
            "analysis_region",
            "SA22023_V1_00",
            "SA22023_V1_00_NAME",
            "population_2023",
            "demand_peak_mva",
        ],
    ].copy()
    sources = sources.loc[~drop].copy().reset_index(drop=True)
    source_reconciliation, candidate_sa3_coverage = _source_reconciliation(
        sa2_to_sa3,
        sa3,
        {
            str(edb): set(group["SA32023_V1_00"].dropna().astype(str))
            for edb, group in ledger.groupby("edb", sort=True)
        },
        dropped,
    )
    sources["residential_percent"] = _share_within_region(sources, "population_2023")
    for category in ("agricultural", "industrial", "commercial", "others"):
        sources[f"{category}_percent"] = _share_within_region(
            sources, f"{category}_count"
        )
    analysis = sources.dissolve(by=["edb", "analysis_region"], as_index=False)[
        ["edb", "analysis_region", "geometry"]
    ]

    site_group = [
        "edb",
        "matched_d5_name",
        "analysis_region",
        "SA22023_V1_00",
        "SA22023_V1_00_NAME",
        "SA32023_V1_00",
        "SA32023_V1_00_NAME",
        "longitude",
        "latitude",
    ]
    sites = (
        ledger.drop(columns="geometry")
        .groupby(site_group, dropna=False, sort=True)
        .agg(
            section_names=(
                "sub_category", lambda values: " | ".join(sorted(map(str, values)))
            ),
            sections_at_site=("sub_category", "size"),
            security_classes=(
                "security_class", lambda values: " | ".join(sorted(set(map(str, values))))
            ),
            actual_peak_mva=("actual_peak_mva", "sum"),
            firm_capacity_mva=("firm_capacity_mva", "sum"),
        )
        .reset_index()
    )
    sites["noncoincident_conservative"] = sites["sections_at_site"].gt(1)
    sites = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites["longitude"], sites["latitude"]),
        crs=4326,
    )
    return FreshCore9Frames(
        ledger=ledger,
        sites=sites,
        sources=sources,
        analysis_regions=analysis,
        dropped_suppressed=dropped,
        truth_audit=truth_audit,
        section_reconciliation=section_reconciliation,
        source_reconciliation=source_reconciliation,
        candidate_sa3_coverage=candidate_sa3_coverage,
    )


def verify_fresh_manifest(paths: FreshSourcePaths) -> dict[str, Any]:
    if not paths.manifest.is_file():
        raise NZFreshSourceError(
            f"fresh NZ source manifest is missing: {paths.manifest}; run acquire first"
        )
    document = json.loads(paths.manifest.read_text(encoding="utf-8"))
    if document.get("schema_version") != "sg_nz_fresh_sources_v1":
        raise NZFreshSourceError("unexpected NZ fresh-source manifest schema")
    if document.get("complete") is not True or document.get("anchor_used") is not False:
        raise NZFreshSourceError("NZ fresh-source manifest is not complete and anchor-free")
    expected = {
        "comcom_geospatial": paths.d5,
        "comcom_disclosure": paths.d6,
        "statsnz_sa2": paths.sa2,
        "statsnz_sa3": paths.sa3,
        "census_population": paths.census_population,
        "census_industry": paths.census_industry,
        "determination_2023": paths.determination_2023,
        "determination_2026": paths.determination_2026,
    }
    sources = document.get("sources", {})
    for name, path in expected.items():
        item = sources.get(name, {})
        if (
            item.get("fresh_official_download") is not True
            or not path.is_file()
            or item.get("sha256") != sha256_file(path)
        ):
            raise NZFreshSourceError(f"fresh source identity failed for {name}")
        if name.startswith("statsnz_") or name.startswith("census_"):
            audit_path = Path(str(item.get("pagination_audit", "")))
            if not audit_path.is_absolute():
                audit_path = paths.raw_root / audit_path
            if not audit_path.is_file():
                raise NZFreshSourceError(f"pagination audit is missing for {name}")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit.get("complete") is not True:
                raise NZFreshSourceError(f"pagination is incomplete for {name}")
    truth = document.get("d6_truth_2024", {})
    if truth.get("forecast_2026_excluded") is not True or truth.get("disclosure_year") != 2024:
        raise NZFreshSourceError("D6 truth audit does not prove the 2024 selection")
    return document


def acquire_nz_fresh_sources(
    repo_root: Path | str,
    *,
    refresh: bool = False,
    requester: RequesterLike | None = None,
    acquired_at_utc: str | None = None,
) -> FreshSourcePaths:
    """Download every formal NZ source and write one atomic landing manifest."""

    repository = Path(repo_root).resolve()
    paths = FreshSourcePaths.from_repo(repository)
    paths.raw_root.mkdir(parents=True, exist_ok=True)
    request_client = requester or requests.Session()
    configured = _configured_sources(repository)
    previous = _existing_manifest(paths) if not refresh else {}
    source_records: dict[str, dict[str, Any]] = {}
    static_targets = {
        "comcom_geospatial": paths.d5,
        "comcom_disclosure": paths.d6,
        "determination_2023": paths.determination_2023,
        "determination_2026": paths.determination_2026,
    }
    for dataset, target in static_targets.items():
        file_spec = configured[dataset]["files"][0]
        url = str(file_spec["url"])
        source_records[dataset] = _stream_download(
            url=url,
            target=target,
            requester=request_client,
            reuse=(not refresh and _can_reuse(target, url, previous, dataset)),
        )

    d5 = parse_d5(paths.d5)
    envelopes: list[tuple[float, float, float, float]] = []
    for _, group in d5.loc[d5["edb"].isin(FOCUS_EDBS)].groupby("edb", sort=True):
        envelopes.append(
            (
                float(group["longitude"].min()) - 0.20,
                float(group["latitude"].min()) - 0.20,
                float(group["longitude"].max()) + 0.20,
                float(group["latitude"].max()) + 0.20,
            )
        )
    if len(envelopes) != len(FOCUS_EDBS):
        raise NZFreshSourceError("D5 does not contain all four acquisition EDB extents")

    arcgis_targets = {
        "statsnz_sa2": (paths.sa2, "OBJECTID", True, envelopes),
        "statsnz_sa3": (paths.sa3, "OBJECTID", True, envelopes),
        "census_population": (paths.census_population, "OBJECTID", False, [None]),
        "census_industry": (paths.census_industry, "OBJECTID", False, [None]),
    }
    for dataset, (target, identity, geometry, query_envelopes) in arcgis_targets.items():
        query = configured[dataset]["query"]
        url = str(query["url"])
        if not refresh and _can_reuse(target, url, previous, dataset):
            item = dict(previous["sources"][dataset])
            audit_value = str(item.get("pagination_audit", ""))
            audit_path = Path(audit_value)
            if not audit_path.is_absolute():
                audit_path = paths.raw_root / audit_path
            if audit_path.is_file():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if audit.get("complete") is True:
                    source_records[dataset] = item
                    source_records[dataset]["status"] = "reused_verified_landing"
                    continue
        source_records[dataset] = fetch_arcgis_complete(
            configured_url=url,
            target=target,
            identity_field=str(query.get("identity_field", identity)),
            requester=request_client,
            geometry=geometry,
            envelopes=query_envelopes,
            page_size=int(query.get("page_size", 1000)),
        )

    truth, truth_audit = select_d6_truth_2024(paths.d6)
    if not set(CORE_EDBS) <= set(truth["edb"].astype(str)):
        raise NZFreshSourceError("D6 2024 truth is missing a core EDB")
    # Store paths relative to the raw root so the manifest remains relocatable.
    for item in source_records.values():
        recorded_path = Path(str(item["path"]))
        if not recorded_path.is_absolute():
            recorded_path = paths.raw_root / recorded_path
        item["path"] = recorded_path.resolve().relative_to(paths.raw_root).as_posix()
        if audit_value := item.get("pagination_audit"):
            audit_path = Path(str(audit_value))
            if not audit_path.is_absolute():
                audit_path = paths.raw_root / audit_path
            item["pagination_audit"] = (
                audit_path.resolve().relative_to(paths.raw_root).as_posix()
            )
    manifest = {
        "schema_version": "sg_nz_fresh_sources_v1",
        "complete": True,
        "country": "nz",
        "scope": "core9",
        "anchor_used": False,
        "acquired_at_utc": acquired_at_utc
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": source_records,
        "d5_identity_sha256": sha256_json(
            sorted(d5["source_row_identity"].astype(str).tolist())
        ),
        "d6_truth_2024": truth_audit,
    }
    atomic_json(manifest, paths.manifest)
    verify_fresh_manifest(paths)
    return paths


__all__ = [
    "CORE_EDBS",
    "FOCUS_EDBS",
    "FreshCore9Frames",
    "FreshSourcePaths",
    "NZFreshSourceError",
    "acquire_nz_fresh_sources",
    "derive_core9_frames",
    "fetch_arcgis_complete",
    "parse_d5",
    "select_d6_truth_2024",
    "verify_fresh_manifest",
]
