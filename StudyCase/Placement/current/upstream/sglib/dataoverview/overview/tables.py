"""Summary tables for the numbered DataOverview display notebooks.

Every function reads already-landed processing artifacts and returns a
``pandas.DataFrame``; nothing here downloads, derives, or writes datasets.
The tables follow the ``[display].required_sections`` contract of
``general.toml``: source_and_license, schema, coverage_stats, unit_and_vintage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from sglib.core.infra.config import LoadedConfig

from ..processing.config import CountryPipelineContext
from ..processing.features.cuz_support import load_cuz_support
from ..processing.features.grid_bundle import load_grid_bundle


# --------------------------------------------------------------------------
# source_and_license
# --------------------------------------------------------------------------


def dataset_ledger(
    configuration: Mapping[str, Any],
    *,
    categories: Iterable[str] | None = None,
) -> pd.DataFrame:
    """One row per ``[datasets.*]`` entry of the merged country configuration."""

    wanted = set(categories) if categories is not None else None
    rows: list[dict[str, Any]] = []
    for name, spec in configuration["datasets"].items():
        category = str(spec["category"])
        if wanted is not None and category not in wanted:
            continue
        files = spec.get("files", [])
        protocol = spec.get("query_protocol")
        acquisition = f"query:{protocol}" if protocol else f"static:{len(files)} file(s)"
        rows.append(
            {
                "dataset": name,
                "category": category,
                "vintage": spec.get("vintage", ""),
                "license": spec.get("license", ""),
                "acquisition": acquisition,
                "semantics": spec.get("semantics", ""),
            }
        )
    return pd.DataFrame(rows, columns=["dataset", "category", "vintage", "license", "acquisition", "semantics"])


def product_ledger(configuration: Mapping[str, Any], *, products: Iterable[str] | None = None) -> pd.DataFrame:
    """One row per ``[products.*]`` entry, optionally restricted to ``products``."""

    wanted = set(products) if products is not None else None
    rows = []
    for name, spec in configuration["products"].items():
        if wanted is not None and name not in wanted:
            continue
        rows.append(
            {
                "product": name,
                "category": spec.get("category", ""),
                "depends_on": ", ".join(map(str, spec.get("depends_on", []))),
                "produces": ", ".join(map(str, spec.get("produces", []))),
                "semantics": spec.get("semantics", ""),
            }
        )
    return pd.DataFrame(rows, columns=["product", "category", "depends_on", "produces", "semantics"])


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


def schema_table(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Column name, dtype, and non-null count for every named frame."""

    rows = []
    for table, frame in frames.items():
        for column in frame.columns:
            rows.append(
                {
                    "table": table,
                    "column": column,
                    "dtype": str(frame[column].dtype),
                    "non_null": int(frame[column].notna().sum()),
                    "rows": int(len(frame)),
                }
            )
    return pd.DataFrame(rows, columns=["table", "column", "dtype", "non_null", "rows"])


def npz_schema(paths: Mapping[str, Path]) -> pd.DataFrame:
    """Array name, shape, and dtype for every key in the named ``.npz`` files."""

    rows = []
    for label, path in paths.items():
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                value = archive[key]
                rows.append(
                    {
                        "artifact": label,
                        "key": key,
                        "shape": "x".join(map(str, value.shape)) or "scalar",
                        "dtype": str(value.dtype),
                    }
                )
    return pd.DataFrame(rows, columns=["artifact", "key", "shape", "dtype"])


# --------------------------------------------------------------------------
# unit_and_vintage
# --------------------------------------------------------------------------


def unit_and_vintage(configuration: LoadedConfig | Mapping[str, Any]) -> pd.DataFrame:
    """Flatten the country profile facts that fix units, CRS, and vintages."""

    values = configuration.values if isinstance(configuration, LoadedConfig) else configuration
    rows = []
    for section in ("units", "temporal", "crs"):
        for key, value in values[section].items():
            rows.append({"section": section, "key": key, "value": value})
    contract = values["station_contract"]
    for key in ("capacity_basis", "demand_column", "capacity_column", "region_demand_column"):
        rows.append({"section": "station_contract", "key": key, "value": contract.get(key, "")})
    rows.append({"section": "country", "key": "evaluation_scope", "value": values["country"]["evaluation_scope"]})
    return pd.DataFrame(rows, columns=["section", "key", "value"])


# --------------------------------------------------------------------------
# coverage_stats
# --------------------------------------------------------------------------


def _metric_rows(items: Iterable[tuple[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(items), columns=["metric", "value"])


def regions_summary(regions: pd.DataFrame, station_contract: Mapping[str, Any]) -> pd.DataFrame:
    """Row count, source-key count, and column totals of the canonical regions table."""

    source_key = str(station_contract["source_key"])
    demand = str(station_contract.get("region_demand_column", "")).strip()
    items: list[tuple[str, Any]] = [
        ("region rows", int(len(regions))),
        (f"distinct {source_key}", int(regions[source_key].nunique())),
    ]
    for column in ("population", "area"):
        if column in regions:
            items.append((f"sum {column}", float(pd.to_numeric(regions[column], errors="coerce").sum())))
    if demand and demand in regions:
        items.append((f"sum {demand}", float(pd.to_numeric(regions[demand], errors="coerce").sum())))
    return _metric_rows(items)


def stations_summary(stations: pd.DataFrame, station_contract: Mapping[str, Any]) -> pd.DataFrame:
    """Row count, region assignment, and demand/capacity totals of the stations table."""

    demand = str(station_contract["demand_column"])
    capacity = str(station_contract.get("capacity_column", "")).strip()
    region = str(station_contract["region_column"])
    assigned = stations[region].notna() if region in stations else pd.Series(True, index=stations.index)
    items: list[tuple[str, Any]] = [
        ("station rows", int(len(stations))),
        (f"assigned to a {region}", int(assigned.sum())),
        (f"unassigned ({region} missing)", int((~assigned).sum())),
        (f"sum {demand} (all)", float(pd.to_numeric(stations[demand], errors="coerce").sum())),
        (f"sum {demand} (assigned)", float(pd.to_numeric(stations.loc[assigned, demand], errors="coerce").sum())),
    ]
    if capacity and capacity in stations:
        items.append((f"sum {capacity} (assigned)", float(pd.to_numeric(stations.loc[assigned, capacity], errors="coerce").sum())))
    return _metric_rows(items)


def group_summary(
    regions: pd.DataFrame,
    stations: pd.DataFrame,
    *,
    group_key: str,
    groups: Iterable[str],
    station_contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Per analysis-region counts of source rows, stations, and demand."""

    demand = str(station_contract["demand_column"])
    source_key = str(station_contract["source_key"])
    rows = []
    for group in groups:
        region_rows = regions.loc[regions[group_key].astype(str) == group]
        station_rows = stations.loc[stations[group_key].astype(str) == group] if group_key in stations else stations.iloc[0:0]
        rows.append(
            {
                "region": group,
                f"n_{source_key}": int(region_rows[source_key].nunique()),
                "n_stations": int(len(station_rows)),
                f"sum {demand}": float(pd.to_numeric(station_rows[demand], errors="coerce").sum()),
            }
        )
    return pd.DataFrame(rows)


def grid_profile(context: CountryPipelineContext) -> pd.DataFrame:
    """One row per analysis region from the landed B+ grid bundle metadata."""

    rows = []
    for item in context.region_items:
        name = str(item["id"])
        grid, step, metadata = load_grid_bundle(name, context.grid_root)
        spacing = metadata.get("ground_neighbour_distance_m", {})
        rows.append(
            {
                "region": name,
                "n_cells": int(len(grid)),
                "n_sources": int(metadata["n_sources"]),
                "cells_per_source": float(metadata["cells_per_source"]),
                "area_km2": float(metadata["area_m2"]) / 1e6,
                "projected_step_m": float(step),
                "ground_step_median_m": float(spacing.get("median", float("nan"))),
                "clamp_branch": metadata.get("clamp_branch", ""),
                "reference_latitude": float(metadata.get("reference_latitude", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def feature_coverage(context: CountryPipelineContext) -> pd.DataFrame:
    """Per region C/U/Z partition counts plus mean land-use, Built-S, and NTL values."""

    rows = []
    for item in context.region_items:
        name = str(item["id"])
        support = load_cuz_support(context.artifact_root / f"{name}_cuz_support.npz")
        with np.load(context.artifact_root / f"{name}_landuse.npz", allow_pickle=False) as landuse:
            lu_columns = [str(column) for column in landuse["columns"]]
            lu_mean = landuse["data"].mean(axis=0)
        with np.load(context.artifact_root / f"{name}_ghsl_built_s.npz", allow_pickle=False) as built:
            built_mean = float(built["data"][:, 0].mean())
        with np.load(context.artifact_root / f"{name}_ntl.npz", allow_pickle=False) as ntl:
            ntl_values = ntl["data"][:, 0]
        n_cells = int(len(support.features))
        row: dict[str, Any] = {
            "region": name,
            "n_cells": n_cells,
            "covered": int(support.covered_mask.sum()),
            "unknown": int(support.unknown_mask.sum()),
            "zero": int(support.zero_mask.sum()),
            "zero_nonzero_values": int(np.count_nonzero(support.features[support.zero_mask])),
        }
        for column, value in zip(lu_columns, lu_mean):
            row[f"mean_{column}"] = float(value)
        row["mean_ghsl_built_fraction"] = built_mean
        row["ntl_median"] = float(np.median(ntl_values))
        row["ntl_max"] = float(ntl_values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def inventory_regions(inventory: Mapping[str, Any]) -> pd.DataFrame:
    """Flatten ``inventory['regions']`` without the nested grid metadata."""

    rows = []
    for record in inventory["regions"]:
        metadata = record.get("grid_metadata", {})
        rows.append(
            {
                "region": record["region"],
                "n_cells": record["n_cells"],
                "n_sources": record.get("n_sources"),
                "covered": record.get("covered"),
                "unknown": record.get("unknown"),
                "zero": record.get("zero"),
                "projected_step_m": metadata.get("projected_step_m"),
                "grid_schema": metadata.get("schema_version"),
            }
        )
    return pd.DataFrame(rows)


def inventory_artifacts(inventory: Mapping[str, Any]) -> pd.DataFrame:
    """Path, size, and hash of every artifact recorded in the inventory."""

    return pd.DataFrame(inventory["artifacts"], columns=["path", "bytes", "sha256"])


def inventory_evidence(inventory: Mapping[str, Any]) -> pd.DataFrame:
    """Handoff evidence declared in the country overlay, with required and observed status."""

    rows = []
    for name, item in inventory.get("handoff_artifacts", {}).items():
        rows.append({"evidence": name, **item})
    columns = ["evidence", "path", "schema", "formal_required", "required_status", "observed_status", "bytes", "sha256"]
    return pd.DataFrame(rows, columns=columns)


__all__ = [
    "dataset_ledger",
    "feature_coverage",
    "grid_profile",
    "group_summary",
    "inventory_artifacts",
    "inventory_evidence",
    "inventory_regions",
    "npz_schema",
    "product_ledger",
    "regions_summary",
    "schema_table",
    "stations_summary",
    "unit_and_vintage",
]
