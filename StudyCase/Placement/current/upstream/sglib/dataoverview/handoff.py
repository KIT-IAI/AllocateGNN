"""Typed, fail-closed DataOverview handoff for orchestration code."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np

from sglib.core.infra.config import LoadedConfig, load_dataoverview_config
from sglib.core.infra.schema import validate_artifact
from sglib.core.infra.terms import CountryProfile

from .processing.features.grid_bundle import grid_paths, load_grid_bundle
from .evidence import HandoffEvidence, load_handoff_evidence
from .reader import canonical_paths, read_artifact


class DataOverviewHandoffError(ValueError):
    pass


@dataclass(frozen=True)
class RegionData:
    region: str
    grid: gpd.GeoDataFrame
    grid_metadata: Mapping[str, Any]
    landuse: Mapping[str, np.ndarray]
    built_surface: Mapping[str, np.ndarray]
    cuz_support: Mapping[str, np.ndarray]
    ntl: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class DataOverviewBundle:
    country: str
    profile: CountryProfile
    regions_table: gpd.GeoDataFrame
    stations_table: gpd.GeoDataFrame
    regions: tuple[RegionData, ...]
    inventory: Mapping[str, Any]
    evidence: Mapping[str, HandoffEvidence]
    configuration: LoadedConfig


def _load_config(repo_root: Path, country: str) -> LoadedConfig:
    profile_path = repo_root / "casestudy" / "config" / "countries" / f"{country}.toml"
    from sglib.core.infra.terms import load_country_profile

    profile = load_country_profile(profile_path)
    stage = repo_root / "casestudy" / "1_DataOverview"
    return load_dataoverview_config(
        stage / "general" / "general.toml",
        profile_path,
        stage / profile.directory / f"{country}.toml",
    )


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def load_bundle(repo_root: Path | str, country: str) -> DataOverviewBundle:
    root = Path(repo_root).resolve()
    config = _load_config(root, country)
    schemas = root / "casestudy" / "1_DataOverview" / "schemas"
    canonical = canonical_paths(root, country, configuration=config.values)
    validate_artifact(canonical["regions"], schemas / f"regions_{country}.toml")
    validate_artifact(canonical["stations"], schemas / f"stations_{country}.toml")
    regions_table = read_artifact(canonical["regions"])
    stations_table = read_artifact(canonical["stations"])
    directory = str(config.values["country"]["directory"])
    inventory_path = root / "results" / "1_DataOverview" / directory / "data_inventory.json"
    validate_artifact(inventory_path, schemas / "inventory.toml")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory["country"] != country:
        raise DataOverviewHandoffError("inventory country identity mismatch")
    evidence = load_handoff_evidence(
        root,
        config.values,
        schemas_root=schemas,
    )
    source_key = str(config.values["regions"]["source_key"])
    data: list[RegionData] = []
    expected_names = [str(item["id"]) for item in config.values["regions"]["items"]]
    for item in config.values["regions"]["items"]:
        name = str(item["id"])
        grid_path, metadata_path = grid_paths(name, root / "data" / "datasets" / "2_derived" / country / "grid_bplus")
        validate_artifact(metadata_path, schemas / "grid_bundle.toml")
        grid, _, metadata = load_grid_bundle(name, metadata_path.parents[2])
        paths = {
            "landuse": root / f"data/datasets/2_derived/{country}/features_bplus/extracted/{name}_landuse.npz",
            "built_surface": root / f"data/datasets/2_derived/{country}/features_bplus/extracted/{name}_ghsl_built_s.npz",
            "cuz_support": root / f"data/datasets/2_derived/{country}/features_bplus/extracted/{name}_cuz_support.npz",
            "ntl": root / f"data/datasets/2_derived/{country}/features_bplus/extracted/{name}_ntl.npz",
        }
        for artifact, schema in (
            (paths["landuse"], "landuse.toml"),
            (paths["built_surface"], "built_surface.toml"),
            (paths["cuz_support"], "cuz_support.toml"),
            (paths["ntl"], "ntl.toml"),
        ):
            validate_artifact(artifact, schemas / schema)
        arrays = {key: _npz(path) for key, path in paths.items()}
        n_cells = len(grid)
        if any(value["data"].shape[0] != n_cells for key, value in arrays.items() if key != "cuz_support"):
            raise DataOverviewHandoffError(f"{country}/{name}: feature/grid row mismatch")
        support = arrays["cuz_support"]
        if support["features"].shape[0] != n_cells:
            raise DataOverviewHandoffError(f"{country}/{name}: support/grid row mismatch")
        expected_source_keys = grid[source_key].astype(str).to_numpy()
        if not np.array_equal(support["source_keys"].astype(str), expected_source_keys):
            raise DataOverviewHandoffError(f"{country}/{name}: source row identity mismatch")
        masks = support["covered_mask"].astype(int) + support["unknown_mask"].astype(int) + support["zero_mask"].astype(int)
        if not np.all(masks == 1) or np.count_nonzero(support["features"][support["zero_mask"]]):
            raise DataOverviewHandoffError(f"{country}/{name}: C/U/Z partition invariant failed")
        if list(metadata["source_key_order"]) != list(map(str, item["source_key_order"])):
            raise DataOverviewHandoffError(f"{country}/{name}: configured source order mismatch")
        data.append(
            RegionData(
                region=name,
                grid=grid,
                grid_metadata=metadata,
                landuse=arrays["landuse"],
                built_surface=arrays["built_surface"],
                cuz_support=support,
                ntl=arrays["ntl"],
            )
        )
    if [item.region for item in data] != expected_names:
        raise DataOverviewHandoffError("region order mismatch")
    return DataOverviewBundle(
        country=country,
        profile=config.country_profile,
        regions_table=regions_table,
        stations_table=stations_table,
        regions=tuple(data),
        inventory=inventory,
        evidence=evidence,
        configuration=config,
    )


__all__ = [
    "DataOverviewBundle",
    "DataOverviewHandoffError",
    "HandoffEvidence",
    "RegionData",
    "load_bundle",
]
