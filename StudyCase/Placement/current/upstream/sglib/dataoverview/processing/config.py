"""Resolved, single-country runtime context derived from strict TOML layers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from sglib.core.infra.config import LoadedConfig
from sglib.core.infra.paths import scoped_path


class PipelineConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CountryPipelineContext:
    repo_root: Path
    merged: Mapping[str, Any]

    @property
    def country_code(self) -> str:
        return str(self.merged["country"]["code"])

    def path(self, value: str | Path, *, label: str = "configured path") -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                candidate.resolve().relative_to(self.repo_root)
            except ValueError as exc:
                raise PipelineConfigError(f"{label} escapes repo_root: {candidate}") from exc
            return candidate.resolve()
        return scoped_path(self.repo_root, candidate)

    @property
    def raw_root(self) -> Path:
        return self.repo_root / "data" / "datasets" / "1_raw" / self.country_code

    @property
    def derived_root(self) -> Path:
        return self.repo_root / "data" / "datasets" / "2_derived" / self.country_code

    @property
    def grid_root(self) -> Path:
        return self.derived_root / "grid_bplus"

    @property
    def cache_root(self) -> Path:
        return self.derived_root / "features" / "cache"

    @property
    def artifact_root(self) -> Path:
        return self.derived_root / "features_bplus" / "extracted"

    @property
    def region_items(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.merged["regions"]["items"]]

    def runtime_feature_config(self) -> dict[str, Any]:
        features = self.merged["features"]
        sectors = {
            name: list(spec["osm_values"])
            for name, spec in features["landuse"]["sectors"].items()
        }
        config = {
            "grid": deepcopy(dict(self.merged["grid"])),
            "fetchers": {
                "osm": {
                    "enabled": bool(features["osm"]["enabled"]),
                    "tags": {"landuse": True},
                    "buffer_m": float(features["osm"]["buffer_m"]),
                },
                "ghsl_built_s": deepcopy(dict(features["ghsl_built_s"])),
                "ntl": {
                    "enabled": bool(features["ntl"]["enabled"]),
                    "mode": str(features["ntl"]["mode"]),
                },
            },
            "extractors": {
                "landuse": {
                    "enabled": True,
                    "fetcher": "osm",
                    "mode": str(features["landuse"]["mode"]),
                    "target_grid_crs": str(self.merged["grid"]["generation_crs"]),
                    "categories": sectors,
                    "default_category": str(features["landuse"]["default_category"]),
                },
                "ghsl_built_s": {
                    "enabled": True,
                    "fetcher": "ghsl_built_s",
                    "source_pixel_area_m2": float(features["ghsl_built_s"]["source_pixel_area_m2"]),
                    "target_grid_crs": str(self.merged["grid"]["generation_crs"]),
                },
                "ntl": {
                    "enabled": True,
                    "fetcher": "ntl",
                    "radius_m": float(features["ntl"]["radius_m"]),
                },
            },
        }
        ntl_query = self.merged["datasets"]["ntl"]["query"]
        config["fetchers"]["ntl"]["path"] = str((self.raw_root / str(ntl_query["output"])).resolve())
        config["fetchers"]["ntl"]["output_crs"] = self.merged["crs"]["native"]
        return config


def build_context(repo_root: Path | str, loaded: LoadedConfig | Mapping[str, Any]) -> CountryPipelineContext:
    root = Path(repo_root).resolve()
    document = loaded.values if isinstance(loaded, LoadedConfig) else loaded
    required = {
        "country",
        "crs",
        "grid",
        "features",
        "artifacts",
        "regions",
        "canonical",
        "datasets",
    }
    missing = required - set(document)
    if missing:
        raise PipelineConfigError(f"merged configuration is missing {sorted(missing)}")
    return CountryPipelineContext(root, document)
