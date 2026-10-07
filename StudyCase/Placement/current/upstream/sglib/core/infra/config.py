"""Strict TOML loading, layer merging, and provenance reporting."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any, Mapping

from .terms import CountryProfile, load_country_profile

from sglib.core.infra.paths import resolve_case_path


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedConfig:
    values: Mapping[str, Any]
    sources: Mapping[str, Path]
    country_profile: CountryProfile

    def explain(self) -> list[tuple[str, Any, Path]]:
        leaves = _flatten(self.values)
        return [(key, leaves[key], self.sources[key]) for key in sorted(leaves)]


def load_toml(path: Path | str) -> dict[str, Any]:
    source = resolve_case_path(path)
    with source.open("rb") as handle:
        document = tomllib.load(handle)
    if not isinstance(document, dict):
        raise ConfigError(f"TOML root must be a table: {source}")
    return document


def _flatten(document: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    leaves: dict[str, Any] = {}
    for key, value in document.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            leaves.update(_flatten(value, dotted))
        else:
            leaves[dotted] = value
    return leaves


def _merge_disjoint(layers: list[tuple[Path, Mapping[str, Any]]]) -> tuple[dict[str, Any], dict[str, Path]]:
    merged: dict[str, Any] = {}
    sources: dict[str, Path] = {}
    for path, layer in layers:
        current = _flatten(layer)
        duplicates = set(sources) & set(current)
        if duplicates:
            raise ConfigError(
                f"configuration keys may not be repeated across layers: {sorted(duplicates)}"
            )
        for dotted in current:
            sources[dotted] = path.resolve()
        _deep_update(merged, layer)
    return merged, sources


def _deep_update(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping):
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                raise ConfigError(f"configuration shape collision at {key}")
            _deep_update(child, value)
        else:
            target[key] = deepcopy(value)


_GENERAL_KEYS = {
    "schema_version",
    "grid",
    "categories",
    "features",
    "cuz",
    "artifacts",
    "display",
}
_COUNTRY_OVERLAY_KEYS = {
    "schema_version",
    "osm_snapshot",
    "ntl",
    "regions",
    "canonical",
    "handoff_artifacts",
    "grid_columns",
    "datasets",
    "products",
    "industry_sectors",
}


def _strict_top_level(document: Mapping[str, Any], allowed: set[str], *, label: str) -> None:
    unknown = set(document) - allowed
    if unknown:
        raise ConfigError(f"unknown {label} keys: {sorted(unknown)}")


def _profile_layer(profile: CountryProfile) -> dict[str, Any]:
    return {
        "country": {
            "code": profile.code,
            "directory": profile.directory,
            "label": profile.label,
            "evaluation_scope": profile.evaluation_scope,
        },
        "crs": dict(profile.crs),
        "temporal": dict(profile.temporal),
        "units": dict(profile.units),
        "station_contract": dict(profile.station_contract),
    }


def load_dataoverview_config(
    general_path: Path | str,
    profile_path: Path | str,
    country_path: Path | str,
) -> LoadedConfig:
    general_source = resolve_case_path(general_path)
    country_source = resolve_case_path(country_path)
    general = load_toml(general_source)
    country = load_toml(country_source)
    _strict_top_level(general, _GENERAL_KEYS, label="general configuration")
    _strict_top_level(country, _COUNTRY_OVERLAY_KEYS, label="country overlay")
    if general.get("schema_version") != "sg_dataoverview_general_v1":
        raise ConfigError("unsupported DataOverview general schema_version")
    if country.get("schema_version") != "sg_dataoverview_country_v1":
        raise ConfigError("unsupported DataOverview country schema_version")
    profile = load_country_profile(profile_path)
    if country_source.stem.lower() != profile.code:
        raise ConfigError("country overlay/profile filename mismatch")
    general_body = {key: value for key, value in general.items() if key != "schema_version"}
    country_body = {key: value for key, value in country.items() if key != "schema_version"}
    profile_source = profile.source_path
    merged, sources = _merge_disjoint(
        [(profile_source, _profile_layer(profile)), (general_source, general_body), (country_source, country_body)]
    )
    return LoadedConfig(merged, sources, profile)
