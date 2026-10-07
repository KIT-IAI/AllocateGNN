"""Controlled vocabulary and the central country-profile registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import re
import tomllib
from typing import Any, Mapping

from sglib.core.infra.paths import resolve_case_path


class EligibilityStatus(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE_DEPLOYMENT_SCOPE = "INELIGIBLE_DEPLOYMENT_SCOPE"
    INELIGIBLE_KNOWN_SITE_SIGNAL = "INELIGIBLE_KNOWN_SITE_SIGNAL"
    QA_ONLY = "QA_ONLY"
    RECONSTRUCTION_ONLY = "RECONSTRUCTION_ONLY"


class DataCategory(StrEnum):
    BOUNDARIES_GRID = "boundaries_grid"
    REGIONAL_TOTALS = "regional_totals"
    STATION_REGISTER = "station_register"
    LANDUSE = "landuse"
    BUILT_SURFACE = "built_surface"
    NTL = "ntl"
    FEATURES_CUZ = "features_cuz"


_CODE = re.compile(r"^[a-z]{2}$")
_PROFILE_TOP_LEVEL = {
    "schema_version",
    "country",
    "crs",
    "temporal",
    "units",
    "station_contract",
}


class CountryProfileError(ValueError):
    pass


@dataclass(frozen=True)
class CountryProfile:
    code: str
    directory: str
    label: str
    evaluation_scope: str
    crs: Mapping[str, str]
    temporal: Mapping[str, Any]
    units: Mapping[str, str]
    station_contract: Mapping[str, Any]
    source_path: Path


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CountryProfileError(f"{label} must be a TOML table")
    return dict(value)


def load_country_profile(path: Path | str) -> CountryProfile:
    source = resolve_case_path(path).resolve()
    with source.open("rb") as handle:
        document = tomllib.load(handle)
    unknown = set(document) - _PROFILE_TOP_LEVEL
    if unknown:
        raise CountryProfileError(f"unknown country-profile keys: {sorted(unknown)}")
    if document.get("schema_version") != "sg_country_profile_v1":
        raise CountryProfileError(f"unsupported profile schema in {source}")
    country = _mapping(document.get("country"), label="country")
    required = {"code", "directory", "label", "evaluation_scope"}
    missing = required - set(country)
    unknown_country = set(country) - required
    if missing or unknown_country:
        raise CountryProfileError(
            f"invalid country keys; missing={sorted(missing)}, unknown={sorted(unknown_country)}"
        )
    code = str(country["code"]).lower()
    if not _CODE.fullmatch(code) or source.stem.lower() != code:
        raise CountryProfileError(f"profile code/file mismatch: {code!r} vs {source.name}")
    crs = _mapping(document.get("crs"), label="crs")
    required_crs = {"native", "area", "generation", "storage", "working"}
    if set(crs) != required_crs:
        raise CountryProfileError(f"crs keys must be {sorted(required_crs)}")
    return CountryProfile(
        code=code,
        directory=str(country["directory"]),
        label=str(country["label"]),
        evaluation_scope=str(country["evaluation_scope"]),
        crs=crs,
        temporal=_mapping(document.get("temporal"), label="temporal"),
        units=_mapping(document.get("units"), label="units"),
        station_contract=_mapping(document.get("station_contract"), label="station_contract"),
        source_path=source,
    )


def discover_country_profiles(root: Path | str) -> dict[str, CountryProfile]:
    base = resolve_case_path(root)
    profiles = {profile.code: profile for profile in map(load_country_profile, base.glob("*.toml"))}
    if not profiles:
        raise CountryProfileError(f"no country profiles found under {base}")
    directories = [profile.directory.casefold() for profile in profiles.values()]
    if len(directories) != len(set(directories)):
        raise CountryProfileError("country profile directories must be unique")
    return dict(sorted(profiles.items()))
