"""Experiment stage configuration: general protocol, country overlay, registrations.

The Experiment stage derives regions, fields, sweeps, allocators, units and CRS
from upstream handoffs and the central country profile. What it *registers*
(plan 006a §1) are the scientific parameters that were committed before any
result existed: ``general/registrations.json`` holds the parameters shared by
every country, ``<dir>/registrations.json`` the country-level ones. A unit is
DONE only if its receipt carries exactly these registered values.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tomllib
from typing import Any, Mapping

from sglib.core.infra.hashing import sha256_file, sha256_json
from sglib.core.infra.terms import CountryProfile, load_country_profile

from sglib.core.infra.paths import resolve_case_path


GENERAL_SCHEMA = "sg_experiment_general_v1"
COUNTRY_SCHEMA = "sg_experiment_country_v1"
UNIT_TYPES = ("preflight_connection", "preflight_planning_pool", "observe", "planning",
              "bounds", "defense")


class ExperimentConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedExperimentConfig:
    values: dict[str, Any]
    country_profile: CountryProfile
    registrations: dict[str, dict[str, Any]]
    sources: dict[str, Path]


def _toml(path: Path) -> dict[str, Any]:
    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ExperimentConfigError(f"cannot read {path}") from exc


def _json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(resolve_case_path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ExperimentConfigError(f"cannot read {path}") from exc
    if not isinstance(document, dict):
        raise ExperimentConfigError(f"{path}: registrations must be an object")
    return document


def registration_projection(unit_type: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Project committed scientific parameters onto their registered form.

    One registered value embeds a large derived document; the registration keeps
    its sha256 instead of the document so the authority files stay readable.
    Receipts are projected the same way before comparison.
    """

    projected = {key: value for key, value in parameters.items()}
    if unit_type == "defense" and isinstance(projected.get("specification"), Mapping):
        specification = dict(projected["specification"])
        if "expected_coordinates" in specification:
            digest = sha256_json(specification.pop("expected_coordinates"))
            if specification.get("expected_coordinates_sha256") not in (None, digest):
                raise ExperimentConfigError("defense specification coordinate digest differs from its coordinates")
            specification["expected_coordinates_sha256"] = digest
        projected["specification"] = specification
    return projected


def load_experiment_config(repo_root: Path | str, country: str, *, config_root: Path | str | None = None) -> LoadedExperimentConfig:
    """Load the Experiment configuration of ``country``.

    ``config_root`` selects another tree holding the same repository layout of
    configuration files, such as a closed root's ``setup/config``; every file
    the stage reads then comes from that tree instead of the checkout.
    """

    root = Path(config_root if config_root is not None else repo_root).resolve()
    profile = load_country_profile(root / "casestudy/config/countries" / f"{country}.toml")
    general_path = resolve_case_path(root / "casestudy/3_Experiment/general/experiment.toml")
    overlay_path = resolve_case_path(root / "casestudy/3_Experiment" / profile.directory / f"{country}.toml")
    general = _toml(general_path)
    overlay = _toml(overlay_path)
    if general.get("schema_version") != GENERAL_SCHEMA:
        raise ExperimentConfigError("unsupported Experiment general schema")
    if overlay.get("schema_version") != COUNTRY_SCHEMA:
        raise ExperimentConfigError("unsupported Experiment country schema")
    allowed_general = {"schema_version", "authorities", "node_ids", "observe", "planning", "paths"}
    allowed_overlay = {"schema_version", "regions", "representative_region", "t1_regions", "t1", "defense"}
    if set(general) - allowed_general:
        raise ExperimentConfigError(f"unknown Experiment general keys: {sorted(set(general) - allowed_general)}")
    if set(overlay) - allowed_overlay:
        raise ExperimentConfigError(f"unknown Experiment country keys: {sorted(set(overlay) - allowed_overlay)}")
    regions = overlay.get("regions")
    if not isinstance(regions, list) or not regions or len(set(regions)) != len(regions):
        raise ExperimentConfigError(f"{country}: regions must be a non-empty unique list")
    if overlay.get("representative_region") not in regions:
        raise ExperimentConfigError(f"{country}: representative_region must be one of the regions")
    if any(region not in regions for region in overlay.get("t1_regions", [])):
        raise ExperimentConfigError(f"{country}: t1_regions must be configured regions")
    authorities = general["authorities"]
    candidate_registry = resolve_case_path(root / str(authorities["candidate_registry"]))
    if sha256_file(candidate_registry) != str(authorities["candidate_registry_sha256"]):
        raise ExperimentConfigError("candidate registry authority fingerprint mismatch")
    general_registrations = _json(root / str(authorities["registrations"]))
    country_registrations_path = overlay_path.parent / "registrations.json"
    country_registrations = _json(country_registrations_path) if country_registrations_path.is_file() else {}
    registrations: dict[str, dict[str, Any]] = {}
    for unit_type in UNIT_TYPES:
        merged = dict(general_registrations.get(unit_type, {}))
        for key, value in country_registrations.get(unit_type, {}).items():
            if key in merged:
                raise ExperimentConfigError(f"{country}/{unit_type}: {key} registered both generally and per country")
            merged[key] = value
        registrations[unit_type] = merged
    unknown = (set(general_registrations) | set(country_registrations)) - set(UNIT_TYPES)
    if unknown:
        raise ExperimentConfigError(f"registrations for unknown unit types: {sorted(unknown)}")
    values = {
        **general,
        "country": {"code": profile.code, "directory": profile.directory, "label": profile.label,
                    "evaluation_scope": profile.evaluation_scope},
        "crs": dict(profile.crs), "units": dict(profile.units), "station_contract": dict(profile.station_contract),
        "regions": list(map(str, regions)),
        "representative_region": str(overlay["representative_region"]),
        "t1_regions": list(map(str, overlay.get("t1_regions", []))),
        "t1": dict(overlay.get("t1", {})),
        "defense": dict(overlay.get("defense", {})),
    }
    return LoadedExperimentConfig(values, profile, registrations, {
        "general": general_path, "country": overlay_path, "candidate_registry": candidate_registry,
        "registrations": root / str(authorities["registrations"]), "country_registrations": country_registrations_path,
    })


def registered_values_match(unit_type: str, registrations: Mapping[str, Mapping[str, Any]],
                            parameters: Mapping[str, Any]) -> list[str]:
    """Return the registered keys whose committed value differs (empty = match)."""

    projected = registration_projection(unit_type, parameters)
    expected = registrations.get(unit_type, {})
    return sorted(key for key, value in expected.items()
                  if key not in projected or sha256_json(projected[key]) != sha256_json(value))
