from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import tomllib
from typing import Any, Mapping

from sglib.core.infra.hashing import sha256_file, sha256_json
from sglib.core.infra.terms import CountryProfile, load_country_profile

from sglib.core.infra.paths import resolve_case_path


class GeneratorConfigError(ValueError):
    pass


SCIENTIFIC_CONFIG_PROJECTION_SCHEMA = "sg_generator_scientific_config_v3"
# Historical counts describe one earlier materialisation.  They are neither
# computational inputs nor admission rules, so they never enter scientific identity.
HISTORICAL_COUNT_CHECK_PREFIX = "expected_"
HISTORICAL_STATION_CONTRACT_COUNTS = ("formal_truth_rows", "section_lineage_rows")
SCIENTIFIC_MATRIX_PROJECTION_SCHEMA = "sg_generator_task_matrix_science_v2"
_SCIENTIFIC_MATRIX_COLUMNS = (
    "task_group",
    "stage",
    "country",
    "family",
    "config",
    "signal",
    "parameter_name",
    "parameter_values",
    "seeds",
    "folds",
    "logical_coordinates",
    "reuse_coordinates",
    "new_train_coordinates",
    "compute_threads",
    "depends_on",
    "reuse_source",
)


@dataclass(frozen=True)
class LoadedGeneratorConfig:
    values: Mapping[str, Any]
    sources: Mapping[str, Path]
    country_profile: CountryProfile

    def explain(self) -> list[tuple[str, Any, Path]]:
        leaves = _flatten(self.values)
        return [(key, leaves[key], self.sources[key]) for key in sorted(leaves)]


def _identity_checks(checks: Any) -> Any:
    if not isinstance(checks, Mapping):
        return deepcopy(checks)
    return {
        key: deepcopy(value)
        for key, value in checks.items()
        if key != "admission_contract" and not str(key).startswith(HISTORICAL_COUNT_CHECK_PREFIX)
    }


def _identity_station_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in contract.items()
        if key not in HISTORICAL_STATION_CONTRACT_COUNTS
    }


def _load(path: Path) -> dict[str, Any]:
    with resolve_case_path(path).open("rb") as handle:
        return tomllib.load(handle)


def _flatten(document: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in document.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(_flatten(value, dotted))
        else:
            result[dotted] = value
    return result


def _update(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping):
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                raise GeneratorConfigError(f"configuration shape collision at {key}")
            _update(child, value)
        else:
            target[key] = deepcopy(value)


def scientific_task_matrix_projection(path: Path | str) -> dict[str, Any]:
    """Return the task matrix columns that determine what is computed."""

    import csv

    matrix = resolve_case_path(path)
    try:
        with matrix.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise GeneratorConfigError("task matrix has no header")
            fields = set(reader.fieldnames)
            legacy = set(_SCIENTIFIC_MATRIX_COLUMNS) - {"compute_threads"}
            if not (
                set(_SCIENTIFIC_MATRIX_COLUMNS) <= fields
                or (legacy <= fields and "cpus" in fields)
            ):
                raise GeneratorConfigError(
                    "task matrix lacks the scientific identity projection"
                )
            rows = []
            for row in reader:
                projected = {
                    name: str(
                        row["cpus"]
                        if name == "compute_threads" and name not in row
                        else row[name]
                    )
                    for name in _SCIENTIFIC_MATRIX_COLUMNS
                }
                rows.append(projected)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise GeneratorConfigError(f"cannot project task matrix: {matrix}") from exc
    return {
        "schema_version": SCIENTIFIC_MATRIX_PROJECTION_SCHEMA,
        "columns": list(_SCIENTIFIC_MATRIX_COLUMNS),
        "rows": rows,
    }


def _engineering_authority_projection(path: Path) -> dict[str, Any]:
    authority = _load(path)
    evidence = deepcopy(authority.get("representation_evidence", {}))
    if isinstance(evidence, dict):
        evidence.pop("path", None)
        evidence.pop("location_policy", None)
    return {
        key: deepcopy(value)
        for key, value in authority.items()
        if key != "frozen_at" and key != "representation_evidence"
    } | {"representation_evidence": evidence}


def _country_admission_projection(root: Path, checks: Mapping[str, Any]) -> dict[str, Any]:
    path = resolve_case_path(root / str(checks.get("admission_contract", "")))
    contract = _load(path)
    shared = deepcopy(contract.get("shared_authority", {}))
    if isinstance(shared, dict):
        shared.pop("path", None)
    return {
        key: deepcopy(value)
        for key, value in contract.items()
        if key not in {"frozen_at", "shared_authority"}
    } | {"shared_authority": shared}


def scientific_config_projection(
    config: LoadedGeneratorConfig | Mapping[str, Any],
    *,
    repo_root: Path | str,
) -> dict[str, Any]:
    """Project merged Generator config onto computation-changing identity only.

    Paths, directory layout, scheduler resources, timestamps, and authority
    container hashes are intentionally absent.  Referenced scientific
    authorities are represented by content-aware semantic projections.
    """

    values = config.values if isinstance(config, LoadedGeneratorConfig) else config
    root = Path(repo_root).resolve()
    authorities = values["authorities"]
    matrix_path = resolve_case_path(root / str(authorities["training_task_matrix"]))
    engineering_path = resolve_case_path(root / str(authorities["engineering_admission"]))
    checks = _identity_checks(values.get("checks", {}))
    execution = deepcopy(values.get("execution", {}))
    if isinstance(execution, dict):
        workspace = execution.get("distance_workspace")
        if isinstance(workspace, dict):
            workspace.pop("authority_sha256", None)
    country = values["country"]
    return {
        "schema_version": SCIENTIFIC_CONFIG_PROJECTION_SCHEMA,
        "country": {
            "code": country["code"],
            "evaluation_scope": country["evaluation_scope"],
        },
        "crs": deepcopy(values["crs"]),
        "temporal": deepcopy(values["temporal"]),
        "units": deepcopy(values["units"]),
        "station_contract": _identity_station_contract(values["station_contract"]),
        "regions": deepcopy(values["regions"]),
        "task_groups": deepcopy(values["task_groups"]),
        "training": deepcopy(values["training"]),
        "features": deepcopy(values["features"]),
        "gpm": deepcopy(values["gpm"]),
        "corrections": deepcopy(values["corrections"]),
        "gnn": deepcopy(values["gnn"]),
        "mlp": deepcopy(values["mlp"]),
        "sweeps": deepcopy(values["sweeps"]),
        "priors": deepcopy(values["priors"]),
        "civd": deepcopy(values["civd"]),
        "idr": deepcopy(values["idr"]),
        "checks": checks,
        "execution": execution,
        "authorities": {
            "candidate_registry_sha256": sha256_file(
                resolve_case_path(root / str(authorities["candidate_registry"]))
            ),
            "training_task_matrix": scientific_task_matrix_projection(matrix_path),
            "idr_contract_sha256": sha256_file(
                resolve_case_path(root / str(authorities["idr_contract"]))
            ),
            "engineering_admission": _engineering_authority_projection(
                engineering_path
            ),
            "country_admission": _country_admission_projection(
                root, values.get("checks", {})
            ),
        },
    }


def scientific_config_fingerprint(
    config: LoadedGeneratorConfig | Mapping[str, Any],
    *,
    repo_root: Path | str,
) -> str:
    return sha256_json(scientific_config_projection(config, repo_root=repo_root))


def load_generator_config(
    repo_root: Path | str,
    general_path: Path | str,
    profile_path: Path | str,
    country_path: Path | str,
) -> LoadedGeneratorConfig:
    root = Path(repo_root).resolve()
    general_source = resolve_case_path(general_path).resolve()
    overlay_source = resolve_case_path(country_path).resolve()
    profile = load_country_profile(profile_path)
    general = _load(general_source)
    overlay = _load(overlay_source)
    allowed_general = {
        "schema_version", "authorities", "training", "features", "gpm", "corrections",
        "gnn", "mlp", "sweeps", "priors", "civd", "paths",
    }
    allowed_overlay = {
        "schema_version",
        "regions",
        "task_groups",
        "idr",
        "checks",
        "execution",
    }
    if set(general) - allowed_general:
        raise GeneratorConfigError(f"unknown Generator general keys: {sorted(set(general)-allowed_general)}")
    if set(overlay) - allowed_overlay:
        raise GeneratorConfigError(f"unknown Generator country keys: {sorted(set(overlay)-allowed_overlay)}")
    if general.get("schema_version") != "sg_generator_general_v1":
        raise GeneratorConfigError("unsupported Generator general schema")
    if overlay.get("schema_version") != "sg_generator_country_v1":
        raise GeneratorConfigError("unsupported Generator country schema")
    if overlay_source.stem.lower() != profile.code:
        raise GeneratorConfigError("Generator overlay/profile filename mismatch")
    profile_layer = {
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
    layers = [
        (profile.source_path, profile_layer),
        (general_source, {key: value for key, value in general.items() if key != "schema_version"}),
        (overlay_source, {key: value for key, value in overlay.items() if key != "schema_version"}),
    ]
    merged: dict[str, Any] = {}
    sources: dict[str, Path] = {}
    for source, layer in layers:
        leaves = _flatten(layer)
        duplicate = set(sources) & set(leaves)
        if duplicate:
            raise GeneratorConfigError(f"Generator keys repeat across layers: {sorted(duplicate)}")
        sources.update({key: source for key in leaves})
        _update(merged, layer)
    if not merged["regions"] or len(merged["regions"]) != len(set(merged["regions"])):
        raise GeneratorConfigError("Generator regions must be non-empty and unique")
    if len(merged["task_groups"]) != 11 or len(set(merged["task_groups"])) != 11:
        raise GeneratorConfigError("each active Generator country must declare exactly 11 groups")
    execution = merged.setdefault("execution", {})
    if not isinstance(execution, dict):
        raise GeneratorConfigError("Generator execution must be a table")
    if "civd_enabled" not in execution:
        # Backward-compatible default for the already-running UK/AU authority.
        execution["civd_enabled"] = True
        sources["execution.civd_enabled"] = general_source
    if type(execution["civd_enabled"]) is not bool:
        raise GeneratorConfigError("execution.civd_enabled must be boolean")
    if float(merged["idr"]["b_tv"]) != 0.10:
        raise GeneratorConfigError("active v1 IDR contract requires explicit B_TV=0.10")
    authorities = merged["authorities"]
    authority_paths: dict[str, Path] = {}
    for name in (
        "candidate_registry",
        "training_task_matrix",
        "idr_contract",
        "engineering_admission",
    ):
        path = resolve_case_path(root / str(authorities[name])).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise GeneratorConfigError(f"authority escapes repository: {path}") from exc
        if not path.is_file() or sha256_file(path) != str(authorities[f"{name}_sha256"]):
            raise GeneratorConfigError(f"authority hash mismatch: {name}")
        authority_paths[name] = path
    engineering_authority = _load(authority_paths["engineering_admission"])
    if (
        engineering_authority.get("schema_version")
        != "sg_engineering_admission_authority_v3"
        or engineering_authority.get("status") != "FROZEN"
        or engineering_authority.get("authority_class") != "B"
    ):
        raise GeneratorConfigError("Generator rerun requires frozen v3 class-B engineering authority")
    memory = engineering_authority.get("memory")
    if not isinstance(memory, dict):
        raise GeneratorConfigError("engineering admission authority lacks memory limits")
    try:
        max_workspace_mib = float(memory["max_cdist_workspace_mib"])
        strategy = str(memory["cdist_strategy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GeneratorConfigError("engineering admission cdist authority is invalid") from exc
    if max_workspace_mib <= 0 or strategy != "country_neutral_chunked_rows_v1":
        raise GeneratorConfigError("engineering admission cdist authority is invalid")
    if (
        memory.get("landuse_supervision_representation")
        != "edge_flat_id_scatter_add_v1"
    ):
        raise GeneratorConfigError(
            "Generator rerun requires edge_flat_id_scatter_add_v1"
        )
    checks = merged.get("checks", {})
    if not isinstance(checks, dict) or checks.get("requires_engineering_admission") is not True:
        raise GeneratorConfigError(
            "every Generator rerun country requires engineering admission"
        )
    contract_path = checks.get("admission_contract")
    if not contract_path:
        raise GeneratorConfigError("Generator country lacks a v3 admission contract")
    try:
        from sglib.core.infra.engineering_admission import (
            EngineeringAdmissionError,
            load_engineering_admission,
        )

        admission = load_engineering_admission(root, str(contract_path))
    except (EngineeringAdmissionError, OSError, TypeError) as exc:
        raise GeneratorConfigError("Generator country v3 admission contract is invalid") from exc
    if admission.authority_path != authority_paths["engineering_admission"]:
        raise GeneratorConfigError(
            "country admission contract and Generator general authority differ"
        )
    if "distance_workspace" in execution:
        raise GeneratorConfigError("Generator overlay may not replace authority distance workspace")
    execution["distance_workspace"] = {
        "strategy": strategy,
        "max_workspace_mib": max_workspace_mib,
        "max_workspace_bytes": int(max_workspace_mib * 2**20),
        "authority_sha256": str(authorities["engineering_admission_sha256"]),
    }
    for key in execution["distance_workspace"]:
        sources[f"execution.distance_workspace.{key}"] = authority_paths[
            "engineering_admission"
        ]
    execution["landuse_supervision_representation"] = str(
        memory["landuse_supervision_representation"]
    )
    sources["execution.landuse_supervision_representation"] = authority_paths[
        "engineering_admission"
    ]
    try:
        from .weighter.candidates import validate_candidate_registry

        candidate_registry = validate_candidate_registry(
            json.loads(
                authority_paths["candidate_registry"].read_text(encoding="utf-8-sig")
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise GeneratorConfigError("candidate registry authority is invalid") from exc
    if profile.code.upper() not in candidate_registry["active_countries"]:
        raise GeneratorConfigError(
            f"{profile.code}: country is absent from candidate registry authority"
        )
    return LoadedGeneratorConfig(merged, sources, profile)


def training_params(config: LoadedGeneratorConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Build the self-contained worker document consumed by local/HPC tasks."""

    values = config.values if isinstance(config, LoadedGeneratorConfig) else config
    training = values["training"]
    configs: dict[str, Any] = {}
    for name, spec in training["configs"].items():
        objective_weights = {
            key: float(value)
            for key, value in spec.items()
            if key not in {"epochs", "feature_set"}
        }
        item = {"epochs": int(spec["epochs"]), "objective_weights": objective_weights}
        if "feature_set" in spec:
            item["feature_set"] = str(spec["feature_set"])
        configs[name] = item
    return {
        "schema_version": "sg_generator_worker_params_v1",
        "country": values["country"]["code"],
        "regions": list(values["regions"]),
        "seeds": list(training["seeds"]),
        "n_folds": int(training["n_folds"]),
        "kfold": {"shuffle": bool(training["kfold_shuffle"]), "random_state": "= seed"},
        "agent_feature_cols": list(values["features"]["agent_columns"]),
        "source_feature_cols": list(values["features"]["source_columns"]),
        "lu_cols": list(values["features"]["landuse_columns"]),
        "lu_prop_to_category": dict(values["features"]["landuse_mapping"]),
        "gpm": dict(values["gpm"]),
        "corrections": {**dict(values["corrections"]), "target_crs": values["crs"]["working"]},
        "gnn": dict(values["gnn"]),
        "mlp": dict(values["mlp"]),
        "config_map": configs,
        "sweeps": {key: list(item) for key, item in values["sweeps"].items()},
        "training_weighting_policy": str(training["weighting_policy"]),
        "training_batch_size": int(training["batch_size"]),
        "checks": {
            key: value
            for key, value in dict(values.get("checks", {})).items()
            if not str(key).startswith(HISTORICAL_COUNT_CHECK_PREFIX)
        },
        "run_contract": {
            "schema_version": "sg_generator_country_run_contract_v2",
            "generator_eligible": True,
            "evaluation_eligible": values["country"]["evaluation_scope"] != "dataoverview_only",
            "in_sample": False,
            "seeds": list(training["seeds"]),
            "n_folds": int(training["n_folds"]),
            "temporal_protocol": values["temporal"]["protocol"],
        },
    }
