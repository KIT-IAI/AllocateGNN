"""Shared engineering-admission authority and country-neutral evaluators.

The shared authority owns every numerical B+ grid, graph, and memory limit.
Country contracts may only own regionalisation and realization policies.  The
loader rejects duplicate ownership rather than silently allowing an overlay to
replace a shared engineering limit.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tomllib
from typing import Any, Mapping

import numpy as np

from sglib.core.algorithms.chunked_distance import (
    chunked_nearest_assignment as _core_chunked_nearest_assignment,
)
from sglib.core.infra.hashing import sha256_file

from sglib.core.infra.paths import resolve_case_path


AUTHORITY_SCHEMA_V2 = "sg_engineering_admission_authority_v2"
AUTHORITY_SCHEMA_V3 = "sg_engineering_admission_authority_v3"
COUNTRY_SCHEMA_V2 = "sg_country_engineering_admission_contract_v2"
COUNTRY_SCHEMA_V3 = "sg_country_engineering_admission_contract_v3"

# Backwards-compatible aliases for callers that imported the original names.
# They intentionally continue to identify the archived v2 contract until the
# repository's live authority references are atomically advanced to v3.
AUTHORITY_SCHEMA = AUTHORITY_SCHEMA_V2
COUNTRY_SCHEMA = COUNTRY_SCHEMA_V2

LEGACY_DENSE_LANDUSE_REPRESENTATION = "legacy_dense_mapping_matrix_v2"
INDEXED_LANDUSE_REPRESENTATION = "edge_flat_id_scatter_add_v1"
LANDUSE_SUPERVISION_REPRESENTATIONS = frozenset(
    {
        LEGACY_DENSE_LANDUSE_REPRESENTATION,
        INDEXED_LANDUSE_REPRESENTATION,
    }
)
INDEXED_EQUIVALENCE_EVIDENCE_SCHEMA = (
    "sg_indexed_landuse_equivalence_diagnostic_v1"
)
# Representation evidence is a design basis of the admission rule, not a
# result: it lives in Git beside the authority that binds it.
INDEXED_EQUIVALENCE_EVIDENCE_ROOT = ("casestudy", "config", "authority", "evidence")
INDEXED_EQUIVALENCE_EVIDENCE_PATH = (
    "casestudy/config/authority/evidence/indexed_landuse_equivalence.json"
)
INDEXED_EQUIVALENCE_IMMUTABILITY_MARKER = (
    "casestudy/config/authority/evidence/IMMUTABLE_AUTHORITY_EVIDENCE.txt"
)
_AUTHORITY_COUNTRY_SCHEMA_PAIRS = {
    AUTHORITY_SCHEMA_V2: COUNTRY_SCHEMA_V2,
    AUTHORITY_SCHEMA_V3: COUNTRY_SCHEMA_V3,
}
_SHARED_SECTIONS = frozenset({"grid", "graph", "memory"})


class EngineeringAdmissionError(ValueError):
    """Raised when authority ownership or an admission metric is invalid."""


@dataclass(frozen=True)
class EngineeringAdmission:
    repo_root: Path
    authority_path: Path
    country_path: Path
    authority: Mapping[str, Any]
    country: Mapping[str, Any]

    @property
    def authority_sha256(self) -> str:
        return sha256_file(self.authority_path)

    @property
    def country_sha256(self) -> str:
        return sha256_file(self.country_path)

    @property
    def grid(self) -> Mapping[str, Any]:
        return self.authority["grid"]

    @property
    def graph(self) -> Mapping[str, Any]:
        return self.authority["graph"]

    @property
    def memory(self) -> Mapping[str, Any]:
        return self.authority["memory"]

    @property
    def landuse_supervision_representation(self) -> str:
        """Return the materialized supervision representation owned by authority.

        The archived v2 schema predates an explicit representation field and is
        therefore bound to its historical dense mapping matrix.  Every later
        schema must name the representation explicitly.
        """

        schema = str(self.authority.get("schema_version"))
        if schema == AUTHORITY_SCHEMA_V2:
            return LEGACY_DENSE_LANDUSE_REPRESENTATION
        value = self.memory.get("landuse_supervision_representation")
        if value not in LANDUSE_SUPERVISION_REPRESENTATIONS:
            raise EngineeringAdmissionError(
                f"unknown landuse supervision representation: {value!r}"
            )
        return str(value)


def _inside(root: Path, path: Path, label: str) -> Path:
    resolved = resolve_case_path(path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise EngineeringAdmissionError(f"{label} escapes repository boundary") from exc
    return resolved


def _positive(section: Mapping[str, Any], keys: tuple[str, ...], label: str) -> None:
    for key in keys:
        value = section.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise EngineeringAdmissionError(f"{label}.{key} must be positive")


def _leaf_names(document: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for key, value in document.items():
        if isinstance(value, Mapping):
            names.update(_leaf_names(value))
        else:
            names.add(str(key))
    return names


def _validate_authority(document: Mapping[str, Any]) -> None:
    schema = document.get("schema_version")
    if schema not in _AUTHORITY_COUNTRY_SCHEMA_PAIRS:
        raise EngineeringAdmissionError("unsupported shared engineering authority schema")
    if document.get("status") != "FROZEN" or document.get("authority_class") != "B":
        raise EngineeringAdmissionError("shared engineering authority is not frozen class B")
    if not _SHARED_SECTIONS <= set(document):
        raise EngineeringAdmissionError("shared authority must own grid, graph, and memory")
    grid, graph, memory = (document[name] for name in ("grid", "graph", "memory"))
    _positive(
        grid,
        (
            "target_points",
            "min_ground_step_m",
            "max_ground_step_m",
            "min_cells_per_source_hard",
            "min_cells_per_target_hard",
            "cells_per_source_oversampling_disclosure",
            "cells_per_target_oversampling_disclosure",
            "max_cells_per_region",
        ),
        "grid",
    )
    if float(grid["max_ground_step_m"]) < float(grid["min_ground_step_m"]):
        raise EngineeringAdmissionError("grid step bounds are inverted")
    if int(grid["cells_per_source_oversampling_disclosure"]) < int(grid["min_cells_per_source_hard"]):
        raise EngineeringAdmissionError("source disclosure reference is below its hard gate")
    if int(grid["cells_per_target_oversampling_disclosure"]) < int(grid["min_cells_per_target_hard"]):
        raise EngineeringAdmissionError("target disclosure reference is below its hard gate")
    _positive(
        graph,
        (
            "max_source_nodes_per_region",
            "max_target_nodes_per_region",
            "max_active_agent_nodes_per_region",
            "directed_edges_per_cell_envelope",
            "max_directed_edges_per_region",
            "planning_k_max",
        ),
        "graph",
    )
    derived_edges = int(graph["directed_edges_per_cell_envelope"]) * int(grid["max_cells_per_region"])
    if int(graph["max_directed_edges_per_region"]) != derived_edges:
        raise EngineeringAdmissionError(
            "max_directed_edges_per_region must equal the cell envelope derivation"
        )
    _positive(
        memory,
        (
            "max_dense_tensor_mib",
            "dense_feature_channels",
            "dense_value_bytes",
            "max_cdist_workspace_mib",
            "max_worker_rss_gib",
        ),
        "memory",
    )
    if schema == AUTHORITY_SCHEMA_V2:
        if "representation_evidence" in document:
            raise EngineeringAdmissionError(
                "v2 authority may not declare v3 representation evidence"
            )
        v3_only = {
            "landuse_supervision_representation",
            "flat_index_value_bytes",
            "landuse_ratio_value_bytes",
        } & set(memory)
        if v3_only:
            raise EngineeringAdmissionError(
                f"v2 authority may not declare v3 representation fields: {sorted(v3_only)}"
            )
    else:
        evidence = document.get("representation_evidence")
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "path",
            "schema_version",
            "sha256",
        }:
            raise EngineeringAdmissionError(
                "v3 authority requires an exact representation_evidence reference"
            )
        if evidence["schema_version"] != INDEXED_EQUIVALENCE_EVIDENCE_SCHEMA:
            raise EngineeringAdmissionError(
                "v3 authority references an unsupported representation evidence schema"
            )
        evidence_locator = str(evidence["path"])
        if evidence_locator.startswith("casestudy2/"):
            evidence_locator = "casestudy/" + evidence_locator.removeprefix("casestudy2/")
        if evidence_locator != INDEXED_EQUIVALENCE_EVIDENCE_PATH:
            raise EngineeringAdmissionError(
                "v3 authority references the wrong representation evidence artifact"
            )
        evidence_hash = evidence["sha256"]
        if (
            not isinstance(evidence_hash, str)
            or len(evidence_hash) != 64
            or any(character not in "0123456789abcdef" for character in evidence_hash)
        ):
            raise EngineeringAdmissionError(
                "v3 authority representation evidence sha256 is invalid"
            )
        representation = memory.get("landuse_supervision_representation")
        if representation not in LANDUSE_SUPERVISION_REPRESENTATIONS:
            raise EngineeringAdmissionError(
                f"unknown landuse supervision representation: {representation!r}"
            )
        _positive(
            memory,
            ("flat_index_value_bytes", "landuse_ratio_value_bytes"),
            "memory",
        )
        for key in ("flat_index_value_bytes", "landuse_ratio_value_bytes"):
            if type(memory[key]) is not int:
                raise EngineeringAdmissionError(f"memory.{key} must be an integer byte width")
        if (
            representation == INDEXED_LANDUSE_REPRESENTATION
            and int(memory["flat_index_value_bytes"]) != 8
        ):
            raise EngineeringAdmissionError(
                "edge_flat_id_scatter_add_v1 requires int64 flat indices (8 bytes)"
            )
        if (
            representation == INDEXED_LANDUSE_REPRESENTATION
            and int(memory["landuse_ratio_value_bytes"]) != 4
        ):
            raise EngineeringAdmissionError(
                "edge_flat_id_scatter_add_v1 requires float32 landuse ratios (4 bytes)"
            )
        if schema == AUTHORITY_SCHEMA_V3 and int(memory["dense_feature_channels"]) != 5:
            raise EngineeringAdmissionError(
                "v3 landuse supervision authority requires five frozen channels"
            )
    if memory.get("dense_dry_run_binding") != "advisory_upper_bound":
        raise EngineeringAdmissionError("dense dry-run binding must remain advisory")
    if memory.get("dense_formal_binding") != "fail_closed_real_active_cells_before_training_submission":
        raise EngineeringAdmissionError("formal dense binding must be fail-closed before submission")
    if memory.get("cdist_strategy") != "country_neutral_chunked_rows_v1":
        raise EngineeringAdmissionError("unsupported cdist workspace strategy")


def _validate_representation_evidence(
    repository: Path,
    authority: Mapping[str, Any],
) -> None:
    """Validate the diagnostic evidence bound by a frozen v3 authority."""

    if authority.get("schema_version") == AUTHORITY_SCHEMA_V2:
        return
    reference = authority["representation_evidence"]
    evidence_path = _inside(
        repository,
        repository / str(reference["path"]),
        "representation evidence",
    )
    relative = evidence_path.relative_to(repository)
    if (relative.parts[:1] not in (("casestudy",), ("casestudy2",)) or relative.parts[1:4] != INDEXED_EQUIVALENCE_EVIDENCE_ROOT[1:]) or evidence_path.suffix != ".json":
        raise EngineeringAdmissionError(
            "v3 representation evidence must be a JSON artifact under the authority evidence directory"
        )
    if not evidence_path.is_file():
        raise EngineeringAdmissionError("v3 representation evidence is missing")
    if sha256_file(evidence_path) != reference["sha256"]:
        raise EngineeringAdmissionError("v3 representation evidence hash mismatch")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EngineeringAdmissionError(
            "v3 representation evidence is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(evidence, dict):
        raise EngineeringAdmissionError("v3 representation evidence must be an object")
    if evidence.get("schema_version") != reference["schema_version"]:
        raise EngineeringAdmissionError(
            "v3 representation evidence schema differs from its authority reference"
        )
    if evidence.get("status") != "DIAGNOSTIC_ONLY":
        raise EngineeringAdmissionError(
            "v3 representation evidence must remain DIAGNOSTIC_ONLY"
        )
    if evidence.get("representation") != authority["memory"].get(
        "landuse_supervision_representation"
    ):
        raise EngineeringAdmissionError(
            "v3 representation evidence differs from the authority representation"
        )
    marker = _inside(
        repository,
        repository / INDEXED_EQUIVALENCE_IMMUTABILITY_MARKER,
        "representation evidence immutability marker",
    )
    if not marker.is_file():
        raise EngineeringAdmissionError(
            "v3 representation evidence lacks its immutability marker"
        )
    marker_text = marker.read_text(encoding="utf-8")
    if str(reference["path"]) not in marker_text or reference["sha256"] not in marker_text:
        raise EngineeringAdmissionError(
            "v3 representation evidence immutability marker differs"
        )


def load_engineering_admission(
    repo_root: Path | str,
    country_contract: Path | str,
) -> EngineeringAdmission:
    """Load a country policy and its shared authority without overlay semantics."""

    repository = Path(repo_root).resolve()
    country_path = Path(country_contract)
    if not country_path.is_absolute():
        country_path = repository / country_path
    country_path = _inside(repository, country_path, "country admission contract")
    with country_path.open("rb") as stream:
        country = tomllib.load(stream)
    country_schema = country.get("schema_version")
    if country_schema not in _AUTHORITY_COUNTRY_SCHEMA_PAIRS.values() or country.get("status") != "FROZEN":
        raise EngineeringAdmissionError("country engineering contract has no supported frozen schema")
    duplicated_sections = _SHARED_SECTIONS & set(country)
    if duplicated_sections:
        raise EngineeringAdmissionError(
            f"country contract duplicates shared authority sections: {sorted(duplicated_sections)}"
        )
    reference = country.get("shared_authority")
    if not isinstance(reference, dict) or set(reference) != {"path", "schema_version"}:
        raise EngineeringAdmissionError("country contract must contain an exact shared_authority reference")
    referenced_schema = reference["schema_version"]
    expected_country_schema = _AUTHORITY_COUNTRY_SCHEMA_PAIRS.get(referenced_schema)
    if expected_country_schema is None:
        raise EngineeringAdmissionError("country contract references an unsupported authority schema")
    if country_schema != expected_country_schema:
        raise EngineeringAdmissionError(
            "country and shared engineering authority schemas are not a versioned pair"
        )
    authority_path = _inside(repository, repository / str(reference["path"]), "shared authority")
    with authority_path.open("rb") as stream:
        authority = tomllib.load(stream)
    if authority.get("schema_version") != referenced_schema:
        raise EngineeringAdmissionError(
            "country contract authority schema reference differs from the referenced file"
        )
    _validate_authority(authority)
    _validate_representation_evidence(repository, authority)
    country_policy = {
        key: value
        for key, value in country.items()
        if key not in {"schema_version", "status", "frozen_at", "shared_authority"}
    }
    shared_limits = {
        key: authority[key]
        for key in _SHARED_SECTIONS
    }
    duplicated_leaves = _leaf_names(country_policy) & _leaf_names(shared_limits)
    if duplicated_leaves:
        raise EngineeringAdmissionError(
            f"country contract duplicates shared authority keys: {sorted(duplicated_leaves)}"
        )
    idr = country.get("idr_matched")
    if not isinstance(idr, dict):
        raise EngineeringAdmissionError("country contract is missing idr_matched")
    _positive(idr, ("formal_candidates", "seeds", "max_realizations_all_regions"), "idr_matched")
    return EngineeringAdmission(repository, authority_path, country_path, authority, country)


def chunked_nearest_assignment(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    *,
    max_workspace_mib: float,
) -> tuple[np.ndarray, int, int]:
    """Return stable nearest labels and the actual peak cdist workspace.

    The row chunk is derived from the authority byte budget, so a large full
    distance matrix never needs to be materialised.  ``numpy.argmin`` provides
    stable first-ordinal tie breaking.
    """

    try:
        return _core_chunked_nearest_assignment(
            left_xy,
            right_xy,
            max_workspace_bytes=int(float(max_workspace_mib) * 2**20),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise EngineeringAdmissionError(str(exc)) from exc


def evaluate_region(
    admission: EngineeringAdmission,
    metrics: Mapping[str, int | float],
    *,
    formal_pre_submission: bool = False,
) -> dict[str, Any]:
    """Evaluate hard gates while keeping dry-run memory advisory.

    A formal pre-submission call must provide ``active_agent_nodes_observed``;
    its true materialized land-use supervision footprint becomes a hard gate.
    Geometry-only dry runs use ``active_agent_nodes_upper_bound`` and report
    memory without allowing the conservative bound to reject a region.
    """

    grid, graph, memory = admission.grid, admission.graph, admission.memory
    required = {
        "n_sources",
        "n_targets",
        "n_cells",
        "min_cells_per_source",
        "min_cells_per_target",
        "total_edges_directed",
        "active_agent_nodes_upper_bound",
        "cdist_peak_workspace_bytes",
    }
    missing = required - set(metrics)
    if missing:
        raise EngineeringAdmissionError(f"region metrics missing {sorted(missing)}")
    if formal_pre_submission and "active_agent_nodes_observed" not in metrics:
        raise EngineeringAdmissionError(
            "formal pre-submission admission requires observed active-agent cells"
        )
    active = int(
        metrics["active_agent_nodes_observed"]
        if formal_pre_submission
        else metrics["active_agent_nodes_upper_bound"]
    )
    footprint_bytes = landuse_supervision_footprint_bytes(
        admission,
        active_edges=active,
        n_sources=int(metrics["n_sources"]),
    )
    limit_bytes = int(float(memory["max_dense_tensor_mib"]) * 2**20)
    hard = {
        "min_cells_per_source_hard": int(metrics["min_cells_per_source"]) >= int(grid["min_cells_per_source_hard"]),
        "min_cells_per_target_hard": int(metrics["min_cells_per_target"]) >= int(grid["min_cells_per_target_hard"]),
        "max_cells_per_region": int(metrics["n_cells"]) <= int(grid["max_cells_per_region"]),
        "max_source_nodes_per_region": int(metrics["n_sources"]) <= int(graph["max_source_nodes_per_region"]),
        "max_target_nodes_per_region": int(metrics["n_targets"]) <= int(graph["max_target_nodes_per_region"]),
        "max_active_agent_nodes_per_region": active <= int(graph["max_active_agent_nodes_per_region"]),
        "max_directed_edges_per_region": int(metrics["total_edges_directed"]) <= int(graph["max_directed_edges_per_region"]),
        "planning_k_max": int(metrics["n_targets"]) <= int(graph["planning_k_max"]),
        "max_cdist_workspace_mib": int(metrics["cdist_peak_workspace_bytes"]) <= int(float(memory["max_cdist_workspace_mib"]) * 2**20),
    }
    if formal_pre_submission:
        # Retain the gate key used by v2 receipts and submit guards.  Under v3
        # the value is the actual representation footprint, not a hypothetical
        # dense allocation; the representation and canonical footprint fields
        # below make that distinction explicit and auditable.
        hard["max_dense_tensor_real_active_mib"] = footprint_bytes <= limit_bytes
    disclosures = {
        "cells_per_source_oversampling_reference": int(grid["cells_per_source_oversampling_disclosure"]),
        "cells_per_source_reference_met": int(metrics["min_cells_per_source"]) >= int(grid["cells_per_source_oversampling_disclosure"]),
        "cells_per_target_oversampling_reference": int(grid["cells_per_target_oversampling_disclosure"]),
        "cells_per_target_reference_met": int(metrics["min_cells_per_target"]) >= int(grid["cells_per_target_oversampling_disclosure"]),
        "binding": "report_only",
    }
    advisories = {
        "landuse_supervision_representation": admission.landuse_supervision_representation,
        "landuse_supervision_footprint_bytes": footprint_bytes,
        "landuse_supervision_footprint_mib": footprint_bytes / 2**20,
        "landuse_supervision_limit_mib": float(memory["max_dense_tensor_mib"]),
        "landuse_supervision_limit_met": footprint_bytes <= limit_bytes,
        # Backward-compatible field names consumed by existing v2 evidence and
        # schemas.  Their values deliberately follow the declared representation.
        "dense_tensor_mib": footprint_bytes / 2**20,
        "dense_tensor_limit_mib": float(memory["max_dense_tensor_mib"]),
        "dense_tensor_limit_met": footprint_bytes <= limit_bytes,
        "binding": "hard_pre_submission" if formal_pre_submission else "advisory_dry_run",
    }
    return {
        "hard_checks": hard,
        "hard_pass": all(hard.values()),
        "oversampling_disclosure": disclosures,
        "memory": advisories,
    }


def landuse_supervision_footprint_bytes(
    admission: EngineeringAdmission,
    *,
    active_edges: int,
    n_sources: int,
) -> int:
    """Return the actual materialized supervision footprint for an authority.

    V2 is permanently bound to ``edges * sources * channels * value_bytes``.
    V3 dispatches on its explicit representation.  Indexed supervision stores
    one int64 flat id per active source-agent edge plus the source-by-land-use
    ratio matrix.  Supporting the known dense identifier in v3 is intentional:
    a future rollback then re-enters the original (usually failing) 32 MiB gate
    instead of silently retaining the compact accounting formula.
    """

    if type(active_edges) is not int or active_edges < 0:
        raise EngineeringAdmissionError("active_edges must be a non-negative integer")
    if type(n_sources) is not int or n_sources <= 0:
        raise EngineeringAdmissionError("n_sources must be a positive integer")
    memory = admission.memory
    representation = admission.landuse_supervision_representation
    channels = int(memory["dense_feature_channels"])
    if representation == LEGACY_DENSE_LANDUSE_REPRESENTATION:
        return active_edges * n_sources * channels * int(memory["dense_value_bytes"])
    if representation == INDEXED_LANDUSE_REPRESENTATION:
        return (
            active_edges * int(memory["flat_index_value_bytes"])
            + n_sources * channels * int(memory["landuse_ratio_value_bytes"])
        )
    # The property and authority validator already reject this, but retain an
    # explicit fail-closed branch so this accounting function is safe in isolation.
    raise EngineeringAdmissionError(
        f"unknown landuse supervision representation: {representation!r}"
    )


def validate_pre_submission(
    admission: EngineeringAdmission,
    metrics: Mapping[str, int | float],
) -> dict[str, Any]:
    """Fail closed unless observed formal active-cell metrics pass every gate."""

    decision = evaluate_region(admission, metrics, formal_pre_submission=True)
    failures = sorted(
        key for key, value in decision["hard_checks"].items() if not value
    )
    if failures:
        raise EngineeringAdmissionError(
            f"formal pre-submission engineering admission failed: {failures}"
        )
    return decision


__all__ = [
    "AUTHORITY_SCHEMA",
    "AUTHORITY_SCHEMA_V2",
    "AUTHORITY_SCHEMA_V3",
    "COUNTRY_SCHEMA",
    "COUNTRY_SCHEMA_V2",
    "COUNTRY_SCHEMA_V3",
    "EngineeringAdmission",
    "EngineeringAdmissionError",
    "INDEXED_LANDUSE_REPRESENTATION",
    "INDEXED_EQUIVALENCE_EVIDENCE_SCHEMA",
    "INDEXED_EQUIVALENCE_EVIDENCE_PATH",
    "LANDUSE_SUPERVISION_REPRESENTATIONS",
    "LEGACY_DENSE_LANDUSE_REPRESENTATION",
    "chunked_nearest_assignment",
    "evaluate_region",
    "landuse_supervision_footprint_bytes",
    "load_engineering_admission",
    "validate_pre_submission",
]
