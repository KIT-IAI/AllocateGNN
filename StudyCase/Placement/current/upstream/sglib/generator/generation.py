"""Execution-generation identity for Generator artifacts.

The generation is deliberately derived rather than selected by a runtime
switch.  A formal V3 run is the immutable combination of the input scientific
identity projection, candidate registry, shared engineering authority, and the
land-use supervision representation named by that authority.  V1 receipts are
supported as legacy whole-container bindings; V2/V3 never bind operational
receipt fields.

Legacy V2 material can still be described for diagnostics.  It is never
accepted as an input to a formal V3 product.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
import tomllib

from sglib.core.infra.hashing import sha256_file, sha256_json
from sglib.core.infra.paths import find_repo_root, resolve_case_path


V3_AUTHORITY_SCHEMA = "sg_engineering_admission_authority_v3"
V3_REPRESENTATION = "edge_flat_id_scatter_add_v1"
V3_GENERATION_SCHEMA = "sg_generator_execution_generation_v3"
LEGACY_GENERATION_SCHEMA = "sg_generator_execution_generation_v2_diagnostic"
INPUT_RECEIPT_SCHEMA_V1 = "sg_generator_inputs_receipt_v1"
INPUT_RECEIPT_SCHEMA_V2 = "sg_generator_inputs_receipt_v2"
INPUT_RECEIPT_SCHEMA_V3 = "sg_generator_inputs_receipt_v3"


class GenerationIdentityError(RuntimeError):
    """Raised when an artifact crosses execution generations."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GenerationIdentityError(f"{name} must be a mapping")
    return value


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise GenerationIdentityError(f"cannot load generation authority: {path}") from exc


def _representation(authority: Mapping[str, Any]) -> str:
    memory = _mapping(authority.get("memory", {}), "engineering authority memory")
    value = memory.get("landuse_supervision_representation")
    if authority.get("schema_version") == V3_AUTHORITY_SCHEMA:
        if value != V3_REPRESENTATION:
            raise GenerationIdentityError(
                "V3 engineering authority must freeze "
                f"memory.landuse_supervision_representation={V3_REPRESENTATION!r}"
            )
        evidence = authority.get("representation_evidence")
        if not isinstance(evidence, Mapping) or not evidence:
            raise GenerationIdentityError(
                "V3 engineering authority must bind representation_evidence"
            )
        return str(value)
    return "dense_source_landuse_one_hot_v2"


def input_identity_binding(path: Path | str) -> dict[str, str]:
    """Return the scientific input identity, not the receipt container hash."""

    receipt_path = Path(path)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GenerationIdentityError("Generator input receipt is unreadable") from exc
    schema = receipt.get("schema_version")
    if schema == INPUT_RECEIPT_SCHEMA_V1:
        return {
            "field": "inputs_receipt_sha256",
            "schema": INPUT_RECEIPT_SCHEMA_V1,
            "fingerprint": sha256_file(receipt_path),
        }
    if schema not in {INPUT_RECEIPT_SCHEMA_V2, INPUT_RECEIPT_SCHEMA_V3}:
        raise GenerationIdentityError("Generator input receipt schema differs")
    identity = receipt.get("scientific_identity")
    fingerprint = receipt.get("scientific_fingerprint")
    if not isinstance(identity, Mapping) or fingerprint != sha256_json(identity):
        raise GenerationIdentityError(
            "Generator input scientific identity fingerprint differs"
        )
    return {
        "field": "inputs_scientific_fingerprint",
        "schema": str(identity.get("schema_version", "")),
        "fingerprint": str(fingerprint),
    }


def derive_run_identity(
    output_root: str | Path,
    config: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Derive the current run identity from frozen files only.

    V3 authorities are checked against the hashes frozen in ``config``.
    Legacy authorities retain their declared hashes only to permit read-only
    diagnostics after the live authority has advanced.
    """

    root = Path(output_root).resolve()
    receipt_path = root / "inputs/receipt.json"
    if not receipt_path.is_file():
        raise GenerationIdentityError(f"Generator input receipt is missing: {receipt_path}")
    input_binding = input_identity_binding(receipt_path)

    authorities = _mapping(config.get("authorities"), "Generator authorities")
    repository = (
        Path(repo_root).resolve()
        if repo_root is not None
        else find_repo_root(__file__)
    )
    candidate_relative = str(authorities.get("candidate_registry", ""))
    engineering_relative = str(authorities.get("engineering_admission", ""))
    if not candidate_relative or not engineering_relative:
        raise GenerationIdentityError("Generator authority paths are incomplete")
    candidate_path = resolve_case_path(repository / candidate_relative)
    engineering_path = resolve_case_path(repository / engineering_relative)
    if not candidate_path.is_file() or not engineering_path.is_file():
        raise GenerationIdentityError("Generator generation authority is missing")
    candidate_declared = str(authorities.get("candidate_registry_sha256", ""))
    engineering_declared = str(authorities.get("engineering_admission_sha256", ""))
    engineering = _load_toml(engineering_path)
    authority_schema = str(engineering.get("schema_version", ""))
    representation = _representation(engineering)
    generation_schema = (
        V3_GENERATION_SCHEMA
        if authority_schema == V3_AUTHORITY_SCHEMA
        else LEGACY_GENERATION_SCHEMA
    )
    candidate_observed = sha256_file(candidate_path)
    engineering_observed = sha256_file(engineering_path)
    if generation_schema == V3_GENERATION_SCHEMA:
        if candidate_declared != candidate_observed:
            raise GenerationIdentityError("V3 candidate registry hash differs from config")
        if engineering_declared != engineering_observed:
            raise GenerationIdentityError("V3 engineering authority hash differs from config")

    payload = {
        "schema_version": generation_schema,
        "candidate_registry_sha256": candidate_declared,
        "engineering_authority_schema": authority_schema,
        "engineering_authority_sha256": engineering_declared,
        "landuse_supervision_representation": representation,
    }
    if input_binding["field"] == "inputs_receipt_sha256":
        payload["inputs_receipt_sha256"] = input_binding["fingerprint"]
    else:
        payload["inputs_identity_schema"] = input_binding["schema"]
        payload["inputs_scientific_fingerprint"] = input_binding["fingerprint"]
    return {
        **payload,
        "run_fingerprint": sha256_json(payload),
        "formal_reuse_allowed": generation_schema == V3_GENERATION_SCHEMA,
    }


def require_generation(
    document: Mapping[str, Any],
    expected: Mapping[str, Any],
    artifact: str,
) -> None:
    """Require an artifact to belong to exactly the expected V3 run."""

    if expected.get("schema_version") != V3_GENERATION_SCHEMA:
        raise GenerationIdentityError(
            f"{artifact}: legacy generation is diagnostic-only and cannot authorize formal reuse"
        )
    observed = document.get("run_identity")
    if not isinstance(observed, Mapping):
        raise GenerationIdentityError(f"{artifact}: V3 run identity is missing")
    fields = [
        "schema_version",
        "run_fingerprint",
        "candidate_registry_sha256",
        "engineering_authority_schema",
        "engineering_authority_sha256",
        "landuse_supervision_representation",
    ]
    if "inputs_scientific_fingerprint" in expected:
        fields.extend(
            ("inputs_identity_schema", "inputs_scientific_fingerprint")
        )
    else:
        fields.append("inputs_receipt_sha256")
    for field in fields:
        if observed.get(field) != expected.get(field):
            raise GenerationIdentityError(
                f"{artifact}: execution generation differs at {field}"
            )


def migration_manifest(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    disposition: str,
) -> dict[str, Any]:
    """Describe, but never perform, an artifact-generation migration."""

    if disposition not in {"archive_only", "recompute_required"}:
        raise GenerationIdentityError("migration disposition is invalid")
    return {
        "schema_version": "sg_generator_generation_migration_manifest_v1",
        "source_run_identity": dict(source),
        "target_run_identity": dict(target),
        "disposition": disposition,
        "artifact_moves_performed": False,
    }


__all__ = [
    "GenerationIdentityError",
    "LEGACY_GENERATION_SCHEMA",
    "V3_AUTHORITY_SCHEMA",
    "V3_GENERATION_SCHEMA",
    "V3_REPRESENTATION",
    "derive_run_identity",
    "input_identity_binding",
    "migration_manifest",
    "require_generation",
]
