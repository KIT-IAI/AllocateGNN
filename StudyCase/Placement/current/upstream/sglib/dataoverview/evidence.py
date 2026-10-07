"""Typed, hashed evidence artifacts attached to a DataOverview handoff."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from sglib.core.infra.hashing import sha256_file
from sglib.core.infra.schema import validate_artifact


class HandoffEvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class HandoffEvidence:
    name: str
    path: Path
    repo_relative: str
    sha256: str
    bytes: int
    schema: str | None
    document: Mapping[str, Any] | None
    formal_required: bool
    required_status: str | None


def load_handoff_evidence(
    repo_root: Path | str,
    configuration: Mapping[str, Any],
    *,
    schemas_root: Path | str | None = None,
) -> dict[str, HandoffEvidence]:
    root = Path(repo_root).resolve()
    configured = configuration.get("handoff_artifacts", {})
    if not isinstance(configured, Mapping):
        raise HandoffEvidenceError("handoff_artifacts must be a TOML table")
    schemas = Path(schemas_root).resolve() if schemas_root is not None else None
    result: dict[str, HandoffEvidence] = {}
    for name, raw_spec in configured.items():
        spec = {"path": raw_spec} if isinstance(raw_spec, str) else raw_spec
        if not isinstance(spec, Mapping):
            raise HandoffEvidenceError(f"handoff artifact {name!r} is not a table")
        unknown = set(spec) - {
            "path",
            "schema",
            "formal_required",
            "required_status",
        }
        if unknown or "path" not in spec:
            raise HandoffEvidenceError(
                f"handoff artifact {name!r} has invalid keys: {sorted(unknown)}"
            )
        path = (root / str(spec["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise HandoffEvidenceError(
                f"handoff artifact {name!r} escapes repository"
            ) from exc
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"handoff evidence is missing or empty: {path}")
        schema_name = str(spec["schema"]) if spec.get("schema") else None
        if schema_name is not None:
            if schemas is None:
                raise HandoffEvidenceError(
                    f"handoff artifact {name!r} declares a schema without schemas_root"
                )
            validate_artifact(path, schemas / schema_name)
        document: Mapping[str, Any] | None = None
        if path.suffix.lower() == ".json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, Mapping):
                raise HandoffEvidenceError(
                    f"handoff JSON evidence {name!r} must be an object"
                )
            document = dict(loaded)
        required_status = (
            str(spec["required_status"]) if spec.get("required_status") else None
        )
        if required_status is not None and document is None:
            raise HandoffEvidenceError(
                f"handoff artifact {name!r} requires status but is not JSON"
            )
        result[str(name)] = HandoffEvidence(
            name=str(name),
            path=path,
            repo_relative=path.relative_to(root).as_posix(),
            sha256=sha256_file(path),
            bytes=path.stat().st_size,
            schema=schema_name,
            document=document,
            formal_required=bool(spec.get("formal_required", False)),
            required_status=required_status,
        )
    return result


__all__ = ["HandoffEvidence", "HandoffEvidenceError", "load_handoff_evidence"]
