"""Leaf hashing and comparison, without registry dependencies or writes.

``build`` consumes file records (paths already hashed by ``read_current``), so
existing inventory artifacts can be fingerprinted without consulting disk.
``verify`` reads only the paths independently supplied by the stage adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, TypedDict

from .hashing import sha256_file, sha256_json


SCHEMA_VERSION = "sg_leaf_manifest_v1"


class GateReport(TypedDict):
    stage: str
    country: str
    status: str
    leaves: dict[str, dict[str, Any]]
    baseline_root_fingerprint: str
    root_fingerprint: str
    unclaimed: list[str]


def build(
    stage: str,
    country: str,
    leaves: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Fingerprint existing artifact records; preserve their declared order."""

    result = {}
    for leaf_id, records in sorted(leaves.items()):
        artifacts = [dict(record) for record in records]
        result[leaf_id] = {
            "fingerprint": sha256_json(artifacts),
            "artifacts": artifacts,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "country": country,
        "leaves": result,
        "root_fingerprint": sha256_json({
            leaf_id: leaf["fingerprint"] for leaf_id, leaf in result.items()
        }),
    }


def read_current(
    leaves: Mapping[str, Iterable[str | Path]], root: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Hash the supplied root-relative paths; absent files remain absent."""

    result = {}
    for leaf_id, paths in leaves.items():
        records = []
        for relative in paths:
            path = root / relative
            if path.is_file():
                records.append({
                    "path": Path(relative).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
        result[leaf_id] = records
    return result


def verify(
    manifest: Mapping[str, Any],
    current: Mapping[str, Iterable[str | Path]],
    root: Path,
) -> GateReport:
    """Compare a baseline with independently enumerated current file sets.

    DataOverview supplies precisely the inventory paths, so unrelated files
    never become extras. Generator supplies its complete current partitions.
    """

    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported leaf manifest: {manifest['schema_version']}")
    observed = build(manifest["stage"], manifest["country"], read_current(current, root))
    reports = {}
    for leaf_id in sorted(set(manifest["leaves"]) | set(observed["leaves"])):
        baseline = manifest["leaves"].get(leaf_id)
        actual = observed["leaves"].get(leaf_id)
        before = {item["path"]: item for item in baseline["artifacts"]} if baseline else {}
        after = {item["path"]: item for item in actual["artifacts"]} if actual else {}
        missing = sorted(before.keys() - after.keys())
        extra = sorted(after.keys() - before.keys())
        changed = sorted(
            path for path in before.keys() & after.keys()
            if before[path]["sha256"] != after[path]["sha256"]
            or before[path]["bytes"] != after[path]["bytes"]
        )
        reports[leaf_id] = {
            "status": "FAIL" if missing or extra or changed or baseline is None or actual is None else "PASS",
            "missing": missing,
            "extra": extra,
            "changed": changed,
            "baseline_fingerprint": baseline["fingerprint"] if baseline else None,
            "fingerprint": actual["fingerprint"] if actual else None,
        }
    return {
        "stage": manifest["stage"],
        "country": manifest["country"],
        "status": "PASS" if all(leaf["status"] == "PASS" for leaf in reports.values()) else "FAIL",
        "leaves": reports,
        "baseline_root_fingerprint": manifest["root_fingerprint"],
        "root_fingerprint": observed["root_fingerprint"],
        "unclaimed": [],
    }
