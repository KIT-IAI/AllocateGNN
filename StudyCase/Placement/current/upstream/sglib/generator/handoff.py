"""Typed, fail-closed Generator artifact adapter for orchestration code."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sglib.core.infra.hashing import sha256_file
from sglib.core.infra.schema import validate_artifact

from sglib.core.infra.paths import resolve_case_path


class GeneratorHandoffError(ValueError):
    pass


CIVD_INDEX_SCHEMA_V1 = "sg_civd_index_v1"
CIVD_EXTENSION_INDEX_SCHEMA = "sg_civd_extension_index_v1"
CIVD_INDEX_SCHEMAS = frozenset({CIVD_INDEX_SCHEMA_V1, CIVD_EXTENSION_INDEX_SCHEMA})
CIVD_METHOD_KEYS = (
    "clustering",
    "noise_rule",
    "cluster_assignment",
    "weight_rule",
    "capacity_floor",
    "distance_floor_m",
    "station_prediction_rule",
    "station_identity",
)


def civd_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    """The registered CIVD protocol of one country: shared method keys plus profile facts."""

    method = config.get("civd")
    if not isinstance(method, Mapping):
        raise GeneratorHandoffError("Generator configuration registers no CIVD protocol")
    missing = [key for key in CIVD_METHOD_KEYS if key not in method]
    if missing:
        raise GeneratorHandoffError(f"registered CIVD protocol lacks {missing}")
    contract = config["station_contract"]
    return {
        **{key: method[key] for key in CIVD_METHOD_KEYS},
        "country": str(config["country"]["code"]),
        "working_crs": str(config["crs"]["working"]),
        "capacity_column": str(contract["capacity_column"]),
        "capacity_basis": str(contract["capacity_basis"]),
    }


def verify_civd_index(document: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    """Accept a v1 index or an extension index whose recorded protocol is the registered one."""

    schema = document.get("schema_version")
    if schema not in CIVD_INDEX_SCHEMAS:
        raise GeneratorHandoffError(f"unknown CIVD index schema: {schema}")
    code = str(config["country"]["code"])
    if document.get("country") != code or document.get("working_crs") != str(config["crs"]["working"]):
        raise GeneratorHandoffError("CIVD index country or working CRS differs from the configuration")
    if schema == CIVD_INDEX_SCHEMA_V1:
        return schema
    expected = civd_protocol(config)
    recorded = document.get("protocol")
    if not isinstance(recorded, Mapping):
        raise GeneratorHandoffError("CIVD extension index records no protocol")
    differing = sorted(key for key, value in expected.items() if recorded.get(key) != value)
    if differing:
        raise GeneratorHandoffError(f"CIVD index protocol differs from the registered protocol at {differing}")
    for entry in document.get("entries", []):
        regional = entry.get("protocol")
        if not isinstance(regional, Mapping) or regional.get("region") != entry.get("region"):
            raise GeneratorHandoffError(f"CIVD entry lacks its regional protocol: {entry.get('region')}")
        differing = sorted(key for key, value in expected.items() if regional.get(key) != value)
        if differing:
            raise GeneratorHandoffError(
                f"CIVD entry {entry.get('region')} protocol differs from the registered protocol at {differing}"
            )
    return schema


@dataclass(frozen=True)
class CandidateField:
    label: str
    family: str
    region: str
    values: np.ndarray
    qa_only: bool
    seed: int | None = None
    fold: int | None = None
    lineage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CivdView:
    region: str
    grid_cluster: np.ndarray
    station_cluster: np.ndarray
    raw_labels: np.ndarray
    probabilities: np.ndarray
    metadata: Mapping[str, Any]
    assignment: np.ndarray | None = None


@dataclass(frozen=True)
class IdrView:
    region: str
    assignment: np.ndarray
    raw_assignment: np.ndarray
    gate: Mapping[str, Any]


@dataclass(frozen=True)
class IdrMatchedView:
    candidate: str
    seed: int | None
    region: str
    assignment: np.ndarray
    raw_assignment: np.ndarray
    gate: Mapping[str, Any]


@dataclass(frozen=True)
class SweepField:
    parameter: str
    signal: str
    value: float
    seed: int
    fold: int
    region: str
    values: np.ndarray
    lineage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalVD:
    region: str
    assignment: np.ndarray
    target_ids: np.ndarray
    lineage: Mapping[str, Any]


@dataclass(frozen=True)
class PublicActivity:
    region: str
    values: np.ndarray
    source_keys: np.ndarray
    lineage: Mapping[str, Any]


@dataclass(frozen=True)
class GeneratorBundle:
    country: str
    candidates: tuple[CandidateField, ...]
    civd: tuple[CivdView, ...]
    idr_fixed: tuple[IdrView, ...]
    idr_matched: tuple[IdrMatchedView, ...]
    sweeps: tuple[SweepField, ...]
    authority_fingerprint: str
    vd: tuple[CanonicalVD, ...] = ()
    public_activity: tuple[PublicActivity, ...] = ()
    delivery_coverage: Mapping[str, Any] = field(default_factory=dict)

    def candidate(self, label: str, region: str, *, seed: int | None = None) -> CandidateField:
        matches = [item for item in self.candidates if item.label == label and item.region == region and item.seed == seed]
        if len(matches) != 1:
            raise GeneratorHandoffError(f"candidate does not resolve uniquely: {label}/{region}/seed={seed}")
        return matches[0]

    def sweep(self, parameter: str, signal: str, value: float, seed: int, region: str) -> SweepField:
        matches = [
            item
            for item in self.sweeps
            if item.parameter == parameter
            and item.signal == signal
            and float(item.value) == float(value)
            and item.seed == seed
            and item.region == region
        ]
        if len(matches) != 1:
            raise GeneratorHandoffError(
                f"sweep does not resolve uniquely: {parameter}/{signal}/{value}/seed={seed}/{region}"
            )
        return matches[0]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def load_bundle(
    root: Path | str,
    config: Mapping[str, Any],
    *,
    include_qa: bool = False,
    schemas_root: Path | str,
    repo_root: Path | str,
) -> GeneratorBundle:
    output_root = Path(root).resolve()
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise GeneratorHandoffError("root has no leaf manifest; run 07 first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = {
        item["path"]: (item["bytes"], item["sha256"])
        for leaf in manifest["leaves"].values() for item in leaf["artifacts"]
    }
    schemas = Path(schemas_root).resolve()
    authority_path = resolve_case_path(Path(repo_root).resolve() / str(config["authorities"]["candidate_registry"]))
    expected_authority = str(config["authorities"]["candidate_registry_sha256"])
    if not authority_path.is_file() or sha256_file(authority_path) != expected_authority:
        raise GeneratorHandoffError("candidate authority fingerprint mismatch")
    index_paths = [output_root / "candidates/candidate_index.json"]
    if include_qa:
        index_paths.append(output_root / "candidates/candidate_qa_index.json")
    entries = []
    for index_path in index_paths:
        validate_artifact(index_path, schemas / "candidate_index.toml")
        entries.extend(json.loads(index_path.read_text(encoding="utf-8"))["entries"])
    candidates = []
    identities = set()
    for entry in entries:
        path = output_root / entry["path"]
        validate_artifact(path, schemas / "candidate_field.toml")
        if entry["path"] not in artifacts or artifacts[entry["path"]][1] != entry["sha256"]:
            raise GeneratorHandoffError(f"candidate hash mismatch: {path}")
        arrays = _load_npz(path)
        if not np.array_equal(arrays["grid_row"], np.arange(len(arrays["data"]))):
            raise GeneratorHandoffError(f"candidate grid identity mismatch: {path}")
        identity = (entry["label"], entry["region"], entry.get("seed"), entry.get("fold"))
        if identity in identities:
            raise GeneratorHandoffError(f"duplicate candidate identity: {identity}")
        identities.add(identity)
        candidates.append(
            CandidateField(
                label=str(entry["label"]),
                family=str(entry["family"]),
                region=str(entry["region"]),
                values=arrays["data"],
                qa_only=bool(entry.get("qa_only", False)),
                seed=int(entry["seed"]) if entry.get("seed") is not None else None,
                fold=int(entry["fold"]) if entry.get("fold") is not None else None,
                lineage=dict(entry),
            )
        )
    civd_items = []
    civd_index = output_root / "civd/index.json"
    if civd_index.is_file():
        validate_artifact(civd_index, schemas / "civd_index.toml")
        civd_document = json.loads(civd_index.read_text(encoding="utf-8"))
        verify_civd_index(civd_document, config)
        for entry in civd_document["entries"]:
            path = output_root / entry["path"]
            validate_artifact(path, schemas / "civd.toml")
            if entry["path"] not in artifacts or artifacts[entry["path"]][1] != entry["sha256"]:
                raise GeneratorHandoffError(f"CIVD hash mismatch: {path}")
            arrays = _load_npz(path)
            civd_items.append(CivdView(str(entry["region"]), arrays["grid_cluster"], arrays["station_cluster"], arrays["raw_labels"], arrays["probabilities"], entry, arrays["assignment"]))
    idr_items = []
    idr_index = output_root / "idr_fixed/index.json"
    if idr_index.is_file():
        validate_artifact(idr_index, schemas / "idr_fixed_index.toml")
        for entry in json.loads(idr_index.read_text(encoding="utf-8"))["entries"]:
            path = output_root / entry["path"]
            validate_artifact(path, schemas / "idr_assignment.toml")
            if entry["path"] not in artifacts or artifacts[entry["path"]][1] != entry["sha256"]:
                raise GeneratorHandoffError(f"IDR-fixed hash mismatch: {path}")
            arrays = _load_npz(path)
            idr_items.append(IdrView(str(entry["region"]), arrays["assignment"], arrays["raw_idr_assignment"], entry))
    matched_items = []
    matched_index = output_root / "idr_matched/index.json"
    matched_identities = set()
    if matched_index.is_file():
        validate_artifact(matched_index, schemas / "idr_matched_index.toml")
        for entry in json.loads(matched_index.read_text(encoding="utf-8"))["entries"]:
            path = output_root / entry["path"]
            validate_artifact(path, schemas / "idr_assignment.toml")
            if entry["path"] not in artifacts or artifacts[entry["path"]][1] != entry["sha256"]:
                raise GeneratorHandoffError(f"IDR-matched hash mismatch: {path}")
            identity = (str(entry["candidate"]), int(entry["seed"]), str(entry["region"]))
            if identity in matched_identities:
                raise GeneratorHandoffError(f"duplicate IDR-matched identity: {identity}")
            matched_identities.add(identity)
            arrays = _load_npz(path)
            matched_items.append(
                IdrMatchedView(
                    candidate=identity[0],
                    seed=identity[1] or None,
                    region=identity[2],
                    assignment=arrays["assignment"],
                    raw_assignment=arrays["raw_idr_assignment"],
                    gate=entry,
                )
            )
    sweep_items = []
    sweep_identities = set()
    for index_path in sorted((output_root / "sweeps").glob("index_*.json")):
        validate_artifact(index_path, schemas / "sweep_index.toml")
        document = json.loads(index_path.read_text(encoding="utf-8"))
        parameter = str(document["parameter"])
        for entry in document["entries"]:
            path = output_root / entry["path"]
            validate_artifact(path, schemas / "candidate_field.toml")
            if entry["path"] not in artifacts or artifacts[entry["path"]][1] != entry["sha256"]:
                raise GeneratorHandoffError(f"sweep hash mismatch: {path}")
            identity = (
                parameter,
                str(entry["signal"]),
                float(entry["value"]),
                int(entry["seed"]),
                str(entry["region"]),
            )
            if identity in sweep_identities:
                raise GeneratorHandoffError(f"duplicate sweep identity: {identity}")
            sweep_identities.add(identity)
            arrays = _load_npz(path)
            if not np.array_equal(arrays["grid_row"], np.arange(len(arrays["data"]))):
                raise GeneratorHandoffError(f"sweep grid identity mismatch: {path}")
            sweep_items.append(
                SweepField(
                    parameter=identity[0],
                    signal=identity[1],
                    value=identity[2],
                    seed=identity[3],
                    fold=int(entry["fold"]),
                    region=identity[4],
                    values=arrays["data"],
                    lineage=dict(entry),
                )
            )
    country = str(config["country"]["code"])
    canonical_views, activity_views = [], []
    for component in ("assignments", "public_activity"):
        document = json.loads((output_root / "static" / component / "index.json").read_text(encoding="utf-8"))
        regions = [e["region"] for e in document["regions"]]
        if len(regions) != len(set(regions)) or set(regions) != set(config["regions"]):
            raise GeneratorHandoffError(f"{component}: 区域交付不完整")
        for entry in document["regions"]:
            path = output_root / entry["path"]
            if entry["path"] not in artifacts or artifacts[entry["path"]][1] != entry["sha256"]:
                raise GeneratorHandoffError(f"{component}: 内容不符")
            arrays = _load_npz(path)
            if component == "assignments":
                canonical_views.append(CanonicalVD(entry["region"], arrays["assignment"], arrays["station_id"], entry))
            else:
                activity_views.append(PublicActivity(entry["region"], arrays["data"], arrays["source_keys"], {
                    **entry, "shape_basis": "GPM_source_internal_shape",
                    "mass_basis": "public_source_composition_sum",
                    "source_total_invariance_max_abs": float(arrays["source_total_invariance_max_abs"]),
                }))
    audit = json.loads((output_root / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or "delivery_coverage" not in audit:
        raise GeneratorHandoffError("005 完整交付 audit 尚未通过")
    return GeneratorBundle(
        country,
        tuple(candidates),
        tuple(civd_items),
        tuple(idr_items),
        tuple(matched_items),
        tuple(sweep_items),
        expected_authority,
        tuple(canonical_views), tuple(activity_views), audit["delivery_coverage"],
    )


def oof_fold(regions: list[str], seed: int, n_folds: int, region: str) -> int:
    from sklearn.model_selection import KFold

    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    matches = [index + 1 for index, (_, test) in enumerate(splitter.split(regions)) if region in [regions[i] for i in test]]
    if len(matches) != 1:
        raise GeneratorHandoffError(f"OOF fold is not unique: {region}/seed{seed}")
    return matches[0]


__all__ = [
    "CandidateField",
    "CivdView",
    "GeneratorBundle",
    "GeneratorHandoffError",
    "IdrMatchedView",
    "IdrView",
    "SweepField",
    "load_bundle",
    "oof_fold",
]
