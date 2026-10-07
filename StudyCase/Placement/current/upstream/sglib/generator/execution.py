"""Artifact-producing Generator operations with caller-injected handoff/config/root."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np

from sglib.core.infra.artifacts import atomic_json, atomic_npz
from sglib.core.infra.hashing import sha256_file, sha256_json
from sglib.core.infra.paths import find_repo_root

from .inputs import GeneratorDataBundle, assemble_from_handoff
from .materialize import (
    materialize_assignment,
    materialize_candidate,
    materialize_civd,
    materialize_idr_fixed,
    materialize_idr_matched,
    public_activity_field,
    validate_field,
)
from .weighter.native import GPMWeighter, UniformWeighter, materialize_equal_grid
from .weighter.correction import factor_bundle
from .weighter.correction import apply_standard_multiplicative, compute_factors, compute_prox_scores


class GeneratorExecutionError(RuntimeError):
    pass


def _formal_engineering_pre_submission(
    handoff: Any,
    bundle: GeneratorDataBundle,
    config: Mapping[str, Any],
    *,
    bundle_path: Path,
) -> dict[str, Any] | None:
    """Evaluate the frozen real-active-cell gate for opted-in countries.

    The GNN graph builder defines active agents and source-agent supervision
    edges as cells with ``covered_mask``.  This check deliberately runs after
    feature loading, but before any training task can be prepared.  Every
    binding metric is recomputed from the pinned bundle.  The graph-size check
    uses the authority's conservative directed-edge envelope, so it remains
    fail-closed without materialising a training cache during preflight.
    """

    checks = config.get("checks", {})
    if not isinstance(checks, Mapping) or not bool(
        checks.get("requires_engineering_admission", False)
    ):
        return None
    contract = checks.get("admission_contract")
    if not contract:
        raise GeneratorExecutionError(
            "formal engineering admission requires a configured v3 contract"
        )
    evidence_name = checks.get("engineering_admission_evidence")
    legacy_evidence = (
        getattr(handoff, "evidence", {}).get(str(evidence_name))
        if evidence_name
        else None
    )
    legacy_records = (
        legacy_evidence.document.get("regions")
        if legacy_evidence is not None and legacy_evidence.document is not None
        else None
    )
    by_region = {
        str(record.get("region")): record
        for record in legacy_records or []
        if isinstance(record, Mapping) and record.get("region")
    }

    from sglib.core.infra.engineering_admission import (
        EngineeringAdmissionError,
        load_engineering_admission,
        validate_pre_submission,
    )

    repo_root = find_repo_root(handoff.profile.source_path)
    try:
        admission = load_engineering_admission(repo_root, str(contract))
    except EngineeringAdmissionError as exc:
        raise GeneratorExecutionError(f"formal engineering authority is invalid: {exc}") from exc

    region_results: list[dict[str, Any]] = []
    for region in bundle.regions:
        grid = bundle.grids[region][0]
        sources = bundle.source_regions[region]
        targets = bundle.stations[region]
        source_counts = grid[bundle.source_column].astype(str).value_counts()
        source_ids = set(sources[bundle.source_column].astype(str))
        if set(source_counts.index) != source_ids:
            raise GeneratorExecutionError(
                f"{region}: formal grid/source coverage differs from the source authority"
            )
        assignment = materialize_assignment(
            grid,
            targets,
            working_crs=str(bundle.params["crs"]["working"]),
        )
        target_counts = np.bincount(assignment, minlength=len(targets))
        observed = {
            "n_sources": len(sources),
            "n_targets": len(targets),
            "n_cells": len(grid),
            "min_cells_per_source": int(source_counts.min()),
            "min_cells_per_target": int(target_counts.min()),
            "empty_targets": int(np.count_nonzero(target_counts == 0)),
        }
        active_observed = int(np.count_nonzero(bundle.region_inputs[region].covered_mask))
        region_input = bundle.region_inputs[region]
        proximity = getattr(region_input, "metadata", {}).get("proximity", {})
        cdist_peak = int(proximity.get("peak_workspace_bytes", 0))
        dry = by_region.get(region, {})
        if cdist_peak <= 0:
            cdist_peak = int(dry.get("cdist_peak_workspace_bytes", 0))
        if cdist_peak <= 0 and dry.get("cdist_peak_workspace_mib") is not None:
            cdist_peak = round(float(dry["cdist_peak_workspace_mib"]) * 2**20)
        if cdist_peak <= 0:
            cdist_peak = max(
                int(dry.get("proximity_chunk_peak_bytes", 0)),
                int(dry.get("proximity_score_chunk_peak_bytes", 0)),
            )
        if cdist_peak <= 0:
            raise GeneratorExecutionError(
                f"{region}: pinned bundle lacks observed cdist workspace"
            )
        edge_envelope = int(admission.graph["directed_edges_per_cell_envelope"])
        total_edges = int(
            dry.get(
                "total_edges_directed",
                dry.get("directed_graph_edges", len(grid) * edge_envelope),
            )
        )
        metrics = {
            **{key: value for key, value in observed.items() if key != "empty_targets"},
            "total_edges_directed": int(total_edges),
            "active_agent_nodes_upper_bound": len(grid),
            "active_agent_nodes_observed": active_observed,
            "cdist_peak_workspace_bytes": int(cdist_peak),
        }
        try:
            decision = validate_pre_submission(admission, metrics)
        except EngineeringAdmissionError as exc:
            raise GeneratorExecutionError(f"{region}: {exc}") from exc
        region_results.append(
            {
                "region": region,
                **metrics,
                "empty_targets": observed["empty_targets"],
                "active_source_agent_edges_observed": active_observed,
                "graph_edge_metric_basis": (
                    "n_cells * authority.directed_edges_per_cell_envelope"
                ),
                "hard_checks": decision["hard_checks"],
                "memory": decision["memory"],
                "status": "PASS",
            }
        )
    return {
        "schema_version": "sg_generator_formal_engineering_pre_submission_v1",
        "status": "PASS",
        "formal": True,
        "country": bundle.country,
        "active_agent_definition": "count_nonzero(covered_mask); identical to learned graph construction",
        "input_bundle": {
            "path": "bundle.pkl",
            "sha256": sha256_file(bundle_path),
            "bytes": bundle_path.stat().st_size,
        },
        "source_evidence": (
            {
                "name": str(evidence_name),
                "path": legacy_evidence.repo_relative,
                "sha256": legacy_evidence.sha256,
            }
            if legacy_evidence is not None
            else "Generator receipt DataOverview inventory"
        ),
        "admission_contract": {
            "path": admission.country_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(admission.country_path),
        },
        "shared_authority": {
            "path": admission.authority_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(admission.authority_path),
        },
        "regions": region_results,
    }


def _atomic_pickle(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    try:
        with temporary.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _load_bundle(root: Path) -> GeneratorDataBundle:
    path = root / "inputs" / "bundle.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"prepare Generator inputs first: {path}")
    with path.open("rb") as stream:
        bundle = pickle.load(stream)
    if not isinstance(bundle, GeneratorDataBundle):
        raise GeneratorExecutionError("unknown Generator input bundle")
    return bundle


def prepare_inputs(
    handoff: Any,
    config: Mapping[str, Any],
    root: Path | str,
    *,
    selected_regions: list[str] | None = None,
    worker_params: Mapping[str, Any] | None = None,
    require_formal_evidence: bool = True,
    inventory_path: Path | str | None = None,
) -> Path:
    output_root = Path(root).resolve()
    evidence_records = {}
    for name, item in getattr(handoff, "evidence", {}).items():
        observed_status = (
            item.document.get("status") if item.document is not None else None
        )
        if (
            require_formal_evidence
            and item.formal_required
            and item.required_status is not None
            and observed_status != item.required_status
        ):
            raise GeneratorExecutionError(
                f"DataOverview evidence {name!r} has status {observed_status!r}; "
                f"required={item.required_status!r}"
            )
        evidence_records[name] = {
            "path": item.repo_relative,
            "sha256": item.sha256,
            "bytes": item.bytes,
            "formal_required": item.formal_required,
            "required_status": item.required_status,
            "observed_status": observed_status,
        }
    repo_root = find_repo_root(handoff.profile.source_path)
    inventory_path = (Path(inventory_path).resolve() if inventory_path is not None else
                      repo_root / "results/1_DataOverview" / handoff.profile.directory / "data_inventory.json")
    inventory_root = repo_root if inventory_path.is_relative_to(repo_root) else output_root
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        if inventory["fingerprint"] != sha256_json(inventory["artifacts"]):
            raise ValueError("inventory fingerprint differs from artifacts")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise GeneratorExecutionError(
            f"DataOverview inventory validation failed: {exc}"
        ) from exc
    bundle = assemble_from_handoff(handoff, config, selected_regions=selected_regions)
    bundle_path = _atomic_pickle(output_root / "inputs" / "bundle.pkl", bundle)
    worker_document = dict(worker_params or config)
    worker_document["regions"] = list(bundle.regions)
    worker_path = atomic_json(
        worker_document, output_root / "inputs" / "worker_params.json"
    )
    pre_submission = None
    pre_submission_document = None
    if require_formal_evidence:
        document = _formal_engineering_pre_submission(
            handoff,
            bundle,
            config,
            bundle_path=bundle_path,
        )
        if document is not None:
            pre_submission_document = document
            path = atomic_json(
                document,
                output_root / "inputs" / "engineering_pre_submission.json",
            )
            pre_submission = {
                "path": path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "status": document["status"],
            }
    from .config import (
        SCIENTIFIC_CONFIG_PROJECTION_SCHEMA,
        scientific_config_fingerprint,
    )

    stable_evidence = {
        name: {
            "sha256": item["sha256"],
            "formal_required": item["formal_required"],
            "required_status": item["required_status"],
            "observed_status": item["observed_status"],
        }
        for name, item in evidence_records.items()
    }
    admission_identity = None
    if pre_submission_document is not None:
        admission_identity = {
            key: value
            for key, value in pre_submission_document.items()
            if key
            not in {
                "input_bundle",
                "admission_contract",
                "shared_authority",
                "source_evidence",
            }
        }
    config_fingerprint = scientific_config_fingerprint(
        config, repo_root=repo_root
    )
    scientific_identity = {
        "schema_version": "sg_generator_inputs_scientific_identity_v3",
        "country": bundle.country,
        "regions": list(bundle.regions),
        "config_fingerprint_schema": SCIENTIFIC_CONFIG_PROJECTION_SCHEMA,
        "config_fingerprint": config_fingerprint,
        "worker_params_sha256": sha256_file(worker_path),
        "dataoverview_country_fingerprint": inventory["fingerprint"],
        "dataoverview_evidence": stable_evidence,
        "engineering_admission": admission_identity,
    }
    receipt = {
        "schema_version": "sg_generator_inputs_receipt_v3",
        "country": bundle.country,
        "regions": list(bundle.regions),
        "config_fingerprint_schema": SCIENTIFIC_CONFIG_PROJECTION_SCHEMA,
        "config_fingerprint": config_fingerprint,
        "scientific_identity": scientific_identity,
        "scientific_fingerprint": sha256_json(scientific_identity),
        "dataoverview_inventory_schema": handoff.inventory["schema_version"],
        "dataoverview_inventory": {
            "path": inventory_path.relative_to(inventory_root).as_posix(),
            "path_base": "repo_root" if inventory_root == repo_root else "generator_root",
            "schema": inventory["schema_version"],
            "fingerprint": inventory["fingerprint"],
            "artifact_count": len(inventory["artifacts"]),
        },
        "dataoverview_evidence": evidence_records,
        "bundle": {"path": "bundle.pkl", "sha256": sha256_file(bundle_path), "bytes": bundle_path.stat().st_size},
    }
    if pre_submission is not None:
        receipt["engineering_pre_submission"] = pre_submission
    return atomic_json(receipt, output_root / "inputs" / "receipt.json")


def _field_record(path: Path, root: Path) -> dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def generate_static_component(root: Path | str, component: str) -> Path:
    output_root = Path(root).resolve()
    bundle = _load_bundle(output_root)
    if component not in {"uniform", "gpm", "proximity", "assignments", "public_activity"}:
        raise GeneratorExecutionError(f"unknown static component: {component}")
    records = []
    features = bundle.params["features"]
    for region in bundle.regions:
        grid, _ = bundle.grids[region]
        source = bundle.source_regions[region]
        stations = bundle.stations[region]
        destination = output_root / "static" / component / f"{region}.npz"
        if component == "uniform":
            values = UniformWeighter({"source_column": bundle.source_column, "demand_column": "Demand (MVA)"}).compute(grid, stations, source).weights
            validate_field(values, grid, source, source_column=bundle.source_column)
            atomic_npz(destination, data=values)
        elif component == "gpm":
            values = GPMWeighter(
                {
                    "mode": bundle.params["gpm"]["mode"],
                    "proportion_columns": features["landuse_columns"],
                    "source_feature_columns": features["source_columns"],
                    "source_column": bundle.source_column,
                    "demand_column": "Demand (MVA)",
                }
            ).compute(grid, stations, source).weights
            validate_field(values, grid, source, source_column=bundle.source_column)
            factors = factor_bundle(grid, bundle.ntl[region], bundle.proximity[region], source_key=bundle.source_column)
            atomic_npz(destination, data=values, N=factors["N"], P=factors["P"], NP=factors["NP"])
        elif component == "proximity":
            atomic_npz(destination, data=bundle.proximity[region])
        elif component == "assignments":
            assignment = materialize_assignment(grid, stations, working_crs=bundle.params["crs"]["working"])
            counts = np.bincount(assignment, minlength=len(stations))
            if (counts == 0).any():
                raise GeneratorExecutionError(f"{region}: canonical VD has an empty target")
            atomic_npz(destination, assignment=assignment, station_id=stations["station_id"].to_numpy(dtype=str))
        else:
            gpm_path = output_root / "static" / "gpm" / f"{region}.npz"
            if not gpm_path.is_file():
                raise FileNotFoundError("generate static gpm before public_activity")
            with np.load(gpm_path, allow_pickle=False) as archive:
                gpm = archive["data"]
            activity, invariance = public_activity_field(gpm, grid[bundle.source_column].astype(str).to_numpy(), source, source_column=bundle.source_column)
            atomic_npz(destination, data=activity, source_keys=grid[bundle.source_column].to_numpy(dtype=str), source_total_invariance_max_abs=np.asarray(invariance))
        records.append({"region": region, **_field_record(destination, output_root)})
    return atomic_json(
        {
            "schema_version": "sg_generator_static_index_v1",
            "country": bundle.country,
            "component": component,
            "working_crs": bundle.params["crs"]["working"],
            "regions": records,
        },
        output_root / "static" / component / "index.json",
    )


def materialize_family(
    root: Path | str,
    family: str,
    candidate_registry: Mapping[str, Any],
) -> Path:
    output_root = Path(root).resolve()
    bundle = _load_bundle(output_root)
    definitions = [item for item in candidate_registry["candidates"] if item["family"] == family and item.get("materialize", True)]
    if not definitions:
        raise GeneratorExecutionError(f"candidate family is absent: {family}")
    base_name = {"Uni": "Uni", "GPM": "GPM", "Equal": "EqualGrid"}.get(family)
    if base_name is None:
        return _materialize_learned_family(output_root, bundle, family, definitions)
    entries = []
    for region in bundle.regions:
        grid, _ = bundle.grids[region]
        source = bundle.source_regions[region]
        with np.load(output_root / "static/uniform" / f"{region}.npz", allow_pickle=False) as archive:
            uniform = archive["data"]
        with np.load(output_root / "static/gpm" / f"{region}.npz", allow_pickle=False) as archive:
            gpm = archive["data"]
            factors = {key: archive[key] for key in ("N", "P", "NP")}
        with np.load(output_root / "static/assignments" / f"{region}.npz", allow_pickle=False) as archive:
            assignment = archive["assignment"]
        equal = materialize_equal_grid(grid, uniform, assignment, source_column=bundle.source_column)
        base = {"Uni": uniform, "GPM": gpm, "EqualGrid": equal}[base_name]
        for definition in definitions:
            label = str(definition["label"])
            field = materialize_candidate(definition, base, factors, grid, source, source_column=bundle.source_column)
            destination = output_root / "candidates" / label / f"{region}.npz"
            atomic_npz(destination, data=field, grid_row=np.arange(len(grid), dtype=np.int64))
            entries.append(
                {
                    "label": label,
                    "family": family,
                    "region": region,
                    "qa_only": bool(definition.get("qa_only", False)),
                    **_field_record(destination, output_root),
                }
            )
    index = atomic_json(
        {"schema_version": "sg_candidate_family_index_v1", "country": bundle.country, "family": family, "entries": entries},
        output_root / "candidates" / f"index_{family}.json",
    )
    return index


def _materialize_learned_family(
    root: Path,
    bundle: GeneratorDataBundle,
    family: str,
    definitions: list[Mapping[str, Any]],
) -> Path:
    family_key = family.lower()
    country = bundle.country.upper()
    canonical_groups = {
        "mlp_baseline": f"B-{country}-MLP",
        "gnn_baseline": f"B-{country}-GNN",
        "gnn_prior_ntl": f"P-{country}-N",
        "gnn_prior_proximity": f"P-{country}-P",
        "gnn_prior_ntl_proximity": f"P-{country}-NP",
        "gnn_fusion_ntl": f"F-{country}-N",
        "gnn_fusion_proximity": f"F-{country}-P",
        "gnn_fusion_ntl_proximity": f"F-{country}-NP",
    }
    allowed_groups = set(canonical_groups.values())
    available: dict[tuple[str, int, int, str], tuple[np.ndarray, Mapping[str, Any]]] = {}
    for task_path in sorted((root / "inference/tasks").glob("*/infer-*.json")):
        task = json.loads(task_path.read_text(encoding="utf-8"))
        if task.get("family") != family_key or task.get("group") not in allowed_groups:
            continue
        output = root.parents[1] / task["output_results_relative"]
        completion = output / "inference_completion.json"
        if not completion.is_file():
            continue
        for region in task["regions"]:
            metadata_path = output / "fields" / f"{region}.json"
            field_path = output / "fields" / f"{region}.npz"
            metadata = {
                **json.loads(metadata_path.read_text(encoding="utf-8")),
                "group": task["group"],
                "inference_node_id": task["chain_commitment"]["node_id"],
                "inference_field_sha256": sha256_file(field_path),
            }
            with np.load(field_path, allow_pickle=False) as archive:
                field = np.array(archive["data"], copy=True)
            available[(str(task["config"]), int(task["seed"]), int(task["fold"]), region)] = (field, metadata)
    if not available:
        raise GeneratorExecutionError(f"no completed inference fields for {family}")
    training_config_map = {
        "mlp_baseline": ("baseline", canonical_groups["mlp_baseline"]),
        "gnn_baseline": ("baseline", canonical_groups["gnn_baseline"]),
        "gnn_prior_ntl": ("prior_ntl", canonical_groups["gnn_prior_ntl"]),
        "gnn_prior_proximity": ("prior_proximity", canonical_groups["gnn_prior_proximity"]),
        "gnn_prior_ntl_proximity": ("prior_ntl_proximity", canonical_groups["gnn_prior_ntl_proximity"]),
        "gnn_fusion_ntl": ("fusion_ntl", canonical_groups["gnn_fusion_ntl"]),
        "gnn_fusion_proximity": ("fusion_proximity", canonical_groups["gnn_fusion_proximity"]),
        "gnn_fusion_ntl_proximity": ("fusion_ntl_proximity", canonical_groups["gnn_fusion_ntl_proximity"]),
    }
    entries = []
    for definition in definitions:
        required_config, required_group = training_config_map[str(definition["training_config"])]
        for (config_name, seed, fold, region), (base, metadata) in available.items():
            if config_name != required_config or metadata.get("group") != required_group or metadata.get("role") != "TEST":
                continue
            grid, _ = bundle.grids[region]
            source = bundle.source_regions[region]
            with np.load(root / "static/gpm" / f"{region}.npz", allow_pickle=False) as archive:
                factors = {key: archive[key] for key in ("N", "P", "NP")}
            operator = str(definition["operator"])
            field = (
                materialize_candidate(definition, base, factors, grid, source, source_column=bundle.source_column)
                if operator in {"base", "multiply", "add"}
                else validate_field(base, grid, source, source_column=bundle.source_column)
            )
            label = str(definition["label"])
            destination = root / "candidates" / label / f"seed_{seed}" / f"{region}.npz"
            atomic_npz(destination, data=field, grid_row=np.arange(len(grid), dtype=np.int64))
            entries.append(
                {
                    "label": label,
                    "family": family,
                    "region": region,
                    "seed": seed,
                    "fold": fold,
                    "inference_node_id": metadata["inference_node_id"],
                    "inference_field_sha256": metadata["inference_field_sha256"],
                    "qa_only": bool(definition.get("qa_only", False)),
                    **_field_record(destination, root),
                }
            )
    if not entries:
        raise GeneratorExecutionError(f"no OOF inference fields resolve for {family}")
    index = atomic_json(
        {"schema_version": "sg_candidate_family_index_v1", "country": bundle.country, "family": family, "entries": entries},
        root / "candidates" / f"index_{family}.json",
    )
    return index


def finalize_candidate_indexes(root: Path | str) -> Path:
    """五个 family 的唯一总索引写者；完整集合通过后才发布。"""
    from .delivery import candidate_entries

    root = Path(root).resolve()
    bundle = _load_bundle(root)
    entries = candidate_entries(root, bundle, family_indexes=True)
    formal = [entry for entry in entries if not entry["qa_only"]]
    qa = [entry for entry in entries if entry["qa_only"]]
    atomic_json({"schema_version": "sg_candidate_qa_index_v1", "entries": qa}, root / "candidates/candidate_qa_index.json")
    return atomic_json({"schema_version": "sg_candidate_index_v1", "entries": formal}, root / "candidates/candidate_index.json")


def _audit_qa_candidate_index(
    output_root: Path,
    bundle: GeneratorDataBundle,
    formal_entries: list[Mapping[str, Any]],
) -> tuple[int, list[str]]:
    """Verify the materialized QA-only side index against current authority.

    QA candidates are deliberately excluded from the formal scientific index and
    from IDR matching, but they are still required artifacts.  Checking only for
    the side-index file allowed a stale, empty index to pass a formal audit after
    the registry added new QA identities.
    """

    failures: list[str] = []
    authorities = bundle.params.get("authorities", {})
    registry_relative = authorities.get("candidate_registry")
    if not isinstance(registry_relative, str) or not registry_relative:
        return 0, ["candidate registry authority is missing"]
    registry_path = find_repo_root(__file__) / registry_relative
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    qa_labels = {
        str(item["label"])
        for item in registry.get("candidates", [])
        if item.get("materialize") is True and item.get("qa_only") is True
    }
    expected = {(label, region) for label in qa_labels for region in bundle.regions}
    index_path = output_root / "candidates/candidate_qa_index.json"
    if not index_path.is_file():
        return 0, ["missing QA-only candidate index"]
    document = json.loads(index_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "sg_candidate_qa_index_v1":
        failures.append("QA-only candidate index schema differs")
    entries = document.get("entries")
    if not isinstance(entries, list):
        return 0, [*failures, "QA-only candidate entries are invalid"]
    observed: list[tuple[str, str]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            failures.append("QA-only candidate entry is invalid")
            continue
        identity = (str(entry.get("label")), str(entry.get("region")))
        observed.append(identity)
        if entry.get("qa_only") is not True:
            failures.append(f"QA-only flag differs: {identity[0]}/{identity[1]}")
        path_value = entry.get("path")
        if not isinstance(path_value, str):
            failures.append(f"QA-only candidate path is missing: {identity[0]}/{identity[1]}")
            continue
        artifact = output_root / path_value
        if not artifact.is_file() or sha256_file(artifact) != entry.get("sha256"):
            failures.append(f"QA-only candidate hash mismatch: {path_value}")
    if len(observed) != len(set(observed)):
        failures.append("QA-only candidate identities are duplicated")
    observed_set = set(observed)
    if observed_set != expected:
        failures.append(
            "QA-only candidate identity set differs: "
            f"missing={sorted(expected - observed_set)} extra={sorted(observed_set - expected)}"
        )
    formal_labels = {str(entry.get("label")) for entry in formal_entries}
    overlap = sorted(formal_labels & qa_labels)
    if overlap:
        failures.append(f"QA-only candidates leaked into formal index: {overlap}")
    return len(entries), failures


def run_civd(root: Path | str) -> Path:
    """Materialize the CIVD layer; a registered ``civd`` protocol yields the extension index."""

    from .handoff import CIVD_EXTENSION_INDEX_SCHEMA, CIVD_INDEX_SCHEMA_V1, civd_protocol

    output_root = Path(root).resolve()
    bundle = _load_bundle(output_root)
    entries = []
    capacity_column = str(bundle.params["station_contract"].get("capacity_column", "")) or None
    protocol = civd_protocol(bundle.params) if isinstance(bundle.params.get("civd"), dict) else None
    for region in bundle.regions:
        result = materialize_civd(bundle.grids[region][0], bundle.stations[region], working_crs=bundle.params["crs"]["working"], capacity_column=capacity_column)
        destination = output_root / "civd" / f"{region}.npz"
        arrays = {
            "assignment": result["assignment"],
            "grid_cluster": result["grid_cluster"],
            "station_cluster": result["station_cluster"],
            "raw_labels": result["raw_labels"],
            "probabilities": result["probabilities"],
        }
        if protocol is not None:
            # the extension format records the station identities the clusters refer to
            arrays["station_id"] = bundle.stations[region]["station_id"].to_numpy(dtype=str)
        atomic_npz(destination, **arrays)
        entry = {"region": region, "n_clusters": result["n_clusters"], "n_noise": result["n_noise"], **_field_record(destination, output_root)}
        if protocol is not None:
            entry["protocol"] = {**protocol, "region": region}
        entries.append(entry)
    document = {
        "schema_version": CIVD_INDEX_SCHEMA_V1 if protocol is None else CIVD_EXTENSION_INDEX_SCHEMA,
        "country": bundle.country,
        "working_crs": bundle.params["crs"]["working"],
        "entries": entries,
    }
    if protocol is not None:
        document["protocol"] = protocol
    return atomic_json(document, output_root / "civd/index.json")


def run_idr_fixed(root: Path | str) -> Path:
    output_root = Path(root).resolve()
    bundle = _load_bundle(output_root)
    entries = []
    for region in bundle.regions:
        with np.load(output_root / "static/public_activity" / f"{region}.npz", allow_pickle=False) as archive:
            activity = archive["data"]
        with np.load(output_root / "static/assignments" / f"{region}.npz", allow_pickle=False) as archive:
            assignment = archive["assignment"]
        result = materialize_idr_fixed(
            bundle.grids[region][0],
            bundle.stations[region],
            activity,
            assignment,
            working_crs=bundle.params["crs"]["working"],
            b_tv=float(bundle.params["idr"]["b_tv"]),
        )
        destination = output_root / "idr_fixed" / f"{region}.npz"
        atomic_npz(destination, assignment=result.pop("assignment"), raw_idr_assignment=result.pop("raw_idr_assignment"))
        entries.append({"region": region, **result, **_field_record(destination, output_root)})
    return atomic_json(
        {
            "schema_version": "sg_idr_fixed_index_v1",
            "country": bundle.country,
            "working_crs": bundle.params["crs"]["working"],
            "entries": entries,
        },
        output_root / "idr_fixed/index.json",
    )


def run_sweep(root: Path | str, sweep: str) -> Path:
    output_root = Path(root).resolve()
    bundle = _load_bundle(output_root)
    values = list(bundle.params["sweeps"][sweep])
    entries = []

    def inference_fields(group: str) -> list[dict[str, Any]]:
        records = []
        for task_path in sorted((output_root / "inference/tasks" / group).glob("infer-*.json")):
            task = json.loads(task_path.read_text(encoding="utf-8"))
            task_output = output_root.parents[1] / task["output_results_relative"]
            if not (task_output / "inference_completion.json").is_file():
                continue
            for region in task["regions"]:
                metadata = json.loads((task_output / "fields" / f"{region}.json").read_text(encoding="utf-8"))
                if metadata.get("role") != "TEST":
                    continue
                with np.load(task_output / "fields" / f"{region}.npz", allow_pickle=False) as archive:
                    field = np.array(archive["data"], copy=True)
                records.append({**metadata, "group": group, "value": task["value"], "field": field,
                                "inference_node_id": task["chain_commitment"]["node_id"],
                                "inference_field_sha256": sha256_file(task_output / "fields" / f"{region}.npz")})
        return records

    if sweep in {"lambda", "tau"}:
        country = bundle.country.upper()
        signals = ("N", "P") if sweep == "lambda" else ("base",)
        for signal in signals:
            group = f"L-{country}-{signal}" if sweep == "lambda" else f"T-{country}"
            alias_group = f"P-{country}-{signal}" if sweep == "lambda" else f"B-{country}-GNN"
            records = inference_fields(group)
            aliases = inference_fields(alias_group)
            for raw_value in values:
                value = float(raw_value)
                source_records = aliases if value in ({0.05} if sweep == "lambda" else {0.01}) else records
                matches = [
                    record for record in source_records
                    if int(record["seed"]) == 42
                    and (sweep != "lambda" or int(record["fold"]) == 1)
                    and (source_records is aliases or float(record["value"]) == value)
                ]
                for record in matches:
                    region = str(record["region"])
                    grid, _ = bundle.grids[region]
                    source = bundle.source_regions[region]
                    field = validate_field(record["field"], grid, source, source_column=bundle.source_column)
                    destination = output_root / "sweeps" / sweep / signal / str(raw_value) / "seed_42" / f"{region}.npz"
                    atomic_npz(destination, data=field, grid_row=np.arange(len(grid), dtype=np.int64))
                    entries.append({"parameter": sweep, "signal": signal, "value": raw_value, "seed": 42, "fold": int(record["fold"]), "region": region,
                                    "inference_node_id": record["inference_node_id"],
                                    "inference_field_sha256": record["inference_field_sha256"],
                                    "anchor_reused": source_records is aliases,
                                    **_field_record(destination, output_root)})
                if sweep == "lambda":
                    from .weighter.learned.common import kfold_splits

                    expected = len(kfold_splits(bundle.regions, 42, int(bundle.params["training"]["n_folds"]))[0][1])
                else:
                    expected = len(bundle.regions)
                if len(matches) != expected:
                    raise GeneratorExecutionError(f"{group}/{raw_value}: expected {expected} OOF fields, got {len(matches)}")
    else:
        candidate_index = json.loads((output_root / "candidates/candidate_index.json").read_text(encoding="utf-8"))["entries"]
        base_entries = [entry for entry in candidate_index if entry["label"] == "GNN"]
        expected_bases = len(bundle.regions) * len(bundle.params["training"]["seeds"])
        if len(base_entries) != expected_bases:
            raise GeneratorExecutionError(f"{sweep}: expected {expected_bases} GNN OOF base fields, got {len(base_entries)}")
        for base_entry in base_entries:
            region = str(base_entry["region"])
            seed = int(base_entry["seed"])
            grid, _ = bundle.grids[region]
            source = bundle.source_regions[region]
            stations = bundle.stations[region]
            with np.load(output_root / base_entry["path"], allow_pickle=False) as archive:
                base = archive["data"]
            with np.load(output_root / "static/gpm" / f"{region}.npz", allow_pickle=False) as archive:
                ntl_factor = archive["N"]
            for raw_value in values:
                value = float(raw_value)
                from .materialize.core import sweep_field

                field = sweep_field(base, grid, stations, bundle.ntl[region], ntl_factor,
                                    source_column=bundle.source_column, working_crs=bundle.params["crs"]["working"],
                                    parameter=sweep, value=value)
                validate_field(field, grid, source, source_column=bundle.source_column)
                destination = output_root / "sweeps" / sweep / "base" / str(raw_value) / f"seed_{seed}" / f"{region}.npz"
                atomic_npz(destination, data=field, grid_row=np.arange(len(grid), dtype=np.int64))
                entries.append({"parameter": sweep, "signal": "base", "value": raw_value, "seed": seed, "fold": int(base_entry["fold"]), "region": region, **_field_record(destination, output_root)})
    return atomic_json({"schema_version": "sg_sweep_index_v1", "country": bundle.country, "parameter": sweep, "entries": entries}, output_root / "sweeps" / f"index_{sweep}.json")


def run_idr_matched(root: Path | str) -> Path:
    output_root = Path(root).resolve()
    bundle = _load_bundle(output_root)
    candidate_index = output_root / "candidates/candidate_index.json"
    if not candidate_index.is_file():
        raise FileNotFoundError(candidate_index)
    candidate_entries = json.loads(candidate_index.read_text(encoding="utf-8"))["entries"]
    output_entries = []
    by_region: dict[str, list[dict[str, Any]]] = {}
    for entry in candidate_entries:
        by_region.setdefault(str(entry["region"]), []).append(entry)
    for region, entries in by_region.items():
        with np.load(output_root / "static/assignments" / f"{region}.npz", allow_pickle=False) as archive:
            canonical = archive["assignment"]
        fields = {}
        identities = {}
        for entry in entries:
            seed = int(entry["seed"]) if entry.get("seed") is not None else 0
            key = (str(entry["label"]), seed)
            if key in fields:
                raise GeneratorExecutionError(f"duplicate matched IDR field: {key}/{region}")
            with np.load(output_root / entry["path"], allow_pickle=False) as archive:
                fields[key] = archive["data"]
            identities[key] = entry
        results = materialize_idr_matched(
            bundle.grids[region][0],
            bundle.stations[region],
            fields,
            canonical,
            working_crs=bundle.params["crs"]["working"],
            b_tv=float(bundle.params["idr"]["b_tv"]),
        )
        for (label, seed), result in results.items():
            destination = output_root / "idr_matched" / label / f"seed_{seed}" / f"{region}.npz"
            atomic_npz(destination, assignment=result.pop("assignment"), raw_idr_assignment=result.pop("raw_idr_assignment"))
            output_entries.append(
                {
                    "candidate": label,
                    "seed": seed,
                    "region": region,
                    "candidate_path": identities[(label, seed)]["path"],
                    "candidate_sha256": identities[(label, seed)]["sha256"],
                    **result,
                    **_field_record(destination, output_root),
                }
            )
    return atomic_json(
        {
            "schema_version": "sg_idr_matched_index_v1",
            "country": bundle.country,
            "working_crs": bundle.params["crs"]["working"],
            "entries": output_entries,
        },
        output_root / "idr_matched/index.json",
    )


def prepare_graph_cache(root: Path | str, feature_set: str) -> Path:
    output_root = Path(root).resolve()
    bundle = _load_bundle(output_root)
    from .weighter.learned.inputs import build_graphs, save_graph_cache

    representation = str(
        bundle.params.get("execution", {}).get(
            "landuse_supervision_representation", ""
        )
    )
    if representation != "edge_flat_id_scatter_add_v1":
        raise GeneratorExecutionError(
            "Generator rerun graph cache requires edge_flat_id_scatter_add_v1"
        )
    graphs = build_graphs(
        bundle,
        feature_set=feature_set,
        inject_priors=True,
        representation=representation,
    )
    input_receipt = output_root / "inputs/receipt.json"
    from .generation import input_identity_binding

    input_binding = input_identity_binding(input_receipt)
    return save_graph_cache(
        output_root / "training/graphs" / f"{feature_set}.pkl",
        bundle,
        graphs,
        feature_set=feature_set,
        input_fingerprints={
            input_binding["field"]: input_binding["fingerprint"]
        },
    )


def prepare_training_group(
    root: Path | str,
    *,
    repo_root: Path | str,
    results_root: Path | str,
    group: str,
    execution_backend: str,
) -> Path:
    output_root = Path(root).resolve()
    bundle = _load_bundle(output_root)
    from .weighter.learned.training.matrix import load_task_matrix, tasks_for_group, validate_country_task_matrix
    from .weighter.learned.training.preparation import prepare_tasks

    matrix_path = Path(repo_root).resolve() / str(bundle.params["authorities"]["training_task_matrix"])
    matrix = load_task_matrix(matrix_path)
    params_path = output_root / "inputs/worker_params.json"
    validate_country_task_matrix(matrix, bundle.country, params_path)
    overlay_groups = tuple(map(str, bundle.params["task_groups"]))
    matrix_groups = tuple(
        item.task_group for item in matrix if item.country == bundle.country
    )
    if matrix_groups != overlay_groups:
        raise GeneratorExecutionError(
            f"{bundle.country}: overlay task_groups differ from training matrix authority"
        )
    tasks = tasks_for_group(matrix, country=bundle.country, group=group)
    feature_sets = sorted({task.feature_set for task in tasks})
    graph_paths = {}
    for feature_set in feature_sets:
        path = output_root / "training/graphs" / f"{feature_set}.pkl"
        if not path.is_file():
            prepare_graph_cache(output_root, feature_set)
        graph_paths[feature_set] = path
    input_receipt = output_root / "inputs/receipt.json"
    from .generation import derive_run_identity, input_identity_binding

    run_identity = derive_run_identity(
        output_root, bundle.params, repo_root=repo_root
    )
    input_binding = input_identity_binding(input_receipt)
    prepared = prepare_tasks(
        tasks,
        frozen_params_path=params_path,
        graph_cache_by_feature_set=graph_paths,
        repo_root=repo_root,
        results_root=results_root,
        config_fingerprint=sha256_file(params_path),
        input_fingerprints={
            input_binding["field"]: input_binding["fingerprint"]
        },
        execution_backend=execution_backend,
        execution_identity={
            "prepared_by": "sglib.generator.execution",
            "run_identity": run_identity,
        },
    )
    task_root = output_root / "training/tasks" / group
    entries = []
    for task in prepared:
        path = atomic_json(task.to_dict(), task_root / f"{task.task_id}.json")
        entries.append({"task_id": task.task_id, **_field_record(path, output_root)})
    return atomic_json(
        {"schema_version": "sg_prepared_training_group_v1", "country": bundle.country, "group": group, "execution_backend": execution_backend, "tasks": entries},
        task_root / "index.json",
    )


def run_training_group_local(
    root: Path | str,
    group: str,
    *,
    limit: int | None = None,
    profile: str = "formal",
) -> Path:
    output_root = Path(root).resolve()
    from .weighter.learned.training import run_training_task
    from .weighter.learned.training.preparation import load_prepared_task

    index_path = output_root / "training/tasks" / group / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"prepare group first: {index_path}")
    entries = json.loads(index_path.read_text(encoding="utf-8"))["tasks"]
    selected = entries if limit is None else entries[:limit]
    completions = []
    repo_root = find_repo_root(output_root)
    results_root = output_root.parents[1]
    for entry in selected:
        task_path = output_root / entry["path"]
        task = load_prepared_task(
            task_path, repo_root=repo_root, results_root=results_root
        )
        result = run_training_task(task, backend="local", device="auto")
        completion = result / "task_completion.json"
        completions.append({"task_id": entry["task_id"], **_field_record(completion, output_root)})
    return atomic_json(
        {"schema_version": "sg_training_group_receipt_v1", "group": group, "profile": profile, "tasks": completions, "complete": len(selected) == len(entries) or profile == "smoke"},
        output_root / "training/receipts" / f"{group}.json",
    )


def verify_training_group(
    root: Path | str,
    group: str,
    *,
    limit: int | None = None,
    profile: str = "formal",
) -> Path:
    output_root = Path(root).resolve()
    from .weighter.learned.training.preparation import load_prepared_task
    from .weighter.learned.training.verify import verify_task_artifacts

    index = json.loads((output_root / "training/tasks" / group / "index.json").read_text(encoding="utf-8"))
    verified = []
    entries = index["tasks"] if limit is None else index["tasks"][:limit]
    repo_root = find_repo_root(output_root)
    results_root = output_root.parents[1]
    for entry in entries:
        task_path = output_root / entry["path"]
        task = load_prepared_task(
            task_path, repo_root=repo_root, results_root=results_root
        )
        completion = verify_task_artifacts(task)
        verified.append({"task_id": task.task_id, "selected_training_loss": completion["selected_training_loss"]})
    atomic_json(
        {"schema_version": "sg_training_group_receipt_v1", "group": group, "profile": profile, "tasks": verified, "complete": len(entries) == len(index["tasks"]) or profile == "smoke"},
        output_root / "training/receipts" / f"{group}.json",
    )
    return atomic_json({"schema_version": "sg_training_verify_v1", "group": group, "tasks": verified}, output_root / "training/verify" / f"{group}.json")


def prepare_inference_group(
    root: Path | str,
    *,
    repo_root: Path | str,
    results_root: Path | str,
    group: str,
    execution_backend: str,
    limit: int | None = None,
    verified_training: Mapping[str, tuple[Any, Mapping[str, Any]]] | None = None,
) -> Path:
    output_root = Path(root).resolve()
    bundle = _load_bundle(output_root)
    from .weighter.learned.inference.preparation import prepare_inference_tasks

    training_index = json.loads((output_root / "training/tasks" / group / "index.json").read_text(encoding="utf-8"))
    entries = training_index["tasks"] if limit is None else training_index["tasks"][:limit]
    task_paths = [output_root / entry["path"] for entry in entries]
    from .generation import derive_run_identity

    run_identity = derive_run_identity(
        output_root, bundle.params, repo_root=repo_root
    )
    prepared = prepare_inference_tasks(
        task_paths,
        output_root=output_root / "inference/outputs" / group,
        repo_root=repo_root,
        results_root=results_root,
        execution_backend=execution_backend,
        execution_identity={
            "prepared_by": "sglib.generator.execution",
            "run_identity": run_identity,
        },
        verified_training=verified_training,
    )
    task_root = output_root / "inference/tasks" / group
    records = []
    for task in prepared:
        path = atomic_json(task.to_dict(), task_root / f"{task.task_id}.json")
        records.append({"task_id": task.task_id, **_field_record(path, output_root)})
    return atomic_json(
        {"schema_version": "sg_prepared_inference_group_v1", "group": group, "execution_backend": execution_backend, "tasks": records},
        task_root / "index.json",
    )


def run_inference_group_local(root: Path | str, group: str) -> Path:
    output_root = Path(root).resolve()
    from .weighter.learned.inference import run_inference_task
    from .weighter.learned.inference.preparation import load_inference_task

    index = json.loads((output_root / "inference/tasks" / group / "index.json").read_text(encoding="utf-8"))
    completions = []
    chain_receipts = []
    repo_root = find_repo_root(output_root)
    results_root = output_root.parents[1]
    for entry in index["tasks"]:
        task = load_inference_task(
            output_root / entry["path"],
            repo_root=repo_root,
            results_root=results_root,
        )
        result = run_inference_task(task, backend="local", device="cpu")
        completion = result / "inference_completion.json"
        completions.append({"task_id": entry["task_id"], **_field_record(completion, output_root)})
        document = json.loads(completion.read_text(encoding="utf-8"))
        if "chain" in document:
            chain_receipts.append(document["chain"])
    group_document: dict[str, Any] = {
        "schema_version": "sg_inference_group_receipt_v1",
        "group": group,
        "tasks": completions,
    }
    if chain_receipts:
        if len(chain_receipts) != len(completions):
            raise GeneratorExecutionError("inference group mixes legacy and content-chain receipts")
        from sglib.core.infra.content_chain import derive_chain_closure

        group_document["schema_version"] = "sg_inference_group_view_v2"
        group_document["chain_closure"] = derive_chain_closure(chain_receipts)
    return atomic_json(group_document, output_root / "inference/receipts" / f"{group}.json")


def verify_inference_group(
    root: Path | str,
    group: str,
    *,
    repo_root: Path | str | None = None,
    results_root: Path | str | None = None,
) -> Path:
    output_root = Path(root).resolve()
    from .weighter.learned.inference.preparation import load_inference_task
    from .weighter.learned.inference.verify import verify_inference_artifacts

    index = json.loads((output_root / "inference/tasks" / group / "index.json").read_text(encoding="utf-8"))
    verified = []
    chain_receipts = []
    repo_root = Path(repo_root).resolve() if repo_root is not None else find_repo_root(output_root)
    results_root = Path(results_root).resolve() if results_root is not None else output_root.parents[1]
    for entry in index["tasks"]:
        task_path = output_root / entry["path"]
        task = load_inference_task(
            task_path, repo_root=repo_root, results_root=results_root
        )
        completion = verify_inference_artifacts(task)
        completion_path = Path(task.output_path) / "inference_completion.json"
        verified.append(
            {
                "task_id": entry["task_id"],
                # The completion contract records counts and artifact hashes,
                # while the prepared task is the authority for the ordered
                # region identities.  Reading a non-existent completion field
                # silently produced empty provenance in group receipts.
                "regions": list(task.regions),
                **_field_record(completion_path, output_root),
            }
        )
        if "chain" in completion:
            chain_receipts.append(completion["chain"])
    group_document: dict[str, Any] = {
        "schema_version": "sg_inference_group_receipt_v1",
        "group": group,
        "tasks": verified,
    }
    if chain_receipts:
        if len(chain_receipts) != len(verified):
            raise GeneratorExecutionError("inference group mixes legacy and content-chain receipts")
        from sglib.core.infra.content_chain import derive_chain_closure

        group_document["schema_version"] = "sg_inference_group_view_v2"
        group_document["chain_closure"] = derive_chain_closure(chain_receipts)
    atomic_json(group_document, output_root / "inference/receipts" / f"{group}.json")
    return atomic_json({"schema_version": "sg_inference_verify_v1", "group": group, "tasks": verified}, output_root / "inference/verify" / f"{group}.json")


def write_smoke_chain_receipt(root: Path | str, group: str) -> Path:
    """Verify and seal one real V3 train -> verify -> infer smoke chain.

    The receipt is intentionally separate from formal completion.  It carries
    an explicit ``formal=false`` marker and hashes every link so an auditor
    cannot accidentally inspect the preserved V2 compatibility smoke root.
    """

    output_root = Path(root).resolve()
    from .weighter.learned.inference.preparation import load_inference_task
    from .weighter.learned.inference.verify import verify_inference_artifacts
    from .weighter.learned.inputs import GRAPH_CACHE_SCHEMA_V3, load_graph_cache
    from .weighter.learned.training.preparation import load_prepared_task
    from .weighter.learned.training.verify import verify_task_artifacts

    def record(path: Path) -> dict[str, Any]:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(output_root)
        except ValueError as exc:
            raise GeneratorExecutionError(
                f"smoke receipt path escapes execution root: {path}"
            ) from exc
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return {
            "path": relative.as_posix(),
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }

    input_path = output_root / "inputs/receipt.json"
    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    country = str(inputs.get("country", ""))
    if not country or group != f"B-{country.upper()}-GNN":
        raise GeneratorExecutionError("smoke group/country identity differs")
    cache_path = output_root / "training/graphs/lu5.pkl"
    cache = load_graph_cache(
        cache_path,
        expected_country=country,
        expected_feature_set="lu5",
        required_schema=GRAPH_CACHE_SCHEMA_V3,
    )

    training_group_path = output_root / f"training/receipts/{group}.json"
    training_group = json.loads(training_group_path.read_text(encoding="utf-8"))
    if (
        training_group.get("profile") != "smoke"
        or training_group.get("complete") is not True
        or len(training_group.get("tasks", [])) != 1
    ):
        raise GeneratorExecutionError("smoke training group receipt is incomplete")
    task_id = str(training_group["tasks"][0].get("task_id", ""))
    training_task_path = output_root / f"training/tasks/{group}/{task_id}.json"
    repo_root = find_repo_root(output_root)
    results_root = output_root.parents[1]
    training_task = load_prepared_task(
        training_task_path, repo_root=repo_root, results_root=results_root
    )
    training_completion = verify_task_artifacts(training_task)
    if int(training_completion.get("epochs_observed", -1)) != 2:
        raise GeneratorExecutionError("smoke training did not run exactly two epochs")
    training_run = training_completion.get("execution_identity", {}).get("run_identity")
    if (
        not isinstance(training_run, Mapping)
        or training_run.get("schema_version")
        != "sg_generator_execution_generation_v3"
        or training_run.get("landuse_supervision_representation")
        != "edge_flat_id_scatter_add_v1"
    ):
        raise GeneratorExecutionError("smoke training receipt lacks the V3 stamp")
    training_completion_path = Path(training_task.output_path) / "task_completion.json"
    training_verify_path = output_root / f"training/verify/{group}.json"
    training_verify = json.loads(training_verify_path.read_text(encoding="utf-8"))
    if [item.get("task_id") for item in training_verify.get("tasks", [])] != [task_id]:
        raise GeneratorExecutionError("smoke training verification identity differs")

    inference_group_path = output_root / f"inference/receipts/{group}.json"
    inference_group = json.loads(inference_group_path.read_text(encoding="utf-8"))
    if len(inference_group.get("tasks", [])) != 1:
        raise GeneratorExecutionError("smoke inference group receipt is incomplete")
    inference_task_id = f"infer-{task_id}"
    inference_task_path = output_root / f"inference/tasks/{group}/{inference_task_id}.json"
    inference_task = load_inference_task(
        inference_task_path, repo_root=repo_root, results_root=results_root
    )
    inference_completion = verify_inference_artifacts(inference_task)
    inference_run = inference_completion.get("execution_identity", {}).get("run_identity")
    if inference_run != training_run:
        raise GeneratorExecutionError("smoke train/infer V3 identities differ")
    if inference_completion.get("supervision_sanitized") is not True:
        raise GeneratorExecutionError("smoke inference did not sanitize supervision")
    inference_completion_path = (
        Path(inference_task.output_path) / "inference_completion.json"
    )
    inference_verify_path = output_root / f"inference/verify/{group}.json"
    inference_verify = json.loads(inference_verify_path.read_text(encoding="utf-8"))
    if [item.get("task_id") for item in inference_verify.get("tasks", [])] != [
        inference_task_id
    ]:
        raise GeneratorExecutionError("smoke inference verification identity differs")

    return atomic_json(
        {
            "schema_version": "sg_generator_v3_smoke_chain_receipt_v1",
            "status": "PASS",
            "profile": "smoke",
            "formal": False,
            "country": country,
            "group": group,
            "regions": list(cache["regions"]),
            "epochs": 2,
            "graph_cache_schema": cache["schema"],
            "landuse_supervision_representation": cache[
                "landuse_supervision_representation"
            ],
            "run_identity": dict(training_run),
            "supervision_sanitized": True,
            "chain": {
                "inputs": record(input_path),
                "graph_cache": record(cache_path),
                "training_task": record(training_task_path),
                "training_completion": record(training_completion_path),
                "training_verify": record(training_verify_path),
                "inference_task": record(inference_task_path),
                "inference_completion": record(inference_completion_path),
                "inference_verify": record(inference_verify_path),
            },
        },
        output_root / "smoke_receipt.json",
    )


def run_audit(root: Path | str, *, profile: str, full: bool = False, destination: Path | None = None) -> Path:
    output_root = Path(root).resolve()
    bundle = _load_bundle(output_root)
    failures = []
    execution = bundle.params.get("execution", {})
    civd_enabled = execution.get("civd_enabled", True)
    if type(civd_enabled) is not bool:
        failures.append("execution.civd_enabled is not boolean")
        civd_enabled = True
    civd_index = output_root / "civd/index.json"
    if civd_enabled and not civd_index.is_file():
        failures.append("missing enabled CIVD index")
    for component in ("assignments", "uniform", "gpm", "proximity", "public_activity"):
        if not (output_root / "static" / component / "index.json").is_file():
            failures.append(f"missing static component: {component}")
    idr_path = output_root / "idr_fixed/index.json"
    if not idr_path.is_file():
        failures.append("missing idr_fixed index")
        idr_entries = []
    else:
        idr_entries = json.loads(idr_path.read_text(encoding="utf-8"))["entries"]
        for entry in idr_entries:
            if not {"g0_pass", "g1_pass", "tv_mass", "transport_budget", "selected_mode", "allocator_version"} <= set(entry):
                failures.append(f"incomplete IDR gate record: {entry.get('region')}")
            if float(entry.get("transport_budget", -1)) != float(bundle.params["idr"]["b_tv"]):
                failures.append(f"IDR budget mismatch: {entry.get('region')}")
    matched_path = output_root / "idr_matched/index.json"
    if not matched_path.is_file():
        failures.append("missing idr_matched index")
        matched_entries = []
    else:
        matched_entries = json.loads(matched_path.read_text(encoding="utf-8"))["entries"]
        for entry in matched_entries:
            if not {"g0_pass", "g1_pass", "tv_mass", "transport_budget", "selected_mode", "allocator_version", "uses_station_geometry", "source_total_basis", "prohibited_truth"} <= set(entry):
                failures.append(f"incomplete matched IDR gate record: {entry.get('candidate')}/{entry.get('region')}")
            if entry.get("prohibited_truth") is not False:
                failures.append(f"matched IDR prohibited-truth flag differs: {entry.get('candidate')}")
    training_receipts = []
    for path in (output_root / "2_Weighter").glob("**/task_completion.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        training_receipts.append(document)
        if document.get("training_weighting_policy") != "uniform_region_cyclic_sgd__source_mean_v1":
            failures.append(f"weighting policy mismatch: {path}")
        if document.get("training_batch_size") != 1:
            failures.append(f"training batch size mismatch: {path}")
    inference_receipts = []
    for path in (output_root / "inference/outputs").glob("**/inference_completion.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        inference_receipts.append(document)
        if document.get("supervision_sanitized") is not True:
            failures.append(f"unsanitized inference: {path}")
    candidate_index = output_root / "candidates/candidate_index.json"
    if not candidate_index.is_file():
        failures.append("missing formal candidate index")
        candidate_count = 0
        entries = []
    else:
        entries = json.loads(candidate_index.read_text(encoding="utf-8"))["entries"]
        candidate_count = len(entries)
        for entry in entries:
            path = output_root / entry["path"]
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                failures.append(f"candidate hash mismatch: {entry['path']}")
    qa_candidate_count, qa_failures = _audit_qa_candidate_index(
        output_root, bundle, entries
    )
    failures.extend(qa_failures)
    if profile == "formal" or full:
        from .delivery import audit_delivery

        coverage = audit_delivery(output_root, bundle)
        expected_groups = len(bundle.params["task_groups"])
        if len(list((output_root / "training/verify").glob("*.json"))) != expected_groups:
            failures.append("formal training groups are incomplete")
        if len(list((output_root / "inference/verify").glob("*.json"))) != expected_groups:
            failures.append("formal inference groups are incomplete")
    document = {
        "schema_version": "sg_generator_audit_v1",
        "status": "PASS" if not failures else "FAIL",
        "profile": profile,
        "country": bundle.country,
        "civd_enabled": civd_enabled,
        "regions": list(bundle.regions),
        "candidate_entries": candidate_count,
        "qa_candidate_entries": qa_candidate_count,
        "training_receipts": len(training_receipts),
        "inference_receipts": len(inference_receipts),
        "idr_gate_entries": len(idr_entries),
        "idr_matched_gate_entries": len(matched_entries),
        "failures": failures,
    }
    if profile == "formal" or full:
        document["delivery_coverage"] = coverage
    path = atomic_json(document, destination if destination is not None else output_root / "audit.json")
    if failures:
        raise GeneratorExecutionError(f"Generator audit failed: {failures}")
    return path
