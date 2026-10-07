from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from sglib.dataoverview.engineering_admission import (
    EngineeringAdmissionError,
    chunked_nearest_assignment,
    evaluate_region,
    load_engineering_admission,
    validate_pre_submission,
)
from sglib.core.infra.engineering_admission import (
    INDEXED_LANDUSE_REPRESENTATION,
    LEGACY_DENSE_LANDUSE_REPRESENTATION,
    landuse_supervision_footprint_bytes,
)

pytestmark = pytest.mark.gate


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "casestudy/1_DataOverview/5_NZ/admission_contract.toml"


def _load_v3(
    tmp_path: Path,
    *,
    representation: str = INDEXED_LANDUSE_REPRESENTATION,
    max_mib: str = "32.0",
):
    authority_relative = Path(
        "casestudy/config/authority/engineering_admission_v3.toml"
    )
    authority = tmp_path / authority_relative
    contract = tmp_path / CONTRACT
    evidence_relative = Path(
        "casestudy/config/authority/evidence/indexed_landuse_equivalence.json"
    )
    evidence = tmp_path / evidence_relative
    authority.parent.mkdir(parents=True)
    contract.parent.mkdir(parents=True)
    evidence.parent.mkdir(parents=True)

    evidence.write_text(
        json.dumps(
            {
                "schema_version": "sg_indexed_landuse_equivalence_diagnostic_v1",
                "status": "DIAGNOSTIC_ONLY",
                "representation": representation,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    evidence_sha256 = sha256(evidence.read_bytes()).hexdigest()
    (evidence.parent / "IMMUTABLE_AUTHORITY_EVIDENCE.txt").write_text(
        f"{evidence_relative.as_posix()}\n{evidence_sha256}\n",
        encoding="utf-8",
    )

    authority_payload = (
        (ROOT / "casestudy/config/authority/engineering_admission.toml")
        .read_text(encoding="utf-8")
        .replace(
            'schema_version = "sg_engineering_admission_authority_v2"',
            'schema_version = "sg_engineering_admission_authority_v3"',
            1,
        )
        .replace("max_dense_tensor_mib = 32.0", f"max_dense_tensor_mib = {max_mib}")
        + (
            f'\nlanduse_supervision_representation = "{representation}"\n'
            "flat_index_value_bytes = 8\n"
            "landuse_ratio_value_bytes = 4\n"
            "\n[representation_evidence]\n"
            f'path = "{evidence_relative.as_posix()}"\n'
            'schema_version = "sg_indexed_landuse_equivalence_diagnostic_v1"\n'
            f'sha256 = "{evidence_sha256}"\n'
        )
    )
    authority.write_text(authority_payload, encoding="utf-8")

    contract_payload = (
        (ROOT / CONTRACT)
        .read_text(encoding="utf-8")
        .replace(
            'schema_version = "sg_country_engineering_admission_contract_v2"',
            'schema_version = "sg_country_engineering_admission_contract_v3"',
            1,
        )
        .replace(
            'path = "casestudy2/config/authority/engineering_admission.toml"',
            f'path = "{authority_relative.as_posix()}"',
            1,
        )
        .replace(
            'schema_version = "sg_engineering_admission_authority_v2"',
            'schema_version = "sg_engineering_admission_authority_v3"',
            1,
        )
    )
    contract.write_text(contract_payload, encoding="utf-8")
    return load_engineering_admission(tmp_path, CONTRACT)


def _region_metrics(*, active: int, n_sources: int) -> dict[str, int]:
    return {
        "n_sources": n_sources,
        "n_targets": 4,
        "n_cells": active,
        "min_cells_per_source": 4,
        "min_cells_per_target": 4,
        "total_edges_directed": active * 10,
        "active_agent_nodes_upper_bound": active,
        "active_agent_nodes_observed": active,
        "cdist_peak_workspace_bytes": 1_000,
    }


def test_shared_v2_authority_owns_every_numerical_engineering_gate() -> None:
    admission = load_engineering_admission(ROOT, CONTRACT)
    assert (
        admission.landuse_supervision_representation
        == LEGACY_DENSE_LANDUSE_REPRESENTATION
    )
    assert admission.grid["min_cells_per_source_hard"] == 4
    assert admission.grid["min_cells_per_target_hard"] == 4
    assert admission.grid["cells_per_source_oversampling_disclosure"] == 256
    assert admission.grid["cells_per_target_oversampling_disclosure"] == 200
    assert admission.grid["max_cells_per_region"] == 100_000
    assert admission.graph["max_directed_edges_per_region"] == 1_000_000
    assert admission.graph["max_active_agent_nodes_per_region"] == 60_000
    assert admission.graph["planning_k_max"] == 1_000
    assert admission.memory["max_dense_tensor_mib"] == 32.0
    assert admission.memory["max_cdist_workspace_mib"] == 128.0
    assert admission.country["idr_matched"]["max_realizations_all_regions"] == 2_000
    assert not ({"grid", "graph", "memory"} & set(admission.country))


def test_v2_uses_its_archived_dense_formula_without_an_explicit_field() -> None:
    admission = load_engineering_admission(ROOT, CONTRACT)
    assert "landuse_supervision_representation" not in admission.memory
    assert (
        landuse_supervision_footprint_bytes(
            admission, active_edges=60_000, n_sources=100
        )
        == 60_000 * 100 * 5 * 4
    )


def test_v3_requires_paired_country_schema_and_uses_the_frozen_indexed_name(
    tmp_path: Path,
) -> None:
    admission = _load_v3(tmp_path)
    assert (
        admission.landuse_supervision_representation
        == "edge_flat_id_scatter_add_v1"
        == INDEXED_LANDUSE_REPRESENTATION
    )
    assert admission.authority["schema_version"] == "sg_engineering_admission_authority_v3"
    assert admission.country["schema_version"] == "sg_country_engineering_admission_contract_v3"
    # edges * int64 + sources * five-channel float32 land-use ratio
    assert (
        landuse_supervision_footprint_bytes(
            admission, active_edges=49_350, n_sources=100
        )
        == 49_350 * 8 + 100 * 5 * 4
    )


def test_v3_unknown_representation_and_cross_version_pair_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(EngineeringAdmissionError, match="unknown landuse supervision"):
        _load_v3(tmp_path / "unknown", representation="compact-ish")

    admission = _load_v3(tmp_path / "mismatch-source")
    contract = admission.country_path
    payload = contract.read_text(encoding="utf-8").replace(
        'schema_version = "sg_country_engineering_admission_contract_v3"',
        'schema_version = "sg_country_engineering_admission_contract_v2"',
        1,
    )
    contract.write_text(payload, encoding="utf-8")
    with pytest.raises(EngineeringAdmissionError, match="not a versioned pair"):
        load_engineering_admission(tmp_path / "mismatch-source", CONTRACT)


def test_v3_representation_evidence_is_hash_and_semantics_bound(
    tmp_path: Path,
) -> None:
    admission = _load_v3(tmp_path / "valid")
    reference = admission.authority["representation_evidence"]
    assert reference["schema_version"] == "sg_indexed_landuse_equivalence_diagnostic_v1"

    evidence = admission.repo_root / reference["path"]
    evidence.write_text(
        evidence.read_text(encoding="utf-8").replace(
            '"status": "DIAGNOSTIC_ONLY"', '"status": "PASS"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(EngineeringAdmissionError, match="evidence hash mismatch"):
        load_engineering_admission(admission.repo_root, CONTRACT)

    semantic = _load_v3(tmp_path / "semantic")
    evidence = semantic.repo_root / semantic.authority["representation_evidence"]["path"]
    bad_payload = json.loads(evidence.read_text(encoding="utf-8"))
    bad_payload["status"] = "PASS"
    evidence.write_text(json.dumps(bad_payload, sort_keys=True), encoding="utf-8")
    authority = semantic.authority_path
    authority_payload = authority.read_text(encoding="utf-8")
    authority_payload = authority_payload.replace(
        semantic.authority["representation_evidence"]["sha256"],
        sha256(evidence.read_bytes()).hexdigest(),
    )
    authority.write_text(authority_payload, encoding="utf-8")
    with pytest.raises(EngineeringAdmissionError, match="must remain DIAGNOSTIC_ONLY"):
        load_engineering_admission(semantic.repo_root, CONTRACT)

    representation = _load_v3(tmp_path / "representation")
    evidence = (
        representation.repo_root
        / representation.authority["representation_evidence"]["path"]
    )
    bad_payload = json.loads(evidence.read_text(encoding="utf-8"))
    bad_payload["representation"] = LEGACY_DENSE_LANDUSE_REPRESENTATION
    evidence.write_text(json.dumps(bad_payload, sort_keys=True), encoding="utf-8")
    authority = representation.authority_path
    authority.write_text(
        authority.read_text(encoding="utf-8").replace(
            representation.authority["representation_evidence"]["sha256"],
            sha256(evidence.read_bytes()).hexdigest(),
        ),
        encoding="utf-8",
    )
    with pytest.raises(EngineeringAdmissionError, match="differs from the authority"):
        load_engineering_admission(representation.repo_root, CONTRACT)


def test_v3_indexed_footprint_is_inclusive_at_the_32mib_guard_boundary(
    tmp_path: Path,
) -> None:
    # Use a 1 KiB limit so the exact inclusive boundary is small enough for a
    # focused unit test: 118*8 + 4*5*4 == 1024 bytes.
    admission = _load_v3(tmp_path, max_mib="0.0009765625")
    exact = evaluate_region(
        admission,
        _region_metrics(active=118, n_sources=4),
        formal_pre_submission=True,
    )
    assert exact["hard_pass"] is True
    assert exact["memory"]["landuse_supervision_footprint_bytes"] == 1024
    assert exact["memory"]["landuse_supervision_limit_met"] is True

    over = evaluate_region(
        admission,
        _region_metrics(active=119, n_sources=4),
        formal_pre_submission=True,
    )
    assert over["hard_pass"] is False
    assert over["memory"]["landuse_supervision_footprint_bytes"] == 1032
    assert over["hard_checks"]["max_dense_tensor_real_active_mib"] is False


def test_v3_future_dense_rollback_reenters_dense_guard_instead_of_compact_accounting(
    tmp_path: Path,
) -> None:
    indexed = _load_v3(tmp_path / "indexed")
    dense = _load_v3(
        tmp_path / "dense",
        representation=LEGACY_DENSE_LANDUSE_REPRESENTATION,
    )
    metrics = _region_metrics(active=60_000, n_sources=100)
    indexed_decision = evaluate_region(indexed, metrics, formal_pre_submission=True)
    dense_decision = evaluate_region(dense, metrics, formal_pre_submission=True)

    assert indexed_decision["hard_checks"]["max_dense_tensor_real_active_mib"] is True
    assert indexed_decision["memory"]["landuse_supervision_footprint_bytes"] == 482_000
    assert dense_decision["hard_checks"]["max_dense_tensor_real_active_mib"] is False
    assert dense_decision["memory"]["landuse_supervision_footprint_bytes"] == 120_000_000


def test_country_contract_cannot_duplicate_shared_sections(tmp_path: Path) -> None:
    authority = tmp_path / "casestudy/config/authority/engineering_admission.toml"
    contract = tmp_path / CONTRACT
    authority.parent.mkdir(parents=True)
    contract.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "casestudy/config/authority/engineering_admission.toml", authority)
    payload = (ROOT / CONTRACT).read_text(encoding="utf-8")
    contract.write_text(payload + "\n[grid]\nmax_cells_per_region = 1\n", encoding="utf-8")
    with pytest.raises(EngineeringAdmissionError, match="duplicates shared authority"):
        load_engineering_admission(tmp_path, CONTRACT)


def test_country_neutral_cdist_chunks_to_the_authority_workspace() -> None:
    left = np.column_stack([np.arange(101, dtype=float), np.zeros(101)])
    right = np.column_stack([np.arange(25, dtype=float) * 4, np.zeros(25)])
    labels, peak_bytes, rows = chunked_nearest_assignment(
        left,
        right,
        max_workspace_mib=0.001,
    )
    assert labels.shape == (101,)
    assert peak_bytes <= int(0.001 * 2**20)
    assert rows < len(left)
    assert labels[0] == 0


def test_dense_32mib_is_advisory_at_dry_run_but_fail_closed_for_formal() -> None:
    admission = load_engineering_admission(ROOT, CONTRACT)
    metrics = {
        "n_sources": 100,
        "n_targets": 20,
        "n_cells": 60_000,
        "min_cells_per_source": 4,
        "min_cells_per_target": 4,
        "total_edges_directed": 600_000,
        "active_agent_nodes_upper_bound": 60_000,
        "cdist_peak_workspace_bytes": 1_000_000,
    }
    dry = evaluate_region(admission, metrics)
    assert dry["hard_pass"] is True
    assert dry["memory"]["dense_tensor_limit_met"] is False
    assert dry["memory"]["binding"] == "advisory_dry_run"
    with pytest.raises(EngineeringAdmissionError, match="observed active-agent"):
        evaluate_region(admission, metrics, formal_pre_submission=True)
    formal = evaluate_region(
        admission,
        {**metrics, "active_agent_nodes_observed": 60_000},
        formal_pre_submission=True,
    )
    assert formal["hard_pass"] is False
    assert formal["hard_checks"]["max_dense_tensor_real_active_mib"] is False
    with pytest.raises(EngineeringAdmissionError, match="max_dense_tensor_real_active_mib"):
        validate_pre_submission(
            admission,
            {**metrics, "active_agent_nodes_observed": 60_000},
        )


def test_nz_bplus_metadata_carries_disclosure_not_a_false_hard_gate() -> None:
    from sglib.core.infra.config import load_dataoverview_config
    from sglib.dataoverview.processing.config import build_context
    from sglib.dataoverview.processing.pipeline import _engineering_admission_metadata

    loaded = load_dataoverview_config(
        ROOT / "casestudy/1_DataOverview/general/general.toml",
        ROOT / "casestudy/config/countries/nz.toml",
        ROOT / "casestudy/1_DataOverview/5_NZ/nz.toml",
    )
    metadata = _engineering_admission_metadata(build_context(ROOT, loaded))
    disclosure = metadata["engineering_admission"]
    assert disclosure["min_cells_per_source_hard"] == 4
    assert disclosure["min_cells_per_target_hard"] == 4
    assert disclosure["cells_per_source_oversampling_disclosure"] == 256
    assert disclosure["cells_per_target_oversampling_disclosure"] == 200
    assert disclosure["oversampling_binding"] == "report_only"
