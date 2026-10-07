from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import pytest
from sglib.core.infra.content_chain import (
    ContentChainError,
    code_projection,
    derive_chain_closure,
    derive_chain_commitment,
    derive_chain_receipt,
    verify_chain,
    verify_chain_link,
)
from sglib.generator.generation import input_identity_binding
from sglib.generator.config import load_generator_config, scientific_config_fingerprint, training_params
from sglib.core.infra.hashing import sha256_json
from sglib.core.infra.hashing import sha256_file
from sglib.core.infra.hashing import sha256_json
pytestmark = pytest.mark.consume
ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "casestudy/2_Generator"
REGISTRY = GENERATOR / "general/candidate_registry.json"
MATRIX = GENERATOR / "general/training_task_matrix_scientific_v2.csv"
LEGACY_MATRIX = GENERATOR / "general/training_task_matrix.csv"


def test_scientific_config_fingerprint_ignores_operational_paths() -> None:
    loaded = load_generator_config(
        ROOT,
        GENERATOR / "general/generator.toml",
        ROOT / "casestudy/config/countries/au.toml",
        GENERATOR / "2_AU/au.toml",
    )
    baseline = scientific_config_fingerprint(loaded, repo_root=ROOT)
    operational = deepcopy(dict(loaded.values))
    operational["paths"] = {
        "formal_root": "elsewhere/formal",
        "smoke_root": "elsewhere/smoke",
        "diagnostic_root": "elsewhere/diagnostic",
    }
    operational["country"] = dict(operational["country"])
    operational["country"]["directory"] = "moved-country-directory"
    assert scientific_config_fingerprint(operational, repo_root=ROOT) == baseline
    scientific = deepcopy(operational)
    scientific["training"] = deepcopy(scientific["training"])
    scientific["training"]["configs"] = deepcopy(
        scientific["training"]["configs"]
    )
    scientific["training"]["configs"]["baseline"] = deepcopy(
        scientific["training"]["configs"]["baseline"]
    )
    scientific["training"]["configs"]["baseline"]["epochs"] += 1
    assert scientific_config_fingerprint(scientific, repo_root=ROOT) != baseline


def _country_config(code: str, directory: str):
    return load_generator_config(
        ROOT,
        GENERATOR / "general/generator.toml",
        ROOT / f"casestudy/config/countries/{code}.toml",
        GENERATOR / f"{directory}/{code}.toml",
    )


def test_historical_counts_stay_outside_scientific_identity() -> None:
    loaded = _country_config("nz", "5_NZ")
    baseline = scientific_config_fingerprint(loaded, repo_root=ROOT)
    counted = deepcopy(dict(loaded.values))
    counted["checks"] = {**counted["checks"], "expected_n_targets": 135}
    counted["station_contract"] = {**counted["station_contract"], "formal_truth_rows": 135}
    assert scientific_config_fingerprint(counted, repo_root=ROOT) == baseline
    assert "expected_n_targets" not in training_params(counted)["checks"]
    admission = deepcopy(dict(loaded.values))
    admission["checks"] = {**admission["checks"], "truth_granularity": "section"}
    assert scientific_config_fingerprint(admission, repo_root=ROOT) != baseline


def test_content_chain_has_one_projection_and_operations_are_observations() -> None:
    parent_commitment = derive_chain_commitment(
        "train.task",
        inputs={"graph": "a" * 64},
        scientific_parameters={"seed": 42, "fold": 1},
        code_sha256="b" * 64,
    )
    parent = derive_chain_receipt(
        parent_commitment,
        outputs={"model.pth": "c" * 64},
        observations={"backend": "hpc", "gres": "gpu:1g.5gb:1"},
    )
    child_commitment = derive_chain_commitment(
        "infer.task",
        inputs={"checkpoint": "c" * 64},
        scientific_parameters={"seed": 42, "fold": 1},
        code_sha256="d" * 64,
    )
    child = derive_chain_receipt(
        child_commitment,
        outputs={"field": "e" * 64},
        observations={"backend": "local", "path": "machine-a/one"},
    )
    moved = derive_chain_receipt(
        child_commitment,
        outputs={"field": "e" * 64},
        observations={"backend": "local", "path": "machine-b/moved"},
    )
    assert child["receipt_sha256"] == moved["receipt_sha256"]
    changed_science = derive_chain_commitment(
        "infer.task",
        inputs={"checkpoint": "c" * 64},
        scientific_parameters={"seed": 123, "fold": 1},
        code_sha256="d" * 64,
    )
    assert changed_science["commitment_sha256"] != child_commitment["commitment_sha256"]
    verify_chain_link(parent, child, {"checkpoint": "model.pth"})
    with pytest.raises(ContentChainError, match="hash differs"):
        verify_chain_link(
            parent,
            derive_chain_commitment(
                "infer.bad",
                inputs={"checkpoint": "f" * 64},
                scientific_parameters={"seed": 42},
                code_sha256="d" * 64,
            ),
            {"checkpoint": "model.pth"},
        )
    with pytest.raises(ContentChainError, match="non-empty"):
        verify_chain_link(parent, child, {})
    closure = derive_chain_closure([child])
    assert verify_chain(closure)["node_count"] == 1
    with pytest.raises(ContentChainError, match="unique"):
        derive_chain_closure([child, child])


def test_code_projection_hashes_symbols_not_their_container_file() -> None:
    def plus_one(value):
        return value + 1

    def renamed_with_docs(value):
        """A locator/docstring change is not computation."""

        return value + 1

    def plus_two(value):
        return value + 2

    first = code_projection({"calculation": plus_one})
    same = code_projection({"calculation": renamed_with_docs})
    changed = code_projection({"calculation": plus_two})
    assert first["code_sha256"] == same["code_sha256"]
    assert first["locators"] != same["locators"]
    assert first["code_sha256"] != changed["code_sha256"]


def test_v3_input_identity_ignores_receipt_container_operations(
    tmp_path: Path,
) -> None:
    identity = {
        "schema_version": "sg_generator_inputs_scientific_identity_v3",
        "country": "au",
        "regions": ["R"],
        "worker_params_sha256": "a" * 64,
    }
    first = {
        "schema_version": "sg_generator_inputs_receipt_v3",
        "scientific_identity": identity,
        "scientific_fingerprint": sha256_json(identity),
        "bundle": {"path": "old/bundle.pkl", "sha256": "b" * 64},
    }
    second = deepcopy(first)
    second["bundle"]["path"] = "moved/bundle.pkl"
    for name, document in (("first.json", first), ("second.json", second)):
        (tmp_path / name).write_text(json.dumps(document), encoding="utf-8")
    first_binding = input_identity_binding(tmp_path / "first.json")
    second_binding = input_identity_binding(tmp_path / "second.json")
    assert first_binding == second_binding
    assert sha256_file(tmp_path / "first.json") != sha256_file(
        tmp_path / "second.json"
    )
