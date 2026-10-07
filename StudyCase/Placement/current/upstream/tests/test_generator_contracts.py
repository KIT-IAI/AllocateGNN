from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from sglib.generator.config import load_generator_config, training_params
from sglib.generator.execution import GeneratorExecutionError, _audit_qa_candidate_index, prepare_inputs
from sglib.generator.weighter.candidates import (
    candidate_labels,
    load_candidate_registry,
    qa_only_candidates,
)
from sglib.core.infra.schema import load_schema
pytestmark = pytest.mark.gate
ROOT = Path(__file__).resolve().parents[1]


def _config(country: str, directory: str):
    return load_generator_config(
        ROOT,
        ROOT / "casestudy/2_Generator/general/generator.toml",
        ROOT / f"casestudy/config/countries/{country}.toml",
        ROOT / f"casestudy/2_Generator/{directory}/{country}.toml",
    )


@pytest.mark.parametrize(
    "country,directory,n_regions,civd_enabled",
    [
        ("uk", "1_UK", 16, True),
        ("au", "2_AU", 12, True),
        ("nl", "4_NL", 16, True),
        ("nz", "5_NZ", 9, True),
    ],
)
def test_generator_config_and_worker_contract(
    country: str,
    directory: str,
    n_regions: int,
    civd_enabled: bool,
) -> None:
    config = _config(country, directory)
    worker = training_params(config)
    assert len(config.values["task_groups"]) == 11
    assert len(worker["regions"]) == n_regions
    assert worker["training_weighting_policy"] == "uniform_region_cyclic_sgd__source_mean_v1"
    assert worker["training_batch_size"] == 1
    assert config.values["idr"]["b_tv"] == 0.10
    assert config.values["execution"]["civd_enabled"] is civd_enabled


def test_qa_candidates_materialize_but_never_enter_formal_layers() -> None:
    config = _config("uk", "1_UK")
    registry = load_candidate_registry(
        ROOT / config.values["authorities"]["candidate_registry"]
    )
    qa = qa_only_candidates(registry)
    assert tuple(item["label"] for item in qa) == (
        "UniAddN",
        "UniAddP",
        "UniAddNP",
    )
    assert all(item["materialize"] and item["qa_only"] for item in qa)
    assert not {item["label"] for item in qa} & set(
        candidate_labels(registry, "L31")
    )


def test_formal_qa_index_is_required_complete_and_disjoint(tmp_path: Path) -> None:
    regions = ["r1", "r2"]
    bundle = SimpleNamespace(
        regions=regions,
        params={
            "authorities": {
                "candidate_registry": "casestudy/2_Generator/general/candidate_registry.json"
            }
        },
    )
    qa_labels = ("UniAddN", "UniAddP", "UniAddNP")
    entries = []
    for label in qa_labels:
        for region in regions:
            artifact = tmp_path / "candidates" / label / f"{region}.npz"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(f"{label}/{region}".encode())
            entries.append(
                {
                    "label": label,
                    "region": region,
                    "qa_only": True,
                    "path": artifact.relative_to(tmp_path).as_posix(),
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            )
    qa_index = tmp_path / "candidates/candidate_qa_index.json"
    qa_index.write_text(
        json.dumps({"schema_version": "sg_candidate_qa_index_v1", "entries": entries}),
        encoding="utf-8",
    )
    count, failures = _audit_qa_candidate_index(tmp_path, bundle, [])
    assert count == 6 and failures == []
    qa_index.write_text(
        json.dumps({"schema_version": "sg_candidate_qa_index_v1", "entries": []}),
        encoding="utf-8",
    )
    count, failures = _audit_qa_candidate_index(tmp_path, bundle, [])
    assert count == 0
    assert any("identity set differs" in item for item in failures)


def test_formal_generator_inputs_fail_closed_on_unpassed_handoff_gate(
    tmp_path: Path,
) -> None:
    from sglib.dataoverview.evidence import HandoffEvidence

    gate = tmp_path / "gate.json"
    gate.write_text('{"status":"PROVISIONAL_REVIEW_REQUIRED"}', encoding="utf-8")
    handoff = SimpleNamespace(
        evidence={
            "engineering_admission": HandoffEvidence(
                name="engineering_admission",
                path=gate,
                repo_relative="gate.json",
                sha256="0" * 64,
                bytes=gate.stat().st_size,
                schema=None,
                document={"status": "PROVISIONAL_REVIEW_REQUIRED"},
                formal_required=True,
                required_status="PASS",
            )
        }
    )
    with pytest.raises(GeneratorExecutionError, match="required='PASS'"):
        prepare_inputs(handoff, {}, tmp_path / "generator")


def test_generator_notebooks_use_shared_stage_and_keep_outputs_empty() -> None:
    notebooks = sorted((ROOT / "casestudy/2_Generator").glob("[1245]_*/*.ipynb"))
    assert len(notebooks) == 20
    for path in notebooks:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert all(cell.get("outputs", []) == [] for cell in document["cells"])
        assert all(cell.get("execution_count") is None for cell in document["cells"] if cell["cell_type"] == "code")
        source = "".join("".join(cell.get("source", [])) for cell in document["cells"])
        if path.name == "07_audit_handoff.ipynb":
            assert "report = run_gate(ctx)" in source
            assert "raise SystemExit(1)" in source
            assert not any(token in source for token in ("run_step", "run_audit", "load_bundle", "hash_manifest"))
        else:
            assert "stage.run_step(" in source
        assert not any(token in source for token in ("run_training_task", "materialize_family", "atomic_", ".to_file("))


def test_all_generator_artifact_schemas_parse() -> None:
    schemas = sorted((ROOT / "casestudy/2_Generator/schemas").glob("*.toml"))
    assert len(schemas) >= 14
    for path in schemas:
        assert load_schema(path)["artifact_type"] in {"json", "npz", "table"}
