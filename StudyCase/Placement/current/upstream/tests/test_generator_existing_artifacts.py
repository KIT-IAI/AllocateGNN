from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import pytest
from sglib.generator.config import load_generator_config
from sglib.generator.execution import _load_bundle
from sglib.generator.handoff import load_bundle
from sglib.generator.registry import build_registry
from sglib.generator.weighter.candidates import load_candidate_registry
from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.leaf_manifest import build, read_current
pytestmark = [pytest.mark.consume, pytest.mark.local_results]
ROOT = Path(__file__).resolve().parents[1]


def _config(country: str, directory: str):
    return load_generator_config(
        ROOT,
        ROOT / "casestudy/2_Generator/general/generator.toml",
        ROOT / f"casestudy/config/countries/{country}.toml",
        ROOT / f"casestudy/2_Generator/{directory}/{country}.toml",
    )


def test_generator_registry_and_smoke_handoff(tmp_path: Path) -> None:
    config = _config("uk", "1_UK")
    candidates = load_candidate_registry(ROOT / config.values["authorities"]["candidate_registry"])
    registry = build_registry("uk", config.values, candidates)
    assert "uk.idr_fixed.idr_fixed" in registry
    assert "uk.idr_matched.idr_matched" in registry
    selected_root = os.environ.get("SG_GENERATOR_SMOKE_ROOT")
    smoke = ROOT / (selected_root or "results/_smoke") / "2_Generator/1_UK"
    audit_path = smoke / "audit.json"
    if selected_root is None and (
        not audit_path.is_file() or "delivery_coverage" not in json.loads(audit_path.read_text(encoding="utf-8"))
    ):
        pytest.skip("full Generator smoke is absent; set SG_GENERATOR_SMOKE_ROOT to a completed 01–07 results base")
    assert audit_path.is_file()
    params = _load_bundle(smoke).params
    smoke = Path(shutil.copytree(smoke, tmp_path / "generator"))
    paths = [path.relative_to(smoke).as_posix() for path in smoke.rglob("*")
             if path.is_file() and path != smoke / "manifest.json"]
    atomic_json(build("2_Generator", "uk", read_current({"handoff": paths}, smoke)),
                smoke / "manifest.json")
    bundle = load_bundle(
        smoke,
        params,
        schemas_root=ROOT / "casestudy/2_Generator/schemas",
        repo_root=ROOT,
    )
    assert len(bundle.candidates) == 312
    assert bundle.candidate("MLP", "TLH2", seed=42).fold == 1
    assert len(bundle.civd) == 4
    assert len(bundle.idr_fixed) == 4
    assert len(bundle.idr_matched) == 312
    assert {item.parameter for item in bundle.sweeps} == set(params["sweeps"])


def test_country_smoke_audits_and_inference_receipts() -> None:
    for directory in ("1_UK", "2_AU"):
        root = ROOT / "results/_smoke/2_Generator" / directory
        audit = root / "audit.json"
        if not audit.is_file():
            pytest.skip(f"{directory} smoke has not been executed")
        assert json.loads(audit.read_text(encoding="utf-8"))["status"] == "PASS"
        receipts = list((root / "inference/receipts").glob("*.json"))
        assert receipts
        for receipt in receipts:
            document = json.loads(receipt.read_text(encoding="utf-8"))
            assert document["tasks"]
            assert all(item["path"].endswith("inference_completion.json") for item in document["tasks"])
            assert all(item["regions"] for item in document["tasks"])
