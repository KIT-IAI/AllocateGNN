"""Reuse a real tiny run, and reject missing, changed or disconnected artifacts."""
from contextlib import contextmanager
import json
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from sglib.core.infra.hashing import sha256_file
from sglib.core.infra.terms import load_country_profile
from sglib.dataoverview.handoff import DataOverviewBundle
from sglib.examples import nl

pytestmark = pytest.mark.consume
ROOT = Path(__file__).resolve().parents[1]


def _synthetic_nl_handoff(repo):
    regions = [f"fixture_{i}" for i in range(4)]
    sources, stations, grids = [], [], []
    for i, region in enumerate(regions):
        x, y = 120000 + i * 5000, 480000 + i * 5000
        keys = [region + "_A", region + "_B"]
        source = gpd.GeoDataFrame({
            "wijk_code": keys, "analysis_region": [region] * 2,
            **{name + "_percent": [.2, .2] for name in
               ("residential", "commercial", "industrial", "agricultural", "others")},
        }, geometry=[box(x, y, x + 800, y + 800), box(x + 1000, y, x + 1800, y + 800)], crs="EPSG:28992")
        station = gpd.GeoDataFrame({
            "station_id": [region + "_s" + str(j) for j in range(4)],
            "wijk_code": [keys[0]] * 2 + [keys[1]] * 2,
            "peak_kw": [3., 7., 5., 15.], "capacity_kw": [100.] * 4,
        }, geometry=[Point(x + 200 + (j // 2) * 1000, y + 200 + (j % 2) * 300)
                     for j in range(4)], crs="EPSG:28992")
        source, station, grid = nl._fixture_region(region, source.to_crs(4326), station.to_crs(4326))
        sources.append(source)
        stations.append(station)
        grids.append(grid)
    profile = load_country_profile(repo / "casestudy/config/countries/nl.toml")
    return DataOverviewBundle(
        "nl", profile,
        gpd.GeoDataFrame(pd.concat(sources, ignore_index=True), crs="EPSG:4326"),
        gpd.GeoDataFrame(pd.concat(stations, ignore_index=True), crs="EPSG:4326"),
        tuple(grids), {"schema_version": "sg_dataoverview_inventory_v1", "formal": False}, {}, None,
    ), regions


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    repo = tmp_path_factory.mktemp("real-smoke-reuse") / "public-repo"
    repo.mkdir()
    shutil.copytree(ROOT / "casestudy", repo / "casestudy")
    (repo / "data").mkdir()
    for name in ("pyproject.toml", "data/metadata.toml"):
        shutil.copyfile(ROOT / name, repo / name)
    root = repo / "results/_smoke/complete"
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("SLURM_JOB_ID", raising=False)
        patch.setattr(nl, "_handoff", lambda *args: _synthetic_nl_handoff(repo))
        receipt = nl.run(repo, root)
    return repo, root, receipt


def _state(root):
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns, sha256_file(p))
            for p in root.rglob("*") if p.is_file()}


@contextmanager
def _changed(path, *, payload=None, missing=False):
    original = path.read_bytes()
    try:
        if missing:
            path.unlink()
        else:
            path.write_bytes(payload if payload is not None else original + b"corrupted")
        yield
    finally:
        path.write_bytes(original)


def test_complete_actual_run_reuses_without_computation_or_writes(completed_run, monkeypatch):
    repo, root, receipt = completed_run
    before = _state(root)

    def forbidden(*args, **kwargs):
        raise AssertionError("reuse must not read upstream data, train, or infer")

    for name in ("_handoff", "prepare_inputs", "run_training_task", "run_inference_task"):
        monkeypatch.setattr(nl, name, forbidden)
    assert nl.run(repo, root) == receipt
    assert _state(root) == before


@pytest.mark.parametrize("relative", (
    "inputs/receipt.json", "inputs/bundle.pkl", "inputs/worker_params.json",
    "inputs/fixture/fixture_0/grid.parquet", "training/graphs/lu5.pkl",
    "2_Weighter/base/gnn/seed_42/fold1/model.pth",
    "2_Weighter/base/mlp/seed_42/fold1/model.pth",
    "inference/outputs/B-NL-GNN/infer-B-NL-GNN-S42-F1/fields/fixture_0.npz",
    "inference/outputs/B-NL-MLP/infer-B-NL-MLP-S42-F1/fields/fixture_0.npz",
    "civd/fixture_0.npz",
))
@pytest.mark.parametrize("missing", (True, False))
def test_reuse_rejects_missing_or_changed_real_products(completed_run, relative, missing):
    repo, root, _ = completed_run
    with _changed(root / relative, missing=missing):
        before = _state(root)
        with pytest.raises(ValueError, match="artifacts"):
            nl.run(repo, root)
        assert _state(root) == before


@pytest.mark.parametrize("relative", (
    "2_Weighter/base/gnn/seed_42/fold1/model.pth",
    "static/uniform/fixture_0.npz",
    "civd/fixture_0.npz",
    "candidates/index_Uni.json",
))
def test_missing_required_product_cannot_be_hidden_by_removing_its_manifest_entry(completed_run, relative):
    repo, root, receipt = completed_run
    document = json.loads(receipt.read_text(encoding="utf-8"))
    document["artifacts"].pop(relative)
    with _changed(root / relative, missing=True), _changed(receipt, payload=json.dumps(document).encode()):
        before = _state(root)
        with pytest.raises(ValueError, match="artifacts"):
            nl.run(repo, root)
        assert _state(root) == before


def test_matching_file_digest_does_not_hide_a_broken_embedded_chain(completed_run):
    repo, root, receipt = completed_run
    relative = "inference/outputs/B-NL-GNN/infer-B-NL-GNN-S42-F1/inference_completion.json"
    completion = json.loads((root / relative).read_text(encoding="utf-8"))
    completion["chain"]["commitment"]["inputs"]["checkpoint"] = "0" * 64
    payload = json.dumps(completion).encode()
    document = json.loads(receipt.read_text(encoding="utf-8"))
    import hashlib
    document["artifacts"][relative] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    with _changed(root / relative, payload=payload), _changed(receipt, payload=json.dumps(document).encode()):
        before = _state(root)
        with pytest.raises(ValueError, match="(?i)(chain|commitment)"):
            nl.run(repo, root)
        assert _state(root) == before


def test_reuse_rejects_an_extra_unreceipted_product(completed_run):
    repo, root, _ = completed_run
    extra = root / "unreceipted.txt"
    extra.write_text("unexpected product", encoding="utf-8")
    try:
        before = _state(root)
        with pytest.raises(ValueError, match="artifacts"):
            nl.run(repo, root)
        assert _state(root) == before
    finally:
        extra.unlink()
