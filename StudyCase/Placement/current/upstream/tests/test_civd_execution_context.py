"""Synthetic CIVD production and execution-context dependency isolation."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from sglib.core.infra.hashing import sha256_file
from sglib.experiment import stage
from sglib.experiment import allocator_observations
from sglib.experiment.civd_observations import check_equal_split, checked_receipt
from sglib.experiment.reconstruction import ReconstructionRegion
from sglib.generator.civd_extension import extend_country
from sglib.generator.downstream import GeneratorHandoff
from sglib.generator.handoff import CandidateField, CanonicalVD, GeneratorBundle, IdrMatchedView, IdrView

pytestmark = pytest.mark.consume
ROOT = Path(__file__).resolve().parents[1]


def test_country_handoffs_are_scoped_to_execution_context(tmp_path):
    calls = []
    contexts = []
    for label in ("first", "second"):
        result = tmp_path / label
        result.mkdir()
        (result / "input.txt").write_text(label, encoding="utf-8")
        def loader(repo, country, output=result):
            value = (output / "input.txt").read_text(encoding="utf-8")
            calls.append((country, output))
            return SimpleNamespace(root=output, value=value)
        contexts.append(stage.country_context(ROOT, "nl", profile="smoke", results_root=result,
            upstream={"generator": loader}))
    for ctx, label in zip(contexts, ("first", "second"), strict=True):
        first = stage.load_upstream(ctx, "generator")
        assert stage.load_upstream(ctx, "generator") is first
        assert first.root == ctx.results and first.value == label
        (first.root / "output.txt").write_text(first.value, encoding="utf-8")
    assert len(calls) == 2 and contexts[0].handoffs is not contexts[1].handoffs
    replacement = SimpleNamespace(root=tmp_path / "extended")
    extended = stage.with_upstream(contexts[0], generator=replacement)
    assert not extended.handoffs
    assert stage.load_upstream(extended, "generator") is replacement
    assert stage.load_upstream(contexts[0], "generator").value == "first"
    assert (contexts[1].results / "output.txt").read_text(encoding="utf-8") == "second"


def _synthetic_inputs(repo):
    station_xy = np.array([[0., 0.], [5., 0.], [500., 0.], [505., 0.]])
    grid_xy = np.array([[1., 0.], [3., 0.], [501., 0.], [503., 0.]])
    ids = np.array(["s0", "s1", "s2", "s3"])
    stations = gpd.GeoDataFrame({"id": ids, "capacity": [1., 2., 3., 4.]},
        geometry=gpd.points_from_xy(*station_xy.T), crs="EPSG:3857")
    grid = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*grid_xy.T), crs=stations.crs)
    contract = {"id_column": "id", "capacity_column": "capacity", "capacity_basis": "fixture"}
    profile = SimpleNamespace(directory="4_NL", crs={"working": "EPSG:3857"}, station_contract=contract)
    data = SimpleNamespace(profile=profile, stations_table=stations,
        regions=(SimpleNamespace(region="r", grid=grid),))
    vd = CanonicalVD("r", np.arange(4), ids, {"sha256": "a" * 64})
    field = CandidateField("GNN", "learned", "r", np.array([2., 3., 7., 8.]), False,
        seed=42, lineage={"sha256": "b" * 64})
    gate = {"sha256": "c" * 64, "g0_pass": True, "g1_pass": True,
        "selected_mode": "candidate", "candidate_sha256": field.lineage["sha256"]}
    fixed = IdrView("r", vd.assignment, vd.assignment, gate)
    matched = IdrMatchedView("GNN", 42, "r", vd.assignment, vd.assignment, gate)
    bundle = GeneratorBundle("nl", (field,), (), (fixed,), (matched,), (), "d" * 64, vd=(vd,))
    generator = GeneratorHandoff(bundle, repo / "results/2_Generator/4_NL",
        {"crs": profile.crs, "station_contract": contract}, "e" * 64, "f" * 64)
    region = ReconstructionRegion("nl", "r", "MW", ids, np.repeat("source", 4), np.array([3., 4., 6., 7.]),
        {"source": 20.}, np.repeat("source", 4), vd.assignment, (field,), "e" * 64,
        station_xy, grid_xy, (), True)
    return data, generator, region


def test_local_civd_materialization_observation_and_receipts_are_isolated(tmp_path):
    data, original, region = _synthetic_inputs(tmp_path)
    handoffs, tables = [], []
    for label in ("first", "second"):
        results = tmp_path / "results/_staging" / label
        extended = extend_country(tmp_path, "nl", results, data=data, generator=original, regions=("r",))
        resumed = extend_country(tmp_path, "nl", results, data=data, generator=original, regions=("r",))
        np.testing.assert_array_equal(extended.bundle.civd[0].assignment, resumed.bundle.civd[0].assignment)
        view = extended.bundle.civd[0]
        artifact = tmp_path / view.metadata["artifact_root"] / view.metadata["path"]
        assert artifact.is_relative_to(results) and sha256_file(artifact) == view.metadata["sha256"]
        receipt = checked_receipt(results / "2_Generator/4_NL/civd")
        assert receipt["commitment"]["inputs"]["formal_generator_inputs"] == original.inputs_receipt_sha256
        assert not list(results.rglob("_closures"))
        output = allocator_observations.observe([region], extended.bundle.idr_fixed,
            extended.bundle.idr_matched, extended.bundle.civd, extended.bundle.vd)
        checks = check_equal_split(region, view, output["predictions"])
        assert checks[0]["status"] == "PASS" and checks[0]["prediction_total"] == 20.
        handoffs.append(extended)
        tables.append(output)
    assert original.bundle.civd == ()
    assert handoffs[0].bundle.civd[0].metadata["artifact_root"] != handoffs[1].bundle.civd[0].metadata["artifact_root"]
    # Generated arrays and numerical output agree; lineage records correctly differ by result root.
    for name in ("predictions", "metrics", "gates", "maps"):
        pd.testing.assert_frame_equal(tables[0][name], tables[1][name])
