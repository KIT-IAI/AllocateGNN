import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import box
from types import SimpleNamespace

from sglib.analysis import support
from sglib.experiment import boundary_diagnostics, defenses

pytestmark = pytest.mark.consume


def _t1_fixture():
    xy = np.c_[np.arange(6) * 10., np.zeros(6)]
    stations = gpd.GeoDataFrame({"station_name": [f"Site {i}" for i in range(6)]},
        geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs="EPSG:3857")
    reference = gpd.GeoDataFrame({"reference_id": np.arange(6), "reference_name": stations.station_name,
        "provenance": "independent"}, geometry=[box(x-5, y-5, x+5, y+5) for x, y in xy], crs=stations.crs)
    views = [{"allocator": "VD", "candidate": None, "assignment_kind": "station", "assignment": np.arange(6)},
             {"allocator": "IDR-fixed", "candidate": "PUBLIC_GPM_SHAPE", "assignment_kind": "station", "assignment": np.arange(6)},
             {"allocator": "CIVD", "candidate": None, "assignment_kind": "cluster", "assignment": np.array([0, 0, 0, 1, 1, 1]),
              "station_cluster": np.array([0, 0, 1, 1, 1, 1])}]
    return xy, np.ones(6, bool), stations, reference, views


def test_civd_t1_preserves_matching_without_fabricating_station_footprints():
    mapping, station, regional = boundary_diagnostics.evaluate(*_t1_fixture())
    assert mapping.accepted.all() and mapping.region_eligible.all()
    for table in (station, regional):
        civd = table[table.allocator.eq("CIVD")]
        assert civd.status.eq("METRIC_NOT_ASSESSABLE").all()
        assert civd.reason.eq("CIVD_CLUSTER_ALLOCATION_HAS_NO_UNIQUE_STATION_FOOTPRINT").all()
        assert civd.assignment_kind.eq("cluster").all()
        other = table[~table.allocator.eq("CIVD")]
        assert other.status.eq("VALID").all()
        metric = "iou_loss" if table is station else "iou_loss_q80"
        assert civd[metric].isna().all() and other[metric].eq(0.).all()
        assert "assignment" not in table and "station_cluster" not in table
    assert station[station.allocator.eq("CIVD")].reference_area_m2.eq(100.).all()
    assert station[station.allocator.eq("CIVD")].estimated_area_m2.isna().all()
    assert regional[regional.allocator.eq("CIVD")].iloc[0].n_matched == 6
    assert regional[regional.allocator.eq("CIVD")].iloc[0].n_valid == 0


@pytest.mark.parametrize("change, message", [
    ({"assignment_kind": None}, "explicit assignment_kind"),
    ({"assignment_kind": "station"}, "must be cluster"),
    ({"station_cluster": np.zeros(6, dtype=int)}, "every assigned cluster"),
    ({"station_cluster": np.zeros(5, dtype=int)}, "every assigned cluster"),
])
def test_t1_rejects_ambiguous_or_invalid_cluster_identity(change, message):
    fixture = _t1_fixture()
    fixture[-1][-1].update(change)
    with pytest.raises(ValueError, match=message):
        boundary_diagnostics.evaluate(*fixture)


def test_t1_cluster_labels_are_not_station_ordinals():
    fixture = _t1_fixture()
    view = fixture[-1][-1]
    # Assignments index the ordered labels; only the labels may be noncontiguous.
    view["station_cluster"] += 100
    _, _, regional = boundary_diagnostics.evaluate(*fixture)
    assert regional[regional.allocator.eq("CIVD")].status.eq("METRIC_NOT_ASSESSABLE").all()


def test_c3_t1_summary_retains_nonassessable_status_and_reason():
    _, station, regional = boundary_diagnostics.evaluate(*_t1_fixture())
    for table in (station, regional):
        table["country"], table["region"] = "uk", "r"
    gates = pd.DataFrame(columns=["country", "candidate", "allocator", "assignment_sha256", "region", "changed_grid_count",
        "fallback", "g0_pass", "g1_pass", "candidate_tv_mass", "max_station_mass_change"])
    metrics = pd.DataFrame(columns=["allocator"])
    result = support.c3_support(gates, regional, station, metrics)["C3_T1_summary"]
    civd = result[result.allocator.eq("CIVD")].iloc[0]
    assert civd.status == "METRIC_NOT_ASSESSABLE" and civd.assignment_kind == "cluster"
    assert civd.reason == "CIVD_CLUSTER_ALLOCATION_HAS_NO_UNIQUE_STATION_FOOTPRINT"
    assert civd.valid_regions == 0 and np.isnan(civd.mean_iou_loss_q80)


def test_production_t1_passes_explicit_assignment_identity(monkeypatch, tmp_path):
    from sglib.experiment import production

    xy, cu_support, stations, reference, views = _t1_fixture()
    reference = reference.rename(columns={"reference_id": "UPID", "reference_name": "PRIMARY_NAME",
                                          "provenance": "DNO_LICENCE_AREA_NAME"})
    monkeypatch.setattr(production.gpd, "read_file", lambda _: reference)
    monkeypatch.setattr(production, "sha256_file", lambda _: "a" * 64)
    ctx = SimpleNamespace(repo=tmp_path, loaded=SimpleNamespace(values={"t1": {
        "reference": "reference.gpkg", "reference_sha256": "a" * 64}}))
    region = SimpleNamespace(country="uk", region="r", grid_xy=xy)
    arrays = {"station_names": stations.station_name.to_numpy(), "station_xy": xy, "cu_support": cu_support}
    allocators = {}
    for kind, view in zip(("vd", "fixed", "civd", "matched"), [*views, views[0]], strict=True):
        allocators[kind] = [{**view, "seed": None, "sha256": "b" * 64,
                            "object": SimpleNamespace(station_cluster=views[-1]["station_cluster"])}]
    result, _ = production.t1_tables(ctx, region, arrays, allocators,
        {"reference_filter": ["independent"], "working_crs": "EPSG:3857"})
    regional = result["region_metrics"]
    assert regional[regional.allocator.eq("CIVD")].assignment_kind.eq("cluster").all()
    assert regional[regional.allocator.eq("CIVD")].status.eq("METRIC_NOT_ASSESSABLE").all()
    assert regional[~regional.allocator.eq("CIVD")].assignment_kind.eq("station").all()
    assert regional[~regional.allocator.eq("CIVD")].iou_loss_q80.eq(0.).all()


def test_fixed_load_and_connection_map_keep_candidate_identity():
    surface = {
        "candidate_id": np.arange(20), "field_labels": np.array(["GPM", "GNN"]),
        "field_seeds": np.array([0, 42]), "unit": np.array("MVA"),
        "radii_km": np.array([10., 20.]), "lambdas": np.array([.25, .5, 1.]),
        "X": np.array([10., 20., 40.]), "G": np.vstack([np.arange(20), np.arange(20)+1]),
        "F": np.ones((2, 20))*5, "Ghat": np.stack([
            np.vstack([np.arange(20)+.5, np.arange(20)+1.5]),
            np.vstack([np.arange(20)+1, np.arange(20)+2])])}
    fixed = defenses.fixed_load_observations("uk", {"r": surface})
    assert len(fixed) == 4 and fixed.status.eq("VALID").all()
    mapped = defenses.connection_map_observations("uk", "r", surface, np.c_[np.arange(20), np.arange(20)])
    assert len(mapped) == 40 and set(mapped.candidate) == {"GPM", "GNN"}
    assert mapped.groupby("candidate").selected.sum().eq(20).all()








def test_c2_support_preserves_six_protocols_and_does_not_select_scan_parameter():
    protocols = [(station, eps) for station in ("all", "positive_truth") for eps in (1e-8, 1e-6, 1e-4)]
    correction = pd.DataFrame([{"country": "uk", "region": "r", "base": "GPM", "corrected": "GPMpostN",
        "operator": "multiply", "signal": "N", "seed": np.nan, "fold": np.nan, "station_set": station,
        "epsilon_ratio": eps, "identifiable": True, "identity_pass": True,
        "old_threshold_agrees_log": eps != 1e-4, "old_threshold_agrees_rmse": True,
        "log_mse_help": eps != 1e-4, "linear_rmse_help": True, "old_threshold_help": True,
        "delta_rmse": -1., "rho": .5, "old_threshold": .2, "mean_term": .1,
        "variance_term": -.2, "identity_tolerance": 1e-12, "deployment_status": "not_assessable"}
        for station, eps in protocols])
    sweeps = pd.DataFrame([{"country": "uk", "region": "r", "parameter_name": "alpha", "parameter_value": value,
        "signal": "N", "metric": "rmse", "evidence": "fixed", "value": value, "status": "VALID"}
        for value in (0., .5, 1.)])
    output = support.c2_support(correction, sweeps)
    assert output["C2_protocol_flips"].iloc[0].n_protocols == 6
    assert output["C2_protocol_flips"].iloc[0].log_direction_flipped
    assert not output["C2_sweep_curves"].parameter_selected.any()
