"""Descriptive CIVD statistics and provenance from synthetic regional tables."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.consume


def _comparison_fixture(monkeypatch, tmp_path):
    from sglib.analysis import civd_comparison as module
    units = {"uk": "MVA", "au": "MVA", "nl": "%_industrial_sites", "nz": "MW"}
    specifications = {country: {"unit": units[country], "regions": ["r0", "r1", "r2"],
        "method_order": ["MLP", "GPM", "EqualStation"], "methods": {"MLP": {"seeds": [42, 123, 456]},
        "GPM": {"seeds": [None]}, "EqualStation": {"seeds": [None], "evidence": ["direct_reference"]}}}
        for country in module.COUNTRIES}
    monkeypatch.setattr(module, "load_analysis_config", lambda repo, country: SimpleNamespace(specification=specifications[country]))
    results = tmp_path / "results/_staging/corrected"
    for country in module.COUNTRIES:
        rows = []
        for candidate in ("GPM", "MLP"):
            for allocator in module.ALLOCATORS:
                errors = [1., 4., 8.] if allocator == "CIVD" else [3., 5., 7.] if allocator == "IDR-fixed" else [2., 4., 6.]
                scores = [.3, .4, .4] if allocator == "CIVD" else [.2, .4, .6]
                for metric in module.METRICS:
                    values = errors if metric == "rmse" else [v/2 for v in errors] if metric == "mae" else [v/10 for v in errors] if metric == "wape" else scores
                    for region, value in zip(specifications[country]["regions"], values, strict=True):
                        invalid = metric == "corr" and allocator == "CIVD" and region == "r2"
                        rows.append({"country": country, "region": region, "candidate": candidate, "allocator": allocator,
                            "metric": metric, "value": np.nan if invalid else value,
                            "status": "METRIC_NOT_ASSESSABLE" if invalid else "VALID",
                            "unit": units[country] if metric in ("rmse", "mae") else "dimensionless",
                            "realization_count": 3 if candidate == "MLP" else 1, "n_targets": 6})
        frame = pd.DataFrame(rows)
        destination = results / "4_Analysis" / module.DIRECTORIES[country] / "C3/region_metrics.csv"
        destination.parent.mkdir(parents=True)
        frame.to_csv(destination, index=False)
        commitment = module.derive_chain_commitment(f"Analysis.{country}.C3", inputs={},
            scientific_parameters={"fixture": True}, code_sha256="a" * 64)
        receipt = module.derive_chain_receipt(commitment, outputs={"region_metrics.csv": module.sha256_file(destination)})
        module.atomic_json(receipt, destination.parent / "receipt.json")
        if country in ("uk", "au"):
            old = tmp_path / "results/4_Analysis" / module.DIRECTORIES[country] / "C3/region_metrics.csv"
            old.parent.mkdir(parents=True)
            frame.loc[frame.allocator.eq("CIVD") & frame.metric.isin(["rmse", "mae"]), "value"] += 10
            frame.to_csv(old, index=False)
            module.atomic_json(module.derive_chain_receipt(commitment,
                outputs={"region_metrics.csv": module.sha256_file(old)}), old.parent / "receipt.json")
    return module, results


def test_civd_four_country_comparison_keeps_region_pairs_units_and_historical_values(monkeypatch, tmp_path):
    import inspect
    import json

    module, results = _comparison_fixture(monkeypatch, tmp_path)
    result = module.build_comparison(tmp_path, results)
    destination = results / "civd_comparison"
    assert result["rows"] == {"country_summary.csv": 160, "paired_comparisons.csv": 120,
                              "region_comparisons.csv": 360, "implementation_before_after.csv": 8}
    assert result["scope"] == "posthoc_correction_comparison"
    summary = pd.read_csv(destination / "country_summary.csv")
    assert summary.candidate.iloc[0] == "MLP"
    assert summary[summary.metric.eq("rmse")].groupby("country").unit.first().to_dict() == {
        "au": "MVA", "nl": "%_industrial_sites", "nz": "MW", "uk": "MVA"}
    paired = pd.read_csv(destination / "paired_comparisons.csv")
    rmse = paired.query("country == 'uk' and candidate == 'MLP' and baseline == 'VD' and metric == 'rmse'").iloc[0]
    assert rmse.n_pairs == 3 and rmse.n_improved == rmse.n_tied == rmse.n_worse == 1
    assert rmse.mean_delta == pytest.approx(1/3)
    assert rmse.relative_mean_pct == pytest.approx(100/12)
    expected = module._bootstrap(np.array([1., 4., 8.]), np.array([2., 4., 6.]), "rmse")
    assert rmse.delta_ci_low == expected["delta_ci_low"] and rmse.delta_ci_high == expected["delta_ci_high"]
    corr = paired.query("country == 'uk' and candidate == 'MLP' and baseline == 'VD' and metric == 'corr'").iloc[0]
    assert corr.n_pairs == 2 and corr.n_excluded == 1 and corr.status == "PARTIAL_VALID"
    assert corr.n_improved == 1 and corr.n_tied == 1 and corr.n_worse == 0
    assert np.isnan(corr.relative_mean_pct) and np.isnan(corr.relative_ci_low_pct)
    assert corr.relative_reason == "RELATIVE_PERCENT_NOT_DEFINED_FOR_SIGNED_SCORE"
    history = pd.read_csv(destination / "implementation_before_after.csv")
    assert history.correction_delta.eq(-10.).all() and not history.old_implementation_valid.any()
    for filename in ("four_country_rmse.png", "four_country_rmse.pdf", "README.md", "summary.md", "provenance.json", "receipt.json"):
        assert (destination / filename).stat().st_size > 100
    receipt = module.verify_chain(json.loads((destination / "receipt.json").read_text(encoding="utf-8")))
    assert receipt["receipt_sha256"] == result["receipt_sha256"]
    expected_symbols = {name for name, fn in vars(module).items() if inspect.isfunction(fn) and fn.__module__ == module.__name__}
    projection = receipt["observations"]["code_projection"]
    assert set(projection["symbols"]) == expected_symbols
    assert receipt["commitment"]["code_sha256"] == projection["code_sha256"]
    assert len(receipt["commitment"]["inputs"]) == 10
    assert set(receipt["outputs"]) == {path.name for path in destination.iterdir()} - {"receipt.json"}
    assert all(module.sha256_file(destination / name) == digest for name, digest in receipt["outputs"].items())
    pdf = destination / "four_country_rmse.pdf"
    assert b"/CreationDate (D:20260914000000" in pdf.read_bytes()
    assert b"/ModDate (D:20260914000000" in pdf.read_bytes()
    repeated = tmp_path / "repeated_figure"
    repeated.mkdir()
    specs = {cc: module.load_analysis_config(tmp_path, cc).specification for cc in module.COUNTRIES}
    module._plot(summary, specs, repeated)
    assert module.sha256_file(pdf) == module.sha256_file(repeated / pdf.name)


def test_civd_four_country_comparison_rejects_missing_allocator_coordinate(monkeypatch, tmp_path):
    module, results = _comparison_fixture(monkeypatch, tmp_path)
    path = results / "4_Analysis/5_NZ/C3/region_metrics.csv"
    frame = pd.read_csv(path)
    frame.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="incomplete regional allocator coordinates"):
        module.build_comparison(tmp_path, results)
    assert not (results / "civd_comparison").exists()


def test_civd_four_country_comparison_rejects_entire_omitted_candidate(monkeypatch, tmp_path):
    module, results = _comparison_fixture(monkeypatch, tmp_path)
    path = results / "4_Analysis/5_NZ/C3/region_metrics.csv"
    frame = pd.read_csv(path)
    frame[frame.candidate.ne("GPM")].to_csv(path, index=False)
    with pytest.raises(ValueError, match="omitted expected candidate"):
        module.build_comparison(tmp_path, results)
    assert not (results / "civd_comparison").exists()


def test_civd_four_country_comparison_rejects_csv_outside_parent_receipt(monkeypatch, tmp_path):
    module, results = _comparison_fixture(monkeypatch, tmp_path)
    path = results / "4_Analysis/5_NZ/C3/region_metrics.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "value"] += 1.
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="differs from its content-chain receipt"):
        module.build_comparison(tmp_path, results)
    assert not (results / "civd_comparison").exists()


def test_civd_bootstrap_records_zero_baseline_and_one_region(monkeypatch, tmp_path):
    module, _ = _comparison_fixture(monkeypatch, tmp_path)
    result = module._bootstrap(np.array([2.]), np.array([0.]), "rmse")
    assert result["delta_ci_low"] == result["delta_ci_high"] == 2.
    assert result["interval_status"] == "DEGENERATE_ONE_REGION"
    assert result["zero_denominator_draws"] == 10000 and result["relative_ci_low_pct"] is None




def test_prior_corrected_formal_result_is_not_mislabeled_as_invalid(monkeypatch, tmp_path):
    import json
    module, results = _comparison_fixture(monkeypatch, tmp_path)
    for country in ("uk", "au"):
        root = tmp_path / "results/4_Analysis" / module.DIRECTORIES[country] / "C3"
        prior = module.verify_chain(json.loads((root / "receipt.json").read_text(encoding="utf-8")))
        commitment = module.derive_chain_commitment(prior["node_id"], inputs={},
            scientific_parameters={"correction_scope": "four_country_posthoc_bugfix_comparison"},
            code_sha256="a" * 64)
        module.atomic_json(module.derive_chain_receipt(commitment, outputs=prior["outputs"]), root / "receipt.json")
    corrected = pd.concat([module._read_country(tmp_path, results, country)[0] for country in module.COUNTRIES])
    comparison, sources = module._historical_comparison(tmp_path, corrected)
    assert comparison.old_implementation_valid.all()
    assert comparison.old_invalid_implementation_mean.isna().all()
    assert comparison.prior_implementation_mean.notna().all()
    assert comparison.withdrawal_reason.isna().all()
    assert all(source["implementation_status"] == "prior_corrected" for source in sources.values())
