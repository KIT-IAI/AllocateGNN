"""Recompute current-paper statistics from the fixedload_20260925_r2 exports.

This is CSV-to-statistics reproduction. It does not retrain the allocation
models, reconstruct spatial inputs, export the original task results, or rerun
the fixed-load connection calculation. Historical provenance and geometric
metadata are read from the frozen release and identified separately.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = REPO_ROOT / "study_materials/placement/current"
CODE_ROOT = REPO_ROOT / "StudyCase/Placement/current"
DEFAULT_OUTPUT = REPO_ROOT / "results/placement_current_reproduction"
COUNTRIES = {"uk": "GB", "au": "AU"}
BOUNDARY = (
    "Frozen task-seed/task-region and Step B CSVs to current-paper statistics "
    "and claim ledger. No model training, spatial-input reconstruction, "
    "task export, or Step B connection recomputation."
)
REUSED_METADATA = [
    "stats/stats_meta.json: Voronoi radii, input hashes and historical cross-checks",
    "stats/holm_family_disclosure.csv: historical nine-contrast family columns",
]


def _load_script(name: str):
    path = CODE_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"placement_current_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load current paper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _primitives():
    upstream = (CODE_ROOT / "upstream").resolve()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from sglib.analysis import paired_inference
    from sglib.experiment.connection_observations import rank_agreement

    for module_name in ("sglib.analysis.paired_inference", "sglib.experiment.connection_observations"):
        loaded = Path(sys.modules[module_name].__file__).resolve()
        if not loaded.is_relative_to(upstream):
            raise RuntimeError(f"{module_name} was loaded from {loaded}, expected bundled {upstream}")
    return paired_inference, rank_agreement


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(release_root: Path = RELEASE_ROOT) -> None:
    """Validate the public subset, not the original full-release SHA256SUMS."""
    release_root = Path(release_root).resolve()
    manifest = json.loads((release_root / "manifest.json").read_text(encoding="utf-8"))
    declared = set()
    for entry in manifest["files"]:
        relative = entry["path"]
        path = (release_root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(release_root):
            raise AssertionError(f"manifest path escapes release root: {relative}")
        if relative in declared:
            raise AssertionError(f"duplicate manifest path: {relative}")
        declared.add(relative)
        if not path.is_file():
            raise AssertionError(f"missing manifest file: {relative}")
        if path.stat().st_size != entry["bytes"]:
            raise AssertionError(f"byte-count mismatch: {relative}")
        if sha256(path).lower() != entry["sha256"].lower():
            raise AssertionError(f"SHA-256 mismatch: {relative}")
    actual = {p.relative_to(release_root).as_posix() for p in (release_root / "frozen").rglob("*") if p.is_file()}
    expected = {p for p in declared if p.startswith("frozen/")}
    if actual != expected:
        raise AssertionError(f"frozen allowlist mismatch: extra={sorted(actual - expected)}, missing={sorted(expected - actual)}")


def verify_source(code_root: Path = CODE_ROOT) -> int:
    """Verify the shipped upstream code and paper scripts before importing them."""
    code_root = Path(code_root).resolve()
    source = json.loads((code_root / "SOURCE.json").read_text(encoding="utf-8"))
    count = 0
    for key, base in (("upstream_files", code_root / "upstream"), ("paper_scripts", code_root)):
        declared = set()
        for entry in source[key]:
            relative = entry["path"]
            path = (base / relative).resolve()
            if Path(relative).is_absolute() or not path.is_relative_to(base):
                raise AssertionError(f"source path escapes {key}: {relative}")
            if relative in declared:
                raise AssertionError(f"duplicate source path in {key}: {relative}")
            declared.add(relative)
            if not path.is_file() or sha256(path).lower() != entry["sha256"].lower():
                raise AssertionError(f"source SHA-256 mismatch: {key}/{relative}")
            count += 1
    return count


def _task_csv(path: Path) -> pd.DataFrame:
    # Step C originally consumes in-memory task exports. Round-trip parsing
    # preserves those binary values, including tied regional percentages.
    result = pd.read_csv(path, float_precision="round_trip")
    result["matching"] = result["matching"].fillna("")
    return result


def _task_inputs(root: Path, cc: str):
    seeds = _task_csv(root / "tasks" / f"{cc}_task_seed_values.csv")
    keys = ["country", "region", "module", "metric", "matching", "method"]
    region = seeds.groupby(keys, dropna=False, sort=True).agg(
        value=("value", "mean"), n_seeds=("value", "size"), unit=("unit", "first")
    ).reset_index()
    exported = _task_csv(root / "tasks" / f"{cc}_task_region_values.csv")
    pd.testing.assert_frame_equal(region, exported, check_dtype=False, check_exact=False, rtol=1e-14, atol=1e-14)
    # Step B products were read with pandas' default parser by the original
    # paper2_statistics driver. Its parsing choice is part of reproduction.
    cost = pd.read_csv(root / f"{cc}_cost_oof.csv")
    elig = pd.read_csv(root / f"{cc}_ref_eligibility.csv")
    return seeds, region, cost, elig


def _module_tables(root: Path):
    statistics = _load_script("paper2_statistics")
    pi, _ = _primitives()
    modules, per_seed = [], []
    for cc in COUNTRIES:
        seeds, region, cost, elig = _task_inputs(root, cc)
        table, seed_table = statistics.module_rows(pi, region, seeds, cost, cc, set(elig[elig.ref_eligible].region))
        modules.append(table)
        per_seed.append(seed_table)
    return pd.concat(modules, ignore_index=True), pd.concat(per_seed, ignore_index=True)


def _endpoints(cost: pd.DataFrame, eligible: set, cc: str) -> list[dict]:
    # Same endpoint aggregation as paper2_statistics.main.
    rows = []
    for (radius, x), group in cost[cost.region.isin(eligible)].groupby(["radius_km", "dc_mw"]):
        for loss in ("L_E", "L_S"):
            means = group.groupby(["region", "method"])[loss].mean().unstack().mean()
            span = means["Uni"] - means["Ref"]
            rows.append({"country": cc, "radius_km": radius, "dc_mw": x, "loss": loss,
                         "n_regions": group.region.nunique(), **{m: means[m] for m in ("Uni", "LU", "GNN", "Ref")},
                         "endpoint_span_rel_to_Uni": span / means["Uni"] if means["Uni"] else None,
                         "pi_reported": cc == "uk",
                         "Pi_LU_pct": 100 * (means["Uni"] - means["LU"]) / span if span else None,
                         "Pi_GNN_pct": 100 * (means["Uni"] - means["GNN"]) / span if span else None})
    return rows


def _t25(root: Path) -> pd.DataFrame:
    rows = []
    for method, group in pd.read_csv(root / "uk_t25_sensitivity.csv").groupby("method"):
        means = group.groupby("region")[["L_S_T9", "L_S_T25", "selected_overlap", "oracle_overlap", "oracle_shift_km"]].mean()
        rows.append({"country": "uk", "method": method, "n_regions": len(means),
                     "regions_oracle_changed": int((means.oracle_overlap < 20).sum()),
                     "regions_selection_changed": int((means.selected_overlap < 20).sum()),
                     "regions_L_S_changed": int((~np.isclose(means.L_S_T9, means.L_S_T25, rtol=0, atol=1e-6)).sum()),
                     "median_L_S_T9": float(means.L_S_T9.median()), "median_L_S_T25": float(means.L_S_T25.median()),
                     "mean_L_S_T9": float(means.L_S_T9.mean()), "mean_L_S_T25": float(means.L_S_T25.mean()),
                     "max_oracle_shift_km": float(means.oracle_shift_km.max())})
    return pd.DataFrame(rows)


def recompute_tables(root: Path) -> dict[str, pd.DataFrame]:
    statistics = _load_script("paper2_statistics")
    pi, rank_agreement = _primitives()
    modules, seeds = _module_tables(root)
    tables = {"module_contrasts.csv": modules, "table3_four_module.csv": modules[modules.in_family].reset_index(drop=True),
              "per_seed_contrasts.csv": seeds}
    groups = {name: [] for name in ("panel_region_associations.csv", "panel_summary.csv", "panel_levels.csv",
                                  "scale_curves.csv", "adequacy_summary.csv", "adequacy_pass_curves.csv")}
    endpoints = []
    for cc in COUNTRIES:
        elig = pd.read_csv(root / f"{cc}_ref_eligibility.csv")
        panel = statistics.panel_stats(rank_agreement, pi, pd.read_csv(root / f"{cc}_controls.csv"), elig, cc)
        for name, table in zip(("panel_region_associations.csv", "panel_summary.csv", "panel_levels.csv"), panel):
            groups[name].append(table)
        totals = pd.read_csv(root / f"{cc}_region_support.csv").set_index("region").source_total
        for centres in ("station", "grid"):
            groups["scale_curves.csv"].append(statistics.scale_stats(pi, pd.read_csv(root / f"{cc}_scale_{centres}.csv"), cc, centres, totals))
        adequacy, curves = statistics.adequacy_stats(pd.read_csv(root / f"{cc}_adequacy.csv"), cc)
        groups["adequacy_summary.csv"].append(adequacy)
        groups["adequacy_pass_curves.csv"].append(curves)
        endpoints.extend(_endpoints(pd.read_csv(root / f"{cc}_cost_oof.csv"), set(elig[elig.ref_eligible].region), cc))
    tables.update({name: pd.concat(parts, ignore_index=True) for name, parts in groups.items()})
    tables["connection_endpoints.csv"] = pd.DataFrame(endpoints)
    tables["t25_summary.csv"] = _t25(root)
    # The current-family columns are recomputed. Upstream nine-family entries
    # remain explicitly identified historical metadata, not new calculations.
    disclosure = pd.read_csv(root / "stats/holm_family_disclosure.csv", float_precision="round_trip")
    for i, row in disclosure.iterrows():
        match = modules[modules.in_family & modules.country.eq(row.country) & modules.module.eq(row.module)].iloc[0]
        disclosure.loc[i, ["n", "p_raw_this_paper", "holm_p_four_module"]] = [match.n, match.p_raw, match.holm_p_4]
    tables["holm_family_disclosure.csv"] = disclosure
    return tables


def _compare_csv(actual: Path, expected: Path) -> None:
    """Check schema/order/labels exactly and numerical values to tight tolerance."""
    left = pd.read_csv(actual, float_precision="round_trip")
    right = pd.read_csv(expected, float_precision="round_trip")
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False, rtol=1e-11, atol=1e-10)
    except AssertionError as error:
        raise AssertionError(f"recomputed table differs from frozen reference: {expected.name}\n{error}") from error


def _write_tables(tables: dict[str, pd.DataFrame], output: Path) -> None:
    (output / "stats").mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / "stats" / name, index=False, float_format="%.17g", lineterminator="\n")


def _write_ledger(frozen: Path, output: Path) -> int:
    # Supply the original metadata to the original ledger builder alongside
    # freshly recomputed statistics. Nothing is written into the frozen package.
    for relative in ("uk_region_support.csv", "au_region_support.csv", "config.json", "uk_tariff_mapping.csv", "stats/stats_meta.json"):
        shutil.copyfile(frozen / relative, output / relative)
    ledger = _load_script("paper2_ledger")
    rows = ledger.build(output)
    (output / "ledger").mkdir(exist_ok=True)
    with (output / "ledger/claim_ledger.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ledger.COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _task_summary(row: pd.Series) -> dict:
    result = row.to_dict()
    result.update(n_regions=int(row.n), base_mean=float(row.lu_mean), median_change_pct=float(row.median_rel_pct),
                  regions_improved=int(row.wins), p_signflip=float(row.p_raw),
                  aggregate_relative_effect_pct=float(row.rel_effect_pct),
                  aggregate_relative_effect_ci95_pct=[float(row.rel_ci_lo), float(row.rel_ci_hi)])
    if "holm_p_4" in row and pd.notna(row.holm_p_4):
        result["p_holm_four_module"] = float(row.holm_p_4)
    return _clean(result)


def _tasks(modules: pd.DataFrame, seeds: pd.DataFrame) -> dict:
    output = {}
    for cc, label in COUNTRIES.items():
        items = {}
        relevant = modules[modules.country.eq(cc) & (modules.in_family | (modules.module.eq("sizing") & modules.matching.eq("one_to_one")))]
        for _, row in relevant.iterrows():
            key = "sizing_1to1" if row.matching == "one_to_one" else row.module
            record = _task_summary(row)
            match = seeds[seeds.country.eq(cc) & seeds.module.eq(row.module) & seeds.metric.eq(row.metric) & seeds.matching.eq(row.matching)]
            if row.module == "connection":
                match = match[match.radius_km.eq(row.radius_km) & match.dc_mw.eq(row.dc_mw)]
            record["per_seed"] = {str(int(s.seed)): _task_summary(s) for _, s in match.iterrows()}
            items[key] = record
        output[label] = items
    return output


def _country_records(frame: pd.DataFrame) -> dict:
    return {label: _clean(frame[frame.country.eq(cc)].to_dict(orient="records")) for cc, label in COUNTRIES.items()}


def reproduce_tasks(release_root: Path = RELEASE_ROOT) -> dict:
    return _tasks(*_module_tables(Path(release_root) / "frozen"))


def reproduce_controls(release_root: Path = RELEASE_ROOT) -> dict:
    statistics = _load_script("paper2_statistics")
    pi, rank = _primitives()
    root = Path(release_root) / "frozen"
    output = {}
    for cc, label in COUNTRIES.items():
        tables = statistics.panel_stats(rank, pi, pd.read_csv(root / f"{cc}_controls.csv"), pd.read_csv(root / f"{cc}_ref_eligibility.csv"), cc)
        output[label] = {key: _clean(table.to_dict(orient="records")) for key, table in zip(("regions", "summary", "levels"), tables)}
    return output


def reproduce_scale(release_root: Path = RELEASE_ROOT) -> dict:
    statistics = _load_script("paper2_statistics")
    pi, _ = _primitives()
    root = Path(release_root) / "frozen"
    output = {}
    for cc, label in COUNTRIES.items():
        totals = pd.read_csv(root / f"{cc}_region_support.csv").set_index("region").source_total
        frame = statistics.scale_stats(pi, pd.read_csv(root / f"{cc}_scale_station.csv"), cc, "station", totals)
        output[label] = _clean(frame.to_dict(orient="records"))
    return output


def reproduce_adequacy(release_root: Path = RELEASE_ROOT) -> dict:
    statistics = _load_script("paper2_statistics")
    root = Path(release_root) / "frozen"
    return {label: _clean(statistics.adequacy_stats(pd.read_csv(root / f"{cc}_adequacy.csv"), cc)[0].to_dict(orient="records")) for cc, label in COUNTRIES.items()}


def verify_headlines(data: dict) -> None:
    gb, au = data["tasks"]["GB"], data["tasks"]["AU"]
    for actual, expected in ((gb["reconstruction"]["base_mean"], 10.462268014399758),
                             (gb["reconstruction"]["gnn_mean"], 9.635992845183765),
                             (gb["connection"]["base_mean"], 11592140.985882353),
                             (gb["connection"]["gnn_mean"], 8629419.950455109)):
        if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-10):
            raise AssertionError(f"current-paper headline mismatch: {actual} != {expected}")
    if gb["connection"]["regions_improved"] != 16 or gb["connection"]["n_regions"] != 16:
        raise AssertionError("current GB fixed-location connection must improve in 16/16 regions")
    for module in ("reconstruction", "siting", "sizing", "connection"):
        if au[module]["gnn_mean"] <= au[module]["base_mean"]:
            raise AssertionError(f"current AU {module} must favour LU on the country mean")
    for country in (gb, au):
        family = {name for name, values in country.items() if values["in_family"]}
        if family != {"reconstruction", "siting", "sizing", "connection"}:
            raise AssertionError(f"unexpected four-module Holm family: {family}")


def reproduce(*, release_root: Path = RELEASE_ROOT, output: Path | None = None, verify: bool = False) -> dict:
    release_root = Path(release_root)
    source_count = None
    if verify:
        verify_manifest(release_root)
        source_count = verify_source()
    frozen = release_root / "frozen"
    tables = recompute_tables(frozen)
    result = {
        "release": {"id": "fixedload_20260925_r2", "reproducibility_boundary": BOUNDARY,
                    "reused_frozen_metadata": REUSED_METADATA,
                    "verification": "not requested"},
        "tasks": _tasks(tables["module_contrasts.csv"], tables["per_seed_contrasts.csv"]),
        "controls": {"summary": _country_records(tables["panel_summary.csv"]), "levels": _country_records(tables["panel_levels.csv"])},
        "scale": _country_records(tables["scale_curves.csv"]),
        "adequacy": _country_records(tables["adequacy_summary.csv"]),
    }
    if output is None:
        DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        workspace = tempfile.TemporaryDirectory(prefix=".placement_current_", dir=DEFAULT_OUTPUT.parent)
    else:
        workspace = nullcontext(str(Path(output).resolve()))
    with workspace as temporary:
        destination = Path(temporary).resolve()
        # A caller must never direct generated statistics into the reference package.
        if destination == release_root.resolve() or destination.is_relative_to(release_root.resolve()):
            raise ValueError("output must be outside the frozen release package")
        _write_tables(tables, destination)
        result["release"]["claim_count"] = _write_ledger(frozen, destination)
        if verify:
            for name in tables:
                _compare_csv(destination / "stats" / name, frozen / "stats" / name)
            _compare_csv(destination / "ledger/claim_ledger.csv", frozen / "ledger/claim_ledger.csv")
            verify_headlines(result)
            result["release"].update(verification="PASS", verified_statistical_tables=len(tables),
                                     verified_source_files=source_count,
                                     byte_identical_artifacts=sum(
                                         sha256(destination / "stats" / name) == sha256(frozen / "stats" / name)
                                         for name in tables
                                     ) + int(sha256(destination / "ledger/claim_ledger.csv") == sha256(frozen / "ledger/claim_ledger.csv")),
                                     comparison="all cells, row order and schema, rtol=1e-11 and atol=1e-10 for numerical cells")
        if output is not None:
            (destination / "paper_numbers.json").write_text(json.dumps(_clean(result), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return _clean(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify subset hashes, recomputed tables, ledger and current headline contracts")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--release-root", type=Path, default=RELEASE_ROOT)
    parser.add_argument("--stdout", action="store_true", help="print JSON without retaining generated tables")
    args = parser.parse_args()
    result = reproduce(release_root=args.release_root, output=None if args.stdout else args.output, verify=args.verify)
    if args.stdout:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"paper_numbers={args.output / 'paper_numbers.json'}")
    if args.verify:
        print("verification=PASS", file=sys.stderr if args.stdout else sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
