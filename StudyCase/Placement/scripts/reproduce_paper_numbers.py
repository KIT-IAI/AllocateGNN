"""Reproduce the numerical evidence in the task-and-scale placement paper.

This entry point consumes only the manuscript-scoped frozen CSV package in
``study_materials/placement``.  It deliberately does not read
the development result directories, model checkpoints, held-out fields, or
the historical 10-feature graph products.

The command reproduces tables and numerical statements from derived result
CSVs.  It does not retrain the lu5 GNN or regenerate its held-out fields; those
artifacts are no longer available in the source workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from StudyCase.Placement.core.stats import exact_sign_flip_p


RELEASE_ROOT = REPO_ROOT / "study_materials" / "placement"
RESULTS = RELEASE_ROOT / "results"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "placement_reproduction"
SEEDS = (42, 123, 456)
GNN_ARMS = tuple(f"GNN-{seed}" for seed in SEEDS)
ADEQUACY_ARMS = ("Uni", "LU", *GNN_ARMS)
EPS = 1e-9

EXPECTED_HEADLINES = {
    "gb.reconstruction.median_change_pct": -26.182790015227475,
    "gb.siting.median_change_pct": -1.6936465680546422,
    "gb.sizing.median_change_pct": 4.498615349660467,
    "gb.connection.median_change_pct": -43.24643524800423,
    "au.reconstruction.median_change_pct": 33.91764711364958,
    "au.siting.median_change_pct": 3.122653957689327,
    "au.sizing.median_change_pct": 68.06071271903178,
    "au.connection.median_change_pct": -6.663739158049074,
    "controls.gnn_r10.cost_mae": 8582242.89523013,
    "scale.gb.r10.decision_median_change_pct": -44.57124695543105,
    "adequacy.gb.eta90_realised": 402,
    "adequacy.au.eta90_realised": 300,
    "adequacy.gb.eta90_zero_bound_pct": 3.482587064676617,
    "adequacy.au.eta90_zero_bound_pct": 35.333333333333336,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _normalise_arm(value: str) -> str:
    arm = str(value).replace("GNN-oof-", "GNN-")
    return {"GPM": "LU", "REF": "Ref"}.get(arm, arm)


def _normalise_arms(frame: pd.DataFrame, column: str = "arm") -> pd.DataFrame:
    out = frame.copy()
    out[column] = out[column].map(_normalise_arm)
    return out


def _signflip(values: Iterable[float]) -> float:
    return float(exact_sign_flip_p(np.asarray(list(values), dtype=float))["p"])


def _holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for index, (name, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (m - index) * float(pvalue)))
        adjusted[name] = running
    return adjusted


def _paired_summary(
    frame: pd.DataFrame,
    *,
    metric: str,
    base: str = "LU",
    gnn_arms: tuple[str, ...] = GNN_ARMS,
) -> dict[str, Any]:
    wide = frame.pivot(index="region", columns="arm", values=metric)
    wide = wide[[base, *gnn_arms]].dropna()
    baseline = wide[base].to_numpy(float)
    per_seed = wide[list(gnn_arms)].to_numpy(float)
    treatment = per_seed.mean(axis=1)
    difference = treatment - baseline
    relative = np.divide(
        difference,
        baseline,
        out=np.full_like(difference, np.nan),
        where=baseline != 0,
    )

    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(baseline), size=(10_000, len(baseline)))
    boot = difference[indices].sum(axis=1) / baseline[indices].sum(axis=1) * 100.0
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    seed_rows: dict[str, Any] = {}
    for column, seed in enumerate(SEEDS):
        values = per_seed[:, column]
        delta = values - baseline
        ratio = np.divide(
            delta,
            baseline,
            out=np.full_like(delta, np.nan),
            where=baseline != 0,
        )
        seed_rows[str(seed)] = {
            "mean": float(values.mean()),
            "median_change_pct": float(np.nanmedian(ratio) * 100.0),
            "regions_improved": int((values < baseline).sum()),
            "p_signflip": _signflip(delta),
        }

    return {
        "n_regions": int(len(baseline)),
        "base_mean": float(baseline.mean()),
        "gnn_mean": float(treatment.mean()),
        "median_change_pct": float(np.nanmedian(relative) * 100.0),
        "regions_improved": int((treatment < baseline).sum()),
        "p_signflip": _signflip(difference),
        "aggregate_relative_effect_pct": float(
            difference.sum() / baseline.sum() * 100.0
        ),
        "aggregate_relative_effect_ci95_pct": [float(ci_lo), float(ci_hi)],
        "per_seed": seed_rows,
    }


def reproduce_tasks() -> dict[str, Any]:
    gb_recon = pd.read_csv(RESULTS / "gb_full_matrix_long.csv")
    gb_recon = gb_recon[(~gb_recon["in_sample"]) & gb_recon["arm"].isin(["GPM", "GNN"])]
    gb_recon = gb_recon.copy()
    gb_recon["arm"] = gb_recon.apply(
        lambda row: f"GNN-{int(row['seed'])}" if row["arm"] == "GNN" else "LU",
        axis=1,
    )

    gb_ss = pd.read_csv(RESULTS / "gb_siting_sizing.csv")
    gb_ss = gb_ss.rename(columns={"method": "arm"})
    gb_ss = _normalise_arms(gb_ss)
    gb_siting = gb_ss[gb_ss["eval_weight"] == "real"]
    gb_sizing = gb_ss[gb_ss["eval_weight"] == "SIZING"]
    gb_sizing_1to1 = gb_ss[gb_ss["eval_weight"] == "SIZING_1TO1"]

    gb_cost = _normalise_arms(pd.read_csv(RESULTS / "gb_cost_oof.csv"))
    gb_cost = gb_cost[(gb_cost["radius_km"] == 10.0) & (gb_cost["dc_mw"] == 300.0)]

    au_recon = _normalise_arms(pd.read_csv(RESULTS / "au_recon.csv"))
    au_siting = _normalise_arms(pd.read_csv(RESULTS / "au_siting.csv"))
    au_sizing = _normalise_arms(pd.read_csv(RESULTS / "au_sizing.csv"))
    au_cost = _normalise_arms(pd.read_csv(RESULTS / "au_cost.csv"))
    au_cost = au_cost[~au_cost["coverage_dropped"].astype(bool)]

    gb = {
        "reconstruction": _paired_summary(gb_recon, metric="rmse"),
        "siting": _paired_summary(gb_siting, metric="WSD"),
        "sizing": _paired_summary(gb_sizing, metric="RSD_median"),
        "sizing_1to1": _paired_summary(gb_sizing_1to1, metric="RSD_median"),
        "connection": _paired_summary(gb_cost, metric="cost_mae_gbp"),
    }
    au = {
        "reconstruction": _paired_summary(au_recon, metric="rmse_mva"),
        "siting": _paired_summary(au_siting, metric="WSD_ref_km"),
        "sizing": _paired_summary(
            au_sizing[au_sizing["protocol"] == "multi"], metric="RSD_median"
        ),
        "sizing_1to1": _paired_summary(
            au_sizing[au_sizing["protocol"] == "1to1"], metric="RSD_median"
        ),
        "connection": _paired_summary(au_cost, metric="L_E_mva"),
    }

    au_raw = {
        name: au[name]["p_signflip"]
        for name in ("reconstruction", "siting", "sizing", "connection")
    }
    au_adjusted = _holm(au_raw)
    for name, value in au_adjusted.items():
        au[name]["p_holm_four_module"] = value

    return {"GB": gb, "AU": au}


def reproduce_controls() -> dict[str, Any]:
    frame = _normalise_arms(pd.read_csv(RESULTS / "gb_controls.csv"))
    r10 = frame[frame["radius_km"] == 10.0]
    level_columns = ["rho_pointwise", "rho_aggregate", "cost_mae_gbp", "regret_gbp"]
    levels = r10.groupby("arm")[level_columns].mean().to_dict(orient="index")

    association_rows: list[dict[str, Any]] = []

    def rho(left: pd.Series, right: pd.Series) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            return float(spearmanr(left, right).statistic)

    for (region, radius), group in frame.groupby(["region", "radius_km"]):
        association_rows.append(
            {
                "region": region,
                "radius_km": float(radius),
                "cell_to_fixed": rho(group["rho_pointwise"], group["cost_mae_gbp"]),
                "cell_to_selection": rho(group["rho_pointwise"], group["regret_gbp"]),
                "aggregate_to_fixed": rho(group["rho_aggregate"], group["cost_mae_gbp"]),
                "aggregate_to_selection": rho(group["rho_aggregate"], group["regret_gbp"]),
            }
        )
    associations = pd.DataFrame(association_rows)
    medians: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for radius in (10.0, 20.0):
        subset = associations[associations["radius_km"] == radius]
        medians[str(int(radius))] = {
            column: float(subset[column].median())
            for column in (
                "cell_to_fixed",
                "cell_to_selection",
                "aggregate_to_fixed",
                "aggregate_to_selection",
            )
        }
        delta = (
            subset["aggregate_to_selection"] - subset["cell_to_selection"]
        ).dropna()
        comparisons[str(int(radius))] = {
            "n_defined": int(len(delta)),
            "aggregate_more_negative": int((delta < 0).sum()),
            "p_signflip": _signflip(delta),
        }
    return {"levels_r10": levels, "association_medians": medians, "comparison": comparisons}


def reproduce_scale() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for country, filename in (
        ("GB", "uk_scale_station.csv"),
        ("AU", "au_scale_station.csv"),
    ):
        frame = _normalise_arms(pd.read_csv(RESULTS / filename))
        frame["method"] = frame["arm"].str.replace(r"GNN-\d+", "GNN", regex=True)
        country_rows: dict[str, Any] = {}
        for radius, group in frame.groupby("radius_km"):
            radius_rows: dict[str, Any] = {}
            for metric, label in (
                ("mae_decision", "decision"),
                ("mae_representation", "representation"),
            ):
                wide = group.groupby(["region", "method"])[metric].mean().unstack()
                wide = wide[["LU", "GNN"]].dropna()
                delta = wide["GNN"] - wide["LU"]
                relative = delta / wide["LU"] * 100.0
                radius_rows[label] = {
                    "median_change_pct": float(relative.median()),
                    "q25_change_pct": float(relative.quantile(0.25)),
                    "q75_change_pct": float(relative.quantile(0.75)),
                    "regions_improved": int((delta < 0).sum()),
                    "p_signflip": _signflip(delta),
                }
            radius_key = str(int(radius)) if float(radius).is_integer() else str(float(radius))
            country_rows[radius_key] = radius_rows
        output[country] = country_rows
    return output


def reproduce_adequacy() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for country, filename, scale in (
        ("GB", "uk_adequacy.csv", 1e6),
        ("AU", "au_adequacy.csv", 1.0),
    ):
        frame = _normalise_arms(pd.read_csv(RESULTS / filename))
        frame = frame[frame["arm"].isin(ADEQUACY_ARMS)].copy()
        frame["budget_realised"] = (
            frame["budget_realised"].astype(str).str.lower() == "true"
        )
        realised = frame[frame["budget_realised"]]
        eta90 = realised[realised["eta"] == 0.90]
        violations = (
            (realised["L_E"] > realised["L_E_bound"] + 1e-6)
            | (realised["L_S"] > realised["L_S_bound"] + 1e-6)
        )
        country_rows: dict[str, Any] = {
            "total_units": int(len(frame)),
            "budget_realised_units": int(len(realised)),
            "eta90_realised_units": int(len(eta90)),
            "bound_violations": int(violations.sum()),
            "eta90_zero_regret_pct": float((eta90["L_S"].abs() < EPS).mean() * 100),
            "eta90_zero_bound_pct": float((eta90["L_S_bound"].abs() < EPS).mean() * 100),
            "eta90_zero_bound_by_incoming_load": {},
            "alpha_observed": {},
            "eta90_by_method": {},
            "median_slack_by_eta": {},
        }
        for incoming_load, group in eta90.groupby("dc_mw"):
            country_rows["eta90_zero_bound_by_incoming_load"][str(int(incoming_load))] = {
                "share": float((group["L_S_bound"].abs() < EPS).mean()),
                "percent": float((group["L_S_bound"].abs() < EPS).mean() * 100),
            }
        for eta, group in frame.groupby("eta"):
            country_rows["alpha_observed"][str(float(eta))] = {
                "GNN": float(group[group["arm"].str.startswith("GNN")]["budget_realised"].mean()),
                "LU": float(group[group["arm"] == "LU"]["budget_realised"].mean()),
                "Uni": float(group[group["arm"] == "Uni"]["budget_realised"].mean()),
            }
            valid = realised[realised["eta"] == eta]
            country_rows["median_slack_by_eta"][str(float(eta))] = {
                "fixed": float(np.median(valid["L_E_bound"] - valid["L_E"]) / scale),
                "selection": float(np.median(valid["L_S_bound"] - valid["L_S"]) / scale),
            }
        for name, selector in (
            ("GNN", eta90["arm"].str.startswith("GNN")),
            ("LU", eta90["arm"] == "LU"),
            ("Uni", eta90["arm"] == "Uni"),
        ):
            group = eta90[selector]
            positive_e = group[group["L_E"] > EPS]
            positive_s = group[group["L_S"] > EPS]
            country_rows["eta90_by_method"][name] = {
                "n": int(len(group)),
                "zero_bound_share": float((group["L_S_bound"].abs() < EPS).mean()),
                "zero_bound_pct": float((group["L_S_bound"].abs() < EPS).mean() * 100),
                "bound_over_actual_fixed_median": float(
                    np.median(positive_e["L_E_bound"] / positive_e["L_E"])
                ),
                "bound_over_actual_selection_median": float(
                    np.median(positive_s["L_S_bound"] / positive_s["L_S"])
                ),
            }
        output[country] = country_rows
    return output


def reproduce() -> dict[str, Any]:
    return {
        "release": {
            "id": "task-scale-lu5",
            "manuscript_sha256": "F7C1B9330909D7952E794D8184D39C0354CA69C5D336CD69D65CAC6285A6021B",
            "reproducibility_boundary": "derived frozen CSVs to paper numbers",
        },
        "controls": reproduce_controls(),
        "scale": reproduce_scale(),
        "tasks": reproduce_tasks(),
        "adequacy": reproduce_adequacy(),
    }


def _get_path(data: dict[str, Any], dotted: str) -> Any:
    value: Any = data
    for part in dotted.split("."):
        value = value[part]
    return value


def verify_manifest() -> None:
    manifest_path = RELEASE_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = {entry["path"] for entry in manifest["files"]}
    actual = {path.name for path in RESULTS.iterdir() if path.is_file()}
    if actual != declared:
        raise AssertionError(
            f"result allowlist mismatch: actual={sorted(actual)}, declared={sorted(declared)}"
        )
    for entry in manifest["files"]:
        path = RESULTS / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise AssertionError(f"SHA-256 mismatch: {entry['path']}")
        frame = pd.read_csv(path)
        if len(frame) != entry["rows"]:
            raise AssertionError(f"row-count mismatch: {entry['path']}")
        if list(frame.columns) != entry["columns"]:
            raise AssertionError(f"schema mismatch: {entry['path']}")


def verify_headlines(data: dict[str, Any]) -> None:
    paths = {
        "gb.reconstruction.median_change_pct": "tasks.GB.reconstruction.median_change_pct",
        "gb.siting.median_change_pct": "tasks.GB.siting.median_change_pct",
        "gb.sizing.median_change_pct": "tasks.GB.sizing.median_change_pct",
        "gb.connection.median_change_pct": "tasks.GB.connection.median_change_pct",
        "au.reconstruction.median_change_pct": "tasks.AU.reconstruction.median_change_pct",
        "au.siting.median_change_pct": "tasks.AU.siting.median_change_pct",
        "au.sizing.median_change_pct": "tasks.AU.sizing.median_change_pct",
        "au.connection.median_change_pct": "tasks.AU.connection.median_change_pct",
        "controls.gnn_r10.cost_mae": "controls.levels_r10.GNN.cost_mae_gbp",
        "scale.gb.r10.decision_median_change_pct": "scale.GB.10.decision.median_change_pct",
        "adequacy.gb.eta90_realised": "adequacy.GB.eta90_realised_units",
        "adequacy.au.eta90_realised": "adequacy.AU.eta90_realised_units",
        "adequacy.gb.eta90_zero_bound_pct": "adequacy.GB.eta90_zero_bound_pct",
        "adequacy.au.eta90_zero_bound_pct": "adequacy.AU.eta90_zero_bound_pct",
    }
    for name, expected in EXPECTED_HEADLINES.items():
        actual = _get_path(data, paths[name])
        if isinstance(expected, int):
            if actual != expected:
                raise AssertionError(f"headline mismatch {name}: {actual} != {expected}")
        elif not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10):
            raise AssertionError(f"headline mismatch {name}: {actual} != {expected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify manifest and headline contracts")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stdout", action="store_true", help="print JSON instead of writing it")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = reproduce()
    if args.verify:
        verify_manifest()
        verify_headlines(data)
    rendered = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.stdout:
        print(rendered, end="")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        output = args.output / "paper_numbers.json"
        output.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"paper_numbers={output}")
    if args.verify:
        print("verification=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
