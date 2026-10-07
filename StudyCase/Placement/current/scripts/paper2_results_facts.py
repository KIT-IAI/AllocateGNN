"""Results-section facts that the claim ledger does not state directly, generated from the frozen run.

Two groups of numbers in Sec. 5 are derived from frozen per-evaluation tables rather than read from a
single ledger row:

* the share of evaluation locations whose reported-demand neighborhood sum needs reinforcement for the
  incoming load, q_y = max(0, G_y + X - F_y) > 0, per region and setting (``uk_cost_oof.csv`` /
  ``au_cost_oof.csv``, field ``true_zero_q_fraction``; q depends on reported demand and substation capacity only, so the Ref rows
  are used), averaged over regions with equal weight, for each of the six (R, X) settings;
* the number of bound-audit evaluations whose empirical budget is exceeded and, among them, how many
  violate either bound (``uk_adequacy.csv`` / ``au_adequacy.csv``), cross-checked against the ledger
  rows ``ADQ.*.units_total``, ``ADQ.*.units_budget_realised`` and ``ADQ.*.violations_not_realised``;
* the zero-selection-bound and zero-regret shares at eta = 0.90 in percent (pooled over methods, seeds,
  radii and loads, and split by load and by method), recomputed from the adequacy tables and
  cross-checked against the ledger shares ``ADQ.*.zero_bound_S_share.eta0.9.*`` and
  ``ADQ.*.zero_regret_share.eta0.9.all``;
* the number of regions over which each median regional change of ``stats/module_contrasts.csv`` is
  defined (``n_pct_defined``, regions with a nonzero LU value), keyed ``module|metric[|matching][|R|X]``.

Writes ``figures_new/results_facts.json`` (full precision, with the SHA-256 of every file read).
``check_manuscript_numbers.py`` accepts these values as generated facts (channel F) and as table bindings.

    python scripts/paper2_results_facts.py --run-id fixedload_20260925_r2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--backup", type=Path, default=here.parents[2] / "results" / "_backup")
    ap.add_argument("--out", type=Path, default=here.parents[1] / "figures_new" / "results_facts.json")
    args = ap.parse_args()
    backup = args.backup.resolve()
    frozen = backup / "frozen" / args.run_id
    used: dict[str, str] = {}

    def use(path: Path) -> Path:
        used[path.relative_to(backup).as_posix()] = sha(path)
        return path

    with use(frozen / "ledger/claim_ledger.csv").open(encoding="utf-8") as stream:
        ledger = {row["claim_id"]: row["value"] for row in csv.DictReader(stream)}

    facts: dict[str, dict] = {}
    for country, key in (("uk", "GB"), ("au", "AU")):
        cost = pd.read_csv(use(frozen / f"{country}_cost_oof.csv"))
        ref = cost[cost.method == "Ref"]
        if ref.groupby(["radius_km", "dc_mw"]).region.nunique().nunique() != 1:
            raise ValueError(f"{country}: settings cover different region sets")
        share = (1.0 - ref.true_zero_q_fraction).groupby([ref.radius_km, ref.dc_mw]).mean()
        # the reported-demand benchmark quantity must not depend on the method row it is read from
        for method, rows in cost.groupby("method"):
            other = (1.0 - rows.true_zero_q_fraction).groupby([rows.radius_km, rows.dc_mw]).mean()
            if not (other - share).abs().max() < 1e-12:
                raise ValueError(f"{country}: reinforcement share differs for method {method}")
        by_setting = {f"R{r:g}_X{x:g}": 100.0 * float(v) for (r, x), v in share.items()}
        adequacy = pd.read_csv(use(frozen / f"{country}_adequacy.csv"))
        exceeded = adequacy[~adequacy.budget_realised.astype(bool)]
        realised = adequacy[adequacy.budget_realised.astype(bool)]
        violating = int((~exceeded.valid_E.astype(bool) | ~exceeded.valid_S.astype(bool)).sum())
        checks = {
            "units_total": (len(adequacy), f"ADQ.{key}.units_total"),
            "units_budget_realised": (len(realised), f"ADQ.{key}.units_budget_realised"),
            "violations_not_realised": (violating, f"ADQ.{key}.violations_not_realised"),
        }
        for name, (value, claim) in checks.items():
            if int(float(ledger[claim])) != value:
                raise ValueError(f"{claim}: ledger {ledger[claim]} != recomputed {value}")
        # zero selection bound / zero realized regret at the main budget level, in percent
        main = realised[realised.eta == 0.9]
        zero = {"n_realised_eta0.9": int(len(main)),
                "zero_bound_S_pct": 100.0 * float(main.L_S_bound_zero.astype(bool).mean()),
                "zero_regret_pct": 100.0 * float(main.L_S_zero.astype(bool).mean())}
        shares = {"zero_bound_S_pct": f"ADQ.{key}.zero_bound_S_share.eta0.9.all",
                  "zero_regret_pct": f"ADQ.{key}.zero_regret_share.eta0.9.all"}
        for x, rows in main.groupby("dc_mw"):
            zero[f"zero_bound_S_pct_X{x:g}"] = 100.0 * float(rows.L_S_bound_zero.astype(bool).mean())
            shares[f"zero_bound_S_pct_X{x:g}"] = f"ADQ.{key}.zero_bound_S_share.eta0.9.all.X{x:g}"
        for method, rows in main.groupby("method"):
            zero[f"zero_bound_S_pct_{method}"] = 100.0 * float(rows.L_S_bound_zero.astype(bool).mean())
            shares[f"zero_bound_S_pct_{method}"] = f"ADQ.{key}.zero_bound_S_share.eta0.9.{method}"
        if zero["n_realised_eta0.9"] != int(float(ledger[f"ADQ.{key}.units_realised_eta0.9"])):
            raise ValueError(f"{key}: eta=0.9 realised count disagrees with the ledger")
        for name, claim in shares.items():
            if abs(zero[name] - 100.0 * float(ledger[claim])) > 1e-9:
                raise ValueError(f"{claim}: ledger {ledger[claim]} != recomputed {zero[name] / 100}")
        # regions over which each median regional change is defined (nonzero LU value)
        contrasts = pd.read_csv(use(frozen / "stats/module_contrasts.csv"))
        median_regions = {}
        for row in contrasts[contrasts.country == country].itertuples():
            parts = [row.module, row.metric]
            if isinstance(row.matching, str):
                parts.append(row.matching)
            if pd.notna(row.radius_km):
                parts += [f"R{row.radius_km:g}", f"X{row.dc_mw:g}"]
            median_regions["|".join(parts)] = {"n_pct_defined": int(row.n_pct_defined), "n": int(row.n)}
        facts[key] = {
            "median_regions": median_regions,
            "n_regions": int(ref.region.nunique()),
            "reinforcement_share_pct": by_setting,
            "reinforcement_share_pct_main": by_setting["R10_X300"],
            "reinforcement_share_pct_min": min(by_setting.values()),
            "reinforcement_share_pct_max": max(by_setting.values()),
            "units_budget_exceeded": int(len(exceeded)),
            "violations_budget_exceeded": violating,
            "violations_budget_exceeded_E": int((~exceeded.valid_E.astype(bool)).sum()),
            "violations_budget_exceeded_S": int((~exceeded.valid_S.astype(bool)).sum()),
            "eta0.9": zero,
        }

    out = {"run_id": args.run_id, "script_sha256": sha(here), "facts": facts, "inputs_sha256": used}
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
