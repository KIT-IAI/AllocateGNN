"""Figure 5 (conditional bound pass rates) from the frozen revision run; legacy make_fig_adequacy.py style.

Rows: selection bound (top, tolerance tau_S) and fixed-location error bound (bottom,
tolerance tau_E); columns: Britain (GBP m, T9) and Australia (PF=1 MVA-equivalent). Curves: P(bound <= tau |
budget held) at eta = 0.90, evaluations whose budget was exceeded excluded (the bounds are conditional on the
budget holding); GNN pools its three seeds' evaluations and the three loads X = 100/300/500 MW and both radii
are pooled for every method. The per-method evaluation counts go to the caption. Because the tau axis starts
at 0.3, the left plateau is not the numerical-zero rate; bounds classified as zero under the numerical
tolerance of the statistical protocol are reported per country in the caption. The shaded band in the British
panels is the tolerance range of interest fixed in the protocol (GBP 1m-10m). The dotted GB line is the full
reinforcement cost of the 300 MW reference load only.

All caption numbers are written to figures_new/fig5_sources.json (facts) by this script.

    python scripts/make_fig5_adequacy.py --run-id fixedload_20260925_r2
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ETA_MAIN = 0.90
FACTS: dict = {}  # caption values generated here, recorded in figures_new/fig5_sources.json
REFERENCE_LOAD = 300.0
TOLERANCE_RANGE_GBP = (1e6, 10e6)  # tau_S range of interest fixed in the statistical protocol
ARMS = [("Uni", "#777777", "-."), ("LU", "#E69F00", "--"), ("GNN", "#0072B2", "-")]
METHODS = [name for name, _, _ in ARMS]
CASES = {"GB": ("uk", 1e6, "£m"), "AU": ("au", 1.0, "MVA")}
COUNTRY_LABEL = {"GB": "Britain", "AU": "Australia"}
ROWS = (("L_S_bound", "Selection bound", "S"), ("L_E_bound", "Fixed-location error bound", "E"))
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper2_style  # noqa: E402

paper2_style.apply()
WIDTH_IN = paper2_style.WIDTH_IN


def main() -> None:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--backup", type=Path, default=here.parents[2] / "results" / "_backup")
    ap.add_argument("--root", type=Path, help="Frozen run directory, overriding --backup/frozen/--run-id")
    ap.add_argument("--out", type=Path, default=here.parents[1] / "figures_new")
    args = ap.parse_args()
    root = args.root.resolve() if args.root else args.backup.resolve() / "frozen" / args.run_id
    led = pd.read_csv(root / "ledger/claim_ledger.csv", dtype={"value": str}).set_index("claim_id")
    full_reinforcement = REFERENCE_LOAD * 430_000.0
    data, caption, loads, radii = {}, [], set(), set()
    for case, (cc, unit, ulab) in CASES.items():
        d = pd.read_csv(root / f"{cc}_adequacy.csv")
        realised = d[d.budget_realised]
        at = realised[realised.eta == ETA_MAIN]
        n_title = int(float(led.loc[f"ADQ.{case}.units_realised_eta0.9", "value"]))
        assert len(at) == n_title, f"{case}: plotted {len(at)} != ledger {n_title}"
        data[case] = (realised, at)
        loads.update(at.dc_mw.unique()); radii.update(at.radius_km.unique())
        counts = at.method.value_counts()
        caption.append((COUNTRY_LABEL[case], n_title, {m: int(counts.get(m, 0)) for m in METHODS}))
        FACTS.setdefault(case, {})["n_eta0.9"] = {"all": n_title, **{m: int(counts.get(m, 0)) for m in METHODS}}
        for bound, _, _ in ROWS:
            v, tol = at[bound].to_numpy(float), at.zero_tolerance.to_numpy(float)
            FACTS[case].setdefault(bound, {})["zero_bound_pct_pooled"] = 100 * float(np.mean(v <= tol))
            if case == "GB":
                for t in TOLERANCE_RANGE_GBP:
                    FACTS[case][bound][f"exceed_pct_at_{t / 1e6:g}m"] = 100 * float(np.mean(v > t))

    LOADS, RADII = sorted(loads), sorted(radii)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 3.6), sharey=True,
                             gridspec_kw=dict(left=0.085, right=0.985, top=0.94, bottom=0.18, hspace=0.66, wspace=0.06))
    for col, case in enumerate(("GB", "AU")):
        cc, unit, ulab = CASES[case]
        realised, at = data[case]
        taus = np.logspace(np.log10(0.3), np.log10(1000.0), 200) * unit
        counts = at.method.value_counts()
        for row, (bound, label, sub) in enumerate(ROWS):
            ax = axes[row, col]
            if case == "AU" and row == 1:
                label = "Fixed-location requirement-error bound"
            for name, colour, style in ARMS:
                g = at[at.method == name]
                v = g[bound].to_numpy(float)
                ax.plot(taus / unit, [(v <= t).mean() for t in taus], style, color=colour,
                        label=f"{name} ($n$ = {counts.get(name, 0)})")
                FACTS[case][bound][f"zero_bound_pct_{name}"] = 100 * float(np.mean(v <= g.zero_tolerance.to_numpy(float)))
            ax.set_xscale("log"); ax.set_xlim(0.3, 1000); ax.set_ylim(-0.02, 1.02)
            ax.grid(axis="y", color="0.9", linewidth=0.4)
            if case == "GB":
                ax.axvspan(TOLERANCE_RANGE_GBP[0] / 1e6, TOLERANCE_RANGE_GBP[1] / 1e6, color="0.88", lw=0, zorder=0)
                ax.axvline(full_reinforcement / 1e6, color="0.25", ls=":", lw=0.9)
            ax.set_xlabel(rf"Planner tolerance, $\tau_{sub}$ ({ulab})")
            if col == 0:
                ax.set_ylabel("Pass rate\n(evaluations whose budget held)")
            ax.set_title(f"({'abcd'[2 * row + col]}) {COUNTRY_LABEL[case]}: {label.lower()}", loc="left", fontsize=8)
            if row == 1:
                # method key in the empty upper left of the bottom row (the counts are the same in both rows)
                ax.legend(loc="upper left", frameon=False, fontsize=7.5, handlelength=1.8, borderaxespad=0.4, labelspacing=0.25)
            if row == 0:
                inset = ax.inset_axes([0.08, 0.56, 0.36, 0.27])
                etas = sorted(realised.eta.unique())
                for name, colour, style in ARMS:
                    inset.plot(etas, [realised[(realised.eta == e) & (realised.method == name)].slack_S.median() / unit for e in etas],
                               style, marker="o", ms=2.4, lw=1.0, color=colour)
                inset.set_title(rf"Median slack ({ulab}) by $\eta$", loc="left", fontsize=7.2, pad=2)
                inset.set_xticks([0.5, 0.9, 1.0], ["0.5", "0.9", "1"])
                inset.margins(y=0.3)
                inset.tick_params(labelsize=7, length=2, pad=1, which="both", top=False, right=False)
                inset.minorticks_off()
                inset.patch.set_facecolor("white")
    handles = [mpl.patches.Patch(color="0.88", label="Tolerance range £1m–£10m (Britain)"),
               mpl.lines.Line2D([], [], color="0.25", ls=":", label="Full reinforcement at $X$ = 300 MW (Britain)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.0, columnspacing=1.6)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "fig5_adequacy.pdf", bbox_inches=None)
    fig.savefig(args.out / "fig5_adequacy.png", bbox_inches=None)
    plt.close(fig)
    for stale in ("fig5b_slack.pdf", "fig5b_slack.png", "captions/fig5b_slack.md"):
        (args.out / stale).unlink(missing_ok=True)

    # method similarity of the zero-bound shares, in percentage points, over all country x bound cells
    spread = max(max(FACTS[c][b][f"zero_bound_pct_{m}"] for m in METHODS) - min(FACTS[c][b][f"zero_bound_pct_{m}"] for m in METHODS)
                 for c in CASES for b, _, _ in ROWS)
    FACTS["zero_bound_method_spread_pp"] = spread
    gb_exceed = min(FACTS["GB"][b][f"exceed_pct_at_{t / 1e6:g}m"] for b, _, _ in ROWS for t in TOLERANCE_RANGE_GBP)
    FACTS["GB"]["exceed_pct_min_over_range"] = gb_exceed
    if gb_exceed <= 90:
        raise ValueError("caption first sentence assumes British bounds exceed the whole tolerance range in more than nine tenths")
    au_zero = FACTS["AU"]["L_S_bound"]["zero_bound_pct_pooled"]
    if not 30 <= au_zero <= 40:
        raise ValueError("caption first sentence assumes about a third of Australian selection bounds are zero")
    (args.out / "fig5_sources.json").write_text(json.dumps({"run_id": args.run_id, "facts": FACTS}, indent=2), encoding="utf-8")
    pct = lambda x: f"{x:.1f}~\\%"
    z = lambda c, b: FACTS[c][b]["zero_bound_pct_pooled"]
    ex = lambda b: FACTS["GB"][b]["exceed_pct_at_10m"]
    counts = " and ".join(f"{n} in {name}" for name, n, m in caption)
    paper2_style.write_caption(args.out, "fig5_adequacy", "fig:adequacy",
        "In Britain both connection-specific bounds exceed every tolerance from \\pounds1m to \\pounds10m in more than nine tenths of the "
        f"evaluations (at \\pounds10m, {pct(ex('L_S_bound'))} for the selection bound and {pct(ex('L_E_bound'))} for the "
        f"fixed-location error bound). In Australia the selection bound is zero in {pct(au_zero)}. "
        "Each curve gives the share of evaluations whose bound is at most the planner tolerance. "
        f"The curves include evaluations in which the budget held ({counts}). "
        "They are taken at $\\eta=0.90$ and pooled over "
        f"loads of {', '.join(f'{x:g}' for x in LOADS[:-1])} and {LOADS[-1]:g}~MW, radii of {RADII[0]:g} and {RADII[1]:g}~km "
        "and, for \\acrfull{gnn}-based allocation, the three seeds. "
        f"The fixed-location error bound is zero in {pct(z('GB', 'L_E_bound'))} of the British and {pct(z('AU', 'L_E_bound'))} of the "
        f"Australian evaluations, and the three methods differ by at most {spread:.1f} percentage points in each zero-bound share. "
        "The shaded band is the British tolerance range of Sec.~\\ref{sec:protocol}, and the dotted line marks "
        f"\\pounds{full_reinforcement / 1e6:.0f}m, the cost of fully reinforcing a {REFERENCE_LOAD:g}~MW load. "
        "Insets: median slack of the selection bound (bound minus realized regret) against $\\eta$.",
        f"frozen/{args.run_id}/{{uk,au}}_adequacy.csv; ledger ADQ.*.units_realised_eta0.9; facts figures_new/fig5_sources.json")
    print("wrote fig5_adequacy")


if __name__ == "__main__":
    main()
