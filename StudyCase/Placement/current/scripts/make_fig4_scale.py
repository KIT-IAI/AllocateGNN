"""Figure 4 (scale curves) from the frozen revision run; legacy make_fig_scale.py style.

Both curves share the same station centres, radii and candidates and differ only in the
comparison target: the reported-demand neighbourhood sum G (decision) and the VD-Ref
aggregate (representation). GNN seeds are averaged within region before the LU->GNN
regional percentage change; the band is the regional IQR of the decision curve.

The curves are read from the frozen run's ``stats/scale_curves.csv`` (D9 zero rule) and the
Voronoi-equivalent radius from its claim ledger (D10); the figure computes nothing. Every point whose
defined-region count is below the region count is reported in an external information band.
The star is the station-level RMSE reconstruction change (a different metric from the
neighbourhood MAE of the curves), placed at the Voronoi-equivalent radius for reference.

    python scripts/make_fig4_scale.py --run-id fixedload_20260925_r2   (one row, two columns)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

CASE = {"uk": "GB", "au": "AU"}
CASE_LABEL = {"GB": "Britain", "AU": "Australia"}
REPORTED_STYLE = dict(color="black", ls="-")        # comparison with reported-demand sums (method colours stay reserved)
REF_STYLE = dict(color="0.5", ls="--")              # comparison with Ref
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
    backup = args.backup.resolve()
    root = args.root.resolve() if args.root else backup / "frozen" / args.run_id
    led = pd.read_csv(root / "ledger/claim_ledger.csv", dtype={"value": str}).set_index("claim_id")
    value = lambda cid: float(led.loc[cid, "value"])
    curves = pd.read_csv(root / "stats/scale_curves.csv")

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.0),
                             gridspec_kw=dict(left=0.085, right=0.985, top=0.91, bottom=0.3, wspace=0.16))
    caption_n = []
    signs, span = {}, {}
    for ax, cc in zip(axes, ("uk", "au")):
        case = CASE[cc]
        c = curves[(curves.country == cc) & (curves.centres == "station")]
        # Frozen schema key for the reported-demand neighborhood sum G.
        dec = c[c.target == "ledger_G"].sort_values("radius_km")
        rep = c[c.target == "VD_Ref"].sort_values("radius_km")
        ax.axhline(0, color="0.55", lw=0.6, zorder=1)
        ax.fill_between(dec.radius_km, dec.q25, dec.q75, color="0.6", alpha=0.25, lw=0, zorder=2)
        ax.plot(dec.radius_km, dec.median_rel_pct, marker="o", ms=3.2, zorder=4, **REPORTED_STYLE,
                label="Reported-demand comparison (median, IQR)")
        ax.plot(rep.radius_km, rep.median_rel_pct, marker="s", ms=2.8, markerfacecolor="white",
                markeredgewidth=0.7, zorder=3, **REF_STYLE, label="Ref comparison (median)")
        # defined-region counts only where they fall below n: one small label per affected point
        for frame, colour, dy in ((dec, "black", -9), (rep, "0.4", 5)):
            for r in frame[frame.n_pct_defined < frame.n].itertuples():
                ax.annotate(f"{r.n_pct_defined}", xy=(r.radius_km, r.median_rel_pct), xytext=(-9 if r.radius_km == frame.radius_km.max() else 4, dy),
                            textcoords="offset points", fontsize=7.2, color=colour)
        vr = value(f"SCALE.{case}.voronoi_radius_km")
        recon = value(f"T3.{case}.reconstruction.rmse.median_rel_pct")
        ax.axvline(vr, color="0.35", ls=":", lw=0.8, zorder=1)
        ax.text(vr * 1.03, 0.97, f"{vr:.2f} km", transform=ax.get_xaxis_transform(), fontsize=7.2, va="top", color="0.35")
        ax.plot([vr], [recon], marker="*", ms=8, color="#009E73", mec="0.2", mew=0.4, ls="none", zorder=5,
                label="Reconstruction RMSE (reference)")
        ax.margins(y=0.13)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 7.5, 10, 15, 20])
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.get_xaxis().set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
        n = int(value(f"DATA.{case}.n_regions"))
        ax.set_title(f"({'ab'[cc == 'au']}) {CASE_LABEL[case]}, {n} regions", loc="left")
        partial = {}
        for t, fr in (("solid", dec), ("dashed", rep)):
            for r in fr[fr.n_pct_defined < fr.n].itertuples():
                partial.setdefault((r.radius_km, int(r.n_pct_defined), int(r.n)), []).append(t)
        caption_n.append((CASE_LABEL[case], vr, partial))
        signs[case] = (dec.median_rel_pct < 0).all(), (dec.median_rel_pct > 0).all()
        span[case] = float(dec.radius_km.min()), float(dec.radius_km.max())
    for ax in axes:
        ax.set_xlabel("Geographic-support radius, $R$ (km)")
    axes[0].set_ylabel("Change in neighborhood-demand\nMAE, LU to GNN (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, handlelength=2.0, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.035))
    args.out.mkdir(parents=True, exist_ok=True)
    stem = "fig4_scale"
    for stale in ("fig4_scale_2row.pdf", "fig4_scale_2row.png", "captions/fig4_scale_2row.md", "captions/fig4_scale_2row.tex"):
        (args.out / stale).unlink(missing_ok=True)
    fig.savefig(args.out / f"{stem}.pdf", bbox_inches=None)
    fig.savefig(args.out / f"{stem}.png", bbox_inches=None)
    plt.close(fig)
    radii = {name: vr for name, vr, _ in caption_n}
    if not (signs["GB"][0] and signs["AU"][1]):
        raise ValueError("caption first sentence assumes GB curve below zero and AU curve above zero at every radius")
    if span["GB"] != span["AU"]:
        raise ValueError("caption assumes the same radius range in both countries")
    if not any(partial for _, _, partial in caption_n):
        raise ValueError("caption mentions region counts beside points, but every point uses all regions")
    r0, r1 = span["GB"]
    # batch 13: caption cut to about 150 words; the per-radius region counts are drawn beside the points and the
    # zero-error rule is stated in Sec. 5.1
    paper2_style.write_caption(args.out, stem, "fig:scale",
        f"In Britain, \\acrfull{{gnn}}-based allocation lowers the neighborhood-demand error relative to \\acrfull{{lu}} at every radius from {r0:g} to {r1:g}~km. In Australia "
        "it raises the error at every radius (RQ1). "
        "For each radius $R$, allocated demand is summed within $R$ of every existing substation. "
        "The solid curve compares these sums with the reported-demand neighborhood sums and the dashed curve with "
        "those under the \\acrfull{ref}. Each point is the median regional percentage change in the "
        "\\acrfull{mae} from \\acrshort{lu} to \\acrshort{gnn}-based allocation, with \\acrshort{gnn} seeds averaged per region. "
        "Negative values favor \\acrshort{gnn}-based allocation. The band is the interquartile range (IQR) of the solid curve over regions. "
        "Small numbers give the regions used where zero-error regions drop out. "
        "The dotted line marks the Voronoi-equivalent radius ("
        + ", ".join(f"{vr:.2f}~km in {name}" for name, vr in radii.items()) + "), and the star the substation-level "
        "\\acrfull{rmse} change of peak-demand reconstruction there.",
        f"frozen/{args.run_id}/stats/scale_curves.csv; ledger SCALE.*, T3.*.reconstruction.rmse.median_rel_pct, DATA.*.n_regions")
    print(f"wrote {args.out / (stem + '.pdf')}")


if __name__ == "__main__":
    main()
