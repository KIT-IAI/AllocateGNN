"""Evaluation-design diagram (Fig. pipeline) and its generated caption.

Adapted from the earlier navigation figure (``allocategnn-vibe/StudyCase_placement/docs/paper3/figures/
make_fig0_pipeline.py``; layout kept: a shared top row input -> allocation methods -> task-specific
aggregation, a left branch box with the four tasks and the two cases, and a right branch box with the
bound chain). Content follows the current manuscript: no internal codes, the siting -> sizing arrow is
the only dependency drawn between tasks, and the right branch applies to the connection task only.
Style: ``paper2_style`` (SciencePlots, 183 mm width), Okabe-Ito colours, text at 7 pt or larger.

The two region counts come from the frozen claim ledger (``DATA.GB.n_regions``, ``DATA.AU.n_regions``).
Geometry checks run before export: every label stays inside its own box and inside the page, and no two
labels overlap.

    python scripts/make_fig0_pipeline.py --run-id fixedload_20260925_r2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper2_style  # noqa: E402

paper2_style.apply()

W_MM, H_MM = 183.0, 86.0  # half-page limit (batch 12): branch boxes and case boxes compressed
FS = 7.2          # body text (pt)
FS_TITLE = 7.6    # container titles (pt)
INK = "#262626"
SHARED = dict(face="#F2F2F2", edge="#4D4D4D")
SIDE = dict(face="#FAFAFA", edge="#9A9A9A")
TASKS = dict(face="#E6F3FB", edge="#56B4E9", title="#1F5F86")    # Okabe-Ito sky blue
BOUNDS = dict(face="#F8EAF2", edge="#CC79A7", title="#8A3D68")   # Okabe-Ito reddish purple


class Canvas:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(W_MM / 25.4, H_MM / 25.4))
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax.set_xlim(0, W_MM); self.ax.set_ylim(0, H_MM); self.ax.set_axis_off()
        self.texts, self.boxed = [], []

    def box(self, x, y, w, h, text, face, edge, fs=FS, lw=0.7, ls="-", text_xy=None, ha="center", va="center",
            color=INK, pad=0.35):
        patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=0.9", facecolor=face,
                               edgecolor=edge, linewidth=lw, linestyle=ls, clip_on=False)
        self.ax.add_patch(patch)
        tx, ty = text_xy if text_xy else (x + w / 2, y + h / 2)
        label = self.ax.text(tx, ty, text, ha=ha, va=va, fontsize=fs, color=color, linespacing=1.12, clip_on=False)
        self.texts.append(label); self.boxed.append((patch, label))
        return patch

    def arrow(self, start, end, color="#333333", ls="-", lw=0.8, ms=7.0):
        self.ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=ms, linewidth=lw, linestyle=ls,
                                          color=color, shrinkA=0.5, shrinkB=0.5, clip_on=False))

    def line(self, xs, ys, color="#888888", lw=0.55, ls="-"):
        self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, clip_on=False, solid_capstyle="butt")

    def validate(self):
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        page = self.fig.bbox
        boxes = [t.get_window_extent(renderer=r) for t in self.texts]
        for t, b in zip(self.texts, boxes):
            if b.x0 < page.x0 or b.y0 < page.y0 or b.x1 > page.x1 or b.y1 > page.y1:
                raise AssertionError(f"text leaves the page: {t.get_text()!r}")
        inset = 0.8 * self.fig.dpi / 72
        for patch, label in self.boxed:
            pb, lb = patch.get_window_extent(renderer=r), label.get_window_extent(renderer=r)
            if lb.x0 < pb.x0 + inset or lb.y0 < pb.y0 + inset or lb.x1 > pb.x1 - inset or lb.y1 > pb.y1 - inset:
                raise AssertionError(f"text leaves its box: {label.get_text()!r}")
        for i, a in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                if a.overlaps(boxes[j]):
                    raise AssertionError(f"text overlap: {self.texts[i].get_text()!r} / {self.texts[j].get_text()!r}")


def draw(n_gb: int, n_au: int):
    c = Canvas()
    # shared top row
    top, th = 65.0, 19.0
    c.box(2, top, 40, th, "Source-area demand totals\nfrom reported peaks\n+ land-use and\nbuilt-up inputs", **SHARED)
    c.box(50, top, 38, th, "Spatial demand\nallocation\nUni | LU | GNN", **SHARED)
    c.box(96, top, 85, th,
          "Task-specific aggregation (RQ1)\n"
          "Reconstruction: sum over substation Voronoi regions\n"
          "Siting: demand-weighted distance to selected sites\n"
          "Sizing: sum over catchments of the sited substations\n"
          "Connection: sum within radius $R$ of a candidate location", **SHARED)
    c.arrow((42.4, top + th / 2), (49.6, top + th / 2))
    c.arrow((88.4, top + th / 2), (95.6, top + th / 2))
    # evaluation reference and diagnostics
    sy, sh = 52.5, 7.5
    c.box(2, sy, 44, sh, "Ref: reported-demand\nreference allocation", fs=7.0, lw=0.5, **SIDE)
    c.box(50, sy, 38, sh, "Controlled variations\nof Ref (RQ1)", fs=7.0, lw=0.5, **SIDE)
    c.line([62, 24], [top - 0.4, sy + sh + 0.4]); c.line([72, 69], [top - 0.4, sy + sh + 0.4])
    # split from the aggregation node into the two branches
    split = (138.5, 52.0)
    c.line([138.5, split[0]], [top - 0.4, split[1]], color="#333333", lw=0.7)
    c.arrow(split, (104.0, 49.4), color=TASKS["edge"], lw=0.9)
    c.arrow(split, (151.5, 49.4), color=BOUNDS["edge"], lw=0.9, ls=(0, (4, 2)))
    # left branch: four tasks and two cases
    c.box(2, 3, 116, 46, "Four evaluation tasks  |  RQ2: change from LU to GNN in each task", face=TASKS["face"],
          edge=TASKS["edge"], fs=FS_TITLE, lw=0.9, text_xy=(4.5, 46.2), ha="left", va="center", color=TASKS["title"])
    ty, tht = 19.5, 21.0
    c.box(5, ty, 24, tht, "Peak-demand\nreconstruction\n\nRMSE", face="white", edge=TASKS["edge"])
    c.box(32, ty, 22, tht, "Substation\nsiting\n\nWSD", face="white", edge=TASKS["edge"])
    c.box(57, ty, 22, tht, "Rule-based\nsizing at\nsited substations\n\nRSD", face="white", edge=TASKS["edge"])
    c.arrow((54.4, ty + tht / 2), (56.6, ty + tht / 2), color=TASKS["title"], ms=6)
    c.box(82, 16.0, 33, 27.5, "Large-demand connection", face="white", edge=TASKS["edge"], text_xy=(98.5, 40.8))
    c.box(84, 27.0, 29, 10.5, "Fixed-location\ncost error\nRQ1, RQ2", face="#F5FAFD", edge="#8FC9EC", lw=0.5)
    c.box(84, 18.0, 29, 7.5, "Selection regret\nRQ1", face="#F5FAFD", edge="#8FC9EC", lw=0.5)
    c.box(5, 5.0, 52, 10.0, f"Britain: {n_gb} regions,\ncost in £", face="white", edge="#8FC9EC", lw=0.55)
    c.box(62, 5.0, 53, 10.0, f"Australia: {n_au} regions,\nrequirement in MVA equivalents", face="white", edge="#8FC9EC", lw=0.55)
    # right branch: connection bounds
    c.box(122, 3, 59, 46, "Connection-specific bounds  |  RQ3\n(connection task only)", face=BOUNDS["face"], edge=BOUNDS["edge"],
          fs=FS_TITLE, lw=0.9, ls=(0, (4, 2)), text_xy=(151.5, 45.0), color=BOUNDS["title"])
    chain = [(35.0, 5.5, "Error limit at every candidate location"),
             (27.4, 5.5, "Worst case within the interval"),
             (19.8, 5.5, "Bounds on both connection outputs"),
             (4.2, 13.6, "Result at a tolerance:\ninsensitive at tolerance, not certified\nor not assessable\n"
                         "Audit: budget held or exceeded")]
    for y, h, text in chain:
        c.box(125, y, 53, h, text, face="white", edge=BOUNDS["edge"], lw=0.6)
    for (y0, _, _), (y1, h1, _) in zip(chain[:-1], chain[1:]):
        c.arrow((151.5, y0 - 0.4), (151.5, y1 + h1 + 0.4), color=BOUNDS["title"], ms=6)
    c.validate()
    return c.fig


def main() -> None:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--backup", type=Path, default=here.parents[2] / "results" / "_backup")
    ap.add_argument("--root", type=Path, help="Frozen run directory, overriding --backup/frozen/--run-id")
    ap.add_argument("--out", type=Path, default=here.parents[1] / "figures_new")
    args = ap.parse_args()
    root = args.root.resolve() if args.root else args.backup.resolve() / "frozen" / args.run_id
    led = pd.read_csv(root / "ledger/claim_ledger.csv",
                      dtype={"value": str}).drop_duplicates("claim_id").set_index("claim_id")
    n_gb, n_au = (int(float(led.loc[f"DATA.{c}.n_regions", "value"])) for c in ("GB", "AU"))
    fig = draw(n_gb, n_au)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "fig0_pipeline.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(args.out / "fig0_pipeline.png")
    plt.close(fig)
    paper2_style.write_caption(args.out, "fig0_pipeline", "fig:pipeline",
        "Evaluation design. Three allocation methods, \\acrfull{uni}, \\acrfull{lu} and \\acrfull{gnn}-based allocation, spread source-area "
        "totals derived from reported substation peaks over raster cells. Each task sums the allocation over its own geographic support "
        "(RQ1). The left branch holds the four tasks, scored by the \\acrfull{rmse} of substation peaks, the "
        "\\acrfull{wsd}, the \\acrfull{rsd} and, for large-demand connection, the fixed-location "
        "cost error and the selection regret. Sizing uses the sites chosen by siting, which is the only sequential dependency "
        "among the tasks. RQ2 follows the change from \\acrshort{lu} to \\acrshort{gnn}-based allocation through the four tasks, and RQ1 relates "
        "the agreement cell by cell and the agreement of the neighborhood sums to both connection outputs. The right branch (RQ3) applies only to the connection task. An error "
        "limit that holds at every candidate location at once (the error budget) bounds both connection outputs. At a "
        "planner's tolerance each bound is insensitive at tolerance, not certified or not assessable. The \\acrfull{ref}, built "
        "from reported substation peaks, is the evaluation reference, and its controlled variations enter RQ1.",
        f"frozen/{args.run_id}/ledger (DATA.GB.n_regions, DATA.AU.n_regions); layout adapted from the earlier navigation figure")
    print("wrote fig0_pipeline")


if __name__ == "__main__":
    main()
