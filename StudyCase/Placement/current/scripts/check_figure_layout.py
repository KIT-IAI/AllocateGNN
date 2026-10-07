"""Layout check for the exported figures: page width, text inside the page, frozen run intact; .tex edits are listed, not failed.

Scans every PDF in ``figures_new/`` (so deleted figures cannot linger in the report), records page
size in mm and the minimum distance of any text anchor to the page edge, re-verifies the frozen
run against its SHA256SUMS, and lists tracked .tex files with changes for information only
(the manuscript is edited after the figure stage, so a .tex change is not a figure failure). Writes
``figures_new/layout_check.json`` and exits non-zero on any failure.

    python scripts/check_figure_layout.py --run-id fixedload_20260925_r1
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from pypdf import PdfReader

PT_TO_MM = 25.4 / 72
TARGET_WIDTH_MM = 183.0
COLUMN_WIDTH_MM = 88.0  # single-column figures (figure, not figure*)


def text_anchors(page):
    points = []

    def visit(text, cm, tm, font_dict, font_size):
        if text.strip():
            x = cm[0] * tm[4] + cm[2] * tm[5] + cm[4]
            y = cm[1] * tm[4] + cm[3] * tm[5] + cm[5]
            points.append((x, y))
    page.extract_text(visitor_text=visit)
    return points


def main() -> None:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--figures", type=Path, default=here.parents[1] / "figures_new")
    ap.add_argument("--backup", type=Path, default=here.parents[2] / "results" / "_backup")
    args = ap.parse_args()
    checks, failures = [], []
    for pdf in sorted(args.figures.glob("*.pdf")):
        page = PdfReader(pdf).pages[0]
        w, h = float(page.mediabox.width), float(page.mediabox.height)
        pts = text_anchors(page)
        outside = [(round(x, 1), round(y, 1)) for x, y in pts if not (0 <= x <= w and 0 <= y <= h)]
        margin = min((min(x, y, w - x, h - y) for x, y in pts), default=None)
        row = {"file": pdf.name, "width_mm": round(w * PT_TO_MM, 2), "height_mm": round(h * PT_TO_MM, 2),
               "text_anchors": len(pts), "text_outside_page": outside,
               "minimum_text_edge_margin_mm": None if margin is None else round(margin * PT_TO_MM, 3)}
        if min(abs(row["width_mm"] - TARGET_WIDTH_MM), abs(row["width_mm"] - COLUMN_WIDTH_MM)) > 0.5 or outside:
            failures.append(pdf.name)
        checks.append(row)
    frozen = args.backup.resolve() / "frozen" / args.run_id
    bad = []
    for line in (frozen / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, rel = line.strip().split("  ", 1)
        if hashlib.sha256((frozen / rel).read_bytes()).hexdigest() != digest:
            bad.append(rel)
    repo = here.parents[1]
    tex = [p for p in subprocess.check_output(["git", "-C", str(repo), "diff", "--name-only", "HEAD"], text=True).split()
           if p.endswith(".tex")]
    record = {"checked_at": datetime.now(timezone.utc).isoformat(), "target_width_mm": [TARGET_WIDTH_MM, COLUMN_WIDTH_MM], "checks": checks,
              "scope": "PDF page size and text-anchor positions; visual overlap is reviewed on the rendered PNG",
              "frozen_hash_check": {"run_id": args.run_id, "checked_files": len(list(filter(None, (frozen / "SHA256SUMS").read_text().splitlines()))),
                                    "failures": bad},
              "tex_changes": tex, "failures": failures + bad}
    (args.figures / "layout_check.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"figures": [c["file"] for c in checks], "failures": record["failures"]}, indent=1))
    if record["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
