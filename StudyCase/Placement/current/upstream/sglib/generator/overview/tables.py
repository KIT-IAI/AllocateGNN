from __future__ import annotations

import json

import pandas as pd

from sglib.generator import execution, stage
from sglib.generator.registry import unit_status


def inputs(ctx) -> pd.DataFrame:
    bundle = execution._load_bundle(ctx.root)
    return pd.DataFrame([{
        "region": region,
        "grid_cells": len(bundle.grids[region][0]),
        "sources": len(bundle.source_regions[region]),
        "targets": len(bundle.stations[region]),
    } for region in bundle.regions])


def inventory(ctx, members=None) -> pd.DataFrame:
    selected = stage.select_units(ctx.units, members) if members else sorted(ctx.units)
    return pd.DataFrame([{"unit": key, "step": ctx.units[key].step,
                          "state": unit_status(ctx.units[key], ctx.root)} for key in selected])


def candidates(ctx) -> pd.DataFrame:
    document = json.loads((ctx.root / "candidates/candidate_index.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame(document["entries"])
    return frame.groupby(["family", "label"], sort=False).agg(fields=("region", "size"),
        regions=("region", "nunique")).reset_index()


def allocators(ctx) -> pd.DataFrame:
    rows = []
    for name in ("idr_fixed", "idr_matched"):
        path = ctx.root / name / "index.json"
        entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
        rows.append({"allocator": name, "status": "DONE", "fields": len(entries),
                     "regions": len({e["region"] for e in entries}),
                     "gate_pass": sum(e.get("g0_pass", False) and e.get("g1_pass", False) for e in entries),
                     "reason": ""})
    return pd.DataFrame(rows)


def audit(ctx) -> pd.DataFrame:
    document = json.loads((ctx.root / "audit.json").read_text(encoding="utf-8"))
    return pd.DataFrame([{"item": key, "value": value} for key, value in document.items()
                         if key not in {"regions", "delivery_coverage"}])


def field_concentration(ctx, region=None, labels=("Uni", "GPM", "MLP", "GNN", "GNNpriorNP", "GNNpostNP", "GNNfusionNP"), seed=None) -> pd.DataFrame:
    """Concentration statistics of materialized demand fields for one region."""

    import numpy as np

    from . import figures

    _, region, _, _, _ = figures._region_context(ctx, region)
    seed = figures._default_seed(ctx) if seed is None else int(seed)
    uniform = figures._uniform(ctx, region)
    covered = uniform > 0
    rows = []
    for label in labels:
        try:
            values, used_seed = figures._field(ctx, region, label, seed)
        except ValueError as exc:
            rows.append({"label": label, "region": region, "note": str(exc)})
            continue
        ratio = values[covered] / uniform[covered]
        rows.append({
            "label": label, "region": region, "seed": used_seed,
            "total": float(values.sum()),
            "gini": round(figures._gini(values), 3),
            "top10_share": round(figures._top_share(values), 3),
            "max_over_uniform": round(float(ratio.max()), 2),
            "cells_below_half_uniform": int((ratio < 0.5).sum()),
            "cells_zero": int((values == 0).sum()),
        })
    return pd.DataFrame(rows)


def partition_summary(ctx, region=None, candidate="GNN", seed=None) -> pd.DataFrame:
    """Voronoi variants for one region: unit counts, cells moved, and IDR gate results."""

    import numpy as np

    from . import figures

    _, region, _, _, _ = figures._region_context(ctx, region)
    seed = figures._matched_seed(ctx, region, candidate, seed)
    vd, variants = figures.partition_variants(ctx, region, candidate, seed)
    rows = []
    for title, station_of_cell, categories, unit_label, note in variants:
        rows.append({
            "variant": title, "region": region, "status": "DONE",
            "units": int(categories.max()) + 1, "unit_label": unit_label,
            "cells_moved_vs_vd_pct": round(100 * float(np.mean(station_of_cell != vd)), 2),
            "note": note.replace("\n", " "),
        })
    return pd.DataFrame(rows)
