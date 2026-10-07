"""Read-only tables for the numbered Experiment notebooks (no figures: plan 006 keeps those in Analysis)."""

from __future__ import annotations

import json

import pandas as pd

from sglib.experiment import production, stage


def status(ctx) -> pd.DataFrame:
    return pd.DataFrame(stage.status_table(ctx))


def _csv(path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"region": str, "country": str}, float_precision="round_trip", **kwargs)


def connection_preflight(ctx) -> pd.DataFrame:
    """Per (lambda, radius) admission summary of the fixed-candidate connection scenarios."""

    return _csv(ctx.root / "preflight/connection/connection_scenario_summary.csv")


def connection_not_assessable(ctx) -> pd.DataFrame:
    return _csv(ctx.root / "preflight/connection/not_assessable.csv")


def planning_pool(ctx) -> pd.DataFrame:
    """Per-region fixed siting pool: k stations, requested and actual pool size, admission status."""

    table = _csv(ctx.root / "preflight/planning_pool/planning_pool_preflight.csv")
    keep = [c for c in ("country", "region", "k", "n_buildable", "requested_M", "actual_M", "status", "reason") if c in table.columns]
    return table[keep]


def coverage(ctx) -> pd.DataFrame:
    """Expected versus produced coordinates per unit, read from receipts and markers."""

    return pd.DataFrame(production.coverage(ctx)["units"])


def audit(ctx) -> pd.DataFrame:
    document = json.loads((ctx.root / "audit.json").read_text(encoding="utf-8"))
    return pd.DataFrame([{"item": key, "value": value} for key, value in document.items() if key != "units"])
