"""Netherlands country adapter for DataOverview.

The shared transform dispatcher discovers this mapping dynamically.  Every runner
accepts a resolved :class:`CountryPipelineContext`; no country branch is added to a
core algorithm.
"""

from __future__ import annotations

from sglib.dataoverview.overview.inventory import write_inventory
from sglib.dataoverview.processing.pipeline import run_features, run_grid

from .pipeline import derive_nl, run_grid_skeleton, write_gate_a


def _regions(context, *, force: bool = False):
    return derive_nl(context, force=force)


def _stations(context, *, force: bool = False):
    return derive_nl(context, force=force)


def _gate_a(context, *, force: bool = False):
    return write_gate_a(context, force=force)


def _grid_skeleton(context, *, force: bool = False):
    return run_grid_skeleton(context, force=force)


def _grid(context, *, force: bool = False):
    return run_grid(context, force=force)


def _features(context, *, force: bool = False):
    return run_features(context, force=force)


def _inventory(context, *, force: bool = False):
    del force
    return write_inventory(context)


PRODUCT_RUNNERS = {
    "regions": _regions,
    "stations": _stations,
    "gate_a": _gate_a,
    "grid_skeleton": _grid_skeleton,
    "grid": _grid,
    "features": _features,
    "inventory": _inventory,
}

__all__ = ["PRODUCT_RUNNERS", "derive_nl", "run_grid_skeleton", "write_gate_a"]
