"""Assignment-aware equal-cell grid baseline for known-site reconstruction."""

from __future__ import annotations

import geopandas as gpd
import numpy as np


class EqualGridError(ValueError):
    """Raised when the EqualGrid construction is not identifiable."""


def materialize_equal_grid(
    grid: gpd.GeoDataFrame,
    uniform_field: np.ndarray,
    assignment: np.ndarray,
    *,
    source_column: str,
) -> np.ndarray:
    """Equalise Uniform mass over active VD-cell intersections per source/C/U block.

    The Uniform field supplies the authoritative mass of every source/support
    block.  EqualGrid changes only its within-block spatial distribution.  It
    therefore remains source- and C/U-conserving while the Z block stays exact
    zero.  Because ``assignment`` is built from observed station locations, the
    resulting field is a transductive Reconstruction-only candidate.
    """

    required = {source_column, "covered_mask", "unknown_mask", "zero_mask"}
    missing = sorted(required - set(grid.columns))
    if missing:
        raise EqualGridError(f"EqualGrid grid is missing columns: {missing}")
    if not np.array_equal(grid.index.to_numpy(), np.arange(len(grid))):
        raise EqualGridError("EqualGrid requires a RangeIndex-aligned grid")

    uniform = np.asarray(uniform_field, dtype=np.float64)
    targets = np.asarray(assignment, dtype=np.int64)
    if uniform.shape != (len(grid),) or targets.shape != (len(grid),):
        raise EqualGridError("EqualGrid field and assignment must align to grid rows")
    if not np.isfinite(uniform).all() or (uniform < 0).any():
        raise EqualGridError("EqualGrid Uniform authority must be finite and non-negative")
    if (targets < 0).any():
        raise EqualGridError("EqualGrid assignment contains a negative target")

    covered = grid["covered_mask"].to_numpy(dtype=bool)
    unknown = grid["unknown_mask"].to_numpy(dtype=bool)
    zero = grid["zero_mask"].to_numpy(dtype=bool)
    partition = covered.astype(np.int8) + unknown.astype(np.int8) + zero.astype(np.int8)
    if not np.array_equal(partition, np.ones(len(grid), dtype=np.int8)):
        raise EqualGridError("EqualGrid C/U/Z masks must form an exact partition")
    if np.count_nonzero(uniform[zero]) != 0:
        raise EqualGridError("EqualGrid Uniform authority must be exact-zero on Z")

    result = np.zeros(len(grid), dtype=np.float64)
    sources = grid[source_column].astype(str).to_numpy()
    for source in dict.fromkeys(sources.tolist()):
        source_mask = sources == source
        for support_mask in (covered, unknown):
            indices = np.flatnonzero(source_mask & support_mask)
            target_mass = float(uniform[indices].sum())
            if target_mass == 0.0:
                continue
            if not len(indices):
                raise EqualGridError(f"source {source}: positive block mass has no cells")
            active_targets = np.unique(targets[indices])
            if not len(active_targets):
                raise EqualGridError(f"source {source}: block has no active VD cell")
            cell_mass = target_mass / float(len(active_targets))
            for target in active_targets:
                intersection = indices[targets[indices] == target]
                if not len(intersection):
                    raise EqualGridError("EqualGrid active target has an empty intersection")
                result[intersection] = cell_mass / float(len(intersection))

    if np.count_nonzero(result[zero]) != 0:
        raise EqualGridError("EqualGrid wrote non-zero demand to Z")
    return result


__all__ = ["EqualGridError", "materialize_equal_grid"]


