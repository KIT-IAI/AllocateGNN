# -*- coding: utf-8 -*-
"""
Shared post-correction utility module.

Provides the demand-correction pipeline (NTL/proximity factor construction,
Voronoi aggregation, and post-hoc correction schemes) shared by the
experiment scripts in this directory.

Key numerical conventions:
1. Standard deviation uses numpy's `.std()` (ddof=0). Function entry points
   coerce array-like inputs to numpy float64 arrays, since a pandas Series
   (which defaults to ddof=1) would silently shift the additive-correction
   alpha scaling.
2. Coordinate reference systems are intentionally kept distinct: proximity
   distance is computed in EPSG:27700 (TARGET_CRS), while Voronoi
   nearest-neighbour assignment uses EPSG:3857 (VoronoiAllocator's default
   working_crs). Unifying these would flip grid-point assignment near
   substation boundaries.
3. The RCI (residential/commercial/industrial) mask only affects how the
   epsilon and median normalisation constants are computed; the resulting
   factor is still applied to every agent and is strictly positive.
4. The NTL factor uses a log1p form; gamma = 2.0; DIST_CLAMP_KM = 0.01.
5. Functions assume `grid_gdf` / `subs_sub` use a RangeIndex, since
   `group.index` is used directly as a numpy positional index
   (`load_grid_and_subs` resets the index accordingly).

Two behavioral notes on this module's aggregation/loading logic:
A. `load_grid_and_subs` raises `FileNotFoundError` if `{loc}_ntl.npz` is
   missing, rather than silently substituting zeros, to avoid masking a
   missing-input error.
B. Voronoi aggregation is split into `compute_voronoi_assignment` (computed
   once per region, cacheable via a dict) and `aggregate_by_assignment`
   (aggregates demand given an assignment). This avoids tens of thousands
   of repeated `sjoin_nearest` calls at larger experiment scales.
   `voronoi_aggregate` is kept as a combined convenience function with
   equivalent semantics.
"""

import sys
import pickle
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import geopandas as gpd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SpatialAllocation.Allocator import allocator_registry  # noqa: E402

warnings.filterwarnings('ignore', category=FutureWarning)

# ════════════════════════════════════════════════════════════
# Paths and constants
# ════════════════════════════════════════════════════════════

DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
ASSEMBLED_DIR = DATA_DIR / 'features' / 'assembled'
EXTRACTED_DIR = DATA_DIR / 'features' / 'extracted'

ALL_LOCATIONS = [
    'London',
    'TLH2', 'TLH3', 'TLJ1', 'TLF1', 'TLF2',
    'TLC1', 'TLC2', 'TLD6', 'TLG1', 'TLG2', 'TLE4',
    'TLH1', 'TLE3', 'TLD3', 'TLD4',
]

RCI_THRESHOLD = 0.5
PROXIMITY_GAMMA = 2.0
DIST_CLAMP_KM = 0.01
TARGET_CRS = 'EPSG:27700'


# ════════════════════════════════════════════════════════════
# Evaluation and data loading (see module docstring, note A)
# ════════════════════════════════════════════════════════════

def evaluate_allocation(subs_result: gpd.GeoDataFrame,
                        actual_col: str = 'Demand (MVA)',
                        alloc_col: str = 'allocated_demand') -> Dict[str, float]:
    """Computes allocation evaluation metrics."""
    actual = subs_result[actual_col].values
    allocated = subs_result[alloc_col].values
    corr, _ = pearsonr(actual, allocated)
    rmse = np.sqrt(mean_squared_error(actual, allocated))
    mae = mean_absolute_error(actual, allocated)
    return {'corr': corr, 'rmse': rmse, 'mae': mae}


def load_grid_and_subs(loc: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame,
                                          gpd.GeoDataFrame, np.ndarray]:
    """Loads grid data and substations. Raises an error if the NTL feature file is missing, rather than defaulting to zero."""
    with open(ASSEMBLED_DIR / f'{loc}_grid_points.pickle', 'rb') as f:
        grid_gdf, step_size_m = pickle.load(f)

    region_gdf = gpd.read_file(str(DATA_DIR / 'ITL3_region.gpkg'))
    substations_gdf = gpd.read_file(str(DATA_DIR / 'substations.gpkg'))

    study_itl3 = grid_gdf['ITL3'].unique()
    region_sub = region_gdf[region_gdf['ITL3'].isin(study_itl3)].copy()
    subs_sub = substations_gdf[substations_gdf['ITL3'].isin(study_itl3)].copy().reset_index(drop=True)

    # Load NTL — raises immediately if missing rather than silently defaulting to zero
    ntl_path = EXTRACTED_DIR / f'{loc}_ntl.npz'
    if not ntl_path.exists():
        raise FileNotFoundError(
            f'{ntl_path} does not exist. Silently defaulting to zero here would '
            f'mask a missing-input error, so this module raises immediately instead. '
            f'Verify the upstream input precondition checks first.')
    ntl_npz = np.load(str(ntl_path), allow_pickle=True)
    ntl_values = ntl_npz['data'][:, 0]

    return grid_gdf, region_sub, subs_sub, ntl_values


# ════════════════════════════════════════════════════════════
# Factor construction (inputs coerced to numpy arrays at entry)
# ════════════════════════════════════════════════════════════

def compute_prox_scores(grid_gdf: gpd.GeoDataFrame,
                        subs_sub: gpd.GeoDataFrame,
                        gamma: float = PROXIMITY_GAMMA) -> np.ndarray:
    """Computes proximity scores (distances computed in EPSG:27700)."""
    grid_proj = grid_gdf.to_crs(TARGET_CRS)
    subs_proj = subs_sub.to_crs(TARGET_CRS)
    grid_coords = np.column_stack([grid_proj.geometry.x.values, grid_proj.geometry.y.values])
    subs_coords = np.column_stack([subs_proj.geometry.x.values, subs_proj.geometry.y.values])
    dist_km = cdist(grid_coords, subs_coords, metric='euclidean') / 1000.0
    dist_km = np.maximum(dist_km, DIST_CLAMP_KM)
    return np.sum(dist_km ** (-gamma), axis=1)


def compute_factors(grid_gdf: gpd.GeoDataFrame,
                    ntl_values: np.ndarray,
                    prox_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the NTL factor and proximity factor per ITL3 region, returning per-agent factor arrays.

    epsilon is the 5th percentile of the NTL values within the RCI-masked,
    NTL>0 subset (falling back through two tiers to 0.1 if that subset is
    empty); the median is computed over the RCI-masked subset only. If the
    median is <= 0, it falls back to epsilon (for NTL) or 1e-6 (for
    proximity). Both factors use a log1p form.
    The RCI mask only affects epsilon/median; the factor itself is applied
    to every agent and is strictly positive.
    """
    # Coerce inputs to numpy float64 arrays to avoid pandas Series ddof/alignment semantics leaking in
    ntl_values = np.asarray(ntl_values, dtype=float)
    prox_scores = np.asarray(prox_scores, dtype=float)

    rci_sum = (grid_gdf['lu_residential_prop'].values
              + grid_gdf['lu_commercial_prop'].values
              + grid_gdf['lu_industrial_prop'].values)
    rci_mask = rci_sum > RCI_THRESHOLD

    ntl_factor = np.ones(len(grid_gdf))
    prox_factor = np.ones(len(grid_gdf))

    for itl3, group in grid_gdf.groupby('ITL3'):
        idx = group.index

        # NTL factor
        ntl_group = ntl_values[idx]
        rci_group = rci_mask[idx]
        rci_nonzero_ntl = ntl_group[rci_group & (ntl_group > 0)]
        if len(rci_nonzero_ntl) > 0:
            epsilon = np.percentile(rci_nonzero_ntl, 5)
        else:
            nonzero_ntl = ntl_group[ntl_group > 0]
            epsilon = np.percentile(nonzero_ntl, 5) if len(nonzero_ntl) > 0 else 0.1
        rci_ntl = ntl_group[rci_group]
        ntl_median = np.median(rci_ntl) if len(rci_ntl) > 0 else np.median(ntl_group)
        if ntl_median <= 0:
            ntl_median = epsilon
        ntl_factor[idx] = np.log(1 + ntl_group + epsilon) / np.log(1 + ntl_median)

        # Proximity factor
        prox_group = prox_scores[idx]
        rci_prox = prox_group[rci_group]
        prox_median = np.median(rci_prox) if len(rci_prox) > 0 else np.median(prox_group)
        if prox_median <= 0:
            prox_median = 1e-6
        prox_factor[idx] = np.log(1 + prox_group) / np.log(1 + prox_median)

    return ntl_factor, prox_factor


# ════════════════════════════════════════════════════════════
# Voronoi aggregation (split per module docstring, note B)
# ════════════════════════════════════════════════════════════

def compute_voronoi_assignment(grid_gdf: gpd.GeoDataFrame,
                               subs_sub: gpd.GeoDataFrame,
                               cache: Optional[dict] = None,
                               cache_key: Optional[str] = None) -> np.ndarray:
    """Computes the Voronoi nearest-neighbour assignment (once per region; result can be cached via a dict).

    Uses VoronoiAllocator's default working_crs = EPSG:3857 (must not be
    changed to EPSG:27700). sjoin_nearest can return multiple rows for
    points exactly equidistant between substations, so the assignment
    length is asserted against the grid point count.

    Args:
        cache: optional cache dict; used together with cache_key to
            short-circuit if already computed.
        cache_key: cache key (e.g. region name). Both cache and cache_key
            must be provided for caching to take effect.
    """
    if cache is not None and cache_key is not None and cache_key in cache:
        return cache[cache_key]

    alloc_voronoi = allocator_registry.create('voronoi')
    voronoi_res = alloc_voronoi.allocate(grid_gdf, subs_sub)
    assignment = voronoi_res.assignment

    assert len(assignment) == len(grid_gdf), (
        f'assignment length {len(assignment)} != grid point count {len(grid_gdf)} '
        f'— sjoin_nearest returned multiple rows for exactly-equidistant points')

    if cache is not None and cache_key is not None:
        cache[cache_key] = assignment
    return assignment


def aggregate_by_assignment(subs_sub: gpd.GeoDataFrame,
                            assignment: np.ndarray,
                            demand_arr: np.ndarray) -> gpd.GeoDataFrame:
    """Aggregates demand to substations according to the assignment array, returning a per-substation table with an added allocated_demand column.

    Note: `.loc[target_idx, ...]` relies on subs_sub having a RangeIndex
    (load_grid_and_subs resets the index accordingly); violating this
    assumption would silently misassign values.
    """
    assignment = np.asarray(assignment)
    demand_arr = np.asarray(demand_arr, dtype=float)

    subs_result = subs_sub.copy()
    subs_result['allocated_demand'] = 0.0
    for target_idx in range(len(subs_sub)):
        mask = assignment == target_idx
        subs_result.loc[target_idx, 'allocated_demand'] = demand_arr[mask].sum()
    return subs_result


def voronoi_aggregate(grid_gdf: gpd.GeoDataFrame,
                      subs_sub: gpd.GeoDataFrame,
                      demand_arr: np.ndarray,
                      cache: Optional[dict] = None,
                      cache_key: Optional[str] = None) -> Dict[str, float]:
    """Aggregates demand to substations via Voronoi partitioning and evaluates the result.

    Combines compute_voronoi_assignment + aggregate_by_assignment + evaluate_allocation.
    """
    assignment = compute_voronoi_assignment(grid_gdf, subs_sub, cache=cache, cache_key=cache_key)
    subs_result = aggregate_by_assignment(subs_sub, assignment, demand_arr)
    return evaluate_allocation(subs_result)


# ════════════════════════════════════════════════════════════
# Three post-correction schemes (inputs coerced to numpy arrays at entry)
# ════════════════════════════════════════════════════════════

def apply_correction_no_renorm(base_demand: np.ndarray,
                               combined_factor: np.ndarray,
                               grid_gdf: gpd.GeoDataFrame,
                               region_sub: gpd.GeoDataFrame) -> np.ndarray:
    """Multiplicative post-correction without per-ITL3 renormalisation. Demand conservation no longer holds under this scheme.

    grid_gdf / region_sub are unused in this variant but kept for a uniform
    signature across the three correction schemes.
    """
    base_demand = np.asarray(base_demand, dtype=float)
    combined_factor = np.asarray(combined_factor, dtype=float)
    result = base_demand * combined_factor
    return result


def apply_additive_correction(base_demand: np.ndarray,
                              combined_factor: np.ndarray,
                              grid_gdf: gpd.GeoDataFrame,
                              region_sub: gpd.GeoDataFrame) -> np.ndarray:
    """Additive post-correction: demand_i = base_i + alpha * (factor_i - mean_factor), followed by renormalisation.

    alpha scales the additive offset so its magnitude is comparable to the
    multiplicative scheme's effect. `.std()` uses numpy's ddof=0 convention;
    inputs are coerced to numpy arrays at the function entry point to
    guarantee this.
    """
    base_demand = np.asarray(base_demand, dtype=float)
    combined_factor = np.asarray(combined_factor, dtype=float)

    region_info = region_sub.set_index('ITL3')
    result = np.zeros_like(base_demand)

    for itl3, group in grid_gdf.groupby('ITL3'):
        if itl3 not in region_info.index:
            continue
        total_demand = region_info.loc[itl3, 'Demand (MVA)']
        idx = group.index

        factor_group = combined_factor[idx]
        base_group = base_demand[idx]

        # Additive offset: factor centred around its mean, then added to base demand
        offset = factor_group - factor_group.mean()
        # Scale the offset so its std matches that of base_demand
        base_std = base_group.std()
        offset_std = offset.std()
        if offset_std > 0 and base_std > 0:
            alpha = base_std / offset_std
        else:
            alpha = 0.0

        raw = base_group + alpha * offset
        raw = np.maximum(raw, 1e-6)  # Guard against negative values

        raw_sum = raw.sum()
        if raw_sum > 0:
            result[idx] = total_demand * raw / raw_sum
        else:
            result[idx] = total_demand / len(group)

    return result


def apply_standard_multiplicative(base_demand: np.ndarray,
                                  combined_factor: np.ndarray,
                                  grid_gdf: gpd.GeoDataFrame,
                                  region_sub: gpd.GeoDataFrame) -> np.ndarray:
    """Standard multiplicative post-correction: demand_i = base_i * factor_i / sum(base_j * factor_j) * total."""
    base_demand = np.asarray(base_demand, dtype=float)
    combined_factor = np.asarray(combined_factor, dtype=float)

    region_info = region_sub.set_index('ITL3')
    result = np.zeros_like(base_demand)

    for itl3, group in grid_gdf.groupby('ITL3'):
        if itl3 not in region_info.index:
            continue
        total_demand = region_info.loc[itl3, 'Demand (MVA)']
        idx = group.index

        raw = base_demand[idx] * combined_factor[idx]
        raw_sum = raw.sum()
        if raw_sum > 0:
            result[idx] = total_demand * raw / raw_sum
        else:
            result[idx] = total_demand / len(group)

    return result
