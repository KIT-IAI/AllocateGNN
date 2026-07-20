# -*- coding: utf-8 -*-
"""AU third case study - feature-fusion arm training script (015).

This script is adapted from `012_train_au_gnn.py` (which remains unmodified);
the only scientific change is moving NTL and Proximity from a
"loss-prior / post-correction" channel into an **input-feature** channel
(feature-level fusion, following the same approach as the UK case's
`027_train_feature_fusion.py`): AGENT_FEATURE_COLS gains two extra columns,
'ntl_feat' / 'prox_feat', taking the agent feature dimension from 5 to 7.
- Transform = log1p (matches the same pointwise form used internally by
  NTLPriorLoss and ProximityPriorLoss: torch.log(1 + clamp(x, min=0)));
- Standardization = per-region z-score (within each location's agent set,
  numpy convention with ddof=0); standardizing independently within each
  region means no statistics are shared across train/test folds, so there
  is no cross-fold leakage by construction.

=== Deviations from 012 (read before reusing this pipeline) ===
1. AGENT_FEATURE_COLS = the 5-dim lu_* columns + ntl_feat + prox_feat
   (7 dims total); the fusion columns are written inside build_graphs(),
   before feature-column selection, at the same point and with the same
   formula as in the UK case's `027_train_feature_fusion.py`. The raw
   values for the fusion columns come from the B2 precomputed npz files
   (ntl_dict / proximity_dict) - the same tensors already used as priors
   in 012, so AU needs no extra intermediate products.
2. CONFIG_MAP keeps only 'fusion' (same semantics as the UK case's 027
   baseline config: the fusion arm's signal enters as an input feature
   rather than a loss prior, trained from scratch for 200 epochs with just
   the base loss; prior-loss configs would conflict with this experiment's
   semantics, so they are all removed). The config directory name is
   'fusion' (not 'baseline' as in the UK case) - the AU arms share the same
   data/processed/training/ tree as the existing baseline/ntl_prox arms,
   and the directory name doubles as the arm name to avoid confusion with
   the real baseline.
3. The graph cache lives in its own directory, `graph_cache_fusion/`
   (7-dim), kept separate from 012's main cache (5-dim); verify_graph_cache
   asserts 7 dims and fails immediately if it hits the 5-dim cache.
4. The run manifest is written to `run_manifest_fusion.json`, so it does
   not overwrite 012's run_manifest.json.
5. Everything else is copied directly from 012: region universe / SA3 star
   topology / hyperparameters (hgt/256/128/3 layers/tau=0.01/lr 1e-3)
   copied from the UK case / 4-fold x 3 seeds / two-stage evaluation via
   the shared correction utilities (7 arms using the same labeling scheme
   as elsewhere in this case study) / the assert_training_complete
   resumption-integrity check / prior-tensor injection (the fusion config
   does not consume these tensors, but they are injected anyway to keep
   the graph structure consistent - the same "always inject" behavior as
   012's baseline).

Usage:
    python 015_train_au_fusion.py --dry-run                  # input check + single-region 7-dim graph smoke test, then exit (no training)
    python 015_train_au_fusion.py --cache-only               # only build/verify the 12-graph fusion cache, then exit (CPU)
    python 015_train_au_fusion.py --config fusion --seed 42
    python 015_train_au_fusion.py --all --seeds 42 123 456   # full run: 1 config x 3 seeds x 4 folds = 12 training runs

Config mapping:
    fusion:     {'landuse_prediction_loss': 1.0}             200 epochs (7-dim input)
"""

import sys
import argparse
import pickle
import warnings
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from shapely.geometry import Point

# Project root directory: walk upward to find the directory containing SpatialAllocation
SCRIPT_DIR = Path(__file__).resolve().parent
_p = SCRIPT_DIR
while not (_p / 'SpatialAllocation').exists():
    if _p.parent == _p:
        raise RuntimeError('Could not find the repository root (SpatialAllocation package)')
    _p = _p.parent
PROJECT_ROOT = _p
# Add the UK experiment directory to sys.path so shared_correction_utils can be
# imported read-only across case studies
UK_EXP_DIR = PROJECT_ROOT / 'StudyCase' / 'British_weighter_experiments'
for _extra in (str(UK_EXP_DIR), str(PROJECT_ROOT)):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

import shared_correction_utils as scu  # noqa: E402  # shared post-hoc correction utilities (read-only import from the UK case study)
from SpatialAllocation.GNN.utils.GraphBuilder import (  # noqa: E402
    preprocess_features, prepare_hetero_graph_from_processed
)
from SpatialAllocation.GNN.core.EdgeWeightSolver import EdgeWeightSolver  # noqa: E402
from SpatialAllocation.GNN.core.ModelConfig import ModelConfig  # noqa: E402

warnings.filterwarnings('ignore', category=FutureWarning)

# The Windows console defaults to cp1252, which cannot encode non-ASCII output - force UTF-8 (does not affect file outputs)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ════════════════════════════════════════════════════════════
# Path constants
# ════════════════════════════════════════════════════════════

PROCESSED = SCRIPT_DIR / 'data' / 'processed'
ASSEMBLED_DIR = PROCESSED / 'features' / 'assembled'
EXTRACTED_DIR = PROCESSED / 'features' / 'extracted'
REGIONS_SA3 = PROCESSED / 'regions_sa3.gpkg'
STATION_TABLE = PROCESSED / 'station_table_fy2009.csv'
STEP_TABLE = PROCESSED / 'grid' / 'grid_step_size_table.csv'
FEATURE_SCHEMA = ASSEMBLED_DIR / 'feature_schema.json'
B2_REGISTRY = SCRIPT_DIR / 'docs' / 'b1b2' / 'b2_features.json'

# Graph cache (deviation 3: a dedicated 7-dim cache directory for the fusion
# arm, kept separate from 012's main cache)
GRAPH_CACHE_DIR = PROCESSED / 'graph_cache_fusion'
# Training output root directory (shares the same tree as 012; the config
# directory name 'fusion' doubles as the arm name, deviation 2)
TRAIN_ROOT = PROCESSED / 'training'

N_FOLDS = 4

# ─── Region configuration (12 SA4 regions; order = feature_schema.json
#     locations, frozen in this exact order since the KFold split depends
#     on it - do not change) ───
ALL_LOCATIONS = [
    'Central_Coast',
    'Hunter_Valley_exc_Newcastle',
    'Newcastle_and_Lake_Macquarie',
    'Sydney_City_and_Inner_South',
    'Sydney_Eastern_Suburbs',
    'Sydney_Inner_South_West',
    'Sydney_Inner_West',
    'Sydney_North_Sydney_and_Hornsby',
    'Sydney_Northern_Beaches',
    'Sydney_Parramatta',
    'Sydney_Ryde',
    'Sydney_Sutherland',
]

EXPECTED_N_SA3 = 34                 # verified count of SA3 regions
EXPECTED_N_SA4 = 12                 # verified count of SA4 regions
N_USABLE = 143                      # usable stations for FY2009
USABLE_STATUSES = {'matched', 'matched_osm'}
DEMAND_COL = 'peak_mw'              # primary demand metric (annual peak, MW)

# ─── Feature control (deviation 1: base 5 dims + 2 fusion dims = 7 dims,
#     same formula as the UK case's 027 script) ───
AGENT_FEATURE_COLS_BASE = [
    'lu_residential_prop', 'lu_commercial_prop', 'lu_industrial_prop',
    'lu_agricultural_prop', 'lu_others_prop',
]
FUSION_FEATURE_COLS = ['ntl_feat', 'prox_feat']
AGENT_FEATURE_COLS = AGENT_FEATURE_COLS_BASE + FUSION_FEATURE_COLS

SOURCE_FEATURE_COLS = [
    'residential_percent', 'commercial_percent', 'industrial_percent',
    'agricultural_percent', 'others_percent',
]

# ─── Constants (copied directly from the UK case; the proximity formula
#     constants are already baked into the B2 precomputed features, so only
#     RCI remains here) ───
RCI_THRESHOLD = 0.5

# tau initialization (copied from the UK case; TAU_START is the single
# constant driving both the resumption branch and the normal-training
# branch's ModelConfig, to keep the two in sync and avoid drift)
TAU_START = 0.01

# ─── Land-use mapping (structurally matches the UK case) ───
LU_PROP_TO_CATEGORY = {
    'lu_residential_prop': 'residential',
    'lu_commercial_prop': 'commercial',
    'lu_industrial_prop': 'industrial',
    'lu_agricultural_prop': 'agricultural',
    'lu_others_prop': 'others',
}

# ─── Config mapping (deviation 2: only 'fusion' is kept - the signal enters
#     as an input feature, with 200 epochs on the base loss, matching the
#     semantics of the UK case's 027 CONFIG_MAP) ───
CONFIG_MAP = {
    'fusion': {
        'objective_weights': {'landuse_prediction_loss': 1.0},
        'epochs': 200,
    },
}

AGENT_CONNECTIVITY = None           # SA3 star topology, no agent-agent grid edges

# ─── Correction arm labels (grid_demands keys and metric row names are
#     derived from this list) ───
ARM_LABELS = ['GNN', 'GNNpostN', 'GNNpostP', 'GNNpostNP',
              'GNNaddN', 'GNNaddP', 'GNNaddNP']
SIGNALS = ['N', 'P', 'NP']

# GNN base per-SA3 conservation tolerance (a relative lower bound; see the
# note by GNN_CONS_RTOL in 012 - the error-model upper bound is
# max(GNN_CONS_RTOL, n*eps32), a necessary consequence of float32
# accumulation order, not a precision regression)
GNN_CONS_RTOL = 1e-4
_EPS32 = float(np.finfo(np.float32).eps)


# ════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════

def load_regions() -> gpd.GeoDataFrame:
    """Load the 34 source SA3 regions (with demand + 5 percentages), injecting alias columns used by the shared module.

    Aliases (pure renaming, no change to numeric semantics):
    - 'SA3' -> str (used to align relation_column; same dtype as the grid's SA3 code)
    - 'ITL3' = SA3 code (the groupby key used by shared_correction_utils)
    - 'Demand (MVA)' = demand_peak_mw (the lookup key used by the shared
      module, and the key GraphBuilder injects into source.y, matching the
      UK graph structure)
    """
    sa3 = gpd.read_file(REGIONS_SA3, layer='regions_sa3')
    assert len(sa3) == EXPECTED_N_SA3, f'SA3 count {len(sa3)} != {EXPECTED_N_SA3}'
    assert sa3['SA4'].nunique() == EXPECTED_N_SA4
    assert sa3[SOURCE_FEATURE_COLS + ['demand_peak_mw']].notna().all().all()
    assert (sa3['demand_peak_mw'] > 0).all(), 'Found non-positive region demand'
    sa3['SA3'] = sa3['SA3'].astype(str)
    sa3['ITL3'] = sa3['SA3']
    sa3['Demand (MVA)'] = sa3['demand_peak_mw']
    return sa3


def load_stations(regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Load the usable FY2009 stations (143) and spatial-join them to inherit SA3/loc_key membership."""
    st = pd.read_csv(STATION_TABLE, encoding='utf-8-sig')
    usable = st[st['status'].isin(USABLE_STATUSES)].copy()
    assert len(usable) == N_USABLE, f'Usable station count {len(usable)} != {N_USABLE}'
    pts = gpd.GeoDataFrame(
        usable,
        geometry=gpd.points_from_xy(usable['lon_wgs84'], usable['lat_wgs84']),
        crs='EPSG:4326',
    )
    joined = gpd.sjoin(pts, regions[['ITL3', 'SA4', 'loc_key', 'geometry']],
                       how='left', predicate='within')
    assert joined['ITL3'].notna().all(), 'Found usable stations that do not fall within any SA3 footprint'
    assert len(joined) == N_USABLE, 'sjoin produced duplicate rows (station on a boundary?)'
    return joined.drop(columns='index_right')


def load_data(locations=None):
    """Load regions/stations/grids + B2-precomputed NTL/Proximity + the RCI mask.

    locations: load only the specified regions (used by the --dry-run single-region
    smoke test); None = load all 12. Returns
    (grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict).
    The fusion columns are not written here (the assembled pickle only
    contains the 5 lu_* columns + 3 wc_* columns) - they are written inside
    build_graphs(), before feature-column selection (deviation 1).
    """
    if locations is None:
        locations = ALL_LOCATIONS

    print('=' * 60)
    print('Loading data + precomputed prior features...')
    print('=' * 60)

    regions = load_regions()
    stations = load_stations(regions)
    print(f'SA3 regions: {len(regions)} rows | usable stations: {len(stations)} rows')

    # Check that the region universe is consistent between the script's
    # frozen location order and the grid step-size table
    step_table = pd.read_csv(STEP_TABLE, encoding='utf-8-sig')
    assert set(step_table['loc_key']) == set(ALL_LOCATIONS), \
        'ALL_LOCATIONS is inconsistent with grid_step_size_table.csv - review any change to the region universe'

    schema = json.loads(FEATURE_SCHEMA.read_text(encoding='utf-8'))
    b2 = json.loads(B2_REGISTRY.read_text(encoding='utf-8')) \
        if B2_REGISTRY.exists() else None

    grids = {}
    ntl_dict = {}
    proximity_dict = {}
    rci_dict = {}
    region_dict = {}
    subs_dict = {}

    for loc in locations:
        # ── Assemble the grid (RangeIndex discipline: the shared utilities use group.index as a numpy positional index) ──
        with open(ASSEMBLED_DIR / f'{loc}_grid_points.pickle', 'rb') as f:
            grid_gdf, step_size_m = pickle.load(f)
        grid_gdf = grid_gdf.reset_index(drop=True)
        assert (grid_gdf.index == np.arange(len(grid_gdf))).all()
        # The fusion columns are not in the assembled pickle - only check the base 5 dims here (deviation 1)
        missing = [c for c in AGENT_FEATURE_COLS_BASE if c not in grid_gdf.columns]
        assert not missing, f'{loc}: grid is missing feature columns {missing}'
        grid_gdf['SA3'] = grid_gdf['SA3'].astype(str)
        grid_gdf['ITL3'] = grid_gdf['SA3']       # alias column (deviation 2)
        n_schema = schema['region_stats'][loc]['n_points']
        assert len(grid_gdf) == n_schema, \
            f'{loc}: grid-point count {len(grid_gdf)} != feature_schema value {n_schema}'
        grids[loc] = (grid_gdf, float(step_size_m))

        # ── Region subset (hard assertion that grid SA3 and region-table SA3 mutually cover each other, to prevent silent zeroing) ──
        region_sub = regions[regions['loc_key'] == loc].reset_index(drop=True)
        assert len(region_sub) >= 1, f'{loc}: region table has no such loc_key'
        assert set(grid_gdf['SA3'].unique()) == set(region_sub['SA3']), \
            f'{loc}: grid SA3 and region-table SA3 do not mutually cover each other'
        region_dict[loc] = region_sub

        # ── Station subset ──
        subs_sub = stations[stations['loc_key'] == loc].reset_index(drop=True)
        assert len(subs_sub) >= 1, f'{loc}: no usable stations'
        if b2 is not None:
            n_b2 = b2['regions'][loc]['proximity_stats']['n_stations']
            assert len(subs_sub) == n_b2, \
                f'{loc}: station count {len(subs_sub)} != recorded B2 value {n_b2} - review any upstream change'
        subs_dict[loc] = subs_sub

        # ── NTL (precomputed and stored to disk in an earlier step; missing values raise an error rather than being silently zeroed) ──
        ntl_path = EXTRACTED_DIR / f'{loc}_ntl.npz'
        if not ntl_path.exists():
            raise FileNotFoundError(
                f'{ntl_path} is missing - silently zeroing NTL would be the most dangerous failure mode in this pipeline, rerun the upstream extraction step first')
        ntl_values = np.load(ntl_path, allow_pickle=True)['data'][:, 0]
        assert ntl_values.shape == (len(grid_gdf),)
        assert np.isfinite(ntl_values).all() and (ntl_values >= 0).all()
        ntl_dict[loc] = ntl_values

        # ── proximity (precomputed npz, EPSG:7856) ──
        prox_path = EXTRACTED_DIR / f'{loc}_proximity.npz'
        if not prox_path.exists():
            raise FileNotFoundError(f'{prox_path} is missing - rerun the upstream extraction step first')
        prox_scores = np.load(prox_path, allow_pickle=True)['data'][:, 0]
        assert prox_scores.shape == (len(grid_gdf),)
        assert np.isfinite(prox_scores).all() and (prox_scores > 0).all()
        proximity_dict[loc] = prox_scores

        # ── RCI mask ──
        rci = (grid_gdf['lu_residential_prop'] + grid_gdf['lu_commercial_prop']
               + grid_gdf['lu_industrial_prop']).values
        rci_dict[loc] = rci > RCI_THRESHOLD

    print(f'Loaded data for {len(locations)} regions')
    return grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict


# ════════════════════════════════════════════════════════════
# Graph construction (7-dim fusion structure + NTL/Proximity/RCI tensor injection)
# ════════════════════════════════════════════════════════════

def _region_zscore_log1p(values: np.ndarray, label: str) -> np.ndarray:
    """Fusion input-column transform: log1p + per-region z-score (matches the UK case's approach exactly).

    - Clip to [0, +inf) then apply log1p - this matches, pointwise, the same
      torch.log(1 + clamp(x, min=0)) transform used internally by
      NTLPriorLoss and ProximityPriorLoss;
    - The z-score is computed **within a single location's full agent set**
      (numpy convention, ddof=0); because each region's statistics are
      independent, no information is shared between train/test folds, so
      there is no cross-fold leakage by construction;
    - Raises if std is zero or non-finite rather than silently continuing
      (silently zeroing NTL would be the most dangerous failure mode in
      this pipeline, so missing/degenerate inputs must fail loudly).
    """
    arr = np.log1p(np.clip(np.asarray(values, dtype=np.float64), 0.0, None))
    std = arr.std()  # numpy default, ddof=0
    if not np.isfinite(std) or std <= 0:
        raise ValueError(f'Fusion input column has zero or invalid within-region variance ({label}), refusing to continue silently')
    return ((arr - arr.mean()) / std).astype(np.float32)


def build_graphs(grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict,
                 locations=None, inject_priors=True):
    """Build the HeteroData graph for each region (relation_column='SA3').

    Structurally matches 012's build_graphs section by section, with exactly
    one scientific change (deviation 1): before feature-column selection,
    write the two fusion input columns ntl_feat / prox_feat
    (log1p + per-region z-score), taking the agent feature dimension from 5 to 7.
    """
    if locations is None:
        locations = list(grids.keys())

    print('\n' + '=' * 60)
    print('Building HeteroData graphs (7-dim fusion)...')
    print('=' * 60)

    graphs = {}

    for loc in locations:
        grid_gdf, step_size_m = grids[loc]
        region_sub = region_dict[loc]
        subs_sub = subs_dict[loc]

        # Derive the landuse category column (argmax over lu_*, needed by the landuse_prediction_loss supervision matrix)
        lu_prop_cols = [c for c in LU_PROP_TO_CATEGORY if c in grid_gdf.columns]
        if 'landuse' not in grid_gdf.columns and lu_prop_cols:
            categories = [LU_PROP_TO_CATEGORY[c] for c in lu_prop_cols]
            max_idx = grid_gdf[lu_prop_cols].values.argmax(axis=1)
            grid_gdf = grid_gdf.copy()
            grid_gdf['landuse'] = [categories[i] for i in max_idx]
            grids[loc] = (grid_gdf, step_size_m)

        # Coordinate projection + normalization (agent/target/source
        # coordinates are jointly fit with a single StandardScaler, EPSG:3857)
        gdf_a = grid_gdf.copy().to_crs('EPSG:3857')
        gdf_t = subs_sub.copy().to_crs('EPSG:3857')
        gdf_s = region_sub.copy()
        gdf_s['geometry'] = gdf_s.geometry.centroid.to_crs('EPSG:3857')

        coords_a = np.column_stack([gdf_a.geometry.x, gdf_a.geometry.y])
        coords_t = np.column_stack([gdf_t.geometry.x, gdf_t.geometry.y])
        coords_s = np.column_stack([gdf_s.geometry.x, gdf_s.geometry.y])

        scaler = StandardScaler().fit(np.vstack([coords_a, coords_t, coords_s]))

        coords_a_scaled = scaler.transform(coords_a)
        coords_s_scaled = scaler.transform(coords_s)

        gdf_a_scaled = gdf_a.copy()
        gdf_a_scaled['geometry'] = [Point(x, y) for x, y in coords_a_scaled]

        gdf_s_scaled = gdf_s.copy()
        gdf_s_scaled['geometry'] = [Point(x, y) for x, y in coords_s_scaled]

        # ── The one scientific change (deviation 1): write the two fusion input columns before feature-column selection ──
        # The raw values are already prepared in load_data() (ntl_dict / proximity_dict, aligned with the grid row positions)
        assert len(ntl_dict[loc]) == len(gdf_a_scaled), f'{loc}: ntl length does not match agent count'
        assert len(proximity_dict[loc]) == len(gdf_a_scaled), f'{loc}: prox length does not match agent count'
        gdf_a_scaled['ntl_feat'] = _region_zscore_log1p(ntl_dict[loc], f'{loc}/ntl_feat')
        gdf_a_scaled['prox_feat'] = _region_zscore_log1p(proximity_dict[loc], f'{loc}/prox_feat')

        agent_cols_for_graph = [c for c in AGENT_FEATURE_COLS if c in gdf_a_scaled.columns]
        source_cols_for_graph = [c for c in SOURCE_FEATURE_COLS if c in gdf_s_scaled.columns]
        assert len(agent_cols_for_graph) == 7 and len(source_cols_for_graph) == 5

        features_a = preprocess_features(gdf_a_scaled[agent_cols_for_graph + ['geometry']])
        features_s = preprocess_features(gdf_s_scaled[source_cols_for_graph + ['geometry']])

        hetero_data = prepare_hetero_graph_from_processed(
            gdf_s_scaled, gdf_a_scaled,
            processed_features_s=features_s,
            processed_features_a=features_a,
            relation_column='SA3',              # the only graph-structure change vs. the UK case (which uses 'ITL3')
            agent_connectivity=AGENT_CONNECTIVITY,
        )

        # Inject NTL / Proximity / RCI (the fusion config does not consume
        # these, but injecting them keeps the graph structure consistent -
        # the same "always inject" behavior as 012's baseline, with no
        # effect on training)
        if inject_priors:
            hetero_data['agent'].ntl_values = torch.tensor(
                ntl_dict[loc], dtype=torch.float32)
            hetero_data['agent'].proximity_scores = torch.tensor(
                proximity_dict[loc], dtype=torch.float32)
            hetero_data['agent'].rci_mask = torch.tensor(
                rci_dict[loc], dtype=torch.bool)

        # ── Hard assertions on the constructed graph (dimension now 7) ──
        n_agents = len(grid_gdf)
        assert hetero_data['agent'].x.shape[1] == len(AGENT_FEATURE_COLS), \
            f'{loc}: agent feature dim {hetero_data["agent"].x.shape[1]} != 7'
        n_edges = hetero_data['source', 'connects_to', 'agent'].edge_index.shape[1]
        assert n_edges == n_agents, \
            f'{loc}: SA3 star-topology edge count {n_edges} != agent count {n_agents}'
        assert hetero_data['source'].num_nodes == len(region_sub)
        assert ('agent', 'near', 'agent') not in hetero_data.edge_types, \
            f'{loc}: AGENT_CONNECTIVITY=None but a near edge was found'
        assert hasattr(hetero_data, 'landuse_mapping_matrix'), \
            f'{loc}: landuse supervision matrix is missing - landuse_prediction_loss would silently fail'

        graphs[loc] = hetero_data
        print(f'  {loc}: agent={hetero_data["agent"].num_nodes}, '
              f'source={hetero_data["source"].num_nodes}, '
              f'edges={n_edges}')

    return graphs


def load_or_cache_all(cache_dir: Path = GRAPH_CACHE_DIR):
    """Load data + build graphs, preferring to read from the on-disk cache when available.

    Note: intentionally does not import load_or_cache_all from 012 -
    default arguments are evaluated at import time, which would bind to the
    wrong path if this module were imported before 012's module-level state
    was set up, so the function is duplicated here instead.

    Returns (graphs, grids, ntl_dict, proximity_dict, region_dict, subs_dict).
    """
    cache_file = cache_dir / 'cached_graphs.pickle'

    if cache_file.exists():
        print(f'Loading graphs from cache: {cache_file}')
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        return (cached['graphs'], cached['grids'], cached['ntl_dict'],
                cached['proximity_dict'], cached['region_dict'], cached['subs_dict'])

    # No cache -> build everything from scratch
    grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict = load_data()
    graphs = build_graphs(
        grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict,
        inject_priors=True,  # always inject priors (attributes unused by fusion do not affect training)
    )

    # Save the cache (also stores proximity_dict, since the AU inference
    # stage's correction factors use the precomputed npz values)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'graphs': graphs,
            'grids': grids,
            'ntl_dict': ntl_dict,
            'proximity_dict': proximity_dict,
            'region_dict': region_dict,
            'subs_dict': subs_dict,
        }, f)
    print(f'Graph cache saved: {cache_file}')

    return graphs, grids, ntl_dict, proximity_dict, region_dict, subs_dict


def verify_graph_cache(graphs, grids, ntl_dict, proximity_dict) -> None:
    """Cache assertions (shared by --cache-only and the training entry point): 12 graphs, agent dim 7,
    star-topology edge count = agent count, no near edges, ntl/prox/rci all injected with matching lengths."""
    assert set(graphs) == set(ALL_LOCATIONS), \
        f'Cache is missing regions: {sorted(set(ALL_LOCATIONS) - set(graphs))}'
    expected_dim = len(AGENT_FEATURE_COLS)
    for loc in ALL_LOCATIONS:
        g = graphs[loc]
        n_agents = len(grids[loc][0])
        dim = g['agent'].x.shape[1]
        assert dim == expected_dim, (
            f'{loc}: agent feature dim {dim} != {expected_dim} - '
            f'this may be hitting 012\'s main cache (5-dim); check that GRAPH_CACHE_DIR points to '
            f'graph_cache_fusion and rebuild with --rebuild-cache')
        n_edges = g['source', 'connects_to', 'agent'].edge_index.shape[1]
        assert n_edges == n_agents, f'{loc}: edge count {n_edges} != agent count {n_agents}'
        assert ('agent', 'near', 'agent') not in g.edge_types, f'{loc}: a near edge was found'
        for attr in ('ntl_values', 'proximity_scores', 'rci_mask'):
            assert hasattr(g['agent'], attr), f'{loc}: agent.{attr} was not injected'
            assert len(getattr(g['agent'], attr)) == n_agents, \
                f'{loc}: agent.{attr} length != agent count'
        assert g['agent'].rci_mask.dtype == torch.bool
        assert len(ntl_dict[loc]) == n_agents
        assert len(proximity_dict[loc]) == n_agents
        print(f'  {loc}: agent dim {expected_dim} / edges {n_edges} = agent count / priors all injected [OK]')


# ════════════════════════════════════════════════════════════
# Prediction + correction arms + Voronoi aggregation (single region; structurally matches 012)
# ════════════════════════════════════════════════════════════

def predict_and_evaluate_location(loc, solver, graphs, grids, ntl_dict,
                                  proximity_dict, region_dict, subs_dict,
                                  assignment_cache,
                                  save_grid_demands_dir=None):
    """Run prediction for a single region -> apply the 7 correction arms -> Voronoi aggregation -> evaluation.

    Returns {method: metrics_dict}, where method = 'voronoi_' + the arm label.
    Evaluation choice: the correction factors come from scu.compute_factors
    (NP = N x P applied as a single combined factor), and aggregation uses
    the shared two-stage pipeline - the exact same numeric path used for
    the static baselines. save_grid_demands_dir: if not None, save the
    grid_demands pickle to that directory (keyed by arm label).
    """
    grid_gdf, step_size_m = grids[loc]
    region_sub = region_dict[loc]
    subs_sub = subs_dict[loc]
    graph = graphs[loc]
    ntl_values = ntl_dict[loc]
    prox_scores = proximity_dict[loc]

    # ── Predict edge weights -> gnn_demand (structurally matches the UK case, with the region key swapped for SA3) ──
    edge_weights_df = solver.predict_edge_weights(graph)

    region_info = region_sub.set_index('SA3')
    source_index_map = graph.source_index_map

    grid_gdf = grid_gdf.copy()
    grid_gdf['gnn_demand'] = 0.0

    for _, row in edge_weights_df.iterrows():
        s_idx = int(row['source_node_idx'])
        a_orig_idx = int(row['agent_original_idx'])
        w = row['predicted_weight']

        s_orig_idx = source_index_map.iloc[s_idx]
        sa3 = region_sub.loc[s_orig_idx, 'SA3']
        total_demand = region_info.loc[sa3, 'Demand (MVA)']
        grid_gdf.loc[a_orig_idx, 'gnn_demand'] += w * total_demand

    gnn_base = grid_gdf['gnn_demand'].values.astype(float)

    # GNN base per-SA3 conservation (a direct consequence of the softmax
    # weights summing to 1; tolerance = the error-model upper bound
    # max(GNN_CONS_RTOL, n*eps32), see the note by the constant definition)
    for sa3, group in grid_gdf.groupby('SA3'):
        total = float(region_info.loc[sa3, 'Demand (MVA)'])
        n_grp = int(len(group))
        dev = abs(float(gnn_base[group.index.to_numpy()].sum()) - total)
        tol_rel = max(GNN_CONS_RTOL, n_grp * _EPS32)
        assert dev <= tol_rel * total, \
            (f'{loc}/SA3 {sa3}: GNN base conservation violated (deviation {dev:.3e} MW / demand {total:.1f} / '
             f'n={n_grp} / tolerance {tol_rel:.3e})')

    # ── Factors and the 7 arms (NP = N x P applied as a single combined factor) ──
    ntl_factor, prox_factor = scu.compute_factors(grid_gdf, ntl_values, prox_scores)
    factors = {'N': ntl_factor, 'P': prox_factor, 'NP': ntl_factor * prox_factor}

    demands = {'GNN': gnn_base}
    for sig in SIGNALS:
        demands[f'GNNpost{sig}'] = scu.apply_standard_multiplicative(
            gnn_base, factors[sig], grid_gdf, region_sub)
        demands[f'GNNadd{sig}'] = scu.apply_additive_correction(
            gnn_base, factors[sig], grid_gdf, region_sub)

    for arm, arr in demands.items():
        assert np.isfinite(arr).all() and (arr >= 0).all(), \
            f'{loc}/{arm}: demand array contains non-finite or negative values'

    # ── Save grid_demands (used by downstream statistical evaluation; keyed by arm label) ──
    if save_grid_demands_dir is not None:
        save_grid_demands_dir.mkdir(parents=True, exist_ok=True)
        grid_demands = {arm: np.asarray(demands[arm], dtype=float).copy()
                        for arm in ARM_LABELS}
        with open(save_grid_demands_dir / f'{loc}_grid_demands.pickle', 'wb') as f:
            pickle.dump(grid_demands, f)

    # ── Two-stage Voronoi aggregation (EPSG:3857; assignment computed once per region) ──
    assignment = scu.compute_voronoi_assignment(
        grid_gdf, subs_sub, cache=assignment_cache, cache_key=loc)

    loc_metrics = {}
    for arm in ARM_LABELS:
        subs_result = scu.aggregate_by_assignment(subs_sub, assignment, demands[arm])
        loc_metrics[f'voronoi_{arm}'] = scu.evaluate_allocation(
            subs_result, actual_col=DEMAND_COL)

    return loc_metrics


# ════════════════════════════════════════════════════════════
# K-Fold training main loop (structurally matches 012; tau is kept in sync via the TAU_START constant)
# ════════════════════════════════════════════════════════════

def _make_model_config(epochs: int, model_path: Path) -> ModelConfig:
    """ModelConfig factory using hyperparameters copied from the UK case -
    shared by both the resumption/inference branch and the normal-training
    branch (a single physical function, so the two stay in sync by
    construction)."""
    warmup_epochs = 20
    decay_epochs = 20
    cosine_epochs = epochs - warmup_epochs - decay_epochs
    return ModelConfig(
        epochs=epochs,
        hidden_dim=256,
        embedding_dim=128,
        num_layers=3,
        conv_type='hgt',
        allocation_temperature_start=TAU_START,
        learning_rate=1e-3,
        weight_decay=1e-4,
        use_scheduler=True,
        warmup_epochs=warmup_epochs,
        decay_epochs=decay_epochs,
        cosine_epochs=cosine_epochs,
        cosine_eta_min=1e-5,
        learnable=False,
        save_path=str(model_path),
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )


def assert_training_complete(model_path: Path, expected_epochs: int) -> None:
    """Guard against resuming from a truncated training run.

    model.pth is the best-so-far checkpoint, written to disk at any point
    during the training loop, so a hard interruption (crash/kill) can leave
    a partial result behind; *_training_log.json is only written once the
    full epoch loop has finished. Consequently, the mere presence of
    model.pth is not sufficient evidence that training completed - the log
    must also exist and show the full epoch count, otherwise this refuses
    to proceed and tells the caller how to recover.
    """
    log_path = model_path.with_name(model_path.name[:-4] + '_training_log.json')
    if not log_path.exists():
        raise SystemExit(f'Truncated training run: {model_path} exists but {log_path.name} does not - '
                         'move this directory aside for inspection and rerun from scratch (do not simply resume).')
    with open(log_path, encoding='utf-8') as f:
        _log = json.load(f)
    n_logged = len(_log.get('train_losses', {}).get('total', []))
    if n_logged != expected_epochs:
        raise SystemExit(f'Incomplete training log: {log_path} ({n_logged}/{expected_epochs}) - '
                         'move this directory aside for inspection and rerun from scratch (do not simply resume).')


def run_kfold_training(config_name: str, seed: int,
                       graphs, grids, ntl_dict, proximity_dict,
                       region_dict, subs_dict, assignment_cache):
    """Run K-fold cross-validation training for a single (config, seed) pair, with checkpoint resumption."""

    exp_config = CONFIG_MAP[config_name]
    objective_weights = exp_config['objective_weights']
    epochs = exp_config['epochs']

    # Set the random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Output directory
    seed_dir = TRAIN_ROOT / f'seed_{seed}' / config_name
    seed_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"=" * 60}')
    print(f'Config: {config_name} | seed: {seed} | epochs: {epochs} | tau_start: {TAU_START}')
    print(f'Loss function: {objective_weights}')
    print(f'Output directory: {seed_dir}')
    print(f'{"=" * 60}\n')

    # K-fold split (random_state follows the seed, exactly matching the UK case)
    location_array = np.array(ALL_LOCATIONS)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    fold_rmse_tables = []
    fold_mae_tables = []
    fold_corr_tables = []

    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(location_array)):
        train_locs = location_array[train_indices].tolist()
        test_locs = location_array[test_indices].tolist()

        print(f'\n{"=" * 60}')
        print(f'Fold {fold_idx + 1}/{N_FOLDS}')
        print(f'  Training regions ({len(train_locs)}): {train_locs}')
        print(f'  Test regions ({len(test_locs)}): {test_locs}')
        print(f'{"=" * 60}')

        fold_dir = seed_dir / f'fold{fold_idx + 1}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        model_path = fold_dir / 'model.pth'

        # ── Check whether this fold is already fully complete (both the model and the metric CSVs exist) ──
        fold_csvs = [fold_dir / f'{name}.csv' for name in ('rmse', 'mae', 'corr')]
        if model_path.exists() and all(f.exists() for f in fold_csvs):
            assert_training_complete(model_path, epochs)
            # Check whether grid_demands is missing - if so, load the model and rerun inference
            grid_demands_dir = fold_dir / 'grid_demands'
            expected_gd = [grid_demands_dir / f'{loc}_grid_demands.pickle'
                           for loc in ALL_LOCATIONS]
            missing_gd = [p for p in expected_gd if not p.exists()]

            if missing_gd:
                print(f'Fold {fold_idx + 1} metrics already complete, but {len(missing_gd)} '
                      f'grid_demands are missing - rerunning inference...')

                # tau parameterization point 1/2 (resumption/inference branch) - shares TAU_START with the other branch via the factory
                config = _make_model_config(epochs, model_path)
                solver = EdgeWeightSolver(config)
                train_graphs_tmp = [graphs[loc] for loc in train_locs]
                train_dl_tmp = DataLoader(train_graphs_tmp, batch_size=1, shuffle=False)
                solver.init_model(train_dl_tmp, objective_weights)
                solver._load_checkpoint()

                for loc in ALL_LOCATIONS:
                    gd_file = grid_demands_dir / f'{loc}_grid_demands.pickle'
                    if gd_file.exists():
                        continue
                    print(f'  Generating missing grid_demands: {loc}')
                    predict_and_evaluate_location(
                        loc, solver, graphs, grids, ntl_dict, proximity_dict,
                        region_dict, subs_dict, assignment_cache,
                        save_grid_demands_dir=grid_demands_dir,
                    )
                print(f'Fold {fold_idx + 1} grid_demands regeneration complete')
            else:
                print(f'Fold {fold_idx + 1} already complete (including grid_demands), skipping')

            saved_rmse = pd.read_csv(fold_dir / 'rmse.csv', index_col=0)
            saved_mae = pd.read_csv(fold_dir / 'mae.csv', index_col=0)
            saved_corr = pd.read_csv(fold_dir / 'corr.csv', index_col=0)
            fold_rmse_tables.append(saved_rmse.T.to_dict())
            fold_mae_tables.append(saved_mae.T.to_dict())
            fold_corr_tables.append(saved_corr.T.to_dict())
            continue

        # ── Normal training / load model + evaluate ──
        # tau parameterization point 2/2 (normal-training branch) - shares TAU_START with the other branch via the factory
        config = _make_model_config(epochs, model_path)

        solver = EdgeWeightSolver(config)

        train_graphs = [graphs[loc] for loc in train_locs]
        test_graphs = [graphs[loc] for loc in test_locs]
        train_dl = DataLoader(train_graphs, batch_size=1, shuffle=False)
        test_dl = DataLoader(test_graphs, batch_size=1, shuffle=False)

        # Check whether a trained model already exists -> skip training (checkpoint resumption)
        if model_path.exists():
            assert_training_complete(model_path, epochs)
            print(f'Found existing model {model_path}, skipping training - initializing model structure and loading checkpoint')
            solver.init_model(train_dl, objective_weights)
            solver._load_checkpoint()
        else:
            print(f'Training config: {config.conv_type}, hidden={config.hidden_dim}, '
                  f'epochs={config.epochs}')
            print(f'Device: {config.device}')
            solver.train_multi_graph(train_dl, test_dataloader=test_dl,
                                     objective_weights=objective_weights)
            print(f'Fold {fold_idx + 1} training complete')

        # Predict & evaluate for all regions
        fold_rmse = {}
        fold_mae = {}
        fold_corr = {}

        grid_demands_dir = fold_dir / 'grid_demands'

        for loc in ALL_LOCATIONS:
            role = 'TRAIN' if loc in train_locs else 'TEST'
            print(f'\n  Evaluating {loc} ({role})...')

            loc_metrics = predict_and_evaluate_location(
                loc, solver, graphs, grids, ntl_dict, proximity_dict,
                region_dict, subs_dict, assignment_cache,
                save_grid_demands_dir=grid_demands_dir,
            )

            for method_name, m in loc_metrics.items():
                if method_name not in fold_rmse:
                    fold_rmse[method_name] = {}
                    fold_mae[method_name] = {}
                    fold_corr[method_name] = {}
                fold_rmse[method_name][loc] = round(m['rmse'], 4)
                fold_mae[method_name][loc] = round(m['mae'], 4)
                fold_corr[method_name][loc] = round(m['corr'], 4)

                print(f'    {method_name}: corr={m["corr"]:.4f}, '
                      f'RMSE={m["rmse"]:.4f}, MAE={m["mae"]:.4f}')

        fold_rmse_tables.append(fold_rmse)
        fold_mae_tables.append(fold_mae)
        fold_corr_tables.append(fold_corr)

        # Save this fold's tables
        pd.DataFrame(fold_rmse).T.to_csv(fold_dir / 'rmse.csv')
        pd.DataFrame(fold_mae).T.to_csv(fold_dir / 'mae.csv')
        pd.DataFrame(fold_corr).T.to_csv(fold_dir / 'corr.csv')

    # ════════════════════════════════════════════════════════════
    # Aggregate K-fold results (each region takes the value from the fold where it was the test set)
    # ════════════════════════════════════════════════════════════

    print(f'\n{"=" * 60}')
    print('Aggregating K-fold results...')
    print(f'{"=" * 60}')

    all_methods = sorted(fold_rmse_tables[0].keys())
    kf_splits = list(KFold(n_splits=N_FOLDS, shuffle=True,
                           random_state=seed).split(location_array))

    test_rmse = {m: {} for m in all_methods}
    test_mae = {m: {} for m in all_methods}
    test_corr = {m: {} for m in all_methods}

    for fold_idx, (_, test_indices) in enumerate(kf_splits):
        test_locs = location_array[test_indices].tolist()
        for loc in test_locs:
            for method in all_methods:
                test_rmse[method][loc] = fold_rmse_tables[fold_idx][method][loc]
                test_mae[method][loc] = fold_mae_tables[fold_idx][method][loc]
                test_corr[method][loc] = fold_corr_tables[fold_idx][method][loc]

    test_rmse_df = pd.DataFrame(test_rmse).T
    test_mae_df = pd.DataFrame(test_mae).T
    test_corr_df = pd.DataFrame(test_corr).T

    test_rmse_df['mean'] = test_rmse_df.mean(axis=1).round(4)
    test_mae_df['mean'] = test_mae_df.mean(axis=1).round(4)
    test_corr_df['mean'] = test_corr_df.mean(axis=1).round(4)

    test_rmse_df.to_csv(seed_dir / 'kfold_test_rmse.csv')
    test_mae_df.to_csv(seed_dir / 'kfold_test_mae.csv')
    test_corr_df.to_csv(seed_dir / 'kfold_test_corr.csv')

    print(f'\n=== K-fold test-set RMSE ===')
    print(test_rmse_df.to_string())

    print(f'\n=== K-fold test-set MAE ===')
    print(test_mae_df.to_string())

    print(f'\n=== K-fold test-set Correlation ===')
    print(test_corr_df.to_string())

    # Save the split info
    split_info = {}
    for fold_idx, (train_indices, test_indices) in enumerate(kf_splits):
        split_info[f'fold_{fold_idx + 1}'] = {
            'train': location_array[train_indices].tolist(),
            'test': location_array[test_indices].tolist(),
        }
    with open(seed_dir / 'kfold_splits.json', 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    print(f'\nAll results saved to: {seed_dir}')


# ════════════════════════════════════════════════════════════
# --dry-run: input check + single-region graph-construction smoke test (no training, no cache writes)
# ════════════════════════════════════════════════════════════

def run_dry_run() -> None:
    """Check that inputs are present + run a single-region 7-dim graph-construction
    smoke test, then exit (no training, no graph-cache writes - a single-region
    graph is not a complete cache, and writing it would corrupt graph_cache_fusion)."""
    print('\n' + '=' * 60)
    print('--dry-run: input check + single-region 7-dim graph-construction smoke test (no training)')
    print('=' * 60)

    problems = []
    checks = [
        ('regions_sa3.gpkg', REGIONS_SA3),
        ('station_table_fy2009.csv', STATION_TABLE),
        ('grid_step_size_table.csv', STEP_TABLE),
        ('feature_schema.json', FEATURE_SCHEMA),
        ('b2_features.json (optional)', B2_REGISTRY),
    ]
    for loc in ALL_LOCATIONS:
        checks.append((f'{loc} assembled', ASSEMBLED_DIR / f'{loc}_grid_points.pickle'))
        checks.append((f'{loc} ntl', EXTRACTED_DIR / f'{loc}_ntl.npz'))
        checks.append((f'{loc} proximity', EXTRACTED_DIR / f'{loc}_proximity.npz'))

    n_ok = 0
    for name, path in checks:
        ok = path.exists()
        n_ok += int(ok)
        if not ok:
            problems.append(f'Missing: {name} -> {path}')
    print(f'  Input files: {n_ok}/{len(checks)} present')
    print(f'  shared_correction_utils: imported successfully ({scu.__file__})')
    print(f'  Config matrix: {list(CONFIG_MAP)} | default seeds {DEFAULT_SEEDS} | '
          f'{N_FOLDS}-fold x {len(ALL_LOCATIONS)} regions')
    print(f'  Agent feature columns (7-dim): {AGENT_FEATURE_COLS}')
    print(f'  tau kept in sync: resumption/normal branches share _make_model_config(TAU_START={TAU_START})')
    print(f'  Graph cache directory: {GRAPH_CACHE_DIR} (present: '
          f'{(GRAPH_CACHE_DIR / "cached_graphs.pickle").exists()})')

    if problems:
        print('\n[dry-run found problems]')
        for p in problems:
            print(f'  - {p}')
        sys.exit(1)

    # ── Single-region graph-construction smoke test (first region; hard assertions run inside build_graphs) ──
    smoke_loc = ALL_LOCATIONS[0]
    print(f'\n[dry-run] Single-region 7-dim graph-construction smoke test: {smoke_loc}')
    grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict = \
        load_data(locations=[smoke_loc])
    graphs = build_graphs(grids, ntl_dict, proximity_dict, rci_dict,
                          region_dict, subs_dict, locations=[smoke_loc])
    g = graphs[smoke_loc]
    n_agents = len(grids[smoke_loc][0])
    print(f'\n[dry-run smoke-test result] {smoke_loc}: '
          f'agent feature dim = {g["agent"].x.shape[1]} (expected 7), '
          f'source = {g["source"].num_nodes}, '
          f'star-topology edges = {g["source", "connects_to", "agent"].edge_index.shape[1]}'
          f' (= agent count {n_agents}), '
          f'ntl/prox/rci all injected, no near edges, landuse supervision matrix present')
    print('\n[dry-run passed] inputs present + single-region 7-dim graph smoke test passed; '
          'run full training with --all when ready.')
    sys.exit(0)


# ════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════

ALL_CONFIGS = list(CONFIG_MAP.keys())
DEFAULT_SEEDS = [42, 123, 456]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AU feature-fusion arm K-fold training (fusion config, following the UK case\'s feature-fusion approach)')
    parser.add_argument('--config', type=str,
                        choices=ALL_CONFIGS,
                        help='Training config (mutually exclusive with --all; the fusion arm only has "fusion")')
    parser.add_argument('--seed', type=int,
                        help='Random seed (mutually exclusive with --all)')
    parser.add_argument('--all', action='store_true',
                        help='Run all configs x all seeds')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help=f'Used together with --all to specify the seed list (default {DEFAULT_SEEDS})')
    parser.add_argument('--configs', type=str, nargs='+',
                        choices=ALL_CONFIGS,
                        default=ALL_CONFIGS,
                        help='Used together with --all to specify the config list (the fusion arm only has "fusion")')
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='Force a rebuild of the graph cache (use after data changes)')
    parser.add_argument('--cache-only', action='store_true',
                        help='Only build/verify the graph cache then exit, without entering the training loop '
                             '(the cache is meant to be built on CPU ahead of time, with training run separately later)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run: check inputs + run a single-region 7-dim graph-construction smoke test, then exit '
                             '(does not load/write the cache, does not train)')
    args = parser.parse_args()

    # ── --dry-run: handled first (does not touch the cache or training paths) ──
    if args.dry_run:
        run_dry_run()

    # Argument validation (--cache-only does not need config/seed)
    if args.cache_only:
        run_configs = []
        run_seeds = []
    elif args.all:
        run_configs = args.configs
        run_seeds = args.seeds
    elif args.config and args.seed is not None:
        run_configs = [args.config]
        run_seeds = [args.seed]
    else:
        parser.error('Use --all to run everything, or specify both --config and --seed, '
                     'or use --cache-only to only build the graph cache, or --dry-run for a smoke test')

    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.cuda.is_available()}')

    # Data & graphs are loaded/built only once (or read from cache)
    cache_dir = GRAPH_CACHE_DIR
    if args.rebuild_cache:
        cache_file = cache_dir / 'cached_graphs.pickle'
        if cache_file.exists():
            cache_file.unlink()
            print('Deleted the old cache, will rebuild')
    graphs, grids, ntl_dict, proximity_dict, region_dict, subs_dict = \
        load_or_cache_all(cache_dir)

    # ── --cache-only: verify the cache then exit directly (does not enter the training loop) ──
    if args.cache_only:
        print('\n' + '=' * 60)
        print('--cache-only: verifying the graph cache...')
        print('=' * 60)
        verify_graph_cache(graphs, grids, ntl_dict, proximity_dict)
        print(f'\nGraph cache ready: {cache_dir / "cached_graphs.pickle"}')
        print('Cache build/verification complete, did not enter the training loop (--cache-only)')
        sys.exit(0)

    # The training entry point also runs the cache assertion first (prevents a stale/partial cache from feeding into training)
    verify_graph_cache(graphs, grids, ntl_dict, proximity_dict)

    # Run manifest (a separate file so it does not overwrite 012's run_manifest.json)
    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        'experiment': 'AU_fusion_training',
        'spec': 'Feature-fusion approach following the UK case\'s feature-fusion script, '
                'combined with the AU-adapted base from 012_train_au_gnn.py',
        'configs': run_configs,
        'seeds': run_seeds,
        'n_folds': N_FOLDS,
        'locations': ALL_LOCATIONS,
        'region_universe': {'sa4': EXPECTED_N_SA4, 'sa3': EXPECTED_N_SA3},
        'relation_column': 'SA3',
        'agent_feature_cols': AGENT_FEATURE_COLS,
        'fusion_transform': 'log1p + per-region z-score (ddof=0; matches the UK '
                            'case exactly, no cross-fold statistic sharing)',
        'source_feature_cols': SOURCE_FEATURE_COLS,
        'tau_start': TAU_START,
        'hyperparams': 'hgt/hidden256/emb128/3 layers/lr1e-3/wd1e-4/'
                       'warmup20+decay20+cosine/learnable=False (copied from the UK case)',
        'demand_col': DEMAND_COL,
        'graph_cache': str(GRAPH_CACHE_DIR / 'cached_graphs.pickle'),
        'evaluation_choice': {
            'aggregation': 'shared_correction_utils two-stage pipeline '
                           '(compute_voronoi_assignment + aggregate_by_assignment'
                           ' + evaluate_allocation; EPSG:3857)',
            'why': 'uses exactly the same numeric path as the static baselines and the main GNN training script, so cross-arm statistics are built on the same aggregation implementation',
            'np_factor': 'NP = ntl_factor x prox_factor applied as a single combined factor',
            'arms': ARM_LABELS,
            'no_wc_arm': 'wc_* is diagnostic only and does not feed into results',
            'no_civd': 'evaluation uses Voronoi aggregation only',
        },
        'started_at': datetime.now().isoformat(timespec='seconds'),
    }
    with open(TRAIN_ROOT / 'run_manifest_fusion.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # In-process cache for Voronoi assignment (computed once per region, reused across folds/configs)
    assignment_cache: dict = {}

    total = len(run_configs) * len(run_seeds)
    current = 0
    for cfg in run_configs:
        for s in run_seeds:
            current += 1
            print(f'\n{"#" * 60}')
            print(f'# Task {current}/{total}: config={cfg}, seed={s}')
            print(f'{"#" * 60}')
            run_kfold_training(cfg, s, graphs, grids, ntl_dict, proximity_dict,
                               region_dict, subs_dict, assignment_cache)

    print('\n' + '=' * 60)
    print(f'All done! Ran {total} task(s)')
    print('=' * 60)
