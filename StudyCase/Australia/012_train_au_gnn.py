# -*- coding: utf-8 -*-
"""AU case study - GNN training script (012).

This script follows the same overall pattern as the UK case's
`027_train_feature_fusion.py` (walk-up project-root discovery + a
--cache-only mode). However, 027 is the 7-dimensional feature-fusion
variant; this script instead reproduces the UK case's original
**5-dimensional lu_* agent features** (AGENT_FEATURE_COLS has no
ntl_feat/prox_feat). NTL/proximity/RCI are injected as **separate tensor
attributes** (hetero_data['agent'].ntl_values / .proximity_scores /
.rci_mask, matching the UK training script's structure) for use by the
ntl_prior / proximity_prior loss terms, matching the UK case's original
design.

═══ Adaptations and deviations relative to the UK training scripts (read before modifying) ═══
1. Region universe = SA4 x 12 / SA3 x 34 (corrected from an earlier
   nominal count of 13/35 after auditing the actual station data; see
   b1_regions.json.deviation_from_b0). The 4-fold split over 12 regions
   gives test folds of size 3/3/3/3, using
   KFold(n_splits=4, shuffle=True, random_state=seed), matching the UK
   training script.
2. relation_column='SA3' (the only graph-structure change relative to the
   UK case, which uses 'ITL3'). The grid and region tables are also given
   an 'ITL3' alias column (= the SA3 code as a string) and a
   'Demand (MVA)' alias column (= demand_peak_mw), so that
   shared_correction_utils's groupby/lookup logic works unmodified -- this
   is a pure renaming adaptation with no change in numerical semantics
   (same idea as deviation item 3 in 011).
3. Proximity scores are read from the precomputed `{loc}_proximity.npz`
   (EPSG:7856), rather than calling the UK case's
   ProximityCorrector.compute_scores (whose TARGET_CRS=EPSG:27700 is fixed
   for the UK case). NTL is likewise read from the precomputed
   `{loc}_ntl.npz` (DMSP F16 2008+2009 median).
4. [Evaluation choice] Post-training inference evaluation uses the
   **shared_correction_utils two-stage pipeline**
   (compute_voronoi_assignment + aggregate_by_assignment +
   evaluate_allocation, with a fixed EPSG:3857 working CRS), rather than
   the UK training script's inline evaluate function. Rationale: this
   keeps the evaluation on the exact same numerical path as the 011
   static-baseline arms, so cross-arm statistics later are built on a
   single consistent aggregation implementation. Correction arms follow
   the shared-module convention: factors = scu.compute_factors (N/P
   factors), with NP applied as a single combined factor N x P;
   multiplicative = apply_standard_multiplicative; additive =
   apply_additive_correction. Note this differs from the UK training
   script's "ntl then proximity" stacked correction
   (ntl_prox_gnn_demand) -- the arm matrix and statistical tests here are
   not defined on that stacked convention.
5. No WC-correction arm is produced (wc_* columns are diagnostic only and
   are not carried into the results, so the wc_*_gnn_demand column from
   the UK script is not produced here); no CIVD aggregation (Voronoi only;
   the SCIP dependency is out of scope); no landuse_demand precomputation
   (that GPM column from the UK script does not feed into the GNN
   inference chain here -- the static baselines are already covered by
   the companion 011 script).
6. grid_demands pickle keys use the **correction-arm label convention**
   (GNN / GNNpostN / GNNpostP / GNNpostNP / GNNaddN / GNNaddP /
   GNNaddNP), rather than the UK script's *_gnn_demand column names -- this
   matches the arm naming used in 011's au_static_metrics.csv so the two
   can be compared directly. Metric-table method names are
   'voronoi_' + arm label (matching the UK kfold_test_*.csv row-naming
   style).
7. Demand column = peak_mw; evaluation uses actual_col='peak_mw'.
8. Note on a documentation mismatch: earlier task notes described the
   assembled pickle as containing ntl + proximity columns, which does not
   match the actual precomputed data: in practice the 5 lu_* columns plus
   3 wc_* columns live in the pickle, while ntl / proximity live in the
   extracted npz files -- this script reads from the actual data layout.
9. Hyperparameters are copied from the UK case: conv=hgt / hidden 256 /
   embedding 128 / 3 layers / tau=0.01 (the TAU_START constant drives the
   ModelConfig in both the "resume-and-infer" branch and the
   "normal-training" branch, via a single shared factory function, so the
   two cannot drift out of sync) / lr 1e-3 / wd 1e-4 / warmup 20 + decay
   20 + remaining epochs on a cosine schedule / learnable=False /
   AGENT_CONNECTIVITY=None. Configured arms: baseline for 200 epochs;
   ntl_prox for 400 epochs with lambda=0.05 (structure copied from the UK
   training script).

Note: training is compute-intensive. Use --dry-run or --cache-only for
lightweight input validation and graph-cache construction (CPU-only,
fast) before launching a full training run.

Usage:
    python 012_train_au_gnn.py --dry-run                  # input validation + single-region graph smoke test, then exit (no training)
    python 012_train_au_gnn.py --cache-only               # build/verify the 12-region graph cache, then exit (CPU only)
    python 012_train_au_gnn.py --config baseline --seed 42
    python 012_train_au_gnn.py --all --seeds 42 123 456   # full run: 4 configs x 3 seeds x 4 folds
    python 012_train_au_gnn.py --all --configs ntl --seeds 42 123 456   # rerun a single arm

Config map:
    baseline:   {'landuse_prediction_loss': 1.0}                       200 epochs
    ntl:        {'landuse_prediction_loss': 1.0, 'ntl_prior': 0.05}    200 epochs
    proximity:  {'landuse_prediction_loss': 1.0,
                 'proximity_prior': 0.05}                              200 epochs
    ntl_prox:   {'landuse_prediction_loss': 1.0,
                 'ntl_prior': 0.05, 'proximity_prior': 0.05}           400 epochs

(The 'ntl' and 'proximity' single-prior arms were added later, matching
 the equivalent entries in the UK training script's CONFIG_MAP
 ('ntl' / 'proximity', 200 epochs, lambda=0.05), to fill out the AU
 phase-diagram placement analysis with a third arm each; they share the
 existing graph cache with no changes needed.)
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
# (a fixed parent-count would break if this script is ever moved)
SCRIPT_DIR = Path(__file__).resolve().parent
_p = SCRIPT_DIR
while not (_p / 'SpatialAllocation').exists():
    if _p.parent == _p:
        raise RuntimeError('Could not find the repository root (SpatialAllocation package)')
    _p = _p.parent
PROJECT_ROOT = _p
# Add the UK experiment directory to sys.path to import shared_correction_utils
# (a read-only import shared across case studies; this module should not be modified here)
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

# The Windows console defaults to cp1252, which cannot encode non-ASCII output -- force UTF-8
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

# Graph cache directory (separate from the UK case's own graph caches)
GRAPH_CACHE_DIR = PROCESSED / 'graph_cache'
# Training artifact root directory (data/ is gitignored; layout = seed_{seed}/{config}/fold*/..., matching the UK training script)
TRAIN_ROOT = PROCESSED / 'training'

N_FOLDS = 4

# ─── Region configuration (12 SA4 regions; order matches feature_schema.json
#     locations, fixed alphabetically -- the KFold split depends on this
#     order, so it must not be changed) ───
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

EXPECTED_N_SA3 = 34                 # corrected after auditing the actual data (nominal count was 35)
EXPECTED_N_SA4 = 12                 # corrected after auditing the actual data (nominal count was 13)
N_USABLE = 143                      # gate: usable FY2009 stations
USABLE_STATUSES = {'matched', 'matched_osm'}
DEMAND_COL = 'peak_mw'              # primary demand column (annual peak MW)

# ─── Feature control (5 dims, matching the UK case's original feature set and order; no fusion columns) ───
AGENT_FEATURE_COLS = [
    'lu_residential_prop', 'lu_commercial_prop', 'lu_industrial_prop',
    'lu_agricultural_prop', 'lu_others_prop',
]

SOURCE_FEATURE_COLS = [
    'residential_percent', 'commercial_percent', 'industrial_percent',
    'agricultural_percent', 'others_percent',
]

# ─── Constants (copied verbatim from the UK case; the proximity formula's
#     own constants are already baked into the precomputed data, so only RCI remains here) ───
RCI_THRESHOLD = 0.5

# Tau initialization (copied from the UK case). TAU_START is the single
# constant that drives ModelConfig in both the resume-and-infer branch and
# the normal-training branch, to prevent the two from drifting out of sync.
TAU_START = 0.01

# ─── Land-use mapping (matches the UK training script) ───
LU_PROP_TO_CATEGORY = {
    'lu_residential_prop': 'residential',
    'lu_commercial_prop': 'commercial',
    'lu_industrial_prop': 'industrial',
    'lu_agricultural_prop': 'agricultural',
    'lu_others_prop': 'others',
}

# ─── Config map (baseline + ntl_prox arms; structure copied from the UK
#     training script, lambda=0.05 matching the UK case. The ntl / proximity
#     single-prior arms were added later, matching the equivalently named
#     entries in the UK training script's CONFIG_MAP (200 epochs), to fill
#     out the AU phase-diagram placement analysis -- see the module
#     docstring) ───
CONFIG_MAP = {
    'baseline': {
        'objective_weights': {'landuse_prediction_loss': 1.0},
        'epochs': 200,
    },
    'ntl': {
        'objective_weights': {'landuse_prediction_loss': 1.0, 'ntl_prior': 0.05},
        'epochs': 200,
    },
    'proximity': {
        'objective_weights': {'landuse_prediction_loss': 1.0, 'proximity_prior': 0.05},
        'epochs': 200,
    },
    'ntl_prox': {
        'objective_weights': {
            'landuse_prediction_loss': 1.0,
            'ntl_prior': 0.05,
            'proximity_prior': 0.05,
        },
        'epochs': 400,
    },
}

AGENT_CONNECTIVITY = None           # SA3 star topology, no agent-agent grid edges

# ─── Correction-arm labels (grid_demands keys and metric-table row names are derived from these) ───
ARM_LABELS = ['GNN', 'GNNpostN', 'GNNpostP', 'GNNpostNP',
              'GNNaddN', 'GNNaddP', 'GNNaddNP']
SIGNALS = ['N', 'P', 'NP']

# GNN-baseline per-SA3 conservation tolerance (relative lower bound). The
# softmax weights are float32, so this is looser than the 1e-6 MW absolute
# tolerance used for the static arms in 011, which is an unavoidable
# consequence of float32 precision rather than a relaxation of rigor.
# The assertion actually uses the error-model upper bound
# max(GNN_CONS_RTOL, n * eps32): the hard upper bound on |sum(w) - 1| from
# sequential float32 summation grows linearly with group size n. Empirically
# in the UK case the observed deviation was 0.7-4.5% of this bound (largest
# group n ~ 38,000 giving |sum(w) - 1| ~ 1.1e-4); the AU SA3 groups are
# larger still (first-fold observation 1.7e-4 at SA3 11802), so a fixed
# 1e-4 tolerance would incorrectly flag pure floating-point noise as a
# conservation violation. A genuine conservation break is an O(1)-scale
# deviation, three orders of magnitude above this bound, so the relative
# error model does not lose any real detection power.
GNN_CONS_RTOL = 1e-4
_EPS32 = float(np.finfo(np.float32).eps)


# ════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════

def load_regions() -> gpd.GeoDataFrame:
    """Load the 34 source SA3 regions (with demand + 5 land-use percentages), and inject alias columns for the shared correction module.

    Aliases (pure renaming, no change in numerical semantics):
    - 'SA3' -> str (used for relation_column alignment; matches the grid's SA3 code dtype)
    - 'ITL3' = SA3 code (the groupby key used by shared_correction_utils)
    - 'Demand (MVA)' = demand_peak_mw (the lookup key used by the shared
      module, and the key GraphBuilder injects into source.y, aligning with
      the UK graph structure)
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
    """Load usable FY2009 stations (143) -> spatial-join to inherit SA3/loc_key membership."""
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
    assert len(joined) == N_USABLE, 'sjoin produced duplicate rows (a station sitting exactly on a boundary?)'
    return joined.drop(columns='index_right')


def load_data(locations=None):
    """Load regions/stations/grids plus the precomputed NTL/Proximity data and the RCI mask.

    locations: load only the given regions (used for the --dry-run single-region smoke test); None = all 12.
    Returns (grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict).
    """
    if locations is None:
        locations = ALL_LOCATIONS

    print('=' * 60)
    print('Loading data + precomputed prior features...')
    print('=' * 60)

    regions = load_regions()
    stations = load_stations(regions)
    print(f'SA3 regions: {len(regions)} rows | usable stations: {len(stations)} rows')

    # Region-universe consistency check: the fixed script order vs. the grid step-size table
    step_table = pd.read_csv(STEP_TABLE, encoding='utf-8-sig')
    assert set(step_table['loc_key']) == set(ALL_LOCATIONS), \
        'ALL_LOCATIONS is inconsistent with grid_step_size_table.csv -- any change to the region universe needs review'

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
        # ── Assemble the grid (RangeIndex discipline: scu uses group.index as a numpy positional index) ──
        with open(ASSEMBLED_DIR / f'{loc}_grid_points.pickle', 'rb') as f:
            grid_gdf, step_size_m = pickle.load(f)
        grid_gdf = grid_gdf.reset_index(drop=True)
        assert (grid_gdf.index == np.arange(len(grid_gdf))).all()
        missing = [c for c in AGENT_FEATURE_COLS if c not in grid_gdf.columns]
        assert not missing, f'{loc}: grid is missing feature columns {missing}'
        grid_gdf['SA3'] = grid_gdf['SA3'].astype(str)
        grid_gdf['ITL3'] = grid_gdf['SA3']       # alias column, see module docstring deviation notes
        n_schema = schema['region_stats'][loc]['n_points']
        assert len(grid_gdf) == n_schema, \
            f'{loc}: grid-point count {len(grid_gdf)} != feature_schema record {n_schema}'
        grids[loc] = (grid_gdf, float(step_size_m))

        # ── Region subset (hard assertion that the grid's SA3 codes and the region table's SA3 codes cover each other exactly, to prevent silent zero-fill) ──
        region_sub = regions[regions['loc_key'] == loc].reset_index(drop=True)
        assert len(region_sub) >= 1, f'{loc}: no rows found for this loc_key in the region table'
        assert set(grid_gdf['SA3'].unique()) == set(region_sub['SA3']), \
            f'{loc}: grid SA3 codes and region-table SA3 codes do not cover each other'
        region_dict[loc] = region_sub

        # ── Station subset ──
        subs_sub = stations[stations['loc_key'] == loc].reset_index(drop=True)
        assert len(subs_sub) >= 1, f'{loc}: no usable stations'
        if b2 is not None:
            n_b2 = b2['regions'][loc]['proximity_stats']['n_stations']
            assert len(subs_sub) == n_b2, \
                f'{loc}: station count {len(subs_sub)} != recorded count {n_b2} -- any upstream change needs review'
        subs_dict[loc] = subs_sub

        # ── NTL (precomputed; DMSP F16 2008+2009 median; raise immediately if missing) ──
        ntl_path = EXTRACTED_DIR / f'{loc}_ntl.npz'
        if not ntl_path.exists():
            raise FileNotFoundError(
                f'{ntl_path} is missing -- silently zero-filling NTL would be the most dangerous failure '
                f'mode in this pipeline, so this raises instead; regenerate the precomputed NTL data first')
        ntl_values = np.load(ntl_path, allow_pickle=True)['data'][:, 0]
        assert ntl_values.shape == (len(grid_gdf),)
        assert np.isfinite(ntl_values).all() and (ntl_values >= 0).all()
        ntl_dict[loc] = ntl_values

        # ── Proximity (read from the precomputed npz, EPSG:7856; does not call
        #    the UK case's ProximityCorrector, whose TARGET_CRS=EPSG:27700 is fixed for the UK case) ──
        prox_path = EXTRACTED_DIR / f'{loc}_proximity.npz'
        if not prox_path.exists():
            raise FileNotFoundError(f'{prox_path} is missing -- regenerate the precomputed proximity data first')
        prox_scores = np.load(prox_path, allow_pickle=True)['data'][:, 0]
        assert prox_scores.shape == (len(grid_gdf),)
        assert np.isfinite(prox_scores).all() and (prox_scores > 0).all()
        proximity_dict[loc] = prox_scores

        # ── RCI mask (same formula as the UK training script) ──
        rci = (grid_gdf['lu_residential_prop'] + grid_gdf['lu_commercial_prop']
               + grid_gdf['lu_industrial_prop']).values
        rci_dict[loc] = rci > RCI_THRESHOLD

    print(f'Loaded data for {len(locations)} regions')
    return grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict


# ════════════════════════════════════════════════════════════
# Graph construction (the UK case's original 5-dimensional structure + NTL/Proximity/RCI tensor injection)
# ════════════════════════════════════════════════════════════

def build_graphs(grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict,
                 locations=None, inject_priors=True):
    """Build the HeteroData graph for each region (relation_column='SA3').

    Matches the UK training script's build_graphs section by section, with
    three AU-specific adaptations:
    - relation_column='ITL3' -> 'SA3' (same underlying values -- ITL3 is just an alias column);
    - no fusion columns are written (agent features are always 5-dimensional);
    - hard assertions are run on each constructed graph (feature dimension /
      star-topology edge count / absence of 'near' edges / presence of the landuse supervision matrix).
    """
    if locations is None:
        locations = list(grids.keys())

    print('\n' + '=' * 60)
    print('Building HeteroData graphs...')
    print('=' * 60)

    graphs = {}

    for loc in locations:
        grid_gdf, step_size_m = grids[loc]
        region_sub = region_dict[loc]
        subs_sub = subs_dict[loc]

        # Derive the landuse category column (argmax over lu_*, needed for the landuse_prediction_loss supervision matrix)
        lu_prop_cols = [c for c in LU_PROP_TO_CATEGORY if c in grid_gdf.columns]
        if 'landuse' not in grid_gdf.columns and lu_prop_cols:
            categories = [LU_PROP_TO_CATEGORY[c] for c in lu_prop_cols]
            max_idx = grid_gdf[lu_prop_cols].values.argmax(axis=1)
            grid_gdf = grid_gdf.copy()
            grid_gdf['landuse'] = [categories[i] for i in max_idx]
            grids[loc] = (grid_gdf, step_size_m)

        # Coordinate projection + normalization (matches the UK training
        # script: the agent/target/source coordinate sets jointly fit a
        # single StandardScaler, EPSG:3857)
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

        agent_cols_for_graph = [c for c in AGENT_FEATURE_COLS if c in gdf_a_scaled.columns]
        source_cols_for_graph = [c for c in SOURCE_FEATURE_COLS if c in gdf_s_scaled.columns]
        assert len(agent_cols_for_graph) == 5 and len(source_cols_for_graph) == 5

        features_a = preprocess_features(gdf_a_scaled[agent_cols_for_graph + ['geometry']])
        features_s = preprocess_features(gdf_s_scaled[source_cols_for_graph + ['geometry']])

        hetero_data = prepare_hetero_graph_from_processed(
            gdf_s_scaled, gdf_a_scaled,
            processed_features_s=features_s,
            processed_features_a=features_a,
            relation_column='SA3',              # the only graph-structure change relative to the UK case (which uses 'ITL3')
            agent_connectivity=AGENT_CONNECTIVITY,
        )

        # Inject NTL / Proximity / RCI (needed by the prior losses, matching
        # the UK training script's structure. The baseline config does not
        # consume these attributes, so injecting them does not affect its
        # training -- they are always injected regardless of config, as in the UK script)
        if inject_priors:
            hetero_data['agent'].ntl_values = torch.tensor(
                ntl_dict[loc], dtype=torch.float32)
            hetero_data['agent'].proximity_scores = torch.tensor(
                proximity_dict[loc], dtype=torch.float32)
            hetero_data['agent'].rci_mask = torch.tensor(
                rci_dict[loc], dtype=torch.bool)

        # ── Hard assertions on the constructed graph (smoke-test checks) ──
        n_agents = len(grid_gdf)
        assert hetero_data['agent'].x.shape[1] == len(AGENT_FEATURE_COLS), \
            f'{loc}: agent feature dim {hetero_data["agent"].x.shape[1]} != 5'
        n_edges = hetero_data['source', 'connects_to', 'agent'].edge_index.shape[1]
        assert n_edges == n_agents, \
            f'{loc}: SA3 star-topology edge count {n_edges} != agent count {n_agents}'
        assert hetero_data['source'].num_nodes == len(region_sub)
        assert ('agent', 'near', 'agent') not in hetero_data.edge_types, \
            f'{loc}: AGENT_CONNECTIVITY=None but a near edge type was found'
        assert hasattr(hetero_data, 'landuse_mapping_matrix'), \
            f'{loc}: landuse supervision matrix is missing -- landuse_prediction_loss would silently fail'

        graphs[loc] = hetero_data
        print(f'  {loc}: agent={hetero_data["agent"].num_nodes}, '
              f'source={hetero_data["source"].num_nodes}, '
              f'edges={n_edges}')

    return graphs


def load_or_cache_all(cache_dir: Path = GRAPH_CACHE_DIR):
    """Load data + build graphs, preferring to read from an on-disk cache when
    available. (Deliberately does not import load_or_cache_all from the UK
    script, since a default argument evaluated at import time would bind to
    a stale path.)

    Returns (graphs, grids, ntl_dict, proximity_dict, region_dict, subs_dict).
    """
    cache_file = cache_dir / 'cached_graphs.pickle'

    if cache_file.exists():
        print(f'Loading graphs from cache: {cache_file}')
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        return (cached['graphs'], cached['grids'], cached['ntl_dict'],
                cached['proximity_dict'], cached['region_dict'], cached['subs_dict'])

    # No cache -> build from scratch
    grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict = load_data()
    graphs = build_graphs(
        grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict,
        inject_priors=True,  # always inject priors (attributes unused by the baseline config do not affect its training)
    )

    # Save the cache (also stores proximity_dict, since AU inference-time
    # correction factors are read from the precomputed npz values rather
    # than being recomputed at predict time)
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
    """Cache assertions (shared by --cache-only and the training entry
    point): 12 graphs, agent features 5-dimensional, star-topology edge
    count equals agent count, no 'near' edges, and ntl/prox/rci are all
    injected with matching lengths."""
    assert set(graphs) == set(ALL_LOCATIONS), \
        f'Cache is missing regions: {sorted(set(ALL_LOCATIONS) - set(graphs))}'
    expected_dim = len(AGENT_FEATURE_COLS)
    for loc in ALL_LOCATIONS:
        g = graphs[loc]
        n_agents = len(grids[loc][0])
        dim = g['agent'].x.shape[1]
        assert dim == expected_dim, (
            f'{loc}: agent feature dim {dim} != {expected_dim} -- '
            f'this may be hitting a feature-fusion cache (7 dims) instead; rebuild with --rebuild-cache')
        n_edges = g['source', 'connects_to', 'agent'].edge_index.shape[1]
        assert n_edges == n_agents, f'{loc}: edge count {n_edges} != agent count {n_agents}'
        assert ('agent', 'near', 'agent') not in g.edge_types, f'{loc}: found a near edge type'
        for attr in ('ntl_values', 'proximity_scores', 'rci_mask'):
            assert hasattr(g['agent'], attr), f'{loc}: agent.{attr} was not injected'
            assert len(getattr(g['agent'], attr)) == n_agents, \
                f'{loc}: agent.{attr} length does not match agent count'
        assert g['agent'].rci_mask.dtype == torch.bool
        assert len(ntl_dict[loc]) == n_agents
        assert len(proximity_dict[loc]) == n_agents
        print(f'  {loc}: agent 5-dim / edges {n_edges} = agent count / priors injected [OK]')


# ════════════════════════════════════════════════════════════
# Prediction + correction arms + Voronoi aggregation (single region)
# ════════════════════════════════════════════════════════════

def predict_and_evaluate_location(loc, solver, graphs, grids, ntl_dict,
                                  proximity_dict, region_dict, subs_dict,
                                  assignment_cache,
                                  save_grid_demands_dir=None):
    """Run prediction for a single region -> apply the 7 correction arms -> Voronoi aggregation -> evaluation.

    Returns {method: metrics_dict}, where method = 'voronoi_' + the arm
    label. Evaluation choice (see module docstring item 4): factors =
    scu.compute_factors (NP applied as a single combined factor N x P),
    aggregation = the shared two-stage pipeline, keeping this on the exact
    same numerical path as the 011 static baselines.
    save_grid_demands_dir: if not None, save the grid_demands pickle to
    this directory (keys = the arm labels, see module docstring item 6).
    """
    grid_gdf, step_size_m = grids[loc]
    region_sub = region_dict[loc]
    subs_sub = subs_dict[loc]
    graph = graphs[loc]
    ntl_values = ntl_dict[loc]
    prox_scores = proximity_dict[loc]

    # ── Predict edge weights -> gnn_demand (matches the UK training script's approach, with the region key swapped to SA3) ──
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

    # GNN-baseline per-SA3 conservation check (a direct consequence of the
    # softmax weights summing to 1; tolerance = the error-model upper bound
    # max(GNN_CONS_RTOL, n * eps32), see the comment where the constant is defined)
    for sa3, group in grid_gdf.groupby('SA3'):
        total = float(region_info.loc[sa3, 'Demand (MVA)'])
        n_grp = int(len(group))
        dev = abs(float(gnn_base[group.index.to_numpy()].sum()) - total)
        tol_rel = max(GNN_CONS_RTOL, n_grp * _EPS32)
        assert dev <= tol_rel * total, \
            (f'{loc}/SA3 {sa3}: GNN-baseline conservation violated (deviation {dev:.3e} MW / demand {total:.1f} / '
             f'n={n_grp} / tolerance {tol_rel:.3e})')

    # ── Factors and the 7 correction arms (NP applied as a single combined factor N x P) ──
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

    # ── Save grid_demands (used by downstream evaluation steps; keys = the arm labels) ──
    if save_grid_demands_dir is not None:
        save_grid_demands_dir.mkdir(parents=True, exist_ok=True)
        grid_demands = {arm: np.asarray(demands[arm], dtype=float).copy()
                        for arm in ARM_LABELS}
        with open(save_grid_demands_dir / f'{loc}_grid_demands.pickle', 'wb') as f:
            pickle.dump(grid_demands, f)

    # ── Two-stage Voronoi aggregation (fixed EPSG:3857 working CRS; assignment computed once per region) ──
    assignment = scu.compute_voronoi_assignment(
        grid_gdf, subs_sub, cache=assignment_cache, cache_key=loc)

    loc_metrics = {}
    for arm in ARM_LABELS:
        subs_result = scu.aggregate_by_assignment(subs_sub, assignment, demands[arm])
        loc_metrics[f'voronoi_{arm}'] = scu.evaluate_allocation(
            subs_result, actual_col=DEMAND_COL)

    return loc_metrics


# ════════════════════════════════════════════════════════════
# K-fold training main loop (matches the UK training script; tau is kept in sync via the TAU_START constant)
# ════════════════════════════════════════════════════════════

def _make_model_config(epochs: int, model_path: Path) -> ModelConfig:
    """Factory for a ModelConfig using the hyperparameters copied from the
    UK case, shared by both the resume-and-infer branch and the
    normal-training branch (implemented as a single physical function so
    the two branches cannot drift out of sync)."""
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
    """Hardened check for whether training actually completed (fixing an
    earlier bug where a resumed run could be treated as complete when it wasn't).

    model.pth is the best-so-far snapshot, written to disk at any point
    during the training loop, so a hard interruption (crash/kill) can leave
    a partially-trained checkpoint behind; *_training_log.json is only
    written once the full epoch loop finishes. So the mere presence of
    model.pth is not sufficient evidence that training completed -- the log
    file must also exist and record the full expected number of epochs,
    otherwise this raises and tells the caller how to recover.
    """
    log_path = model_path.with_name(model_path.name[:-4] + '_training_log.json')
    if not log_path.exists():
        raise SystemExit(f'Truncated training run: {model_path} exists but {log_path.name} does not -- '
                         'move this directory aside for inspection and rerun (do not simply resume).')
    with open(log_path, encoding='utf-8') as f:
        _log = json.load(f)
    n_logged = len(_log.get('train_losses', {}).get('total', []))
    if n_logged != expected_epochs:
        raise SystemExit(f'Incomplete training log: {log_path} ({n_logged}/{expected_epochs} epochs) -- '
                         'move this directory aside for inspection and rerun (do not simply resume).')


def run_kfold_training(config_name: str, seed: int,
                       graphs, grids, ntl_dict, proximity_dict,
                       region_dict, subs_dict, assignment_cache):
    """Run K-fold cross-validation training for a single config x seed combination (resumable)."""

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

    # K-fold split (random_state follows seed, exactly matching the UK training pipeline)
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
            # Check whether grid_demands is missing -- if so, reload the model and rerun inference
            grid_demands_dir = fold_dir / 'grid_demands'
            expected_gd = [grid_demands_dir / f'{loc}_grid_demands.pickle'
                           for loc in ALL_LOCATIONS]
            missing_gd = [p for p in expected_gd if not p.exists()]

            if missing_gd:
                print(f'Fold {fold_idx + 1} metrics are complete, but {len(missing_gd)} '
                      f'grid_demands files are missing -- rerunning inference...')

                # tau parameterization point 1/2 (resume-and-infer branch) -- shares the TAU_START factory
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
        # tau parameterization point 2/2 (normal-training branch) -- shares the TAU_START factory
        config = _make_model_config(epochs, model_path)

        solver = EdgeWeightSolver(config)

        train_graphs = [graphs[loc] for loc in train_locs]
        test_graphs = [graphs[loc] for loc in test_locs]
        train_dl = DataLoader(train_graphs, batch_size=1, shuffle=False)
        test_dl = DataLoader(test_graphs, batch_size=1, shuffle=False)

        # Check whether a trained model already exists -> skip training (resumable)
        if model_path.exists():
            assert_training_complete(model_path, epochs)
            print(f'Found existing model {model_path}, skipping training; initializing model structure and loading checkpoint')
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
    # Aggregate K-fold results (each region's metric is taken from the fold in which it served as the test set)
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
# --dry-run: input validation + single-region graph-construction smoke test (no training, no cache writes)
# ════════════════════════════════════════════════════════════

def run_dry_run() -> None:
    """Validate that inputs are present, run a single-region graph-construction
    smoke test, then exit (no training; does not write to the graph cache,
    since a single-region graph is not a complete cache and writing it would
    corrupt graph_cache)."""
    print('\n' + '=' * 60)
    print('--dry-run: input validation + single-region graph-construction smoke test (no training)')
    print('=' * 60)

    problems = []
    checks = [
        ('regions_sa3.gpkg', REGIONS_SA3),
        ('station_table_fy2009.csv', STATION_TABLE),
        ('grid_step_size_table.csv', STEP_TABLE),
        ('feature_schema.json', FEATURE_SCHEMA),
        ('b2_features.json (soft dependency)', B2_REGISTRY),
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
    print(f'  Tau kept in sync: the resume and normal-training branches share _make_model_config(TAU_START={TAU_START})')
    print(f'  Graph cache directory: {GRAPH_CACHE_DIR} (present: '
          f'{(GRAPH_CACHE_DIR / "cached_graphs.pickle").exists()})')

    if problems:
        print('\n[dry-run found problems]')
        for p in problems:
            print(f'  - {p}')
        sys.exit(1)

    # ── Single-region graph-construction smoke test (first region; hard assertions run inside build_graphs) ──
    smoke_loc = ALL_LOCATIONS[0]
    print(f'\n[dry-run] Single-region graph-construction smoke test: {smoke_loc}')
    grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict = \
        load_data(locations=[smoke_loc])
    graphs = build_graphs(grids, ntl_dict, proximity_dict, rci_dict,
                          region_dict, subs_dict, locations=[smoke_loc])
    g = graphs[smoke_loc]
    n_agents = len(grids[smoke_loc][0])
    print(f'\n[dry-run smoke test result] {smoke_loc}: '
          f'agent feature dim = {g["agent"].x.shape[1]} (expected 5), '
          f'source = {g["source"].num_nodes}, '
          f'star-topology edges = {g["source", "connects_to", "agent"].edge_index.shape[1]}'
          f' (= agent count {n_agents}), '
          f'ntl/prox/rci all injected, no near edges, landuse supervision matrix present')
    print('\n[dry-run passed] Inputs are present and the single-region graph smoke test passed; '
          'launch full training with --all when ready.')
    sys.exit(0)


# ════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════

ALL_CONFIGS = list(CONFIG_MAP.keys())
DEFAULT_SEEDS = [42, 123, 456]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AU case study - GNN K-fold training (baseline + ntl_prox)')
    parser.add_argument('--config', type=str,
                        choices=ALL_CONFIGS,
                        help='training config (mutually exclusive with --all)')
    parser.add_argument('--seed', type=int,
                        help='random seed (mutually exclusive with --all)')
    parser.add_argument('--all', action='store_true',
                        help='run all configs x all seeds')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help=f'used together with --all to specify the seed list (default {DEFAULT_SEEDS})')
    parser.add_argument('--configs', type=str, nargs='+',
                        choices=ALL_CONFIGS,
                        default=ALL_CONFIGS,
                        help='used together with --all to specify the config list (default: all)')
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='force a graph-cache rebuild (use after the underlying data changes)')
    parser.add_argument('--cache-only', action='store_true',
                        help='build/verify the graph cache then exit, without entering the training loop')
    parser.add_argument('--dry-run', action='store_true',
                        help='dry run: validate inputs + single-region graph-construction smoke test, then exit '
                             '(does not load/write the cache, does not train)')
    args = parser.parse_args()

    # ── --dry-run: handled first (does not touch the cache or training paths) ──
    if args.dry_run:
        run_dry_run()

    # Argument validation (--cache-only does not need a config/seed)
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
                     'or use --cache-only to just build the graph cache, or --dry-run for a smoke test')

    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.cuda.is_available()}')

    # Data & graphs are loaded/built only once (or read from cache)
    cache_dir = GRAPH_CACHE_DIR
    if args.rebuild_cache:
        cache_file = cache_dir / 'cached_graphs.pickle'
        if cache_file.exists():
            cache_file.unlink()
            print('Deleted the old cache; it will be rebuilt')
    graphs, grids, ntl_dict, proximity_dict, region_dict, subs_dict = \
        load_or_cache_all(cache_dir)

    # ── --cache-only: verify the cache then exit directly (does not enter the training loop) ──
    if args.cache_only:
        print('\n' + '=' * 60)
        print('--cache-only: verifying the graph cache...')
        print('=' * 60)
        verify_graph_cache(graphs, grids, ntl_dict, proximity_dict)
        print(f'\nGraph cache ready: {cache_dir / "cached_graphs.pickle"}')
        print('Cache build/verification complete; did not enter the training loop (--cache-only)')
        sys.exit(0)

    # The training entry point also runs the cache assertions first (to prevent a stale or partial cache from being used for training)
    verify_graph_cache(graphs, grids, ntl_dict, proximity_dict)

    # Run manifest (records the evaluation choices and conventions used, for downstream traceability)
    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        'experiment': 'AU_B4_gnn_training',
        'spec': 'AU GNN training (baseline + ntl_prox configs)',
        'configs': run_configs,
        'seeds': run_seeds,
        'n_folds': N_FOLDS,
        'locations': ALL_LOCATIONS,
        'region_universe': {'sa4': EXPECTED_N_SA4, 'sa3': EXPECTED_N_SA3,
                            'note': 'corrected after auditing the actual data (nominal counts were 13/35)'},
        'relation_column': 'SA3',
        'agent_feature_cols': AGENT_FEATURE_COLS,
        'source_feature_cols': SOURCE_FEATURE_COLS,
        'tau_start': TAU_START,
        'hyperparams': 'hgt/hidden256/emb128/3 layers/lr1e-3/wd1e-4/'
                       'warmup20+decay20+cosine/learnable=False (copied from the UK case)',
        'demand_col': DEMAND_COL,
        'graph_cache': str(GRAPH_CACHE_DIR / 'cached_graphs.pickle'),
        'evaluation_choice': {
            'aggregation': 'shared_correction_utils two-stage pipeline '
                           '(compute_voronoi_assignment + aggregate_by_assignment'
                           ' + evaluate_allocation; fixed EPSG:3857 working CRS)',
            'why': 'keeps this on the exact same numerical path as the 011 static arms, so cross-arm statistics are built on a single consistent aggregation implementation',
            'np_factor': 'NP = ntl_factor x prox_factor applied as a single combined factor '
                         '(differs from the UK training script\'s sequential ntl-then-proximity stacking)',
            'arms': ARM_LABELS,
            'no_wc_arm': 'wc_* columns are diagnostic only and are not carried into the results',
            'no_civd': 'Voronoi aggregation only',
        },
        'started_at': datetime.now().isoformat(timespec='seconds'),
    }
    with open(TRAIN_ROOT / 'run_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # In-process cache of Voronoi assignments (computed once per region, reused across folds/configs)
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
    print(f'All done! Ran {total} tasks in total')
    print('=' * 60)
