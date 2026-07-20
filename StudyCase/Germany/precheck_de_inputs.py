# -*- coding: utf-8 -*-
"""Precheck: assert that all inputs for the German Börde case training script (005) are ready.

Read-only check: results are dumped to a JSON report; scientific values are
not hard-asserted, only reported.
Checks performed (reference values come from the archived 001/002 notebook
outputs and the 005 docstring):
    1. source_regions.gpkg     -- 34 Gemeinden, including 5 *_percent feature columns
    2. substations.gpkg        -- 13 substations, p_mw all > 0, total load 207.5 MW
    3. boerde_grid_points.pickle -- [grid_gdf, step_size_m], 51619 points,
                                   all feature columns required for training present (lu_x5 + wc_x3)
    4. boerde_ntl.npz          -- length aligned with the grid and non-zero variance
                                   (guards against silent zero-fill)
    5. all_node_features_col.pickle / feature_schema.json
    6. network_distance (required for the 003 static baseline, not required
       for training -- tracked as an independent status flag)
    7. Derived-quantity computability -- Gemeinde demand derivation / proximity / RCI smoke test

Usage:
    python precheck_de_inputs.py
Output:
    results/precheck_de_inputs.json (the all_green field = all training inputs ready)
"""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import geopandas as gpd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
FEATURES_DIR = DATA_DIR / 'features'
ASSEMBLED_DIR = FEATURES_DIR / 'assembled'
EXTRACTED_DIR = FEATURES_DIR / 'extracted'
ND_DIR = FEATURES_DIR / 'network_distance' / 'boerde'
OUT_PATH = SCRIPT_DIR / 'results' / 'precheck_de_inputs.json'

# Reference values (from the initial run archive)
EXPECT_N_GEMEINDEN = 34
EXPECT_N_SUBS = 13
EXPECT_TOTAL_LOAD_MW = 207.5
EXPECT_N_GRID = 51619

# Columns required for 005 training
SOURCE_PCT_COLS = ['residential_percent', 'commercial_percent',
                   'industrial_percent', 'agricultural_percent', 'others_percent']
AGENT_LU_COLS = ['lu_residential_prop', 'lu_commercial_prop', 'lu_industrial_prop',
                 'lu_agricultural_prop', 'lu_others_prop']
AGENT_WC_COLS = ['wc_built_up_ratio', 'wc_agricultural_ratio', 'wc_others_ratio']


def check(report: dict, name: str, ok: bool, detail):
    report['checks'][name] = {'ok': bool(ok), 'detail': detail}
    status = 'OK  ' if ok else 'FAIL'
    print(f'  [{status}] {name}: {detail}')
    return ok


def main():
    report = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'purpose': 'Readiness check for the German (Börde) data and feature pipeline before training',
        'checks': {},
    }
    green = True

    # ── 1. source_regions ──
    p = DATA_DIR / 'source_regions.gpkg'
    if p.exists():
        gdf = gpd.read_file(str(p))
        missing = [c for c in SOURCE_PCT_COLS + ['Name', 'NUTS3'] if c not in gdf.columns]
        green &= check(report, 'source_regions_rows', len(gdf) == EXPECT_N_GEMEINDEN,
                       f'{len(gdf)} rows (expected {EXPECT_N_GEMEINDEN})')
        green &= check(report, 'source_regions_cols', not missing, f'Missing columns: {missing}')
    else:
        green &= check(report, 'source_regions_exists', False, str(p))
        gdf = None

    # ── 2. substations ──
    p = DATA_DIR / 'substations.gpkg'
    if p.exists():
        subs = gpd.read_file(str(p))
        total = float(subs['p_mw'].sum()) if 'p_mw' in subs.columns else float('nan')
        green &= check(report, 'substations_rows', len(subs) == EXPECT_N_SUBS,
                       f'{len(subs)} rows (expected {EXPECT_N_SUBS})')
        green &= check(report, 'substations_pmw_positive',
                       'p_mw' in subs.columns and bool((subs['p_mw'] > 0).all()),
                       f'p_mw>0 holds for all rows, total load {total:.1f} MW')
        green &= check(report, 'substations_total_load',
                       abs(total - EXPECT_TOTAL_LOAD_MW) < 1e-6,
                       f'{total:.1f} MW (expected {EXPECT_TOTAL_LOAD_MW})')
    else:
        green &= check(report, 'substations_exists', False, str(p))
        subs = None

    # ── 3. assembled grid_points ──
    p = ASSEMBLED_DIR / 'boerde_grid_points.pickle'
    grid_gdf = None
    if p.exists():
        with open(p, 'rb') as f:
            loaded = pickle.load(f)
        fmt_ok = isinstance(loaded, list) and len(loaded) == 2
        grid_gdf, step = (loaded if fmt_ok else (None, None))
        green &= check(report, 'grid_points_format', fmt_ok, '[grid_gdf, step_size_m]')
        if fmt_ok:
            missing = [c for c in AGENT_LU_COLS + AGENT_WC_COLS + ['Name']
                       if c not in grid_gdf.columns]
            nan_ct = int(grid_gdf[[c for c in AGENT_LU_COLS + AGENT_WC_COLS
                                   if c in grid_gdf.columns]].isna().sum().sum())
            green &= check(report, 'grid_points_rows', len(grid_gdf) == EXPECT_N_GRID,
                           f'{len(grid_gdf)} points, step={step}m (expected {EXPECT_N_GRID})')
            green &= check(report, 'grid_points_train_cols', not missing, f'Missing columns: {missing}')
            green &= check(report, 'grid_points_no_nan', nan_ct == 0,
                           f'NaN count in training feature columns = {nan_ct}')
    else:
        green &= check(report, 'grid_points_exists', False, str(p))

    # ── 4. NTL (exists and has non-zero variance) ──
    p = EXTRACTED_DIR / 'boerde_ntl.npz'
    if p.exists():
        npz = np.load(p, allow_pickle=True)
        ntl = npz['data'][:, 0]
        var = float(np.var(ntl))
        n_ok = grid_gdf is None or len(ntl) == len(grid_gdf)
        green &= check(report, 'ntl_length', n_ok, f'{len(ntl)} values')
        green &= check(report, 'ntl_variance_nonzero', var > 0,
                       f'var={var:.4f}, max={float(ntl.max()):.2f}')
    else:
        green &= check(report, 'ntl_exists', False, str(p))

    # ── 5. Feature schema ──
    for fname in ['all_node_features_col.pickle', 'feature_schema.json']:
        p = ASSEMBLED_DIR / fname
        green &= check(report, f'assembled_{fname}', p.exists(), str(p))

    # ── 6. network_distance (required for the 003 static baseline; not required for training, so not counted in all_green) ──
    nd_matrix = ND_DIR / 'network_distance_matrix.npy'
    nd_ok = nd_matrix.exists()
    nd_detail = str(nd_matrix)
    if nd_ok:
        shape = np.load(str(nd_matrix)).shape
        nd_detail = f'{nd_matrix.name} shape={shape}'
    check(report, 'network_distance_for_003', nd_ok, nd_detail)
    report['checks']['network_distance_for_003']['required_for_training'] = False

    # ── 7. Derived-quantity smoke test (demand derivation / proximity / RCI) ──
    if gdf is not None and subs is not None and grid_gdf is not None:
        try:
            gem_demand = subs.groupby('Gemeinde')['p_mw'].sum()
            derived = gdf['Name'].map(gem_demand).fillna(0.0)
            cons_ok = abs(float(derived.sum()) - EXPECT_TOTAL_LOAD_MW) < 1e-6
            green &= check(report, 'gemeinde_demand_derivable', cons_ok,
                           f'Derived demand total {derived.sum():.1f} MW (conserved)')

            rci = (grid_gdf['lu_residential_prop'] + grid_gdf['lu_commercial_prop']
                   + grid_gdf['lu_industrial_prop']).values
            green &= check(report, 'rci_computable', bool(np.isfinite(rci).all()),
                           f'RCI>0.5 fraction {(rci > 0.5).mean():.3f}')

            sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
            from SpatialAllocation.FeatureExtractor.correctors.proximity_corrector import ProximityCorrector
            prox = ProximityCorrector.compute_scores(
                grid_gdf.head(100), subs, gamma=1.0,
                target_crs='EPSG:25832', clamp_km=0.01)
            green &= check(report, 'proximity_computable',
                           bool(np.isfinite(prox).all()) and len(prox) == 100,
                           '100-point smoke test passed')
        except Exception as e:
            green &= check(report, 'derived_quantities', False, f'{type(e).__name__}: {e}')

    report['all_green'] = bool(green)
    report['note'] = ('all_green only covers the inputs required for 005 training; '
                      'network_distance_for_003 is an independent status flag for the 003 static baseline')

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'All green' if green else 'FAIL present'} → {OUT_PATH}")
    return 0 if green else 1


if __name__ == '__main__':
    sys.exit(main())
