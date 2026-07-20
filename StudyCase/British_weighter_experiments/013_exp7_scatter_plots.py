"""
013 Predicted vs Actual scatter plots (Exp 7) - Voronoi allocation base

Pools substations across 16 regions, 4 methods + 45-degree line + LOWESS.
Methods: GNN (GNN baseline), GNNpostP (GNN best), GNNpostNP (antagonism), GPMpostNP (static best)

Usage:
    python 013_exp7_scatter_plots.py
"""

import sys
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SpatialAllocation.Allocator import allocator_registry
from SpatialAllocation.Weighter import weighter_registry
from SpatialAllocation.FeatureExtractor.correctors import corrector_registry
from SpatialAllocation.FeatureExtractor.correctors.proximity_corrector import ProximityCorrector

DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
ASSEMBLED_DIR = DATA_DIR / 'features' / 'assembled'
EXTRACTED_DIR = DATA_DIR / 'features' / 'extracted'
STATIC_DIR = SCRIPT_DIR / 'results' / 'static_allocation'
EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'

LU_COLS = [
    'lu_residential_prop', 'lu_commercial_prop', 'lu_industrial_prop',
    'lu_agricultural_prop', 'lu_others_prop',
]
PCT_COLS = [
    'residential_percent', 'commercial_percent', 'industrial_percent',
    'agricultural_percent', 'others_percent',
]
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp7_scatter_plots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_LOCATIONS = [
    'London',
    'TLH2', 'TLH3', 'TLJ1', 'TLF1', 'TLF2',
    'TLC1', 'TLC2', 'TLD6', 'TLG1', 'TLG2', 'TLE4',
    'TLH1', 'TLE3', 'TLD3', 'TLD4',
]

SEED = 42


def collect_substation_data():
    """Collect actual vs predicted values for all substations across regions (Voronoi allocation)."""
    substations_gdf = gpd.read_file(str(DATA_DIR / 'substations.gpkg'))

    # Method -> (config_name, demand_key)
    gnn_methods = {
        'GNN: voronoi_GNN': ('baseline', 'gnn_demand'),
        'GNNpostP: voronoi_prox_GNN': ('baseline', 'prox_gnn_demand'),
        'GNNpostNP: voronoi_ntl_prox_GNN': ('baseline', 'ntl_prox_gnn_demand'),
    }

    all_data = {name: {'actual': [], 'predicted': [], 'region': []}
                for name in gnn_methods}

    for config_name in ['baseline']:
        splits_path = EXP0_DIR / f'seed_{SEED}' / config_name / 'kfold_splits.json'
        if not splits_path.exists():
            continue
        with open(splits_path, 'r') as f:
            splits = json.load(f)

        for fold_name, split_info in splits.items():
            fold_num = fold_name.split('_')[1]
            test_locs = split_info['test']

            for loc in test_locs:
                gd_path = (EXP0_DIR / f'seed_{SEED}' / config_name
                           / f'fold{fold_num}' / 'grid_demands' / f'{loc}_grid_demands.pickle')
                if not gd_path.exists():
                    continue

                with open(gd_path, 'rb') as f:
                    grid_demands = pickle.load(f)

                with open(ASSEMBLED_DIR / f'{loc}_grid_points.pickle', 'rb') as f:
                    grid_gdf, _ = pickle.load(f)

                study_itl3 = grid_gdf['ITL3'].unique()
                subs_sub = substations_gdf[
                    substations_gdf['ITL3'].isin(study_itl3)
                ].copy().reset_index(drop=True)

                # Voronoi allocation
                alloc_voronoi = allocator_registry.create('voronoi')
                voronoi_res = alloc_voronoi.allocate(grid_gdf, subs_sub)
                actual = subs_sub['Demand (MVA)'].values

                for method_name, (cfg, demand_key) in gnn_methods.items():
                    if cfg != config_name:
                        continue
                    if demand_key not in grid_demands:
                        continue

                    demand_arr = grid_demands[demand_key]
                    predicted = np.zeros(len(subs_sub))
                    for t_idx in range(len(subs_sub)):
                        mask = voronoi_res.assignment == t_idx
                        predicted[t_idx] = demand_arr[mask].sum()

                    all_data[method_name]['actual'].extend(actual.tolist())
                    all_data[method_name]['predicted'].extend(predicted.tolist())
                    all_data[method_name]['region'].extend([loc] * len(actual))

    # Static method GPMpostNP (Voronoi allocation base) - computed inline
    region_gdf = gpd.read_file(str(DATA_DIR / 'ITL3_region.gpkg'))
    ntl_corrector = corrector_registry.create('ntl')
    prox_corrector = corrector_registry.create('proximity')

    static_data = {'actual': [], 'predicted': [], 'region': []}
    for loc in ALL_LOCATIONS:
        grid_path = ASSEMBLED_DIR / f'{loc}_grid_points.pickle'
        ntl_path = EXTRACTED_DIR / f'{loc}_ntl.npz'
        if not grid_path.exists() or not ntl_path.exists():
            print(f'  [skipped] {loc}: data not found')
            continue

        with open(grid_path, 'rb') as f:
            grid_gdf, _ = pickle.load(f)

        ntl_values = np.load(ntl_path, allow_pickle=True)['data'][:, 0]

        study_itl3 = grid_gdf['ITL3'].unique()
        region_sub = region_gdf[region_gdf['ITL3'].isin(study_itl3)].copy()
        subs_sub = substations_gdf[
            substations_gdf['ITL3'].isin(study_itl3)
        ].copy().reset_index(drop=True)

        # GPM weights
        gpm = weighter_registry.create('gpm', config={
            'mode': 'categorical', 'proportion_columns': LU_COLS,
        })
        gpm_res = gpm.compute(grid_gdf, target_gdf=subs_sub)
        W = gpm_res.weights
        region_info = region_sub.set_index('ITL3')
        grid_gdf['landuse_demand'] = 0.0
        for itl3, group in grid_gdf.groupby('ITL3'):
            if itl3 not in region_info.index:
                continue
            total_demand = region_info.loc[itl3, 'Demand (MVA)']
            idx = group.index
            pcts = np.array([region_info.loc[itl3, c] for c in PCT_COLS])
            score = W[idx] @ pcts
            score_sum = score.sum()
            if score_sum > 0:
                grid_gdf.loc[idx, 'landuse_demand'] = total_demand * score / score_sum
            else:
                grid_gdf.loc[idx, 'landuse_demand'] = total_demand / len(group)

        # NTL correction
        ntl_corrector.correct(grid_gdf, region_sub, 'landuse_demand',
                              ntl_values, 'ntl_landuse_demand')

        # Proximity correction (gamma=2)
        prox_scores = ProximityCorrector.compute_scores(
            grid_gdf, subs_sub, gamma=2.0,
            target_crs='EPSG:27700', clamp_km=0.01)
        prox_corrector.correct(grid_gdf, region_sub, 'ntl_landuse_demand',
                               prox_scores, 'prox2_ntl_landuse_demand')

        # Voronoi allocation + aggregate to substations
        alloc = allocator_registry.create('voronoi')
        voronoi_res = alloc.allocate(grid_gdf, subs_sub)
        actual = subs_sub['Demand (MVA)'].values
        predicted = np.zeros(len(subs_sub))
        for t_idx in range(len(subs_sub)):
            mask = voronoi_res.assignment == t_idx
            predicted[t_idx] = grid_gdf.loc[mask, 'prox2_ntl_landuse_demand'].sum() \
                if mask.any() else 0.0

        static_data['actual'].extend(actual.tolist())
        static_data['predicted'].extend(predicted.tolist())
        static_data['region'].extend([loc] * len(actual))

    all_data['GPMpostNP: Static Best'] = static_data
    print(f'  GPMpostNP: {len(static_data["actual"])} substations collected')

    return all_data


def plot_scatter():
    """Plot scatter plots (Voronoi allocation base)."""
    print('=' * 60)
    print('Exp 7: Predicted vs Actual scatter plots (Voronoi allocation base)')
    print('=' * 60)

    all_data = collect_substation_data()

    # Define plot order and colors (matching the paper's main narrative)
    plot_order = [
        ('GNN: voronoi_GNN', 'gray', 'o'),
        ('GNNpostP: voronoi_prox_GNN', 'tab:blue', '^'),
        ('GNNpostNP: voronoi_ntl_prox_GNN', 'red', 's'),
        ('GPMpostNP: Static Best', 'green', 'D'),
    ]

    for method_name, color, marker in plot_order:
        data = all_data.get(method_name, {})

        actual = np.array(data.get('actual', []))
        predicted = np.array(data.get('predicted', []))

        fig_s, ax = plt.subplots(figsize=(3.5, 2.5))

        if len(actual) == 0:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=8)
            ax.set_title(method_name, fontsize=8)
            plt.close(fig_s)
            continue

        ax.scatter(actual, predicted, c=color, marker=marker, alpha=0.5,
                   s=10, edgecolor='none')

        # 45-degree reference line
        max_val = max(actual.max(), predicted.max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=0.8, label='Perfect')

        # LOWESS trend line
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            sorted_idx = np.argsort(actual)
            smooth = lowess(predicted[sorted_idx], actual[sorted_idx], frac=0.3)
            ax.plot(smooth[:, 0], smooth[:, 1], color='darkorange',
                    linewidth=1.5, label='LOWESS')
        except ImportError:
            pass

        # Statistics
        from scipy.stats import pearsonr
        corr, _ = pearsonr(actual, predicted)
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        ax.text(0.05, 0.95, f'r={corr:.3f}\nRMSE={rmse:.2f}',
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Actual Demand (MVA)', fontsize=8)
        ax.set_ylabel('Predicted Demand (MVA)', fontsize=8)
        ax.set_title(method_name, fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(loc='lower right', fontsize=8)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.2)

        fig_s.tight_layout()
        safe_name = method_name.split(':')[0].strip()
        fig_s.savefig(OUTPUT_DIR / f'figure3_scatter_{safe_name}.png', dpi=200, bbox_inches='tight')
        fig_s.savefig(OUTPUT_DIR / f'figure3_scatter_{safe_name}.pdf', bbox_inches='tight')
        plt.close(fig_s)

    # Save data
    for method_name, data in all_data.items():
        if data.get('actual'):
            df = pd.DataFrame(data)
            safe_name = method_name.replace(':', '').replace(' ', '_').replace('+', '_')
            df.to_csv(OUTPUT_DIR / f'scatter_data_{safe_name}.csv', index=False)

    # -- 2x2 combined figure (paper Figure 3) --
    from scipy.stats import pearsonr
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess as lowess_func
        has_lowess = True
    except ImportError:
        has_lowess = False

    # Unify axis range: take the global min/max across all methods
    panel_labels = [
        ('GNN', 'GNN: voronoi_GNN', 'gray'),
        ('GNNpostP', 'GNNpostP: voronoi_prox_GNN', 'tab:blue'),
        ('GNNpostNP', 'GNNpostNP: voronoi_ntl_prox_GNN', 'tab:red'),
        ('GPMpostNP', 'GPMpostNP: Static Best', 'tab:green'),
    ]

    global_min, global_max = np.inf, -np.inf
    for _, method_name, _ in panel_labels:
        data = all_data.get(method_name, {})
        a = np.array(data.get('actual', []))
        p = np.array(data.get('predicted', []))
        if len(a) > 0:
            global_min = min(global_min, a.min(), p.min())
            global_max = max(global_max, a.max(), p.max())
    axis_lo = min(global_min, 0) - 2
    axis_hi = global_max * 1.05

    fig, axes = plt.subplots(2, 2, figsize=(6, 5.5))
    fig.subplots_adjust(wspace=0.08, hspace=0.25)

    for idx, (ax, (title, method_name, color)) in enumerate(
            zip(axes.flat, panel_labels)):
        data = all_data.get(method_name, {})
        actual = np.array(data.get('actual', []))
        predicted = np.array(data.get('predicted', []))

        row, col = divmod(idx, 2)

        if len(actual) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10)
            ax.set_title(title, fontsize=10)
            continue

        ax.scatter(actual, predicted, c=color, alpha=0.4, s=8, edgecolor='none')

        # 45-degree reference line
        ax.plot([axis_lo, axis_hi], [axis_lo, axis_hi], '--', color='gray',
                alpha=0.6, linewidth=0.8)

        # LOWESS trend line
        if has_lowess:
            sorted_idx = np.argsort(actual)
            smooth = lowess_func(predicted[sorted_idx], actual[sorted_idx],
                                 frac=0.3)
            ax.plot(smooth[:, 0], smooth[:, 1], color='darkorange',
                    linewidth=1.5)

        # Annotate r only
        corr_val, _ = pearsonr(actual, predicted)
        ax.text(0.05, 0.95, f'r = {corr_val:.3f}',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Unify axis range
        ax.set_xlim(axis_lo, axis_hi)
        ax.set_ylim(axis_lo, axis_hi)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis='both', labelsize=9)

        # Keep Y-axis labels only on the left column, X-axis labels only on the bottom row
        if col == 0:
            ax.set_ylabel('Predicted (MVA)', fontsize=10)
        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)

        if row == 1:
            ax.set_xlabel('Actual (MVA)', fontsize=10)
        else:
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)

    fig.savefig(OUTPUT_DIR / 'figure3_scatter_plots.png', dpi=300,
                bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure3_scatter_plots.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  Combined figure saved: figure3_scatter_plots.png/pdf')

    print(f'Exp 7 complete, output: {OUTPUT_DIR}')


if __name__ == '__main__':
    plot_scatter()
