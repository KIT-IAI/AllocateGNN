"""
009 Strength sweep (Exp 3) — Voronoi baseline

Part A: post-correction strength sweep (no retraining needed, reuses grid demands from Exp 0)
Part B: prior loss weight sweep (requires retraining, fold-1/seed=42)

All aggregation uses Voronoi partitioning to substations.

Usage:
    python 009_exp3_strength_sweep.py --part A
    python 009_exp3_strength_sweep.py --part B
    python 009_exp3_strength_sweep.py --part all
"""

import sys
import argparse
import pickle
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SpatialAllocation.Allocator import allocator_registry

warnings.filterwarnings('ignore', category=FutureWarning)

DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
ASSEMBLED_DIR = DATA_DIR / 'features' / 'assembled'
EXTRACTED_DIR = DATA_DIR / 'features' / 'extracted'
STATIC_DIR = SCRIPT_DIR / 'results' / 'static_allocation'
EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp3_strength_sweep'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

SEEDS = [42, 123, 456]
SEED = SEEDS[0]  # Part B uses only the first seed


def evaluate_allocation(subs_result, actual_col='Demand (MVA)', alloc_col='allocated_demand'):
    """Compute allocation metrics."""
    actual = subs_result[actual_col].values
    allocated = subs_result[alloc_col].values
    corr, _ = pearsonr(actual, allocated)
    rmse = np.sqrt(mean_squared_error(actual, allocated))
    mae = mean_absolute_error(actual, allocated)
    return {'corr': corr, 'rmse': rmse, 'mae': mae}


def load_grid_and_subs(loc):
    """Load grid data and substations."""
    with open(ASSEMBLED_DIR / f'{loc}_grid_points.pickle', 'rb') as f:
        grid_gdf, step_size_m = pickle.load(f)

    region_gdf = gpd.read_file(str(DATA_DIR / 'ITL3_region.gpkg'))
    substations_gdf = gpd.read_file(str(DATA_DIR / 'substations.gpkg'))

    study_itl3 = grid_gdf['ITL3'].unique()
    region_sub = region_gdf[region_gdf['ITL3'].isin(study_itl3)].copy()
    subs_sub = substations_gdf[substations_gdf['ITL3'].isin(study_itl3)].copy().reset_index(drop=True)

    return grid_gdf, region_sub, subs_sub


def apply_scaled_ntl_correction(grid_gdf, region_sub, base_demand, ntl_values, scale):
    """Scale the NTL correction: factor = 1 + scale*(original_factor - 1)."""
    rci_sum = (grid_gdf['lu_residential_prop'].values
              + grid_gdf['lu_commercial_prop'].values
              + grid_gdf['lu_industrial_prop'].values)
    rci_mask = rci_sum > RCI_THRESHOLD

    result = np.zeros_like(base_demand)
    region_info = region_sub.set_index('ITL3')

    for itl3, group in grid_gdf.groupby('ITL3'):
        if itl3 not in region_info.index:
            continue
        total_demand = region_info.loc[itl3, 'Demand (MVA)']
        idx = group.index
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

        ntl_factor = np.log(1 + ntl_group + epsilon) / np.log(1 + ntl_median)
        scaled_factor = 1.0 + scale * (ntl_factor - 1.0)
        scaled_factor = np.maximum(scaled_factor, 0.01)

        raw = base_demand[idx] * scaled_factor
        raw_sum = raw.sum()
        if raw_sum > 0:
            result[idx] = total_demand * raw / raw_sum
        else:
            result[idx] = total_demand / len(group)

    return result


def apply_scaled_prox_correction(grid_gdf, region_sub, base_demand, subs_sub, gamma):
    """Adjust the proximity correction by gamma. gamma=0 means no correction."""
    if gamma <= 0:
        return base_demand.copy()

    grid_proj = grid_gdf.to_crs(TARGET_CRS)
    subs_proj = subs_sub.to_crs(TARGET_CRS)
    grid_coords = np.column_stack([grid_proj.geometry.x.values, grid_proj.geometry.y.values])
    subs_coords = np.column_stack([subs_proj.geometry.x.values, subs_proj.geometry.y.values])
    dist_km = cdist(grid_coords, subs_coords, metric='euclidean') / 1000.0
    dist_km = np.maximum(dist_km, DIST_CLAMP_KM)
    prox_scores = np.sum(dist_km ** (-gamma), axis=1)

    rci_sum = (grid_gdf['lu_residential_prop'].values
              + grid_gdf['lu_commercial_prop'].values
              + grid_gdf['lu_industrial_prop'].values)
    rci_mask = rci_sum > RCI_THRESHOLD

    result = np.zeros_like(base_demand)
    region_info = region_sub.set_index('ITL3')

    for itl3, group in grid_gdf.groupby('ITL3'):
        if itl3 not in region_info.index:
            continue
        total_demand = region_info.loc[itl3, 'Demand (MVA)']
        idx = group.index
        prox_group = prox_scores[idx]
        rci_group = rci_mask[idx]

        rci_prox = prox_group[rci_group]
        prox_median = np.median(rci_prox) if len(rci_prox) > 0 else np.median(prox_group)
        if prox_median <= 0:
            prox_median = 1e-6

        prox_factor = np.log(1 + prox_group) / np.log(1 + prox_median)

        raw = base_demand[idx] * prox_factor
        raw_sum = raw.sum()
        if raw_sum > 0:
            result[idx] = total_demand * raw / raw_sum
        else:
            result[idx] = total_demand / len(group)

    return result


def voronoi_aggregate(grid_gdf, subs_sub, demand_arr):
    """Aggregate to substations using Voronoi partitioning."""
    alloc_voronoi = allocator_registry.create('voronoi')
    voronoi_res = alloc_voronoi.allocate(grid_gdf, subs_sub)

    subs_result = subs_sub.copy()
    subs_result['allocated_demand'] = 0.0
    for target_idx in range(len(subs_sub)):
        mask = voronoi_res.assignment == target_idx
        subs_result.loc[target_idx, 'allocated_demand'] = demand_arr[mask].sum()
    return subs_result


def run_part_a():
    """Part A: post-correction strength sweep (no retraining needed)."""
    print('=' * 60)
    print('Part A: post-correction strength sweep')
    print('=' * 60)

    ntl_scales = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    prox_gammas = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    combo_scales = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

    # Each seed uses its own KFold (random_state=seed); the fold splits differ,
    # so a separate region->fold map is built per seed
    seed_fold_maps = {}
    for seed in SEEDS:
        sp = EXP0_DIR / f'seed_{seed}' / 'baseline' / 'kfold_splits.json'
        if not sp.exists():
            print(f'[Error] splits file not found: {sp}')
            return
        with open(sp, 'r') as f:
            sp_data = json.load(f)
        fold_map = {}
        for fold_name, split_info in sp_data.items():
            fold_num = fold_name.split('_')[1]
            for loc in split_info['test']:
                fold_map[loc] = fold_num
        seed_fold_maps[seed] = fold_map

    # Checkpoint resume: use a distinct filename to avoid reusing results from
    # an older single-seed run
    part_a_csv = OUTPUT_DIR / 'part_a_sweep_results_3seeds.csv'
    if part_a_csv.exists():
        existing_df = pd.read_csv(part_a_csv)
        all_sweep_results = existing_df.to_dict('records')
        done_keys = set(zip(existing_df['sweep_type'], existing_df['strength'],
                            existing_df['region']))
        print(f'Restored {len(all_sweep_results)} existing results')
    else:
        all_sweep_results = []
        done_keys = set()

    # Iterate over regions (each region uses the test-set prediction from that
    # seed's own fold)
    for loc in ALL_LOCATIONS:
        all_points = (
            [('ntl', s, loc) for s in ntl_scales]
            + [('proximity', g, loc) for g in prox_gammas]
            + [('combo', c, loc) for c in combo_scales]
        )
        if all(k in done_keys for k in all_points):
            print(f'  [Skip] {loc}: all sweep points already completed')
            continue

        grid_gdf, region_sub, subs_sub = load_grid_and_subs(loc)
        ntl_npz = np.load(EXTRACTED_DIR / f'{loc}_ntl.npz', allow_pickle=True)
        ntl_values = ntl_npz['data'][:, 0]

        # For each seed, load grid demands using that seed's own fold structure
        seed_bases = []
        for seed in SEEDS:
            fold_num = seed_fold_maps[seed].get(loc)
            if fold_num is None:
                print(f'  [Warning] {loc} seed_{seed}: no matching fold found')
                continue
            gd_path = (EXP0_DIR / f'seed_{seed}' / 'baseline'
                       / f'fold{fold_num}' / 'grid_demands'
                       / f'{loc}_grid_demands.pickle')
            if not gd_path.exists():
                print(f'  [Warning] {loc} seed_{seed}: grid demands not found, skipping this seed')
                continue
            with open(gd_path, 'rb') as f:
                grid_demands = pickle.load(f)
            seed_bases.append(grid_demands['gnn_demand'])

        if not seed_bases:
            print(f'  [Skip] {loc}: no grid demands for any seed')
            continue

        new_count = 0

        def _avg_metrics(metrics_list):
            """Average the metrics list across multiple seeds."""
            return {k: float(np.mean([mi[k] for mi in metrics_list]))
                    for k in ('rmse', 'mae', 'corr')}

        # NTL sweep
        for scale in ntl_scales:
            if ('ntl', scale, loc) in done_keys:
                continue
            metrics_list = []
            for base in seed_bases:
                if scale == 0:
                    corrected = base.copy()
                else:
                    corrected = apply_scaled_ntl_correction(
                        grid_gdf, region_sub, base, ntl_values, scale)
                subs_result = voronoi_aggregate(grid_gdf, subs_sub, corrected)
                metrics_list.append(evaluate_allocation(subs_result))
            m = _avg_metrics(metrics_list)
            all_sweep_results.append({
                'sweep_type': 'ntl', 'strength': scale,
                'region': loc, **m,
            })
            done_keys.add(('ntl', scale, loc))
            new_count += 1

        # Proximity sweep
        for gamma in prox_gammas:
            if ('proximity', gamma, loc) in done_keys:
                continue
            metrics_list = []
            for base in seed_bases:
                corrected = apply_scaled_prox_correction(
                    grid_gdf, region_sub, base, subs_sub, gamma)
                subs_result = voronoi_aggregate(grid_gdf, subs_sub, corrected)
                metrics_list.append(evaluate_allocation(subs_result))
            m = _avg_metrics(metrics_list)
            all_sweep_results.append({
                'sweep_type': 'proximity', 'strength': gamma,
                'region': loc, **m,
            })
            done_keys.add(('proximity', gamma, loc))
            new_count += 1

        # Combined sweep
        for combo_s in combo_scales:
            if ('combo', combo_s, loc) in done_keys:
                continue
            metrics_list = []
            for base in seed_bases:
                if combo_s == 0:
                    corrected = base.copy()
                else:
                    ntl_corrected = apply_scaled_ntl_correction(
                        grid_gdf, region_sub, base, ntl_values, combo_s)
                    corrected = apply_scaled_prox_correction(
                        grid_gdf, region_sub, ntl_corrected, subs_sub,
                        PROXIMITY_GAMMA * combo_s)
                subs_result = voronoi_aggregate(grid_gdf, subs_sub, corrected)
                metrics_list.append(evaluate_allocation(subs_result))
            m = _avg_metrics(metrics_list)
            all_sweep_results.append({
                'sweep_type': 'combo', 'strength': combo_s,
                'region': loc, **m,
            })
            done_keys.add(('combo', combo_s, loc))
            new_count += 1

        # Save incrementally after each region is processed
        if new_count > 0:
            pd.DataFrame(all_sweep_results).to_csv(part_a_csv, index=False)
            print(f'  {loc}: added {new_count} new results (total {len(all_sweep_results)})')

    if not all_sweep_results:
        print('[Warning] no sweep results')
        return

    sweep_df = pd.DataFrame(all_sweep_results)
    sweep_df.to_csv(part_a_csv, index=False)

    # Plotting
    for sweep_type, title, xlabel in [
        ('ntl', 'NTL Post-Correction Strength', 'Strength'),
        ('proximity', r'Proximity $\gamma$', r'$\gamma$'),
        ('combo', 'NTL+Prox Combined', 'Strength'),
    ]:
        sub = sweep_df[sweep_df['sweep_type'] == sweep_type]
        if sub.empty:
            continue
        fig_s, ax = plt.subplots(figsize=(3.5, 2.5))
        grouped = sub.groupby('strength')['rmse'].agg(['mean', 'std']).reset_index()
        ax.plot(grouped['strength'], grouped['mean'], 'o-', color='tab:blue')
        ax.fill_between(grouped['strength'],
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        alpha=0.2, color='tab:blue')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel('RMSE', fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, alpha=0.3)
        fig_s.tight_layout()
        safe_name = sweep_type
        fig_s.savefig(OUTPUT_DIR / f'figure1_strength_sweep_{safe_name}.png', dpi=200, bbox_inches='tight')
        fig_s.savefig(OUTPUT_DIR / f'figure1_strength_sweep_{safe_name}.pdf', bbox_inches='tight')
        plt.close(fig_s)
    # ── 1×3 combined figure (paper Figure 1) ──
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    panel_defs = [
        ('ntl',       r'(a) NTL scaling coefficient $\alpha$', r'$\alpha$'),
        ('proximity', r'(b) Proximity decay exponent $\gamma$', r'$\gamma$'),
        ('combo',     r'(c) Joint scaling factor $\beta$',      r'$\beta$'),
    ]

    baseline_rmse = sweep_df[
        (sweep_df['sweep_type'] == 'ntl') & (sweep_df['strength'] == 0.0)
    ]['rmse'].mean()

    for ax, (stype, title, xlabel) in zip(axes, panel_defs):
        sub = sweep_df[sweep_df['sweep_type'] == stype]
        if sub.empty:
            continue
        grouped = sub.groupby('strength')['rmse'].agg(['mean', 'std']).reset_index()
        ax.plot(grouped['strength'], grouped['mean'], 'o-', color='tab:blue',
                markersize=4, linewidth=1.2)
        ax.fill_between(grouped['strength'],
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        alpha=0.2, color='tab:blue')
        ax.axhline(baseline_rmse, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, alpha=0.3)

    # Keep the y-axis label only on the leftmost panel
    for ax in axes[1:]:
        ax.set_ylabel('')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure1_strength_sweep.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure1_strength_sweep.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  Combined figure saved: figure1_strength_sweep.png/pdf')

    print(f'Part A complete, figures saved to: {OUTPUT_DIR}')


def assert_training_complete(model_path, expected_epochs):
    """Guard against resuming from an incomplete training checkpoint.

    model.pth is a best-snapshot checkpoint that can be written to disk at
    any point during training, so a hard interruption (crash/kill) can leave
    a partial checkpoint behind; *_training_log.json is only written after
    the full epoch loop finishes. Therefore the presence of model.pth alone
    does not prove training completed — the log must exist and report the
    expected number of epochs, otherwise this function aborts and tells the
    caller how to proceed instead of silently resuming from a truncated
    checkpoint.
    """
    log_path = model_path.with_name(model_path.name[:-4] + '_training_log.json')
    if not log_path.exists():
        raise SystemExit(f'Incomplete training run: {model_path} exists but {log_path.name} is missing -- '
                         'move this directory aside for inspection and rerun (do not resume directly).')
    with open(log_path, encoding='utf-8') as f:
        _log = json.load(f)
    n_logged = len(_log.get('train_losses', {}).get('total', []))
    if n_logged != expected_epochs:
        raise SystemExit(f'Training log incomplete: {log_path} ({n_logged}/{expected_epochs} epochs) -- '
                         'move this directory aside for inspection and rerun (do not resume directly).')


def run_part_b():
    """Part B: prior loss weight sweep (requires retraining)."""
    print('=' * 60)
    print('Part B: prior loss weight sweep')
    print('=' * 60)

    from SpatialAllocation.GNN.core.EdgeWeightSolver import EdgeWeightSolver
    from SpatialAllocation.GNN.core.ModelConfig import ModelConfig
    from SpatialAllocation.FeatureExtractor.correctors.proximity_corrector import ProximityCorrector

    # Reuse the data loading and graph construction from 005
    from importlib import import_module
    training_module = import_module('005_kfold_prior_training')

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    weights_ntl = [0.01, 0.05, 0.1, 0.2, 0.5]
    weights_prox = [0.01, 0.05, 0.1, 0.2, 0.5]

    # Load data
    grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict = training_module.load_data()
    graphs = training_module.build_graphs(
        grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict,
        inject_priors=True,
    )

    # Use the fold-1 split
    location_array = np.array(ALL_LOCATIONS)
    kf = KFold(n_splits=4, shuffle=True, random_state=SEED)
    splits = list(kf.split(location_array))
    train_indices, test_indices = splits[0]
    train_locs = location_array[train_indices].tolist()
    test_locs = location_array[test_indices].tolist()

    print(f'Fold-1 train: {train_locs}')
    print(f'Fold-1 test: {test_locs}')

    # Checkpoint resume: load existing results
    part_b_csv = OUTPUT_DIR / 'part_b_weight_sweep.csv'
    if part_b_csv.exists():
        existing_df = pd.read_csv(part_b_csv)
        sweep_results = existing_df.to_dict('records')
        done_keys = set(zip(existing_df['prior'], existing_df['weight']))
        print(f'Restored {len(sweep_results)} existing results, completed weight points: {done_keys}')
    else:
        sweep_results = []
        done_keys = set()

    def _train_or_load(sweep_dir, obj_w):
        """Train or load an existing model, and return the solver."""
        model_path = sweep_dir / 'model.pth'
        config = ModelConfig(
            epochs=200, hidden_dim=256, embedding_dim=128, num_layers=3,
            conv_type='hgt', allocation_temperature_start=0.01,
            learning_rate=1e-3, weight_decay=1e-4,
            use_scheduler=True, warmup_epochs=20, decay_epochs=20,
            cosine_epochs=160, cosine_eta_min=1e-5, learnable=False,
            save_path=str(model_path),
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

        train_graphs_list = [graphs[loc] for loc in train_locs]
        test_graphs_list = [graphs[loc] for loc in test_locs]
        train_dl = DataLoader(train_graphs_list, batch_size=1, shuffle=False)
        test_dl = DataLoader(test_graphs_list, batch_size=1, shuffle=False)

        solver = EdgeWeightSolver(config)
        if model_path.exists():
            assert_training_complete(model_path, config.epochs)
            print(f'  Existing model {model_path}, skipping training')
            solver.init_model(train_dl, obj_w)
            solver._load_checkpoint()
        else:
            solver.train_multi_graph(train_dl, test_dataloader=test_dl,
                                     objective_weights=obj_w)
        return solver

    # NTL weight sweep
    for w in weights_ntl:
        if ('ntl', w) in done_keys:
            print(f'\n--- NTL prior weight = {w} --- [already completed, skipping]')
            continue
        print(f'\n--- NTL prior weight = {w} ---')
        sweep_dir = OUTPUT_DIR / 'part_b' / f'ntl_w{w}'
        sweep_dir.mkdir(parents=True, exist_ok=True)

        obj_w = {'landuse_prediction_loss': 1.0, 'ntl_prior': w}
        solver = _train_or_load(sweep_dir, obj_w)

        for loc in test_locs:
            metrics = training_module.predict_and_evaluate_location(
                loc, solver, graphs, grids, ntl_dict, region_dict, subs_dict)
            for method, m in metrics.items():
                if method == 'voronoi_GNN':
                    sweep_results.append({
                        'prior': 'ntl', 'weight': w, 'region': loc,
                        'rmse': m['rmse'], 'mae': m['mae'], 'corr': m['corr'],
                    })

        # Save incrementally after each weight point completes
        pd.DataFrame(sweep_results).to_csv(part_b_csv, index=False)
        done_keys.add(('ntl', w))
        print(f'  NTL w={w} completed and saved')

    # Proximity weight sweep
    for w in weights_prox:
        if ('proximity', w) in done_keys:
            print(f'\n--- Proximity prior weight = {w} --- [already completed, skipping]')
            continue
        print(f'\n--- Proximity prior weight = {w} ---')
        sweep_dir = OUTPUT_DIR / 'part_b' / f'prox_w{w}'
        sweep_dir.mkdir(parents=True, exist_ok=True)

        obj_w = {'landuse_prediction_loss': 1.0, 'proximity_prior': w}
        solver = _train_or_load(sweep_dir, obj_w)

        for loc in test_locs:
            metrics = training_module.predict_and_evaluate_location(
                loc, solver, graphs, grids, ntl_dict, region_dict, subs_dict)
            for method, m in metrics.items():
                if method == 'voronoi_GNN':
                    sweep_results.append({
                        'prior': 'proximity', 'weight': w, 'region': loc,
                        'rmse': m['rmse'], 'mae': m['mae'], 'corr': m['corr'],
                    })

        # Save incrementally after each weight point completes
        pd.DataFrame(sweep_results).to_csv(part_b_csv, index=False)
        done_keys.add(('proximity', w))
        print(f'  Proximity w={w} completed and saved')

    if sweep_results:
        sweep_df = pd.DataFrame(sweep_results)
        sweep_df.to_csv(part_b_csv, index=False)

        # Plotting
        for prior_type, title in [('ntl', 'NTL Prior Weight'), ('proximity', 'Proximity Prior Weight')]:
            sub = sweep_df[sweep_df['prior'] == prior_type]
            fig_b, ax = plt.subplots(figsize=(3.5, 2.5))
            grouped = sub.groupby('weight')['rmse'].agg(['mean', 'std']).reset_index()
            ax.plot(grouped['weight'], grouped['mean'], 'o-', color='tab:red')
            ax.fill_between(grouped['weight'],
                            grouped['mean'] - grouped['std'],
                            grouped['mean'] + grouped['std'],
                            alpha=0.2, color='tab:red')
            ax.set_xlabel('Prior Weight', fontsize=8)
            ax.set_ylabel('RMSE', fontsize=8)
            ax.set_title(title, fontsize=8)
            ax.tick_params(axis='both', labelsize=8)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            fig_b.tight_layout()
            fig_b.savefig(OUTPUT_DIR / f'figure1b_weight_sweep_{prior_type}.png', dpi=200, bbox_inches='tight')
            fig_b.savefig(OUTPUT_DIR / f'figure1b_weight_sweep_{prior_type}.pdf', bbox_inches='tight')
            plt.close(fig_b)

    print(f'Part B complete, results: {OUTPUT_DIR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exp 3: strength sweep')
    parser.add_argument('--part', type=str, default='all', choices=['A', 'B', 'all'])
    args = parser.parse_args()

    if args.part in ('A', 'all'):
        run_part_a()
    if args.part in ('B', 'all'):
        run_part_b()
