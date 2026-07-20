"""
018 GNN embedding probing experiment (H2)

Extracts agent node embeddings and final allocation weights w_sa from the trained GNN,
and computes their Pearson/Spearman correlation coefficients with the NTL factor / Proximity factor.

If the correlation is significantly higher than random -> the triple-counting hypothesis holds.
If not significant -> the hypothesis is falsified, and the cause of the antagonism needs to be re-examined.

No additional training is required — only a forward pass on the existing model plus statistical analysis.

Also extracts the temperature parameter τ.

Usage:
    python 018_exp_h2_embedding_probing.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import pickle
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from torch_geometric.loader import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SpatialAllocation.GNN.core.EdgeWeightSolver import EdgeWeightSolver
from SpatialAllocation.GNN.core.ModelConfig import ModelConfig

warnings.filterwarnings('ignore', category=FutureWarning)

# ════════════════════════════════════════════════════════════
# Path constants
# ════════════════════════════════════════════════════════════

EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'
GRAPH_CACHE = EXP0_DIR / 'graph_cache' / 'cached_graphs.pickle'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp_h2_embedding_probing'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
N_FOLDS = 4

RCI_THRESHOLD = 0.5
PROXIMITY_GAMMA = 2.0
DIST_CLAMP_KM = 0.01
TARGET_CRS = 'EPSG:27700'


def compute_factors_for_loc(grid_gdf, ntl_values, subs_sub):
    """Compute the NTL factor and Proximity factor for a region."""
    rci_sum = (grid_gdf['lu_residential_prop'].values
              + grid_gdf['lu_commercial_prop'].values
              + grid_gdf['lu_industrial_prop'].values)
    rci_mask = rci_sum > RCI_THRESHOLD

    # Proximity scores
    grid_proj = grid_gdf.to_crs(TARGET_CRS)
    subs_proj = subs_sub.to_crs(TARGET_CRS)
    grid_coords = np.column_stack([grid_proj.geometry.x.values, grid_proj.geometry.y.values])
    subs_coords = np.column_stack([subs_proj.geometry.x.values, subs_proj.geometry.y.values])
    dist_km = cdist(grid_coords, subs_coords, metric='euclidean') / 1000.0
    dist_km = np.maximum(dist_km, DIST_CLAMP_KM)
    prox_scores = np.sum(dist_km ** (-PROXIMITY_GAMMA), axis=1)

    ntl_factor = np.ones(len(grid_gdf))
    prox_factor = np.ones(len(grid_gdf))

    for itl3, group in grid_gdf.groupby('ITL3'):
        idx = group.index

        # NTL factor (same formula as in 005)
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

    return ntl_factor, prox_factor, rci_mask


def run_probing():
    """Run the embedding probing experiment."""
    # Load cached data
    print("Loading cached graph data...")
    with open(GRAPH_CACHE, 'rb') as f:
        cache = pickle.load(f)

    graphs = cache['graphs']
    grids = cache['grids']
    ntl_dict = cache['ntl_dict']
    subs_dict = cache['subs_dict']

    all_probing_results = []
    all_tau_values = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")

        # Load fold split
        splits_path = EXP0_DIR / f'seed_{seed}' / 'baseline' / 'kfold_splits.json'
        with open(splits_path) as f:
            splits = json.load(f)

        for config in ['baseline', 'ntl_prox']:
            print(f"\n  Config: {config}")

            # Determine objective_weights (used to initialize model structure)
            objective_weights = {'landuse_prediction_loss': 1.0}
            if config == 'ntl_prox':
                objective_weights['ntl_prior'] = 0.1
                objective_weights['proximity_prior'] = 0.1

            for fold_key, fold_info in splits.items():
                test_locs = fold_info['test']
                train_locs = fold_info['train']
                fold_dir_name = fold_key.replace('_', '')
                model_path = EXP0_DIR / f'seed_{seed}' / config / fold_dir_name / 'model.pth'

                if not model_path.exists():
                    print(f"    [skip] {fold_key}: model not found ({model_path})")
                    continue

                print(f"    {fold_key}: test locations = {test_locs}")

                # Create solver and load model
                model_config = ModelConfig(
                    hidden_dim=256,
                    embedding_dim=128,
                    num_layers=3,
                    conv_type='hgt',
                    allocation_temperature_start=0.01,
                    learnable=False,
                    save_path=str(model_path),
                    device='cpu',
                )
                solver = EdgeWeightSolver(model_config)

                # Initialize model structure using the graphs from train_locs
                train_graphs = [graphs[loc] for loc in train_locs if loc in graphs]
                train_dl = DataLoader(train_graphs, batch_size=1, shuffle=False)
                solver.init_model(train_dl, objective_weights)
                solver._load_checkpoint()

                # Extract temperature parameter τ
                tau = None
                if solver.edge_weighting_layer is not None:
                    for name, param in solver.edge_weighting_layer.named_parameters():
                        if 'log_temperature' in name or 'temperature' in name:
                            tau_val = torch.exp(param).item() if 'log' in name else param.item()
                            tau = tau_val
                            all_tau_values.append({
                                'seed': seed, 'config': config, 'fold': fold_key,
                                'tau': tau_val
                            })
                            break

                for loc in test_locs:
                    try:
                        graph = graphs[loc]
                        grid_gdf = grids[loc][0]
                        ntl_values = ntl_dict[loc]
                        subs_sub = subs_dict[loc]

                        # Forward pass to obtain edge weights
                        edge_weights_df = solver.predict_edge_weights(graph)

                        # Extract agent embeddings
                        solver.encoder.eval()
                        with torch.no_grad():
                            embeddings_dict = solver.encoder(
                                graph.x_dict, graph.edge_index_dict)
                            agent_embeddings = embeddings_dict['agent'].cpu().numpy()

                        # Get agent index mapping
                        agent_index_map = graph.agent_index_map
                        agent_orig_indices = agent_index_map.values

                        # Compute NTL/Proximity factors
                        ntl_factor, prox_factor, rci_mask = compute_factors_for_loc(
                            grid_gdf, ntl_values, subs_sub)

                        # Map to agents in graph space
                        ntl_factor_graph = ntl_factor[agent_orig_indices]
                        prox_factor_graph = prox_factor[agent_orig_indices]

                        # Embedding L2 norm
                        embedding_norm = np.linalg.norm(agent_embeddings, axis=1)

                        # For each embedding dimension, compute the maximum correlation with NTL/Prox
                        n_dims = agent_embeddings.shape[1]
                        max_ntl_pearson = 0
                        max_prox_pearson = 0
                        max_ntl_spearman = 0
                        max_prox_spearman = 0

                        for dim in range(n_dims):
                            emb_dim = agent_embeddings[:, dim]
                            r_ntl, _ = pearsonr(emb_dim, ntl_factor_graph)
                            r_prox, _ = pearsonr(emb_dim, prox_factor_graph)
                            rho_ntl, _ = spearmanr(emb_dim, ntl_factor_graph)
                            rho_prox, _ = spearmanr(emb_dim, prox_factor_graph)

                            if abs(r_ntl) > abs(max_ntl_pearson):
                                max_ntl_pearson = r_ntl
                            if abs(r_prox) > abs(max_prox_pearson):
                                max_prox_pearson = r_prox
                            if abs(rho_ntl) > abs(max_ntl_spearman):
                                max_ntl_spearman = rho_ntl
                            if abs(rho_prox) > abs(max_prox_spearman):
                                max_prox_spearman = rho_prox

                        # Correlation between embedding norm and NTL/Prox
                        r_norm_ntl, _ = pearsonr(embedding_norm, ntl_factor_graph)
                        r_norm_prox, _ = pearsonr(embedding_norm, prox_factor_graph)
                        rho_norm_ntl, _ = spearmanr(embedding_norm, ntl_factor_graph)
                        rho_norm_prox, _ = spearmanr(embedding_norm, prox_factor_graph)

                        # Correlation between edge weights and NTL/Prox (averaged per-source)
                        weight_ntl_corrs = []
                        weight_prox_corrs = []
                        for s_idx in edge_weights_df['source_node_idx'].unique():
                            mask = edge_weights_df['source_node_idx'] == s_idx
                            sub_df = edge_weights_df[mask]
                            w_vals = sub_df['predicted_weight'].values
                            a_local = sub_df['agent_node_idx'].values
                            ntl_vals = ntl_factor_graph[a_local]
                            prox_vals = prox_factor_graph[a_local]
                            if len(w_vals) > 3:
                                r_w_ntl, _ = spearmanr(w_vals, ntl_vals)
                                r_w_prox, _ = spearmanr(w_vals, prox_vals)
                                if not np.isnan(r_w_ntl):
                                    weight_ntl_corrs.append(r_w_ntl)
                                if not np.isnan(r_w_prox):
                                    weight_prox_corrs.append(r_w_prox)

                        mean_weight_ntl = np.mean(weight_ntl_corrs) if weight_ntl_corrs else 0
                        mean_weight_prox = np.mean(weight_prox_corrs) if weight_prox_corrs else 0

                        result = {
                            'seed': seed, 'config': config, 'fold': fold_key, 'location': loc,
                            'n_agents': len(agent_embeddings),
                            'embedding_dim': n_dims,
                            'tau': tau,
                            'max_emb_ntl_pearson': max_ntl_pearson,
                            'max_emb_prox_pearson': max_prox_pearson,
                            'max_emb_ntl_spearman': max_ntl_spearman,
                            'max_emb_prox_spearman': max_prox_spearman,
                            'emb_norm_ntl_pearson': r_norm_ntl,
                            'emb_norm_prox_pearson': r_norm_prox,
                            'emb_norm_ntl_spearman': rho_norm_ntl,
                            'emb_norm_prox_spearman': rho_norm_prox,
                            'weight_ntl_spearman': mean_weight_ntl,
                            'weight_prox_spearman': mean_weight_prox,
                        }
                        all_probing_results.append(result)

                        tau_str = f"tau={tau:.4f}" if tau else "tau=N/A"
                        print(f"      {loc}: max_emb_ntl_r={max_ntl_pearson:.3f}, "
                              f"max_emb_prox_r={max_prox_pearson:.3f}, "
                              f"w_ntl_rho={mean_weight_ntl:.3f}, "
                              f"w_prox_rho={mean_weight_prox:.3f}, "
                              f"{tau_str}")

                    except Exception as e:
                        print(f"      {loc}: error - {e}")
                        import traceback
                        traceback.print_exc()
                        continue

    # Save results
    results_df = pd.DataFrame(all_probing_results)
    results_df.to_csv(OUTPUT_DIR / 'embedding_probing_raw.csv', index=False)

    if all_tau_values:
        tau_df = pd.DataFrame(all_tau_values)
        tau_df.to_csv(OUTPUT_DIR / 'tau_values.csv', index=False)
        print(f"\nTemperature parameter τ statistics:")
        for cfg in tau_df['config'].unique():
            sub = tau_df[tau_df['config'] == cfg]
            print(f"  {cfg}: mean={sub['tau'].mean():.4f}, std={sub['tau'].std():.4f}, "
                  f"range=[{sub['tau'].min():.4f}, {sub['tau'].max():.4f}]")

    # Summary
    print("\n" + "="*80)
    print("Embedding probing summary")
    print("="*80)

    if len(results_df) > 0:
        for config in results_df['config'].unique():
            sub = results_df[results_df['config'] == config]
            print(f"\nConfig: {config} (n={len(sub)})")
            for col in ['max_emb_ntl_pearson', 'max_emb_prox_pearson',
                        'max_emb_ntl_spearman', 'max_emb_prox_spearman',
                        'emb_norm_ntl_pearson', 'emb_norm_prox_pearson',
                        'weight_ntl_spearman', 'weight_prox_spearman']:
                vals = sub[col].dropna()
                if len(vals) > 0:
                    print(f"  {col}: {vals.mean():.4f} +/- {vals.std():.4f} "
                          f"[{vals.min():.4f}, {vals.max():.4f}]")

        summary = results_df.groupby('config').agg({
            'max_emb_ntl_pearson': ['mean', 'std'],
            'max_emb_prox_pearson': ['mean', 'std'],
            'weight_ntl_spearman': ['mean', 'std'],
            'weight_prox_spearman': ['mean', 'std'],
        })
        summary.to_csv(OUTPUT_DIR / 'embedding_probing_summary.csv')

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    run_probing()
