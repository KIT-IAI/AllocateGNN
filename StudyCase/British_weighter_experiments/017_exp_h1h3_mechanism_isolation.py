"""
017 Mechanism isolation experiments (H1 + H3)

H1a: Skip re-normalization (raw multiplicative) — isolates the double-normalisation artifact
H1b: Post-correction applied to random noise (with re-normalization) — isolates signal redundancy vs. structural effect
H3:  Additive post-correction (in place of multiplicative) — isolates the structural effect of multiplicative combination + conservation

All experiments are pure post-processing based on the existing grid_demands from Exp 0.
No retraining is required.

Usage:
    python 017_exp_h1h3_mechanism_isolation.py
"""

import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SpatialAllocation.Allocator import allocator_registry

warnings.filterwarnings('ignore', category=FutureWarning)

# ════════════════════════════════════════════════════════════
# Path constants
# ════════════════════════════════════════════════════════════

DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
ASSEMBLED_DIR = DATA_DIR / 'features' / 'assembled'
EXTRACTED_DIR = DATA_DIR / 'features' / 'extracted'
EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp_h1h3_mechanism_isolation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_LOCATIONS = [
    'London',
    'TLH2', 'TLH3', 'TLJ1', 'TLF1', 'TLF2',
    'TLC1', 'TLC2', 'TLD6', 'TLG1', 'TLG2', 'TLE4',
    'TLH1', 'TLE3', 'TLD3', 'TLD4',
]

SEEDS = [42, 123, 456]
N_FOLDS = 4

RCI_THRESHOLD = 0.5
PROXIMITY_GAMMA = 2.0
DIST_CLAMP_KM = 0.01
TARGET_CRS = 'EPSG:27700'


def evaluate_allocation(subs_result, actual_col='Demand (MVA)', alloc_col='allocated_demand'):
    """Compute allocation evaluation metrics."""
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

    # Load NTL
    ntl_path = EXTRACTED_DIR / f'{loc}_ntl.npz'
    if ntl_path.exists():
        ntl_npz = np.load(str(ntl_path), allow_pickle=True)
        ntl_values = ntl_npz['data'][:, 0]
    else:
        ntl_values = np.zeros(len(grid_gdf))

    return grid_gdf, region_sub, subs_sub, ntl_values


def compute_prox_scores(grid_gdf, subs_sub, gamma=PROXIMITY_GAMMA):
    """Compute Proximity scores."""
    grid_proj = grid_gdf.to_crs(TARGET_CRS)
    subs_proj = subs_sub.to_crs(TARGET_CRS)
    grid_coords = np.column_stack([grid_proj.geometry.x.values, grid_proj.geometry.y.values])
    subs_coords = np.column_stack([subs_proj.geometry.x.values, subs_proj.geometry.y.values])
    dist_km = cdist(grid_coords, subs_coords, metric='euclidean') / 1000.0
    dist_km = np.maximum(dist_km, DIST_CLAMP_KM)
    return np.sum(dist_km ** (-gamma), axis=1)


def compute_factors(grid_gdf, ntl_values, prox_scores):
    """Compute the NTL factor and Proximity factor (per-ITL3). Returns per-agent factor arrays."""
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


def voronoi_aggregate(grid_gdf, subs_sub, demand_arr):
    """Aggregate to substations via Voronoi partition and evaluate."""
    alloc_voronoi = allocator_registry.create('voronoi')
    voronoi_res = alloc_voronoi.allocate(grid_gdf, subs_sub)

    subs_result = subs_sub.copy()
    subs_result['allocated_demand'] = 0.0
    for target_idx in range(len(subs_sub)):
        mask = voronoi_res.assignment == target_idx
        subs_result.loc[target_idx, 'allocated_demand'] = demand_arr[mask].sum()
    return evaluate_allocation(subs_result)


# ════════════════════════════════════════════════════════════
# H1a: Skip re-normalization (raw multiplicative; demand conservation no longer holds)
# ════════════════════════════════════════════════════════════

def apply_correction_no_renorm(base_demand, combined_factor, grid_gdf, region_sub):
    """Multiplicative post-correction, but skip per-ITL3 re-normalization. Demand conservation no longer holds."""
    result = base_demand * combined_factor
    return result


# ════════════════════════════════════════════════════════════
# H1b: Random-noise post-correction (with re-normalization)
# ════════════════════════════════════════════════════════════

def apply_random_noise_correction(base_demand, grid_gdf, region_sub, rng, noise_std=0.5):
    """Replace NTL+Prox with random spatial noise while keeping the same multiplicative + renorm pipeline.

    noise_std: standard deviation of the noise, matched to the variability of the real factor.
    """
    region_info = region_sub.set_index('ITL3')
    result = np.zeros_like(base_demand)

    for itl3, group in grid_gdf.groupby('ITL3'):
        if itl3 not in region_info.index:
            continue
        total_demand = region_info.loc[itl3, 'Demand (MVA)']
        idx = group.index

        # Generate random multiplicative factor (mean 1, positive)
        noise_factor = np.exp(rng.normal(0, noise_std, size=len(idx)))

        raw = base_demand[idx] * noise_factor
        raw_sum = raw.sum()
        if raw_sum > 0:
            result[idx] = total_demand * raw / raw_sum
        else:
            result[idx] = total_demand / len(group)

    return result


# ════════════════════════════════════════════════════════════
# H3: Additive post-correction (in place of multiplicative)
# ════════════════════════════════════════════════════════════

def apply_additive_correction(base_demand, combined_factor, grid_gdf, region_sub):
    """Additive post-correction: demand_i = base_i + alpha * (factor_i - mean_factor), followed by re-normalization.

    alpha scales the additive offset so its magnitude is comparable to the multiplicative effect.
    """
    region_info = region_sub.set_index('ITL3')
    result = np.zeros_like(base_demand)

    for itl3, group in grid_gdf.groupby('ITL3'):
        if itl3 not in region_info.index:
            continue
        total_demand = region_info.loc[itl3, 'Demand (MVA)']
        idx = group.index

        factor_group = combined_factor[idx]
        base_group = base_demand[idx]

        # Additive offset: center the factor, then add it to base demand
        offset = factor_group - factor_group.mean()
        # Scale the offset so its std is comparable to that of base_demand
        base_std = base_group.std()
        offset_std = offset.std()
        if offset_std > 0 and base_std > 0:
            alpha = base_std / offset_std
        else:
            alpha = 0.0

        raw = base_group + alpha * offset
        raw = np.maximum(raw, 1e-6)  # Prevent negative values

        raw_sum = raw.sum()
        if raw_sum > 0:
            result[idx] = total_demand * raw / raw_sum
        else:
            result[idx] = total_demand / len(group)

    return result


# ════════════════════════════════════════════════════════════
# Standard multiplicative post-correction (control group)
# ════════════════════════════════════════════════════════════

def apply_standard_multiplicative(base_demand, combined_factor, grid_gdf, region_sub):
    """Standard multiplicative post-correction: demand_i = base_i * factor_i / sum(base_j * factor_j) * total."""
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


# ════════════════════════════════════════════════════════════
# Main pipeline
# ════════════════════════════════════════════════════════════

def get_fold_locations(seed):
    """Load fold splits from kfold_splits.json."""
    splits_path = EXP0_DIR / f'seed_{seed}' / 'baseline' / 'kfold_splits.json'
    if splits_path.exists():
        import json
        with open(splits_path) as f:
            splits = json.load(f)
        return splits
    # fallback: manual split
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    splits = {}
    indices = list(range(len(ALL_LOCATIONS)))
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(indices)):
        test_locs = [ALL_LOCATIONS[i] for i in test_idx]
        splits[f'fold{fold_idx + 1}'] = {'test': test_locs}
    return splits


def run_experiments():
    """Run all mechanism isolation experiments."""
    all_results = []
    n_random_repeats = 10  # number of random-noise repetitions

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")

        splits = get_fold_locations(seed)

        for fold_key, fold_info in splits.items():
            test_locs = fold_info['test'] if isinstance(fold_info, dict) else fold_info
            # JSON keys are fold_1, fold_2, etc. but dirs are fold1, fold2
            fold_dir_name = fold_key.replace('_', '')
            fold_dir = EXP0_DIR / f'seed_{seed}' / 'baseline' / fold_dir_name

            print(f"\n  {fold_key}: test locations = {test_locs}")

            for loc in test_locs:
                grid_demands_path = fold_dir / 'grid_demands' / f'{loc}_grid_demands.pickle'
                if not grid_demands_path.exists():
                    print(f"    [skip] {loc}: grid_demands not found")
                    continue

                # Load grid demands
                with open(grid_demands_path, 'rb') as f:
                    grid_demands = pickle.load(f)
                base_demand = grid_demands['gnn_demand']

                # Load grid and substations
                grid_gdf, region_sub, subs_sub, ntl_values = load_grid_and_subs(loc)
                prox_scores = compute_prox_scores(grid_gdf, subs_sub)

                # Compute factors
                ntl_factor, prox_factor = compute_factors(grid_gdf, ntl_values, prox_scores)
                combined_factor = ntl_factor * prox_factor  # NTL x Proximity combined

                # ── Control: uncorrected GNN ──
                m_baseline = voronoi_aggregate(grid_gdf, subs_sub, base_demand)
                all_results.append({
                    'seed': seed, 'fold': fold_key, 'location': loc,
                    'experiment': 'baseline_no_correction',
                    **m_baseline
                })

                # ── Control: standard multiplicative NTL+Prox post-correction (with re-normalization) ──
                standard_demand = apply_standard_multiplicative(
                    base_demand, combined_factor, grid_gdf, region_sub)
                m_standard = voronoi_aggregate(grid_gdf, subs_sub, standard_demand)
                all_results.append({
                    'seed': seed, 'fold': fold_key, 'location': loc,
                    'experiment': 'standard_multiplicative_NP',
                    **m_standard
                })

                # ── Control: standard multiplicative NTL-only post-correction ──
                ntl_only_demand = apply_standard_multiplicative(
                    base_demand, ntl_factor, grid_gdf, region_sub)
                m_ntl_only = voronoi_aggregate(grid_gdf, subs_sub, ntl_only_demand)
                all_results.append({
                    'seed': seed, 'fold': fold_key, 'location': loc,
                    'experiment': 'standard_multiplicative_N',
                    **m_ntl_only
                })

                # ── Control: standard multiplicative Proximity-only post-correction ──
                prox_only_demand = apply_standard_multiplicative(
                    base_demand, prox_factor, grid_gdf, region_sub)
                m_prox_only = voronoi_aggregate(grid_gdf, subs_sub, prox_only_demand)
                all_results.append({
                    'seed': seed, 'fold': fold_key, 'location': loc,
                    'experiment': 'standard_multiplicative_P',
                    **m_prox_only
                })

                # ── H1a: skip re-normalization ──
                no_renorm_demand = apply_correction_no_renorm(
                    base_demand, combined_factor, grid_gdf, region_sub)
                m_no_renorm = voronoi_aggregate(grid_gdf, subs_sub, no_renorm_demand)
                all_results.append({
                    'seed': seed, 'fold': fold_key, 'location': loc,
                    'experiment': 'H1a_no_renorm_NP',
                    **m_no_renorm
                })

                # H1a single-factor variant
                no_renorm_n = apply_correction_no_renorm(
                    base_demand, ntl_factor, grid_gdf, region_sub)
                m_no_renorm_n = voronoi_aggregate(grid_gdf, subs_sub, no_renorm_n)
                all_results.append({
                    'seed': seed, 'fold': fold_key, 'location': loc,
                    'experiment': 'H1a_no_renorm_N',
                    **m_no_renorm_n
                })

                no_renorm_p = apply_correction_no_renorm(
                    base_demand, prox_factor, grid_gdf, region_sub)
                m_no_renorm_p = voronoi_aggregate(grid_gdf, subs_sub, no_renorm_p)
                all_results.append({
                    'seed': seed, 'fold': fold_key, 'location': loc,
                    'experiment': 'H1a_no_renorm_P',
                    **m_no_renorm_p
                })

                # ── H1b: random-noise post-correction ──
                for rep in range(n_random_repeats):
                    rng = np.random.default_rng(seed * 1000 + rep)

                    # Match the variability of the real NTL+Prox combined factor
                    real_log_std = np.log(combined_factor[combined_factor > 0]).std()
                    noise_demand = apply_random_noise_correction(
                        base_demand, grid_gdf, region_sub, rng, noise_std=real_log_std)
                    m_noise = voronoi_aggregate(grid_gdf, subs_sub, noise_demand)
                    all_results.append({
                        'seed': seed, 'fold': fold_key, 'location': loc,
                        'experiment': f'H1b_random_noise_rep{rep}',
                        **m_noise
                    })

                # ── H3: additive post-correction ──
                additive_demand = apply_additive_correction(
                    base_demand, combined_factor, grid_gdf, region_sub)
                m_additive = voronoi_aggregate(grid_gdf, subs_sub, additive_demand)
                all_results.append({
                    'seed': seed, 'fold': fold_key, 'location': loc,
                    'experiment': 'H3_additive_NP',
                    **m_additive
                })

                # H3 single-factor variant
                additive_n = apply_additive_correction(
                    base_demand, ntl_factor, grid_gdf, region_sub)
                m_additive_n = voronoi_aggregate(grid_gdf, subs_sub, additive_n)
                all_results.append({
                    'seed': seed, 'fold': fold_key, 'location': loc,
                    'experiment': 'H3_additive_N',
                    **m_additive_n
                })

                additive_p = apply_additive_correction(
                    base_demand, prox_factor, grid_gdf, region_sub)
                m_additive_p = voronoi_aggregate(grid_gdf, subs_sub, additive_p)
                all_results.append({
                    'seed': seed, 'fold': fold_key, 'location': loc,
                    'experiment': 'H3_additive_P',
                    **m_additive_p
                })

                print(f"    {loc}: baseline={m_baseline['rmse']:.2f}, "
                      f"standard_NP={m_standard['rmse']:.2f}, "
                      f"no_renorm_NP={m_no_renorm['rmse']:.2f}, "
                      f"additive_NP={m_additive['rmse']:.2f}")

    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / 'mechanism_isolation_raw.csv', index=False)

    # ── Summary analysis ──
    print("\n" + "="*80)
    print("Summary analysis")
    print("="*80)

    # Aggregate by experiment type (consistent with the convention in Table 3: first average across seeds per location, then compute the ddof=0 standard deviation across the 16 locations)
    # Exclude individual random-noise draws; aggregate those separately below
    non_noise = results_df[~results_df['experiment'].str.contains('random_noise')]
    # Step 1: average over seeds for each (experiment, location) pair -> 16 location means per experiment
    loc_avg = non_noise.groupby(['experiment', 'location'])[['rmse', 'mae', 'corr']].mean()
    # Step 2: compute the population standard deviation (ddof=0) across the 16 locations
    summary = loc_avg.groupby('experiment').agg(
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', lambda x: x.std(ddof=0)),
        mae_mean=('mae', 'mean'),   mae_std=('mae', lambda x: x.std(ddof=0)),
        corr_mean=('corr', 'mean'), corr_std=('corr', lambda x: x.std(ddof=0)),
    )
    summary = summary.sort_values('rmse_mean')
    print("\nNon-random experiment summary (seed mean + ddof=0):")
    print(summary.to_string())

    # Random-noise summary (average over reps -> average over seeds -> ddof=0 across 16 locations)
    noise_rows = results_df[results_df['experiment'].str.contains('random_noise')]
    if len(noise_rows) > 0:
        noise_rows = noise_rows.copy()
        noise_rows['rep'] = noise_rows['experiment'].str.extract(r'rep(\d+)').astype(int)
        # For each (seed, location), first average over all reps
        noise_per_loc_seed = noise_rows.groupby(['seed', 'location'])[['rmse', 'mae', 'corr']].mean()
        # Then average over seeds to get 16 location means
        noise_per_loc = noise_per_loc_seed.groupby('location').mean()
        noise_mean = noise_per_loc.mean()
        noise_std = noise_per_loc.std(ddof=0)
        print(f"\nRandom-noise post-correction ({n_random_repeats}-run average, seed mean + ddof=0):")
        print(f"  RMSE: {noise_mean['rmse']:.4f} ± {noise_std['rmse']:.4f}")
        print(f"  MAE:  {noise_mean['mae']:.4f} ± {noise_std['mae']:.4f}")
        print(f"  Corr: {noise_mean['corr']:.4f} ± {noise_std['corr']:.4f}")

    summary.to_csv(OUTPUT_DIR / 'mechanism_isolation_summary.csv')
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    run_experiments()
