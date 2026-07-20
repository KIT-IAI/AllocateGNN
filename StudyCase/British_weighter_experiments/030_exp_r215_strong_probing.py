# -*- coding: utf-8 -*-
"""
030 - Strong probing for whether GNN embeddings encode NTL/Proximity
information (depends on the fusion-arm outputs from exp_r212_fusion)

The per-dimension maximum-correlation check used in 018 is not sufficient to
establish whether GNN embeddings encode NTL/Proximity information -- this
requires a **strong probe** (a nonlinear regression probe with held-out R^2).
A naive probe result is ambiguous: a low probe R^2 could mean either (a) the
probe itself is too weak, or (b) the encoder genuinely does not retain the
information (case (b) is the informative outcome for the question of whether
the encoder discards this signal). This is therefore implemented as a
four-rung **probe-strength ladder**, where probe validity is independently
corroborated only at rung (i); rungs (ii)-(iv) are recorded to disk without
any hard pass/fail assertion:

  (i)   Positive control (the only rung with a hard assertion): running the
        same probe on the fusion graph cache's 7-dim input features (which
        literally include ntl_feat/prox_feat) should give R^2 ~= 1, proving
        the probe pipeline itself works;
  (ii)  Fusion-arm embedding probe: exp_r212_fusion models (training may not
        be complete -- seeds are skipped and logged when incomplete; --arms
        lets you select which rungs to run);
  (iii) Baseline-arm embedding probe (the main question): the 48 exp0 models
        (3 seeds x 4 configs x 4 folds), loaded on CPU following the same
        approach as 018, 128-dim embeddings -> 2-layer MLP regression probe
        against the target;
  (iv)  Delta control: the same probe on the original 5-dim land-use
        features (exp0 graph-cache agent.x) -- delta = R^2(embedding) -
        R^2(raw features), i.e. "does the embedding exceed the level already
        implicit in the raw land-use input".

=== Probe targets (shared across all four rungs, comparable between them) ===
target in {ntl, prox}, computed as **log1p followed by per-region z-score**
(ddof=0, float64):
- log1p matches the log(1+clamp(x,0)) form used in NTLPriorLoss / Proximity
  PriorLoss (this ladder explicitly uses log1p(NTL)/log1p(prox));
- the per-region z-score is structurally identical to the fusion input-column
  transform used in 027 -> this makes rung (i)'s "literally includes the NTL
  column" claim exact (feature columns 5/6 are the target itself, differing
  only by float32 storage rounding); it also removes cross-region level
  differences that would otherwise contaminate pooled R^2 under the
  region-block split, keeping the four rungs comparable.
Raw prox scores use shared_correction_utils.compute_prox_scores (EPSG:27700,
gamma=2.0, 0.01 km clamp, the same convention as used in 017).

=== Embedding assembly (test-fold convention) ===
For each (seed, config), the embedding for region r is taken from the model
whose test fold contains r (i.e. a model that never saw r during training;
each region appears in exactly one test fold) -> each assembly gives full
coverage of all 16 regions. Loading follows the same approach as 018:
ModelConfig(hgt/256/128/3 layers, device='cpu') + init_model(train graphs) +
_load_checkpoint(), then a no_grad encoder forward pass to extract agent
embeddings.

=== Probe and splits ===
Probe = StandardScaler + 2-layer MLP (sklearn MLPRegressor, hidden=(64,),
adam, early_stopping, fixed random_state); the training set is
deterministically subsampled when it exceeds the cap (evaluation always uses
the full test set). Two splits are reported side by side:
- region_block (primary): GroupKFold(4) blocked by the 16 regions, so each
  region is held out exactly once -> per-region held-out R^2 (a random split
  would be inflated by spatial autocorrelation, so this is the primary
  convention);
- random (control): a pooled 80/20 split with a fixed seed.

=== (ii)/(iii) difference test (16 paired regions) ===
The fusion arm (ii) vs. the exp0 baseline config (iii) are compared
like-for-like (both use a pure land-use loss and differ only in input
features): per-region R^2 is first averaged over seeds, then the 16 paired
region-level differences are tested with
revision_statistics.exact_sign_flip_permutation (exact over 2^16) plus
paired_region_bootstrap (outer level only, B=1e4); the ntl/prox targets form
one family for Holm correction. When the fusion arm has no complete seed,
this test is recorded with status=skipped and a reason.

=== CKA corroboration ===
For each assembly (and the input features used in (i)/(iv)), linear CKA
against the target is computed; when the sample count exceeds the combined
feature/target dimensionality, the first CCA canonical correlation is also
attached (CCA is only meaningful once samples outnumber dimensions).

=== Determinism ===
Inference and probing run **entirely on CPU**; OMP/MKL thread counts = 4
(set before numpy is imported); torch threads = 4; torch/numpy/sklearn all
use fixed seeds; no DataLoader multiprocessing (batch_size=1, sequential).

=== Outputs (results/exp_r215/, 2 files) ===
  probe_results.csv   (rung, arm, config, seed, target, split_mode, region)
                      -> r2 / n_train / n_test; region includes a
                      '__pooled__' summary row
  probe_ladder.json   metadata + status and summary for all four rungs +
                      delta control + paired tests + CKA + skipped-run log
                      (conclusion fields are generated programmatically from
                      the results, never hand-written)

Runtime note: this script is entirely CPU-bound, and a full run takes **on
the order of hours (roughly 2-4 h)** -- each region has ~50k agents, and a
full run involves loading 48+ models, ~200 per-region forward passes, and
~170 probe fits (the 20k training-subsample cap bounds the cost of each fit)
-- **avoid running this alongside heavy CPU workloads while GPU training is
in progress**; before that, only --dry-run is intended (a single-model,
single-region smoke test that runs in minutes).

Usage:
    python 030_exp_r215_strong_probing.py --dry-run          # tiny end-to-end smoke test (a minute or less)
    python 030_exp_r215_strong_probing.py                    # full four-rung ladder
    python 030_exp_r215_strong_probing.py --arms positive_control raw_landuse
    python 030_exp_r215_strong_probing.py --arms baseline_emb --configs baseline ntl_prox
"""

import os
# CPU-only execution with capped thread counts; must be set before numpy is imported.
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # same guard as 018 (protects against Windows MKL double-loading)

import argparse
import json
import pickle
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# The default Windows console codepage (cp1252) cannot encode non-ASCII text --
# force UTF-8 (does not affect file outputs)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import shared_correction_utils as scu    # noqa: E402  (this module already inserts the repo root into sys.path)
import revision_statistics as rs         # noqa: E402

from SpatialAllocation.GNN.core.EdgeWeightSolver import EdgeWeightSolver  # noqa: E402
from SpatialAllocation.GNN.core.ModelConfig import ModelConfig            # noqa: E402

# ════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════

EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'
FUSION_DIR = SCRIPT_DIR / 'results' / 'exp_r212_fusion'
BASELINE_CACHE = EXP0_DIR / 'graph_cache' / 'cached_graphs.pickle'
FUSION_CACHE = FUSION_DIR / 'graph_cache' / 'cached_graphs.pickle'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp_r215'

ALL_LOCATIONS = scu.ALL_LOCATIONS                       # the 16 study regions (same as 017)
EXP0_SEEDS = [42, 123, 456]
FUSION_SEEDS = [42, 123, 456]                           # checked individually for availability; missing ones are logged
EXP0_CONFIGS = ['baseline', 'ntl', 'proximity', 'ntl_prox']

# objective_weights matches the CONFIG_MAP in 005 verbatim (it determines which loss
# modules init_model builds; the weight values themselves don't affect encoder
# loading, but keeping them identical to the source avoids ambiguity)
CONFIG_OBJECTIVES = {
    'baseline': {'landuse_prediction_loss': 1.0},
    'ntl': {'landuse_prediction_loss': 1.0, 'ntl_prior': 0.05},
    'proximity': {'landuse_prediction_loss': 1.0, 'proximity_prior': 0.05},
    'ntl_prox': {'landuse_prediction_loss': 1.0, 'ntl_prior': 0.05,
                 'proximity_prior': 0.05},
}
FUSION_CONFIG = 'baseline'                              # the fusion arm only has a baseline configuration

TARGETS = ['ntl', 'prox']
SPLIT_MODES = ['region_block', 'random']
ARM_CHOICES = ['positive_control', 'fusion_emb', 'baseline_emb', 'raw_landuse']
RUNG_OF_ARM = {
    'positive_control': 'i_positive_control',
    'fusion_emb': 'ii_fusion_embedding',
    'baseline_emb': 'iii_baseline_embedding',
    'raw_landuse': 'iv_raw_landuse',
}

N_REGION_FOLDS = 4                                      # number of folds for the region-blocked split
RANDOM_TEST_FRACTION = 0.2                              # random control split
PROBE_SEED = 42                                         # random source for the probe (sklearn/torch/numpy)
BOOT_SEED = 20260714                                    # random source for the paired bootstrap
BOOT_B = 10000
POSITIVE_CONTROL_R2_THRESHOLD = 0.99                    # the only hard-assertion threshold, used for rung (i)
TRAIN_SUBSAMPLE_CAP = 20000                             # cap on training samples per probe fit
MIN_REGION_TEST_N = 10                                  # minimum per-region sample count for R^2 under the random split

# 2-layer MLP probe (one hidden layer + one output layer = "2 layers")
MLP_PARAMS = dict(
    hidden_layer_sizes=(64,), activation='relu', solver='adam',
    batch_size=256, max_iter=500, early_stopping=True,
    n_iter_no_change=15, validation_fraction=0.1, tol=1e-6,
)

CSV_COLUMNS = ['rung', 'arm', 'config', 'seed', 'target', 'split_mode',
               'region', 'r2', 'n_train', 'n_test']


# ════════════════════════════════════════════════════════════
# Targets and features
# ════════════════════════════════════════════════════════════

def _zscore_log1p_f64(values, label: str) -> np.ndarray:
    """log1p + within-region z-score (ddof=0, float64) -- structurally identical to
    the fusion column transform used in 027.

    Raises immediately on zero or invalid variance, rather than continuing silently.
    """
    arr = np.log1p(np.clip(np.asarray(values, dtype=np.float64), 0.0, None))
    std = arr.std()   # numpy's default is ddof=0
    if not np.isfinite(std) or std <= 0:
        raise ValueError(f'Probe target has zero or invalid within-region variance ({label}); refusing to continue silently')
    return (arr - arr.mean()) / std


def load_cache(cache_path: Path, expect_dim: int) -> dict:
    """Read-only load of a graph cache, asserting the agent feature dimensionality
    (guards against the exp0 and fusion caches being pointed at each other by mistake)."""
    if not cache_path.exists():
        raise FileNotFoundError(f'Graph cache not found: {cache_path}')
    print(f'Loading graph cache (read-only): {cache_path}')
    with open(cache_path, 'rb') as f:
        cached = pickle.load(f)
    for loc in ALL_LOCATIONS:
        dim = cached['graphs'][loc]['agent'].x.shape[1]
        assert dim == expect_dim, (
            f'{loc}: agent feature dim {dim} != {expect_dim} '
            f'(cache {cache_path.name} is misdirected or corrupted)')
    return cached


def compute_targets(cache: dict, locations) -> dict:
    """Compute probe targets from a cache, in graph agent order: {loc: {'ntl': arr, 'prox': arr}}.

    Raw values come from the cache's ntl_dict / grids+subs (the same source used to
    build that cache's graphs), then go through log1p + per-region z-score and are
    mapped into graph agent order via agent_index_map.
    """
    targets = {}
    for loc in locations:
        grid_gdf = cache['grids'][loc][0]
        subs_sub = cache['subs_dict'][loc]
        ntl = np.asarray(cache['ntl_dict'][loc], dtype=np.float64)
        prox = scu.compute_prox_scores(grid_gdf, subs_sub)
        agent_orig = np.asarray(cache['graphs'][loc].agent_index_map.values)
        targets[loc] = {
            'ntl': _zscore_log1p_f64(ntl, f'{loc}/ntl')[agent_orig],
            'prox': _zscore_log1p_f64(prox, f'{loc}/prox')[agent_orig],
        }
    return targets


def graph_input_features(cache: dict, loc: str) -> np.ndarray:
    """Agent input feature matrix from the graph cache (float64, graph agent order)."""
    return cache['graphs'][loc]['agent'].x.cpu().numpy().astype(np.float64)


# ════════════════════════════════════════════════════════════
# Embedding assembly (CPU loading following 018's approach, test-fold convention)
# ════════════════════════════════════════════════════════════

def _load_solver_cpu(model_path: Path, train_graphs: list, objective_weights: dict):
    """Build the model structure on CPU and load a checkpoint, following the same
    approach as 018 (deterministic inference)."""
    config = ModelConfig(
        hidden_dim=256,
        embedding_dim=128,
        num_layers=3,
        conv_type='hgt',
        allocation_temperature_start=0.01,
        learnable=False,
        save_path=str(model_path),
        device='cpu',
    )
    solver = EdgeWeightSolver(config)
    train_dl = DataLoader(train_graphs, batch_size=1, shuffle=False)
    solver.init_model(train_dl, objective_weights)
    solver._load_checkpoint()
    return solver


def check_assembly(exp_dir: Path, seed: int, config: str):
    """Check whether the test-fold assembly for (seed, config) is complete.
    Returns (splits, None) on success or (None, reason) on failure."""
    seed_dir = exp_dir / f'seed_{seed}' / config
    splits_path = seed_dir / 'kfold_splits.json'
    if not splits_path.exists():
        return None, f'kfold_splits.json not found: {splits_path}'
    with open(splits_path, encoding='utf-8') as f:
        splits = json.load(f)
    missing = []
    for fold_key in splits:
        fold_dir_name = fold_key.replace('_', '')       # 'fold_1' -> 'fold1', same convention as 018
        if not (seed_dir / fold_dir_name / 'model.pth').exists():
            missing.append(fold_key)
    if missing:
        return None, f'Missing models ({seed_dir}): {missing}'
    covered = sorted(loc for info in splits.values() for loc in info['test'])
    if covered != sorted(ALL_LOCATIONS):
        return None, f'Test folds do not cover all 16 regions ({seed_dir})'
    return splits, None


def extract_test_fold_embeddings(exp_dir: Path, seed: int, config: str,
                                 cache: dict, objective_weights: dict,
                                 only_locs=None) -> dict:
    """test-fold assembly: the embedding for region r is taken from the model whose
    test fold contains r (a model that never saw r during training).

    only_locs: only extract the given regions (used by dry-run); None = all 16 regions.
    Returns {loc: (n_agents, 128) float64}.
    """
    splits, reason = check_assembly(exp_dir, seed, config)
    if reason is not None:
        raise FileNotFoundError(reason)

    graphs = cache['graphs']
    embeddings = {}
    for fold_key, fold_info in splits.items():
        wanted = [loc for loc in fold_info['test']
                  if only_locs is None or loc in only_locs]
        if not wanted:
            continue
        fold_dir_name = fold_key.replace('_', '')
        model_path = exp_dir / f'seed_{seed}' / config / fold_dir_name / 'model.pth'
        train_graphs = [graphs[loc] for loc in fold_info['train']]
        solver = _load_solver_cpu(model_path, train_graphs, objective_weights)
        solver.encoder.eval()
        with torch.no_grad():
            for loc in wanted:
                graph = graphs[loc]
                emb = solver.encoder(graph.x_dict, graph.edge_index_dict)['agent']
                embeddings[loc] = emb.cpu().numpy().astype(np.float64)
    return embeddings


# ════════════════════════════════════════════════════════════
# Probe
# ════════════════════════════════════════════════════════════

def make_probe(random_state: int = PROBE_SEED, max_iter: int = None) -> Pipeline:
    """StandardScaler + 2-layer MLP regression probe (fixed random source)."""
    params = dict(MLP_PARAMS)
    params['random_state'] = random_state
    if max_iter is not None:
        params['max_iter'] = max_iter
    return Pipeline([('scaler', StandardScaler()),
                     ('mlp', MLPRegressor(**params))])


def _subsample_train(tr_idx: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic subsampling of the training set when it exceeds the cap; the
    evaluation set is never subsampled."""
    if len(tr_idx) <= cap:
        return tr_idx
    return rng.choice(tr_idx, size=cap, replace=False)


def run_probe_suite(features_by_loc: dict, targets_by_loc: dict, locations: list,
                    target_name: str, max_iter: int = None,
                    subsample_cap: int = TRAIN_SUBSAMPLE_CAP) -> list:
    """Run the probe under both splits for one (features, target) pair, returning a
    list of per-region + pooled row dicts.

    region_block (primary): GroupKFold(4) blocked by region, each region held out
    exactly once; out-of-fold predictions are assembled into per-region and pooled
    held-out R^2.
    random (control): a pooled 80/20 split with a fixed seed; per-region R^2 is
    computed within the test subset (regions with fewer than MIN_REGION_TEST_N
    samples are skipped, which in practice never happens since every region has
    thousands of agents).
    """
    X = np.vstack([features_by_loc[loc] for loc in locations])
    y = np.concatenate([targets_by_loc[loc][target_name] for loc in locations])
    groups = np.concatenate([np.full(len(features_by_loc[loc]), gi)
                             for gi, loc in enumerate(locations)])
    assert len(X) == len(y) == len(groups)
    rng = np.random.default_rng(PROBE_SEED)

    rows = []

    # -- region_block (primary convention) --
    gkf = GroupKFold(n_splits=N_REGION_FOLDS)
    oof_pred = np.full(len(y), np.nan)
    n_train_of_region = {}
    for tr_idx, te_idx in gkf.split(X, y, groups):
        tr_used = _subsample_train(tr_idx, subsample_cap, rng)
        probe = make_probe(PROBE_SEED, max_iter)
        probe.fit(X[tr_used], y[tr_used])
        oof_pred[te_idx] = probe.predict(X[te_idx])
        for gi in np.unique(groups[te_idx]):
            n_train_of_region[int(gi)] = len(tr_used)
    assert not np.isnan(oof_pred).any(), 'GroupKFold did not cover all samples'

    for gi, loc in enumerate(locations):
        m = groups == gi
        rows.append({'split_mode': 'region_block', 'region': loc,
                     'r2': float(r2_score(y[m], oof_pred[m])),
                     'n_train': int(n_train_of_region[gi]), 'n_test': int(m.sum())})
    rows.append({'split_mode': 'region_block', 'region': '__pooled__',
                 'r2': float(r2_score(y, oof_pred)),
                 'n_train': -1, 'n_test': int(len(y))})

    # -- random (control convention) --
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=RANDOM_TEST_FRACTION,
                                      random_state=PROBE_SEED, shuffle=True)
    tr_used = _subsample_train(tr_idx, subsample_cap, rng)
    probe = make_probe(PROBE_SEED, max_iter)
    probe.fit(X[tr_used], y[tr_used])
    pred = probe.predict(X[te_idx])

    for gi, loc in enumerate(locations):
        m = groups[te_idx] == gi
        if m.sum() < MIN_REGION_TEST_N:
            continue
        rows.append({'split_mode': 'random', 'region': loc,
                     'r2': float(r2_score(y[te_idx][m], pred[m])),
                     'n_train': int(len(tr_used)), 'n_test': int(m.sum())})
    rows.append({'split_mode': 'random', 'region': '__pooled__',
                 'r2': float(r2_score(y[te_idx], pred)),
                 'n_train': int(len(tr_used)), 'n_test': int(len(te_idx))})

    return rows


# ════════════════════════════════════════════════════════════
# CKA / CCA corroboration
# ════════════════════════════════════════════════════════════

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Frobenius inner product normalization after centering the features)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Xc.T @ Yc, 'fro') ** 2
    den = (np.linalg.norm(Xc.T @ Xc, 'fro') * np.linalg.norm(Yc.T @ Yc, 'fro'))
    if den == 0:
        raise ValueError('CKA denominator is zero (feature or target has zero variance)')
    return float(num / den)


def first_canonical_corr(X: np.ndarray, Y: np.ndarray):
    """First CCA canonical correlation; returns None when the sample count is less
    than or equal to the combined feature/target dimensionality (CCA is not
    meaningful in that regime)."""
    if X.shape[0] <= X.shape[1] + Y.shape[1]:
        return None
    cca = CCA(n_components=1, max_iter=1000)
    Xt, Yt = cca.fit_transform(X, Y)
    return float(np.corrcoef(Xt[:, 0], np.asarray(Yt).reshape(-1))[0, 1])


def cka_block(features_by_loc: dict, targets_by_loc: dict, locations: list) -> dict:
    """Compute the CKA/CCA corroboration block for one assembly (pooled over all agents)."""
    X = np.vstack([features_by_loc[loc] for loc in locations])
    Y = np.column_stack([
        np.concatenate([targets_by_loc[loc]['ntl'] for loc in locations]),
        np.concatenate([targets_by_loc[loc]['prox'] for loc in locations]),
    ])
    return {
        'cka_ntl': linear_cka(X, Y[:, [0]]),
        'cka_prox': linear_cka(X, Y[:, [1]]),
        'cka_joint': linear_cka(X, Y),
        'cca_first_corr_joint': first_canonical_corr(X, Y),
        'n_samples': int(len(X)),
        'feature_dim': int(X.shape[1]),
    }


# ════════════════════════════════════════════════════════════
# Aggregation and testing
# ════════════════════════════════════════════════════════════

def _annotate(rows: list, rung: str, arm: str, config: str, seed: int, target: str) -> list:
    """Add identifying columns to the rows returned by run_probe_suite."""
    return [{'rung': rung, 'arm': arm, 'config': config, 'seed': seed,
             'target': target, **r} for r in rows]


def _r2_summary(df: pd.DataFrame, mask) -> dict:
    """Summarize pooled and per-region statistics by (target, split_mode); these
    fields are recorded, not asserted on."""
    out = {}
    sub_all = df[mask]
    for target in sorted(sub_all['target'].unique()):
        out[target] = {}
        for split in sorted(sub_all['split_mode'].unique()):
            sub = sub_all[(sub_all['target'] == target) & (sub_all['split_mode'] == split)]
            pooled = sub[sub['region'] == '__pooled__']['r2']
            per_region = sub[sub['region'] != '__pooled__']['r2']
            out[target][split] = {
                'pooled_mean': float(pooled.mean()) if len(pooled) else None,
                'pooled_min': float(pooled.min()) if len(pooled) else None,
                'per_region_mean': float(per_region.mean()) if len(per_region) else None,
                'per_region_min': float(per_region.min()) if len(per_region) else None,
                'per_region_max': float(per_region.max()) if len(per_region) else None,
                'n_rows': int(len(sub)),
            }
    return out


def seed_avg_per_region(df: pd.DataFrame, arm: str, config: str,
                        target: str, split_mode: str):
    """Seed-averaged per-region R^2 (returns a Series only when all 16 regions are
    present, otherwise None)."""
    sub = df[(df['arm'] == arm) & (df['config'] == config)
             & (df['target'] == target) & (df['split_mode'] == split_mode)
             & (df['region'] != '__pooled__')]
    if sub.empty:
        return None
    piv = sub.groupby('region')['r2'].mean()
    if set(piv.index) != set(ALL_LOCATIONS):
        return None
    return piv.loc[ALL_LOCATIONS]


def paired_region_test(piv_a: pd.Series, piv_b: pd.Series) -> dict:
    """Paired 16-region difference test: exact sign-flip permutation plus an outer
    paired bootstrap CI."""
    diffs = (piv_a - piv_b).to_numpy(dtype=float)
    perm = rs.exact_sign_flip_permutation(diffs)
    boot = rs.paired_region_bootstrap(diffs, B=BOOT_B, seed=BOOT_SEED)
    return {
        'status': 'done',
        'n_regions': int(len(diffs)),
        'mean_diff': float(diffs.mean()),
        'p_sign_flip_two_sided': perm['p'],
        'min_attainable_p': perm['min_attainable_p'],
        't_obs': perm['t_obs'],
        'boot_ci_lo': boot['ci_lo'],
        'boot_ci_hi': boot['ci_hi'],
        'boot_B': boot['B'],
        'per_region_diff': {loc: float(d) for loc, d in
                            zip(ALL_LOCATIONS, diffs)},
    }


# ════════════════════════════════════════════════════════════
# Main pipeline
# ════════════════════════════════════════════════════════════

def run_full(arms: list, exp0_configs: list, exp0_seeds: list):
    """Run the full four-rung ladder (outputs are written to results/exp_r215/)."""
    t0 = time.time()
    torch.manual_seed(PROBE_SEED)
    torch.set_num_threads(4)
    np.random.seed(PROBE_SEED)

    all_rows = []
    skipped = []
    cka = {}
    ladder = {}

    need_fusion_cache = bool({'positive_control', 'fusion_emb'} & set(arms))
    need_baseline_cache = bool({'baseline_emb', 'raw_landuse'} & set(arms))

    fusion_cache = fusion_targets = None
    baseline_cache = baseline_targets = None
    if need_fusion_cache:
        fusion_cache = load_cache(FUSION_CACHE, expect_dim=7)
        fusion_targets = compute_targets(fusion_cache, ALL_LOCATIONS)
    if need_baseline_cache:
        baseline_cache = load_cache(BASELINE_CACHE, expect_dim=5)
        baseline_targets = compute_targets(baseline_cache, ALL_LOCATIONS)

    # -- (i) positive control: fusion's 7-dim input features (literally includes ntl_feat/prox_feat) --
    if 'positive_control' in arms:
        print('\n' + '=' * 60)
        print('(i) Positive control: probe on fusion input features (the only rung with a hard assertion)')
        print('=' * 60)
        feats = {loc: graph_input_features(fusion_cache, loc) for loc in ALL_LOCATIONS}
        pooled_vals = {}
        for target in TARGETS:
            rows = run_probe_suite(feats, fusion_targets, ALL_LOCATIONS, target)
            all_rows += _annotate(rows, RUNG_OF_ARM['positive_control'],
                                  'positive_control', 'input', -1, target)
            pooled_vals[target] = {r['split_mode']: r['r2'] for r in rows
                                   if r['region'] == '__pooled__'}
            print(f'  target={target}: pooled R^2 = {pooled_vals[target]}')
        cka['positive_control/input'] = cka_block(feats, fusion_targets, ALL_LOCATIONS)
        all_pooled = [v for t in pooled_vals.values() for v in t.values()]
        ladder['rung_i_positive_control'] = {
            'status': 'done',
            'threshold': POSITIVE_CONTROL_R2_THRESHOLD,
            'pooled_r2': pooled_vals,
            'all_above_threshold': bool(all(v > POSITIVE_CONTROL_R2_THRESHOLD
                                            for v in all_pooled)),
        }
    else:
        ladder['rung_i_positive_control'] = {'status': 'not_run_in_this_invocation'}

    # -- (iv) delta-control baseline: raw 5-dim land-use features --
    if 'raw_landuse' in arms:
        print('\n' + '=' * 60)
        print('(iv) Delta control: probe on raw 5-dim land-use features')
        print('=' * 60)
        feats = {loc: graph_input_features(baseline_cache, loc) for loc in ALL_LOCATIONS}
        for target in TARGETS:
            rows = run_probe_suite(feats, baseline_targets, ALL_LOCATIONS, target)
            all_rows += _annotate(rows, RUNG_OF_ARM['raw_landuse'],
                                  'raw_landuse', 'input', -1, target)
        cka['raw_landuse/input'] = cka_block(feats, baseline_targets, ALL_LOCATIONS)
        ladder['rung_iv_raw_landuse'] = {'status': 'done'}
    else:
        ladder['rung_iv_raw_landuse'] = {'status': 'not_run_in_this_invocation'}

    # -- (iii) baseline-arm embedding probe (the main question: 48 exp0 models, loaded on CPU following 018's approach) --
    if 'baseline_emb' in arms:
        print('\n' + '=' * 60)
        print('(iii) Baseline-arm embedding probe (exp0)')
        print('=' * 60)
        done_assemblies = []
        for config in exp0_configs:
            for seed in exp0_seeds:
                key = f'exp0/{config}/seed_{seed}'
                splits, reason = check_assembly(EXP0_DIR, seed, config)
                if reason is not None:
                    skipped.append({'rung': 'iii', 'assembly': key, 'reason': reason})
                    print(f'  [skipped] {key}: {reason}')
                    continue
                print(f'  Assembling {key} ...')
                emb = extract_test_fold_embeddings(
                    EXP0_DIR, seed, config, baseline_cache, CONFIG_OBJECTIVES[config])
                for target in TARGETS:
                    rows = run_probe_suite(emb, baseline_targets, ALL_LOCATIONS, target)
                    all_rows += _annotate(rows, RUNG_OF_ARM['baseline_emb'],
                                          'baseline_emb', config, seed, target)
                cka[f'baseline_emb/{config}/seed_{seed}'] = cka_block(
                    emb, baseline_targets, ALL_LOCATIONS)
                done_assemblies.append(key)
        ladder['rung_iii_baseline_embedding'] = {
            'status': 'done' if done_assemblies else 'skipped_missing_artifacts',
            'assemblies_done': done_assemblies,
            'n_assemblies_expected': len(exp0_configs) * len(exp0_seeds),
        }
    else:
        ladder['rung_iii_baseline_embedding'] = {'status': 'not_run_in_this_invocation'}

    # -- (ii) fusion-arm embedding probe (exp_r212_fusion outputs, training may not be complete -- missing outputs are skipped and logged) --
    if 'fusion_emb' in arms:
        print('\n' + '=' * 60)
        print('(ii) Fusion-arm embedding probe (exp_r212_fusion)')
        print('=' * 60)
        seeds_used, seeds_skipped = [], {}
        for seed in FUSION_SEEDS:
            key = f'fusion/{FUSION_CONFIG}/seed_{seed}'
            splits, reason = check_assembly(FUSION_DIR, seed, FUSION_CONFIG)
            if reason is not None:
                seeds_skipped[str(seed)] = reason
                skipped.append({'rung': 'ii', 'assembly': key, 'reason': reason})
                print(f'  [skipped] {key}: {reason}')
                continue
            print(f'  Assembling {key} ...')
            emb = extract_test_fold_embeddings(
                FUSION_DIR, seed, FUSION_CONFIG, fusion_cache,
                CONFIG_OBJECTIVES[FUSION_CONFIG])
            for target in TARGETS:
                rows = run_probe_suite(emb, fusion_targets, ALL_LOCATIONS, target)
                all_rows += _annotate(rows, RUNG_OF_ARM['fusion_emb'],
                                      'fusion_emb', FUSION_CONFIG, seed, target)
            cka[f'fusion_emb/{FUSION_CONFIG}/seed_{seed}'] = cka_block(
                emb, fusion_targets, ALL_LOCATIONS)
            seeds_used.append(seed)
        ladder['rung_ii_fusion_embedding'] = {
            'status': ('done' if len(seeds_used) == len(FUSION_SEEDS)
                       else 'partial' if seeds_used else 'skipped_missing_artifacts'),
            'seeds_used': seeds_used,
            'seeds_skipped': seeds_skipped,
        }
    else:
        ladder['rung_ii_fusion_embedding'] = {'status': 'not_run_in_this_invocation'}

    # ════════════════════════════════════════════════════════
    # Aggregation, delta control, and paired tests (all fields below are
    # generated programmatically from the results, never hand-written)
    # ════════════════════════════════════════════════════════

    df = pd.DataFrame(all_rows, columns=CSV_COLUMNS)

    # per-rung summary fields ((ii)-(iv) carry no hard assertion) -- ladder key = 'rung_' + rung id
    for arm_name, rung_id in RUNG_OF_ARM.items():
        entry = ladder[f'rung_{rung_id}']
        if entry.get('status') in ('done', 'partial'):
            entry['r2_summary'] = _r2_summary(df, df['arm'] == arm_name)

    # Delta control: R^2(embedding) - R^2(raw 5-dim features), per-region difference after seed-averaging
    delta_control = {}
    if 'raw_landuse' in arms:
        for target in TARGETS:
            for split in SPLIT_MODES:
                piv_raw = seed_avg_per_region(df, 'raw_landuse', 'input', target, split)
                if piv_raw is None:
                    continue
                # deltas for each (iii) config, plus the (ii) fusion delta
                for arm_name, configs in (('baseline_emb', exp0_configs),
                                          ('fusion_emb', [FUSION_CONFIG])):
                    for config in configs:
                        piv_emb = seed_avg_per_region(df, arm_name, config, target, split)
                        if piv_emb is None:
                            continue
                        diffs = (piv_emb - piv_raw).to_numpy(dtype=float)
                        key = f'{arm_name}/{config}'
                        delta_control.setdefault(key, {}).setdefault(target, {})[split] = {
                            'delta_region_mean': float(diffs.mean()),
                            'delta_region_min': float(diffs.min()),
                            'delta_region_max': float(diffs.max()),
                            'sign_flip_p_recorded_only':
                                rs.exact_sign_flip_permutation(diffs)['p'],
                        }
    ladder['delta_control'] = delta_control

    # (ii)/(iii) difference test: fusion vs. exp0-baseline (like-for-like), 16 paired regions
    paired_tests = {'fusion_vs_baseline_emb': {}}
    for split in SPLIT_MODES:
        block = {}
        raw_ps = {}
        for target in TARGETS:
            piv_f = seed_avg_per_region(df, 'fusion_emb', FUSION_CONFIG, target, split)
            piv_b = seed_avg_per_region(df, 'baseline_emb', 'baseline', target, split)
            if piv_f is None or piv_b is None:
                block[target] = {
                    'status': 'skipped',
                    'reason': ('no complete seed assembly for the fusion arm' if piv_f is None
                               else 'exp0 baseline assembly missing'),
                }
                continue
            block[target] = paired_region_test(piv_f, piv_b)
            block[target]['n_fusion_seeds'] = len(
                ladder['rung_ii_fusion_embedding'].get('seeds_used', []))
            raw_ps[target] = block[target]['p_sign_flip_two_sided']
        if raw_ps:
            adj = rs.holm(raw_ps)
            for target, p_adj in adj.items():
                block[target]['p_holm_within_split'] = p_adj
        paired_tests['fusion_vs_baseline_emb'][split] = block
    ladder['paired_tests'] = paired_tests

    ladder['cka'] = cka
    ladder['skipped'] = skipped
    ladder['meta'] = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'script': '030_exp_r215_strong_probing.py',
        'full_run': set(arms) == set(ARM_CHOICES),
        'arms': list(arms),
        'exp0_configs': list(exp0_configs),
        'exp0_seeds': list(exp0_seeds),
        'fusion_seeds_scanned': list(FUSION_SEEDS),
        'targets': TARGETS,
        'target_transform': 'log1p + per-region z-score (ddof=0, float64)',
        'split_modes': SPLIT_MODES,
        'primary_split_mode': 'region_block',
        'n_region_folds': N_REGION_FOLDS,
        'random_test_fraction': RANDOM_TEST_FRACTION,
        'probe': {'kind': 'StandardScaler + MLPRegressor', **{
            k: (list(v) if isinstance(v, tuple) else v) for k, v in MLP_PARAMS.items()}},
        'train_subsample_cap': TRAIN_SUBSAMPLE_CAP,
        'probe_seed': PROBE_SEED,
        'boot_seed': BOOT_SEED,
        'boot_B': BOOT_B,
        'cpu_only': True,
        'torch_num_threads': 4,
        'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
        'versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'torch': torch.__version__,
        },
    }

    # -- write outputs --
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / 'probe_results.csv', index=False)
    with open(OUTPUT_DIR / 'probe_ladder.json', 'w', encoding='utf-8') as f:
        json.dump(ladder, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 60)
    print(f'Done: {len(df)} probe result rows -> {OUTPUT_DIR / "probe_results.csv"}')
    print(f'Ladder summary -> {OUTPUT_DIR / "probe_ladder.json"}')
    print(f'Elapsed {time.time() - t0:.0f} s | {len(skipped)} skipped entries logged')
    print('=' * 60)


# ════════════════════════════════════════════════════════════
# dry-run: single-model, single-region smoke test (runs in minutes, writes no outputs)
# ════════════════════════════════════════════════════════════

def run_dry_run():
    """Smoke test: load one exp0 model -> forward pass on one region -> subsampled
    probe -> self-check the statistics utilities. Writes no files and does not load
    the fusion cache (only checks its presence on disk)."""
    t0 = time.time()
    torch.manual_seed(PROBE_SEED)
    torch.set_num_threads(4)

    print('=' * 60)
    print('--dry-run: single-model, single-region smoke test (writes no outputs)')
    print('=' * 60)

    # 1. Presence check (stat only)
    print(f'exp0 graph cache present: {BASELINE_CACHE.exists()} ({BASELINE_CACHE})')
    print(f'fusion graph cache present: {FUSION_CACHE.exists()} ({FUSION_CACHE})')
    n_exp0_models = len(list(EXP0_DIR.glob('seed_*/*/fold*/model.pth')))
    n_fusion_models = len(list(FUSION_DIR.glob('seed_*/*/fold*/model.pth')))
    print(f'exp0 models present: {n_exp0_models}/48 | fusion models present: {n_fusion_models}/12')
    for seed in FUSION_SEEDS:
        _, reason = check_assembly(FUSION_DIR, seed, FUSION_CONFIG)
        status = 'OK (assemblable)' if reason is None else f'skipped and logged: {reason}'
        print(f'  fusion seed_{seed}: {status}')

    # 2. Load the exp0 cache + a single-model, single-region forward pass
    cache = load_cache(BASELINE_CACHE, expect_dim=5)
    seed, config = 42, 'baseline'
    splits, reason = check_assembly(EXP0_DIR, seed, config)
    assert reason is None, f'exp0 {config}/seed_{seed} assembly incomplete: {reason}'
    fold_key = 'fold_1'
    # pick the fold-1 test region with the fewest agents, to bound the forward-pass cost
    test_locs = splits[fold_key]['test']
    loc = min(test_locs, key=lambda l: cache['graphs'][l]['agent'].num_nodes)
    print(f'\nSmoke test: exp0/{config}/seed_{seed}/{fold_key} -> region {loc} '
          f'({cache["graphs"][loc]["agent"].num_nodes} agents)')

    emb = extract_test_fold_embeddings(
        EXP0_DIR, seed, config, cache, CONFIG_OBJECTIVES[config], only_locs=[loc])
    assert loc in emb and emb[loc].shape[1] == 128, 'Embedding extraction failed'
    print(f'Embedding extraction OK: {emb[loc].shape}')

    # 3. Target + subsampled smoke probe (80/20 within the single region, max_iter capped)
    targets = compute_targets(cache, [loc])
    X, y = emb[loc], targets[loc]['ntl']
    rng = np.random.default_rng(PROBE_SEED)
    if len(X) > 2000:
        pick = rng.choice(len(X), size=2000, replace=False)
        X, y = X[pick], y[pick]
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=PROBE_SEED)
    probe = make_probe(PROBE_SEED, max_iter=50)
    probe.fit(X[tr], y[tr])
    r2 = r2_score(y[te], probe.predict(X[te]))
    print(f'Smoke probe OK: emb->ntl held-out R^2 = {r2:.4f} '
          f'(subsampled + truncated iterations; not representative of the full run)')

    # 4. CKA + statistics-utility self-check
    c = linear_cka(X, y[:, None])
    print(f'linear CKA OK: {c:.4f}')
    synth = rng.normal(0.02, 0.05, size=16)
    perm = rs.exact_sign_flip_permutation(synth)
    boot = rs.paired_region_bootstrap(synth, B=1000, seed=BOOT_SEED)
    print(f'revision_statistics OK: sign-flip p={perm["p"]:.4f} '
          f'(min={perm["min_attainable_p"]:.2e}), boot CI=[{boot["ci_lo"]:.4f}, {boot["ci_hi"]:.4f}]')

    print(f'\n[dry-run passed] End-to-end path verified in {time.time() - t0:.0f} s; no outputs written.')


# ════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Four-rung strong-probe ladder for GNN embeddings -- CPU-only, outputs to results/exp_r215/')
    parser.add_argument('--arms', type=str, nargs='+', choices=ARM_CHOICES,
                        default=ARM_CHOICES,
                        help='Which ladder rungs to run (default: all four; the fusion arm is '
                             'skipped and logged automatically if its outputs are missing)')
    parser.add_argument('--configs', type=str, nargs='+', choices=EXP0_CONFIGS,
                        default=EXP0_CONFIGS,
                        help='Subset of exp0 configs for rung (iii) (default: all 4 configs = 48 models)')
    parser.add_argument('--seeds', type=int, nargs='+', default=EXP0_SEEDS,
                        help=f'Subset of exp0 seeds for rung (iii) (default: {EXP0_SEEDS})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Single-model, single-region smoke test (runs in minutes, writes no '
                             'outputs) -- the only mode intended for use before a full run')
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
        sys.exit(0)

    run_full(args.arms, args.configs, args.seeds)
