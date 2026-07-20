# -*- coding: utf-8 -*-
"""Precondition check: one-shot validation of frozen input artifacts.

Checks that every read-only input artifact required by the downstream
experiments is present and matches its expected reference values, writing
the result to ``results/exp_r_precheck/precheck.json``. Each check item is
represented as {expected, actual, status}.

Status values:
- ``ok``                  -- matches the reference value;
- ``missing_registered``  -- expected to be missing, and is missing (e.g.
                              the Germany case-study artifacts, which were
                              intentionally removed from scope);
- ``known_incomplete``    -- a known, intentionally incomplete state is
                              recorded as-is (exp0_kfold_prior_0_01);
- ``not_found``           -- unexpectedly missing (flagged by tests);
- ``mismatch``            -- count/value does not match the reference
                              (flagged by tests);
- ``error``               -- the check itself raised an exception (flagged
                              by tests).

Usage::

    python precheck_frozen_inputs.py

Corresponding test: ``tests/test_precheck.py`` (asserts every status is
within the allowed set).

This script performs read-only checks and does not modify any experiment
files. Counting always uses a whitelist of exact file names, which
naturally ignores stray manually-created copies (e.g. files with
"- copy" suffixes).
"""

from __future__ import annotations

import gc
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ════════════════════════════════════════════════════════════
# Path constants (aligned with 005_kfold_prior_training.py)
# ════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent                      # British_weighter_experiments
RESULTS_DIR = SCRIPT_DIR / 'results'
DATA_DIR = RESULTS_DIR / 'intermediate'
ASSEMBLED_DIR = DATA_DIR / 'features' / 'assembled'
EXTRACTED_DIR = DATA_DIR / 'features' / 'extracted'
EXP0_DIR = RESULTS_DIR / 'exp0_kfold_prior'
EXP0_001_DIR = RESULTS_DIR / 'exp0_kfold_prior_0_01'
AGENTCONN_DIR = RESULTS_DIR / 'exp0_kfold_prior_AgentConn'
STATIC_DIR = RESULTS_DIR / 'static_allocation'
STATIC_ND_DIR = RESULTS_DIR / 'static_allocation_nd'
GERMANY_RESULTS_DIR = SCRIPT_DIR.parent / 'Germany' / 'results'   # expected to be missing
OUTPUT_DIR = RESULTS_DIR / 'exp_r_precheck'
OUTPUT_JSON = OUTPUT_DIR / 'precheck.json'

# 16 study regions (must match 005's ALL_LOCATIONS; not imported directly
# from 005, since importing that module triggers heavyweight module-level
# initialisation)
ALL_LOCATIONS = [
    'London',
    'TLH2', 'TLH3', 'TLJ1', 'TLF1', 'TLF2',
    'TLC1', 'TLC2', 'TLD6', 'TLG1', 'TLG2', 'TLE4',
    'TLH1', 'TLE3', 'TLD3', 'TLD4',
]

SEEDS = [42, 123, 456]
CONFIGS = ['baseline', 'ntl', 'proximity', 'ntl_prox']
FOLDS = [1, 2, 3, 4]
METRICS = ['rmse', 'mae', 'corr']

# Whitelisted file name sets (exact match, automatically excludes stray copies)
GRID_DEMAND_NAMES = {f'{loc}_grid_demands.pickle' for loc in ALL_LOCATIONS}
KFOLD_CSV_NAMES = {f'kfold_test_{m}.csv' for m in METRICS}


# ════════════════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════════════════

def _mk(expected: Any, actual: Any, status: str, note: Optional[str] = None) -> dict:
    """Builds a single check-item record."""
    item: dict = {'expected': expected, 'actual': actual, 'status': status}
    if note is not None:
        item['note'] = note
    return item


def _count_whitelisted(base: Path, pattern: str, allowed_names: set) -> int:
    """Enumerates files by glob pattern and counts only those whose name is in the whitelist (ignoring any stray files)."""
    if not base.exists():
        return 0
    return sum(1 for p in base.glob(pattern) if p.name in allowed_names)


def _safe(check_fn, name: str, checks: dict) -> None:
    """Runs a single check and captures any exception, so one failing check does not abort the overall run."""
    try:
        checks[name] = check_fn()
    except Exception as exc:  # noqa: BLE001 -- precondition checks must record failures rather than crash
        checks[name] = _mk(None, None, 'error', f'{type(exc).__name__}: {exc}')


# ════════════════════════════════════════════════════════════
# Individual checks (reference values from direct inspection of the frozen artifacts)
# ════════════════════════════════════════════════════════════

def check_exp0_models() -> dict:
    """exp0 model files = 48 (3 seeds x 4 configs x 4 folds)."""
    n = _count_whitelisted(EXP0_DIR, 'seed_*/*/fold*/model.pth', {'model.pth'})
    return _mk(48, n, 'ok' if n == 48 else ('not_found' if n == 0 else 'mismatch'))


def check_exp0_grid_demands() -> dict:
    """exp0 grid_demands = 768 (48 combinations x 16 regions, .pickle files)."""
    n = _count_whitelisted(EXP0_DIR, 'seed_*/*/fold*/grid_demands/*.pickle',
                           GRID_DEMAND_NAMES)
    return _mk(768, n, 'ok' if n == 768 else ('not_found' if n == 0 else 'mismatch'))


def check_exp0_kfold_csv() -> dict:
    """exp0 kfold summary csv files = 36 (3 seeds x 4 configs x 3 metrics)."""
    n = _count_whitelisted(EXP0_DIR, 'seed_*/*/kfold_test_*.csv', KFOLD_CSV_NAMES)
    return _mk(36, n, 'ok' if n == 36 else ('not_found' if n == 0 else 'mismatch'))


def _load_first_graph(cache_file: Path):
    """Loads a graph cache pickle and returns its first graph (each cache file is ~437-440MB, so it is loaded only once)."""
    with open(cache_file, 'rb') as f:
        cached = pickle.load(f)
    graphs = cached['graphs']
    first_loc = next(iter(graphs))
    return first_loc, graphs[first_loc]


def check_main_graph_cache() -> "tuple[dict, dict]":
    """Main graph_cache: must have no ('agent', 'near', 'agent') edge type, and agent features must be 5-dimensional.

    Returns two items: (no-near-edge check, agent-feature-dimension check).
    """
    cache_file = EXP0_DIR / 'graph_cache' / 'cached_graphs.pickle'
    if not cache_file.exists():
        missing = _mk(None, 'file missing', 'not_found', str(cache_file))
        return missing, dict(missing)

    first_loc, g = _load_first_graph(cache_file)
    near_present = ('agent', 'near', 'agent') in g.edge_types
    agent_dim = int(g['agent'].x.shape[1])
    edge_types = [list(et) for et in g.edge_types]

    item_near = _mk(
        expected='no ("agent","near","agent") edge',
        actual={'first_graph': first_loc, 'edge_types': edge_types},
        status='ok' if not near_present else 'mismatch',
    )
    item_dim = _mk(5, agent_dim, 'ok' if agent_dim == 5 else 'mismatch',
                   note=f'first_graph={first_loc}')

    del g
    gc.collect()  # Free the ~437MB object promptly to leave headroom for the AgentConn cache
    return item_near, item_dim


def check_agentconn_graph_cache() -> dict:
    """AgentConn graph_cache: must include an ('agent', 'near', 'agent') edge type."""
    cache_file = AGENTCONN_DIR / 'graph_cache' / 'cached_graphs.pickle'
    if not cache_file.exists():
        return _mk(None, 'file missing', 'not_found', str(cache_file))

    first_loc, g = _load_first_graph(cache_file)
    near_present = ('agent', 'near', 'agent') in g.edge_types
    edge_types = [list(et) for et in g.edge_types]
    item = _mk(
        expected='has ("agent","near","agent") edge',
        actual={'first_graph': first_loc, 'edge_types': edge_types},
        status='ok' if near_present else 'mismatch',
    )
    del g
    gc.collect()
    return item


def _csv_method_count(csv_path: Path) -> Optional[int]:
    """Reads the number of method rows in all_regions_*.csv (excluding the header); returns None if the file is missing."""
    if not csv_path.exists():
        return None
    import pandas as pd
    df = pd.read_csv(csv_path)
    return int(len(df))


def check_static_allocation_methods() -> dict:
    """static_allocation/all_regions_rmse.csv method count = 27 (reference value from direct inspection)."""
    n = _csv_method_count(STATIC_DIR / 'all_regions_rmse.csv')
    if n is None:
        return _mk(27, None, 'not_found', str(STATIC_DIR / 'all_regions_rmse.csv'))
    return _mk(27, n, 'ok' if n == 27 else 'mismatch')


def check_static_allocation_nd_methods() -> dict:
    """static_allocation_nd/all_regions_rmse.csv method count = 10."""
    n = _csv_method_count(STATIC_ND_DIR / 'all_regions_rmse.csv')
    if n is None:
        return _mk(10, None, 'not_found', str(STATIC_ND_DIR / 'all_regions_rmse.csv'))
    return _mk(10, n, 'ok' if n == 10 else 'mismatch')


def check_substations_gpkg() -> dict:
    """Ground-truth substations.gpkg exists (read directly by the training pipeline)."""
    path = DATA_DIR / 'substations.gpkg'
    exists = path.exists()
    return _mk('exists', 'exists' if exists else 'missing',
               'ok' if exists else 'not_found', str(path))


def check_itl3_region_gpkg() -> dict:
    """ITL3_region.gpkg exists (read directly by the training pipeline; also required for building the Moran's I spatial weights)."""
    path = DATA_DIR / 'ITL3_region.gpkg'
    exists = path.exists()
    return _mk('exists', 'exists' if exists else 'missing',
               'ok' if exists else 'not_found', str(path))


def check_grid_points_16_regions() -> dict:
    """{loc}_grid_points.pickle is present for all 16 study regions (under features/assembled/)."""
    missing = [loc for loc in ALL_LOCATIONS
               if not (ASSEMBLED_DIR / f'{loc}_grid_points.pickle').exists()]
    n = len(ALL_LOCATIONS) - len(missing)
    return _mk(16, {'present': n, 'missing_regions': missing},
               'ok' if not missing else 'not_found')


def check_ntl_npz_16_regions_nonzero_var() -> dict:
    """{loc}_ntl.npz is present for all 16 study regions, and each region's NTL values have nonzero variance.

    Because a missing NTL file would otherwise risk being silently treated
    as all-zero downstream, this check asserts both file presence and
    nonzero variance. NTL values are extracted the same way as in the
    training pipeline: data[:, 0].
    """
    missing: list = []
    zero_var: list = []
    variances: dict = {}
    for loc in ALL_LOCATIONS:
        path = EXTRACTED_DIR / f'{loc}_ntl.npz'
        if not path.exists():
            missing.append(loc)
            continue
        npz = np.load(str(path), allow_pickle=True)
        ntl_values = np.asarray(npz['data'][:, 0], dtype=np.float64)
        var = float(np.var(ntl_values))
        variances[loc] = var
        if not (var > 0.0):
            zero_var.append(loc)
    ok = (not missing) and (not zero_var)
    return _mk(
        expected='16 regions present, all var(NTL) > 0',
        actual={'present': len(ALL_LOCATIONS) - len(missing),
                'missing_regions': missing,
                'zero_variance_regions': zero_var,
                'variance_by_region': variances},
        status='ok' if ok else ('not_found' if missing else 'mismatch'),
    )


def check_germany_results() -> dict:
    """Germany case-study artifacts: StudyCase/Germany/results/ is expected to be missing, since the Germany study arm is out of scope.

    If actually missing -> missing_registered (an allowed status); if
    unexpectedly present -> ok, with a note flagging that the decision to
    keep the Germany arm out of scope should be reconsidered.
    """
    exists = GERMANY_RESULTS_DIR.exists()
    if not exists:
        return _mk('missing (Germany arm out of scope)', 'missing', 'missing_registered',
                   str(GERMANY_RESULTS_DIR))
    return _mk('missing (Germany arm out of scope)', 'exists', 'ok',
               'Unexpectedly found Germany artifacts present -- reconsider whether the Germany arm should remain out of scope')


def check_exp0_001_grid_demands() -> dict:
    """exp0_kfold_prior_0_01 grid_demands is known to be incomplete: 752/768 (seed_456/ntl_prox/fold4 is missing)."""
    n = _count_whitelisted(EXP0_001_DIR, 'seed_*/*/fold*/grid_demands/*.pickle',
                           GRID_DEMAND_NAMES)
    if n == 752:
        return _mk('752/768 (known incomplete)', n, 'known_incomplete',
                   'seed_456/ntl_prox/fold4 is missing one set (recorded as-is based on direct inspection)')
    return _mk('752/768 (known incomplete)', n,
               'not_found' if n == 0 else 'mismatch')


def check_exp0_001_kfold_csv() -> dict:
    """exp0_kfold_prior_0_01 kfold csv files are known to be incomplete: 33/36."""
    n = _count_whitelisted(EXP0_001_DIR, 'seed_*/*/kfold_test_*.csv', KFOLD_CSV_NAMES)
    if n == 33:
        return _mk('33/36 (known incomplete)', n, 'known_incomplete',
                   'seed_456/ntl_prox is missing 3 metric csv files (recorded as-is based on direct inspection)')
    return _mk('33/36 (known incomplete)', n,
               'not_found' if n == 0 else 'mismatch')


def check_gpu_available() -> dict:
    """GPU availability: torch.cuda.is_available() (required by later training runs)."""
    import torch
    available = bool(torch.cuda.is_available())
    device_name = torch.cuda.get_device_name(0) if available else None
    return _mk(True, {'cuda_available': available,
                      'device_name': device_name,
                      'torch_version': torch.__version__},
               'ok' if available else 'mismatch')


# ════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════

def run_all_checks() -> dict:
    """Runs all checks and returns the complete report dictionary."""
    checks: dict = {}

    _safe(check_exp0_models, 'exp0_models', checks)
    _safe(check_exp0_grid_demands, 'exp0_grid_demands', checks)
    _safe(check_exp0_kfold_csv, 'exp0_kfold_csv', checks)

    # Both graph-cache items come from the same load (the ~437MB file is loaded only once)
    try:
        item_near, item_dim = check_main_graph_cache()
        checks['main_graph_cache_no_near_edge'] = item_near
        checks['main_graph_cache_agent_dim'] = item_dim
    except Exception as exc:  # noqa: BLE001
        err = _mk(None, None, 'error', f'{type(exc).__name__}: {exc}')
        checks['main_graph_cache_no_near_edge'] = err
        checks['main_graph_cache_agent_dim'] = dict(err)

    _safe(check_agentconn_graph_cache, 'agentconn_cache_has_near_edge', checks)
    _safe(check_static_allocation_methods, 'static_allocation_methods', checks)
    _safe(check_static_allocation_nd_methods, 'static_allocation_nd_methods', checks)
    _safe(check_substations_gpkg, 'substations_gpkg', checks)
    _safe(check_grid_points_16_regions, 'grid_points_16_regions', checks)
    _safe(check_ntl_npz_16_regions_nonzero_var, 'ntl_npz_16_regions_nonzero_var', checks)
    _safe(check_itl3_region_gpkg, 'itl3_region_gpkg', checks)
    _safe(check_germany_results, 'germany_results', checks)
    _safe(check_exp0_001_grid_demands, 'exp0_prior_0_01_grid_demands', checks)
    _safe(check_exp0_001_kfold_csv, 'exp0_prior_0_01_kfold_csv', checks)
    _safe(check_gpu_available, 'gpu_available', checks)

    allowed = {'ok', 'missing_registered', 'known_incomplete'}
    n_bad = sum(1 for v in checks.values() if v['status'] not in allowed)
    report = {
        'meta': {
            'script': 'precheck_frozen_inputs.py',
            'purpose': 'Precondition check -- one-shot validation of frozen input artifacts',
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'python': sys.version.split()[0],
            'executable': sys.executable,
            'allowed_statuses': sorted(allowed),
        },
        'checks': checks,
        'summary': {
            'total': len(checks),
            'passed': len(checks) - n_bad,
            'failed': n_bad,
            'all_green': n_bad == 0,
        },
    }
    return report


def main() -> int:
    """Entry point: runs the checks, writes the JSON report, prints a summary; returns 0 if all checks pass, otherwise 1."""
    # Force UTF-8 console output encoding (does not affect the JSON file, which is always written as UTF-8)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    t0 = time.time()
    report = run_all_checks()
    report['meta']['elapsed_seconds'] = round(time.time() - t0, 1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'\nCheck report written to: {OUTPUT_JSON}')
    print(f'{"Check":<40s} {"Status"}')
    print('-' * 60)
    for name, item in report['checks'].items():
        print(f'{name:<40s} {item["status"]}')
    print('-' * 60)
    s = report['summary']
    print(f'Total {s["total"]} | Passed {s["passed"]} | Failed {s["failed"]} '
          f'| All green: {s["all_green"]} | Elapsed {report["meta"]["elapsed_seconds"]}s')
    return 0 if s['all_green'] else 1


if __name__ == '__main__':
    sys.exit(main())
