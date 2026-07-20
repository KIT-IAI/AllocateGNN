"""035 In-place retraining of AgentConn seed_42/ntl/fold2.

Background:
The AgentConn training artifact directory
exp0_kfold_prior_AgentConn/seed_42/ntl/fold2 contains only model.pth
(internal best_epoch=79) without model_training_log.json. Because the
directory appears to have been produced by a bulk copy (all file mtimes
cluster around the same timestamp), it is not possible to tell whether the
log was lost during copying or the training run itself was truncated. This
script retrains the fold in place; afterwards the downstream evaluation in
029 should be rerun to refresh the AgentConn downstream artifacts.

Faithful-replication conventions (aligned field-by-field with 005; the
frozen 005 script itself is not modified):
- Graph input: reads the AgentConn graph cache directly
  (exp0_kfold_prior_AgentConn/graph_cache/cached_graphs.pickle, read-only;
  raises if missing) -- the same graph used by the original run;
- Region order / config mapping: imports the 005 module at runtime to
  obtain ALL_LOCATIONS and CONFIG_MAP['ntl'] (epochs=200,
  weights={'landuse_prediction_loss':1.0,'ntl_prior':0.05}), to avoid any
  transcription drift;
- Seed and split: torch/np/cuda seed=42 throughout; KFold(4, shuffle,
  random_state=42), taking fold_idx=1 (= fold2) -- identical to 005's
  run_kfold_training;
- ModelConfig: hgt / hidden 256 / embedding 128 / 3 layers / tau=0.01 /
  lr 1e-3 / warmup 20 + decay 20 + cosine 160 / learnable=False -- copied
  from the 005 training branch.

Known non-reproducibility (disclosed for transparency):
In the original run, fold2 training continued the RNG stream left over
from fold1. This script retrains the fold in isolation, so its RNG stream
differs from the original; combined with GPU non-determinism, the
resulting model is not bit-for-bit reproducible against the original --
handled under the repository's existing tolerance-anchoring convention.
This result is for internal reference only and is not used in the paper
body.

Safety gate:
The script refuses to run if the target model.pth already exists (to
prevent silently resuming on top of an existing model and mixing old and
new artifacts) -- any suspect existing model must be moved aside before
rerunning.

Usage (headless runs require MPLBACKEND=Agg, since EdgeWeightSolver
contains a blocking plt.show call):
    MPLBACKEND=Agg KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8 \
        python 035_retrain_agentconn_fold2.py
Outputs: seed_42/ntl/fold2/{model.pth, model_training_log.json};
Downstream: 029_eval_agentconn_downstream.py (rerun) -> notebooks/r214_disclosure.ipynb.
"""

import sys
import json
import pickle
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold

# Project root: walk upward to find the directory containing SpatialAllocation
SCRIPT_DIR = Path(__file__).resolve().parent
_p = SCRIPT_DIR
while not (_p / 'SpatialAllocation').exists():
    if _p.parent == _p:
        raise RuntimeError('Could not find repository root (SpatialAllocation package)')
    _p = _p.parent
PROJECT_ROOT = _p
for _extra in (str(SCRIPT_DIR), str(PROJECT_ROOT)):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from SpatialAllocation.GNN.core.EdgeWeightSolver import EdgeWeightSolver  # noqa: E402
from SpatialAllocation.GNN.core.ModelConfig import ModelConfig  # noqa: E402

# Windows console default codepage (cp1252) cannot render non-ASCII text -- force UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SEED = 42
CONFIG_NAME = 'ntl'
FOLD_IDX = 1  # 0-based → fold2
AGENTCONN_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior_AgentConn'
CACHE_FILE = AGENTCONN_DIR / 'graph_cache' / 'cached_graphs.pickle'
FOLD_DIR = AGENTCONN_DIR / f'seed_{SEED}' / CONFIG_NAME / f'fold{FOLD_IDX + 1}'
MODEL_PATH = FOLD_DIR / 'model.pth'
LOG_PATH = FOLD_DIR / 'model_training_log.json'


def main() -> None:
    # -- Safety gate: refuse to "resume" on top of an existing model --
    if MODEL_PATH.exists():
        raise SystemExit(f'Refusing to run: {MODEL_PATH} already exists. Move the '
                         'existing model aside before retraining (to avoid mixing '
                         'old and new artifacts).')
    if not CACHE_FILE.exists():
        raise SystemExit(f'Refusing to run: AgentConn graph cache missing at {CACHE_FILE} -- '
                         'this cache is the only graph input aligned with the original run '
                         'and must not be rebuilt as a substitute.')

    # -- Replication settings source: read from 005 at runtime to avoid transcription drift --
    m005 = import_module('005_kfold_prior_training')
    all_locations = m005.ALL_LOCATIONS
    exp_config = m005.CONFIG_MAP[CONFIG_NAME]
    objective_weights = exp_config['objective_weights']
    epochs = exp_config['epochs']
    print(f'Config: {CONFIG_NAME} | seed: {SEED} | epochs: {epochs} | '
          f'weights: {objective_weights}')

    print(f'Loading AgentConn graph cache (~440MB): {CACHE_FILE}')
    with open(CACHE_FILE, 'rb') as f:
        cached = pickle.load(f)
    graphs = cached['graphs']

    # -- Seed and split (copied from 005's run_kfold_training) --
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    location_array = np.array(all_locations)
    kf = KFold(n_splits=4, shuffle=True, random_state=SEED)
    train_indices, test_indices = list(kf.split(location_array))[FOLD_IDX]
    train_locs = location_array[train_indices].tolist()
    test_locs = location_array[test_indices].tolist()
    print(f'Fold {FOLD_IDX + 1} training regions ({len(train_locs)}): {train_locs}')
    print(f'Fold {FOLD_IDX + 1} test regions ({len(test_locs)}): {test_locs}')

    # -- ModelConfig (copied from the 005 training branch) --
    warmup_epochs = 20
    decay_epochs = 20
    cosine_epochs = epochs - warmup_epochs - decay_epochs
    config = ModelConfig(
        epochs=epochs,
        hidden_dim=256,
        embedding_dim=128,
        num_layers=3,
        conv_type='hgt',
        allocation_temperature_start=0.01,
        learning_rate=1e-3,
        weight_decay=1e-4,
        use_scheduler=True,
        warmup_epochs=warmup_epochs,
        decay_epochs=decay_epochs,
        cosine_epochs=cosine_epochs,
        cosine_eta_min=1e-5,
        learnable=False,
        save_path=str(MODEL_PATH),
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    print(f'Device: {config.device}')

    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    solver = EdgeWeightSolver(config)
    train_dl = DataLoader([graphs[loc] for loc in train_locs],
                          batch_size=1, shuffle=False)
    test_dl = DataLoader([graphs[loc] for loc in test_locs],
                         batch_size=1, shuffle=False)
    solver.train_multi_graph(train_dl, test_dataloader=test_dl,
                             objective_weights=objective_weights)

    # -- Completion check: the training log must exist and record the full epoch count --
    if not LOG_PATH.exists():
        raise SystemExit(f'Training finished but {LOG_PATH} was not found -- treated as '
                         'incomplete, needs investigation.')
    with open(LOG_PATH, encoding='utf-8') as f:
        log = json.load(f)
    n_logged = len(log['train_losses']['total'])
    if n_logged != epochs:
        raise SystemExit(f'Training log epoch count incomplete: {n_logged}/{epochs}.')
    print(f'Retraining complete and passed the completion check: best_epoch={log["best_epoch"]}, '
          f'epochs={n_logged}/{epochs}')
    print('Next step: rerun 029_eval_agentconn_downstream.py to refresh the downstream evaluation artifacts.')


if __name__ == '__main__':
    main()
