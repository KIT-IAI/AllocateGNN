"""
016 Börde training dynamics visualization (fix variant)

The only difference from the original 016: load_training_losses adds a
fallback read from model_training_log.json (the log actually written on
disk by 005 during retraining, under train_losses['total']). The original
016 only recognized training_history.json / an embedded history in the
checkpoint, and would report "no training data" for retraining runs that
only produced this log file. The numeric semantics are unchanged -- only
the data source lookup is adapted.

Plots loss curves for 4 configs x multiple seeds.
Multiple seeds are plotted as mean ± shaded std.

Usage:
    python 016_boerde_training_dynamics.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / 'results' / 'models'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GNN_CONFIGS = ['baseline', 'ntl', 'proximity', 'ntl_prox']
SEEDS = [42, 123, 456]

CONFIG_LABELS = {
    'baseline': 'Baseline (LU only)',
    'ntl': 'NTL prior',
    'proximity': 'Proximity prior',
    'ntl_prox': 'NTL + Proximity',
}

CONFIG_COLORS = {
    'baseline': '#1f77b4',
    'ntl': '#ff7f0e',
    'proximity': '#2ca02c',
    'ntl_prox': '#d62728',
}


def load_training_losses(config, seed):
    """Load epoch-wise loss from the log saved by EdgeWeightSolver."""
    seed_dir = MODEL_DIR / config / f'seed_{seed}'

    # Try loading training_history.json (saved by EdgeWeightSolver)
    hist_path = seed_dir / 'training_history.json'
    if hist_path.exists():
        with open(hist_path, 'r') as f:
            history = json.load(f)
        return history

    # fix: 005 retraining writes model_training_log.json (train_losses is a dict of lists)
    mlog_path = seed_dir / 'model_training_log.json'
    if mlog_path.exists():
        with open(mlog_path, 'r', encoding='utf-8') as f:
            mlog = json.load(f)
        tl = mlog.get('train_losses')
        if isinstance(tl, dict) and 'total' in tl:
            return {'total_loss': tl['total']}
        if isinstance(tl, list):
            return {'total_loss': tl}

    # Try loading the history embedded in model.pth
    model_path = seed_dir / 'model.pth'
    if model_path.exists():
        import torch
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'training_history' in checkpoint:
            return checkpoint['training_history']
        if 'loss_history' in checkpoint:
            return {'total_loss': checkpoint['loss_history']}

    return None


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    has_data = False

    for ax_idx, config in enumerate(GNN_CONFIGS):
        ax = axes[ax_idx]
        all_losses = []
        min_len = float('inf')

        for seed in SEEDS:
            history = load_training_losses(config, seed)
            if history is None:
                continue

            # Extract total loss
            if isinstance(history, dict):
                if 'total_loss' in history:
                    losses = history['total_loss']
                elif 'train_loss' in history:
                    losses = history['train_loss']
                else:
                    # Take the first key
                    first_key = list(history.keys())[0]
                    losses = history[first_key]
            elif isinstance(history, list):
                if isinstance(history[0], dict):
                    losses = [h.get('total_loss', h.get('loss', 0))
                              for h in history]
                else:
                    losses = history
            else:
                continue

            all_losses.append(np.array(losses, dtype=float))
            min_len = min(min_len, len(losses))

        if not all_losses:
            ax.text(0.5, 0.5, f'{CONFIG_LABELS[config]}\n(no training data)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='gray')
            ax.set_title(CONFIG_LABELS[config])
            continue

        has_data = True

        # Truncate to the shortest length
        truncated = np.array([l[:min_len] for l in all_losses])
        mean_loss = truncated.mean(axis=0)
        std_loss = truncated.std(axis=0)
        epochs = np.arange(1, min_len + 1)

        color = CONFIG_COLORS[config]
        ax.plot(epochs, mean_loss, color=color, linewidth=1.5,
                label=f'mean (n={len(all_losses)})')
        ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                        color=color, alpha=0.2, label='±1 std')

        # Individual seed curves (faint)
        for i, losses in enumerate(all_losses):
            ax.plot(np.arange(1, len(losses[:min_len]) + 1),
                    losses[:min_len],
                    color=color, alpha=0.15, linewidth=0.5)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title(CONFIG_LABELS[config])
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Börde training dynamics (4 configs × multi-seed)', fontsize=14)
    plt.tight_layout()

    if has_data:
        fig.savefig(OUTPUT_DIR / 'training_dynamics.png', dpi=150,
                    bbox_inches='tight')
        print(f'Saved: {OUTPUT_DIR / "training_dynamics.png"}')
    else:
        print('No training data available, please run 005 first')

    plt.close()


if __name__ == '__main__':
    main()
