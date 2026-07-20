"""
016 Training dynamics visualization (Exp 10)

Extracts loss curves from the Exp 0 training_log.json files.

Usage:
    python 016_exp10_training_dynamics.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])

SCRIPT_DIR = Path(__file__).resolve().parent
EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp10_training_dynamics'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ['baseline', 'ntl', 'proximity', 'ntl_prox']
CONFIG_LABELS = {
    'baseline': 'Baseline (landuse only)',
    'ntl': 'NTL Prior',
    'proximity': 'Proximity Prior',
    'ntl_prox': 'NTL + Proximity Prior',
}
CONFIG_COLORS = {
    'baseline': 'gray',
    'ntl': 'tab:blue',
    'proximity': 'tab:orange',
    'ntl_prox': 'tab:red',
}

SEED = 42  # seed used for the main display
FOLD = 1   # fold used for the main display


def load_training_log(config_name, seed=SEED, fold=FOLD):
    """Load a training log."""
    fold_dir = EXP0_DIR / f'seed_{seed}' / config_name / f'fold{fold}'
    if not fold_dir.exists():
        return None

    log_files = list(fold_dir.glob('*_training_log.json'))
    if not log_files:
        return None

    with open(log_files[0], 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_training_dynamics():
    """Plot training dynamics curves."""
    print('=' * 60)
    print('Exp 10: Training Dynamics')
    print('=' * 60)

    logs = {}
    for config in CONFIGS:
        log = load_training_log(config)
        if log is not None:
            logs[config] = log
            n_epochs = len(log.get('train_losses', {}).get('total', []))
            best_epoch = log.get('best_epoch', '?')
            print(f'  {config}: {n_epochs} epochs, best_epoch={best_epoch}')
        else:
            print(f'  {config}: log not found')

    if not logs:
        print('[Warning] No training logs available')
        return

    # ── Panel 1: Train Total Loss ──
    fig1, ax = plt.subplots(figsize=(3.5, 2.5))
    for config, log in logs.items():
        train_total = log.get('train_losses', {}).get('total', [])
        if train_total:
            epochs = range(1, len(train_total) + 1)
            ax.plot(epochs, train_total, color=CONFIG_COLORS[config],
                    label=CONFIG_LABELS[config], linewidth=1.0, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=8)
    ax.set_ylabel('Train Total Loss', fontsize=8)
    ax.set_title('Training Loss (Total)', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.2)
    ax.set_yscale('log')
    fig1.tight_layout()
    fig1.savefig(OUTPUT_DIR / 'figure_training_dynamics_train.png', dpi=200, bbox_inches='tight')
    fig1.savefig(OUTPUT_DIR / 'figure_training_dynamics_train.pdf', bbox_inches='tight')
    plt.close(fig1)

    # ── Panel 2: Test Total Loss ──
    fig2, ax = plt.subplots(figsize=(3.5, 2.5))
    for config, log in logs.items():
        test_total = log.get('test_losses', {}).get('total', [])
        if test_total:
            epochs = range(1, len(test_total) + 1)
            ax.plot(epochs, test_total, color=CONFIG_COLORS[config],
                    label=CONFIG_LABELS[config], linewidth=1.0, alpha=0.8)

            # Mark best epoch
            best_epoch = log.get('best_epoch', None)
            if best_epoch and best_epoch <= len(test_total):
                ax.axvline(x=best_epoch, color=CONFIG_COLORS[config],
                           linestyle=':', alpha=0.5, linewidth=0.8)

    ax.set_xlabel('Epoch', fontsize=8)
    ax.set_ylabel('Test Total Loss', fontsize=8)
    ax.set_title('Validation Loss (Total)', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.2)
    ax.set_yscale('log')
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / 'figure_training_dynamics_test.png', dpi=200, bbox_inches='tight')
    fig2.savefig(OUTPUT_DIR / 'figure_training_dynamics_test.pdf', bbox_inches='tight')
    plt.close(fig2)

    # ── Panel 3: Prior Loss Components ──
    fig3, ax = plt.subplots(figsize=(3.5, 2.5))
    prior_keys = ['ntl_prior', 'proximity_prior']
    prior_labels = {'ntl_prior': 'NTL Prior Loss', 'proximity_prior': 'Proximity Prior Loss'}
    prior_colors = {'ntl_prior': 'tab:blue', 'proximity_prior': 'tab:orange'}
    linestyles = {'ntl': '-', 'proximity': '-', 'ntl_prox': '--'}

    for config in ['ntl', 'proximity', 'ntl_prox']:
        if config not in logs:
            continue
        log = logs[config]
        train_losses = log.get('train_losses', {})

        for prior_key in prior_keys:
            if prior_key in train_losses:
                values = train_losses[prior_key]
                epochs = range(1, len(values) + 1)
                ls = linestyles.get(config, '-')
                label = f'{CONFIG_LABELS[config]} - {prior_labels[prior_key]}'
                ax.plot(epochs, values, color=prior_colors[prior_key],
                        linestyle=ls, label=label, linewidth=1.0, alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=8)
    ax.set_ylabel('Prior Loss', fontsize=8)
    ax.set_title('Prior Loss Components', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.2)
    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / 'figure_training_dynamics_prior.png', dpi=200, bbox_inches='tight')
    fig3.savefig(OUTPUT_DIR / 'figure_training_dynamics_prior.pdf', bbox_inches='tight')
    plt.close(fig3)

    # Output the numeric summary table
    summary_rows = []
    for config, log in logs.items():
        train_losses = log.get('train_losses', {})
        test_losses = log.get('test_losses', {})

        row = {
            'config': config,
            'best_epoch': log.get('best_epoch'),
            'best_loss': log.get('best_loss'),
            'final_train_total': train_losses.get('total', [None])[-1],
            'final_test_total': test_losses.get('total', [None])[-1],
            'total_epochs': len(train_losses.get('total', [])),
            'duration_seconds': log.get('duration_seconds'),
        }

        # Final values of prior losses
        for key in ['ntl_prior', 'proximity_prior', 'landuse_prediction_loss']:
            values = train_losses.get(key, [])
            row[f'final_train_{key}'] = values[-1] if values else None

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / 'training_summary.csv', index=False)
    print('\n=== Training Summary ===')
    print(summary_df.to_string(index=False))

    # Cross-fold averaged dynamics (if multiple folds are available)
    print('\n--- Cross-fold average best epoch ---')
    for config in CONFIGS:
        best_epochs = []
        for fold in range(1, 5):
            log = load_training_log(config, seed=SEED, fold=fold)
            if log:
                best_epochs.append(log.get('best_epoch', 0))
        if best_epochs:
            print(f'  {config}: best_epochs={best_epochs}, '
                  f'mean={np.mean(best_epochs):.1f}')

    print(f'\nExp 10 complete, output: {OUTPUT_DIR}')


if __name__ == '__main__':
    plot_training_dynamics()
