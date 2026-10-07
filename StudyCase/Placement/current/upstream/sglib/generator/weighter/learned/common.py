"""Checkpoint completeness shared by GNN and pointwise learned weighters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold


def assert_training_complete(model_path: Path, expected_epochs: int) -> None:
    log_path = model_path.with_name(model_path.stem + "_training_log.json")
    if not log_path.exists():
        raise RuntimeError(f"checkpoint has no training log: {model_path}")
    document = json.loads(log_path.read_text(encoding="utf-8"))
    losses = document.get("train_losses")
    if isinstance(losses, dict):
        losses = losses.get("total")
    if not isinstance(losses, list) or len(losses) != int(expected_epochs):
        raise RuntimeError(
            f"incomplete training artifact: {model_path} has "
            f"{len(losses) if isinstance(losses, list) else 'invalid'} / {expected_epochs} epochs"
        )


def kfold_splits(regions, seed: int, n_folds: int):
    values = np.asarray(list(regions))
    splitter = KFold(n_splits=int(n_folds), shuffle=True, random_state=int(seed))
    return [(values[train].tolist(), values[test].tolist()) for train, test in splitter.split(values)]


