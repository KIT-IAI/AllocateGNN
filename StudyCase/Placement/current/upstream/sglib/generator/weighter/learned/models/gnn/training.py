"""Country-neutral GNN fold training, checkpoint loading and score inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from ...common import assert_training_complete
from .config import ModelConfig
from .inference import assert_inference_graph, sanitize_for_inference
from .solver import EdgeWeightSolver


def train_gnn_fold(
    config_name,
    seed,
    fold_idx,
    train_locs,
    graphs,
    params,
    fold_dir: Path,
    epochs_override=None,
    tau_start=None,
    device=None,
):
    config_values = params["config_map"][config_name]
    epochs = epochs_override or config_values["epochs"]
    objective_weights = config_values["objective_weights"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected for GNN training but unavailable")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_dir / "model.pth"
    warmup = params["gnn"]["warmup_epochs"]
    decay = params["gnn"]["decay_epochs"]
    config = ModelConfig(
        epochs=epochs,
        hidden_dim=params["gnn"]["hidden_dim"],
        embedding_dim=params["gnn"]["embedding_dim"],
        num_layers=params["gnn"]["num_layers"],
        conv_type=params["gnn"]["conv_type"],
        allocation_temperature_start=(
            params["gnn"]["allocation_temperature_start"]
            if tau_start is None else float(tau_start)
        ),
        learning_rate=params["gnn"]["learning_rate"],
        weight_decay=params["gnn"]["weight_decay"],
        use_scheduler=True,
        warmup_epochs=warmup,
        decay_epochs=decay,
        cosine_epochs=epochs - warmup - decay,
        cosine_eta_min=params["gnn"]["cosine_eta_min"],
        learnable=False,
        save_path=str(model_path),
        device=device,
    )
    solver = EdgeWeightSolver(config)
    loader = DataLoader(
        [graphs[location] for location in train_locs], batch_size=1, shuffle=False
    )
    if model_path.exists():
        assert_training_complete(model_path, epochs)
        solver.init_model(loader, objective_weights)
        solver._load_checkpoint()
    else:
        solver.train_multi_graph(loader, objective_weights=objective_weights)
    return solver


def load_gnn_checkpoint(config_name, fold_dir: Path, graphs, params, tau_start=None):
    model_path = fold_dir / "model.pth"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    values = params["config_map"][config_name]
    epochs = values["epochs"]
    config = ModelConfig(
        epochs=epochs,
        hidden_dim=params["gnn"]["hidden_dim"],
        embedding_dim=params["gnn"]["embedding_dim"],
        num_layers=params["gnn"]["num_layers"],
        conv_type=params["gnn"]["conv_type"],
        allocation_temperature_start=(
            params["gnn"]["allocation_temperature_start"]
            if tau_start is None else float(tau_start)
        ),
        learning_rate=params["gnn"]["learning_rate"],
        weight_decay=params["gnn"]["weight_decay"],
        use_scheduler=True,
        warmup_epochs=params["gnn"]["warmup_epochs"],
        decay_epochs=params["gnn"]["decay_epochs"],
        cosine_epochs=(
            epochs - params["gnn"]["warmup_epochs"] - params["gnn"]["decay_epochs"]
        ),
        cosine_eta_min=params["gnn"]["cosine_eta_min"],
        learnable=False,
        save_path=str(model_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    solver = EdgeWeightSolver(config)
    sample_loader = DataLoader(
        [next(iter(graphs.values()))], batch_size=1, shuffle=False
    )
    solver.init_model(sample_loader, values["objective_weights"])
    solver._load_checkpoint()
    return solver


def predict_gnn_scores(solver, graph, n_grid: int) -> np.ndarray:
    sanitized = sanitize_for_inference(graph)
    assert_inference_graph(sanitized)
    edge_weights = solver.predict_edge_weights(sanitized)
    scores = np.zeros(int(n_grid), dtype=float)
    for row in edge_weights.itertuples(index=False):
        scores[int(row.agent_original_idx)] += float(row.predicted_weight)
    return scores


