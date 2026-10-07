"""Pointwise learned weighter shared by every country adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..common import assert_training_complete
from .gnn.config import ModelConfig
from .gnn.inference import assert_inference_graph, sanitize_for_inference
from .gnn.layers.edge_weight import DifferentiableEdgeWeighting
from .gnn.losses.base import loss_registry


class PointwiseEncoder(nn.Module):
    """Per-node towers without message passing."""

    def __init__(self, input_dims: dict[str, int], config: ModelConfig):
        super().__init__()
        hidden, embedding, n_layers = (
            config.hidden_dim,
            config.embedding_dim,
            config.num_layers,
        )

        def tower(input_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.ReLU()]
            for _ in range(max(n_layers - 1, 0)):
                layers.extend((nn.Linear(hidden, hidden), nn.ReLU()))
            layers.append(nn.Linear(hidden, embedding))
            return nn.Sequential(*layers)

        self.towers = nn.ModuleDict(
            {node_type: tower(dim) for node_type, dim in input_dims.items()}
        )

    def forward(self, x_dict, edge_index_dict=None):
        return {node_type: self.towers[node_type](x) for node_type, x in x_dict.items()}


def loss_metadata(graph) -> dict:
    metadata = {
        "landuse_ratio": graph.landuse_ratio,
        "num_s": int(graph["source"].num_nodes),
    }
    for name in (
        "landuse_mapping_matrix",
        "landuse_flat_index",
        "landuse_supervision_representation",
    ):
        if hasattr(graph, name):
            metadata[name] = getattr(graph, name)
    for source_name, target_name in (
        ("ntl_values", "agent_ntl"),
        ("proximity_scores", "agent_proximity"),
        ("rci_mask", "agent_rci_mask"),
    ):
        if hasattr(graph["agent"], source_name):
            metadata[target_name] = getattr(graph["agent"], source_name)
    return metadata


def _models(graphs, params: dict, device: str):
    sample = next(iter(graphs.values()))
    config = ModelConfig(
        hidden_dim=params["gnn"]["hidden_dim"],
        embedding_dim=params["gnn"]["embedding_dim"],
        num_layers=params["gnn"]["num_layers"],
        allocation_temperature_start=params["gnn"]["allocation_temperature_start"],
        device=device,
    )
    dims = {
        "source": int(sample["source"].x.shape[1]),
        "agent": int(sample["agent"].x.shape[1]),
    }
    return (
        PointwiseEncoder(dims, config).to(device),
        DifferentiableEdgeWeighting(config).to(device),
    )


def train_mlp_fold(
    config_name,
    seed,
    fold_idx,
    train_locs,
    graphs,
    params,
    fold_dir: Path,
    epochs_override=None,
    device="cpu",
):
    config = params["config_map"][config_name]
    epochs = epochs_override or config["epochs"]
    objective_weights = config["objective_weights"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_dir / "model.pth"
    encoder, edge_layer = _models(graphs, params, device)
    if model_path.exists():
        assert_training_complete(model_path, epochs)
        blob = torch.load(model_path, weights_only=False, map_location=device)
        encoder.load_state_dict(blob["encoder"])
        edge_layer.load_state_dict(blob["edge_layer"])
        return encoder, edge_layer

    losses = {name: loss_registry.get_loss(name)() for name in objective_weights}
    parameters = list(encoder.parameters()) + list(edge_layer.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=params["mlp"]["learning_rate"],
        weight_decay=params["mlp"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=params["mlp"]["cosine_eta_min"]
    )
    train_losses = []
    for _epoch in range(1, epochs + 1):
        encoder.train()
        edge_layer.train()
        total = 0.0
        for location in train_locs:
            graph = graphs[location]
            x_dict = {
                "source": graph["source"].x.to(device),
                "agent": graph["agent"].x.to(device),
            }
            edge_index = graph["source", "connects_to", "agent"].edge_index.to(device)
            encoded = encoder(x_dict)
            weights, _costs = edge_layer(
                encoded["source"], encoded["agent"], edge_index
            )
            metadata = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in loss_metadata(graph).items()
            }
            loss = sum(
                weight * losses[name](weights, edge_index, metadata)
                for name, weight in objective_weights.items()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        scheduler.step()
        train_losses.append(total / len(train_locs))

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "edge_layer": edge_layer.state_dict(),
            "config_name": config_name,
            "seed": seed,
            "fold": fold_idx + 1,
            "epochs": epochs,
            "final_loss": train_losses[-1],
            "n_params": sum(parameter.numel() for parameter in parameters),
        },
        model_path,
    )
    (fold_dir / "model_training_log.json").write_text(
        json.dumps(
            {
                "config": config_name,
                "seed": seed,
                "fold": fold_idx + 1,
                "epochs": epochs,
                "train_losses": train_losses,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return encoder, edge_layer


@torch.no_grad()
def load_mlp_checkpoint(fold_dir: Path, graphs, params, device="cpu"):
    model_path = fold_dir / "model.pth"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    encoder, edge_layer = _models(graphs, params, device)
    blob = torch.load(model_path, weights_only=False, map_location=device)
    encoder.load_state_dict(blob["encoder"])
    edge_layer.load_state_dict(blob["edge_layer"])
    encoder.eval()
    edge_layer.eval()
    return encoder, edge_layer


@torch.no_grad()
def predict_mlp_scores(encoder, edge_layer, graph, n_grid: int, device="cpu"):
    encoder.eval()
    edge_layer.eval()
    sanitized = sanitize_for_inference(graph)
    assert_inference_graph(sanitized)
    x_dict = {
        "source": sanitized["source"].x.to(device),
        "agent": sanitized["agent"].x.to(device),
    }
    edge_index = sanitized["source", "connects_to", "agent"].edge_index.to(device)
    encoded = encoder(x_dict)
    weights, _costs = edge_layer(
        encoded["source"], encoded["agent"], edge_index
    )
    agent_indices = edge_index[1].cpu().numpy()
    original = np.asarray(
        sanitized.agent_index_map.iloc[agent_indices], dtype=np.int64
    )
    scores = np.zeros(int(n_grid), dtype=float)
    np.add.at(scores, original, weights.detach().cpu().numpy())
    return scores

