"""Load trained model state and frozen scientific parameter overrides."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from .tasks import PreparedInferenceTask, InferenceTaskError


def task_parameters(task: PreparedInferenceTask) -> tuple[dict[str, Any], float | None]:
    params = json.loads(Path(task.frozen_params_path).read_text(encoding="utf-8-sig"))
    tau_start = None
    if task.parameter == "lambda":
        objective = params["config_map"][task.config]["objective_weights"]
        names = {
            "N": ("ntl_prior",),
            "P": ("proximity_prior",),
            "NP": ("ntl_prior", "proximity_prior"),
        }.get(task.signal)
        if not names or any(name not in objective for name in names):
            raise InferenceTaskError("lambda inference identity is invalid")
        for name in names:
            objective[name] = float(task.value)
    elif task.parameter == "tau":
        tau_start = float(task.value)
    return params, tau_start



def load_gnn_cpu(
    task: PreparedInferenceTask,
    graphs: Mapping[str, Any],
    params: Mapping[str, Any],
    tau_start: float | None,
) -> Any:
    import torch
    from torch_geometric.loader import DataLoader

    from ..models.gnn.config import ModelConfig
    from ..models.gnn.solver import EdgeWeightSolver

    values = params["config_map"][task.config]
    epochs = int(values["epochs"])
    config = ModelConfig(
        epochs=epochs,
        hidden_dim=params["gnn"]["hidden_dim"],
        embedding_dim=params["gnn"]["embedding_dim"],
        num_layers=params["gnn"]["num_layers"],
        conv_type=params["gnn"]["conv_type"],
        allocation_temperature_start=(
            params["gnn"]["allocation_temperature_start"]
            if tau_start is None
            else float(tau_start)
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
        save_path=task.checkpoint_path,
        device="cpu",
    )
    solver = EdgeWeightSolver(config)
    sample = DataLoader([next(iter(graphs.values()))], batch_size=1, shuffle=False)
    solver.init_model(sample, values["objective_weights"])
    checkpoint = torch.load(task.checkpoint_path, map_location="cpu", weights_only=False)
    modules = {
        "encoder_state_dict": solver.encoder,
        "edge_weighting_layer_state_dict": solver.edge_weighting_layer,
        "projection_head_state_dict": solver.projection_head,
        "recon_head_state_dict": solver.recon_head,
        "agent_gating_state_dict": solver.agent_gating,
    }
    loaded = False
    for key, module in modules.items():
        if module is not None and key in checkpoint:
            module.load_state_dict(checkpoint[key])
            module.eval()
            loaded = True
    if not loaded:
        raise InferenceTaskError("GNN checkpoint contains no recognized model state")
    return solver
