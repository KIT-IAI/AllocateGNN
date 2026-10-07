"""Scientific code projection independent of operational task execution."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
from sglib.core.infra.content_chain import code_projection
from ..paths import absolute_root
from .checkpoint import task_parameters, load_gnn_cpu
from .field import infer_demand_field, write_field


def inference_code_identity(repo_root: str | Path) -> dict[str, Any]:
    """Project only symbols that can change field values or their encoding."""

    repository = absolute_root(repo_root, "repo_root")
    return _inference_code_identity(str(repository))



@lru_cache(maxsize=None)
def _inference_code_identity(_repository: str) -> dict[str, Any]:
    # The normalized repository locator is only the per-process cache key.
    from ...support import compose_support_field
    from ..common import kfold_splits
    from .field import _frame
    from ..models.gnn.inference import assert_inference_graph, sanitize_for_inference
    from ..models.gnn.layers.agent_gating import AgentGating
    from ..models.gnn.layers.edge_weight import DifferentiableEdgeWeighting
    from ..models.gnn.layers.graph_encoder import GraphEncoder
    from ..models.gnn.layers.positional_encoding import sinusoidal_pe
    from ..models.gnn.layers.projection import MacroReconstructionHead, SpectralProjectionHead
    from ..models.gnn.solver import EdgeWeightSolver
    from ..models.gnn.training import predict_gnn_scores
    from ..models.mlp import PointwiseEncoder, _models, load_mlp_checkpoint, predict_mlp_scores

    return code_projection(
        {
            "task_parameterization": task_parameters,
            "gnn_checkpoint_loader": load_gnn_cpu,
            "field_encoding": write_field,
            "field_frame": _frame,
            "field_derivation": infer_demand_field,
            "support_composition": compose_support_field,
            "fold_assignment": kfold_splits,
            "inference_assertion": assert_inference_graph,
            "supervision_sanitizer": sanitize_for_inference,
            "agent_gating_init": AgentGating.__init__,
            "agent_gating_forward": AgentGating.forward,
            "edge_weight_init": DifferentiableEdgeWeighting.__init__,
            "edge_weight_forward": DifferentiableEdgeWeighting.forward,
            "graph_encoder_init": GraphEncoder.__init__,
            "graph_encoder_forward": GraphEncoder.forward,
            "positional_encoding": sinusoidal_pe,
            "macro_head_init": MacroReconstructionHead.__init__,
            "macro_head_forward": MacroReconstructionHead.forward,
            "spectral_head_init": SpectralProjectionHead.__init__,
            "spectral_head_forward": SpectralProjectionHead.forward,
            "gnn_solver_init": EdgeWeightSolver.__init__,
            "gnn_solver_model_init": EdgeWeightSolver.init_model,
            "gnn_predict": EdgeWeightSolver.predict_edge_weights,
            "gnn_state_load": EdgeWeightSolver._load_checkpoint,
            "gnn_scores": predict_gnn_scores,
            "mlp_encoder_init": PointwiseEncoder.__init__,
            "mlp_encoder_forward": PointwiseEncoder.forward,
            "mlp_model_factory": _models,
            "mlp_state_load": load_mlp_checkpoint,
            "mlp_scores": predict_mlp_scores,
        }
    )
