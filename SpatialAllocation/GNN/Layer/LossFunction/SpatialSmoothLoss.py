import torch
from torch import nn

from SpatialAllocation.GNN.Layer.LossFunction.LossFunction import loss_registry, BaseLoss


@loss_registry.register(
    "spatial_smooth",
    default_weight=0.5,
    description="Spatial smoothness loss: penalizes edge weight differences between neighboring agents, encouraging weights to vary continuously in space."
)
class SpatialSmoothLoss(BaseLoss):
    """
    Spatial smoothness loss L_spatial_smooth.

    For agent pairs (i, j) with Moore 8-connectivity, minimize the squared
    difference of their edge weights. This is the only loss function that
    requires agent-agent adjacency edges, giving the GNN's message passing
    actual meaning.

    Formula:
        L_smooth = mean( (w_sa[i] - w_sa[j])^2 )   for all (i,j) in agent_adj_pairs

    Motivation:
        landuse_prediction_loss only constrains the weighted-average
        composition ratio, not the spatial distribution of weights. Uniform
        weights and spatially sharp weights can produce the same landuse
        loss, but the downstream RMSE differs greatly. The spatial
        smoothness loss forces weights to vary continuously in space,
        avoiding winner-take-all degeneracy while preserving meaningful
        spatial differences.

    metadata must contain:
        - 'agent_adj_pairs': (2, num_pairs) agent adjacency edge pairs (Moore 8-connectivity)
    """

    def forward(self, edge_weights, edge_index, metadata):
        device = edge_weights.device

        agent_adj_pairs = metadata.get('agent_adj_pairs')
        if agent_adj_pairs is None or agent_adj_pairs.shape[1] == 0:
            return torch.tensor(0.0, device=device)

        # edge_index[1] is the agent index, in one-to-one correspondence with edge_weights
        # Build a mapping from agent_index -> edge_weight
        a_indices = edge_index[1]  # (num_edges,)
        num_a = metadata.get('num_a')
        if num_a is None:
            num_a = int(a_indices.max().item()) + 1

        # Map edge weights back to the agent dimension
        # Each agent has only one edge under the same source, so it is a one-to-one correspondence
        agent_weights = torch.zeros(num_a, device=device)
        agent_weights[a_indices] = edge_weights

        # Get the indices of neighboring agent pairs
        adj_i = agent_adj_pairs[0]  # (num_pairs,)
        adj_j = agent_adj_pairs[1]  # (num_pairs,)

        # Compute the squared weight difference between neighboring agents
        w_i = agent_weights[adj_i]
        w_j = agent_weights[adj_j]
        diff_sq = (w_i - w_j) ** 2

        loss = diff_sq.mean()

        return loss
