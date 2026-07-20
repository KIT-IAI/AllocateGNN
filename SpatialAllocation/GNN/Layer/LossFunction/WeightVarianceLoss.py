import torch
from torch import nn

from SpatialAllocation.GNN.Layer.LossFunction.LossFunction import loss_registry, BaseLoss


@loss_registry.register(
    "weight_variance",
    default_weight=0.1,
    description="Weight non-uniformity loss: encourages edge weights to maintain moderate non-uniformity under each source, preventing uniform degeneration."
)
class WeightVarianceLoss(BaseLoss):
    """
    Weight non-uniformity loss L_weight_variance.

    For each source node, compute the negative entropy of its agents' edge
    weights (minimizing negative entropy = maximizing the opposite of
    entropy). Uses a negative-variance form to encourage differentiation in
    the weight distribution, preventing all agents from receiving the same
    weight.

    Formula:
        L_var = -mean_over_s( Var(w_sa[i] for i in agents_of(s)) )

    Motivation:
        landuse_prediction_loss allows uniform weights w=1/N as an optimal
        solution (when the agents' average landuse composition is close to
        the regional statistics). Under uniform weights, loss ~= 0, but
        spatial information is completely lost. This loss forces the model
        to learn spatial differences by maximizing weight variance.

    Note:
        This loss runs in the opposite direction to entropy_regularization
        (which encourages uniformity). The two should not be used together.

    metadata must contain:
        - 'num_s': int, number of source nodes
    """

    def forward(self, edge_weights, edge_index, metadata):
        device = edge_weights.device
        s_indices = edge_index[0]
        num_s = metadata.get('num_s')

        if num_s is None or num_s == 0:
            return torch.tensor(0.0, device=device)

        # Compute the mean weight for each source
        weight_sum = torch.zeros(num_s, device=device)
        weight_sum.scatter_add_(0, s_indices, edge_weights)

        edge_count = torch.zeros(num_s, device=device)
        edge_count.scatter_add_(0, s_indices, torch.ones_like(edge_weights))

        valid_mask = edge_count > 1  # At least 2 agents are needed to compute variance
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        weight_mean = weight_sum / edge_count.clamp(min=1.0)

        # Compute the squared deviation of each edge from its source mean
        mean_per_edge = weight_mean[s_indices]  # (num_edges,)
        sq_diff = (edge_weights - mean_per_edge) ** 2

        # Aggregate the variance for each source
        var_sum = torch.zeros(num_s, device=device)
        var_sum.scatter_add_(0, s_indices, sq_diff)

        variance_per_s = var_sum / edge_count.clamp(min=1.0)

        # Take the average variance over valid sources, negate it (minimizing negative variance = maximizing variance)
        loss = -variance_per_s[valid_mask].mean()

        return loss
