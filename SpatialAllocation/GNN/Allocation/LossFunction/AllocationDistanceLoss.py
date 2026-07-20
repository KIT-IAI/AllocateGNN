import torch
from torch import nn

from SpatialAllocation.GNN.Allocation.LossFunction.AllocationLossRegistry import allocation_loss_registry
from SpatialAllocation.GNN.Layer.LossFunction.LossFunction import BaseLoss


@allocation_loss_registry.register(
    "allocation_distance",
    default_weight=1.0,
    description="Distance prior loss: encourages agents to be allocated to nearby targets (soft Voronoi baseline)."
)
class AllocationDistanceLoss(BaseLoss):
    """
    Distance prior loss L_distance: spatial baseline for agent->target allocation.

    Formula:
        d^2_at = ||coord_a - coord_t||^2          # squared Euclidean distance
        d^2_norm = z-score(d^2_at, per-agent)     # per-agent z-score normalization
        L_dist = mean_a [ Sum_t (w_at x d^2_norm) ]

    Design choices:
        - Squared distance: consistent with the k-means/Voronoi objective, giving smoother gradients
        - Per-agent z-score: different agents have different k-NN distance scales; normalization ensures comparability
        - Converges to soft Voronoi when used alone

    metadata requires:
        - 'agent_coords': (num_a, 2) agent coordinates
        - 'target_coords': (num_t, 2) target coordinates
        - 'edge_index_at': (2, num_edges) agent->target edge index
    """

    def forward(self, edge_weights, edge_index, metadata):
        device = edge_weights.device

        agent_coords = metadata.get('agent_coords')
        target_coords = metadata.get('target_coords')
        edge_index_at = metadata.get('edge_index_at')

        if (agent_coords is None or target_coords is None
                or edge_index_at is None):
            return torch.tensor(0.0, device=device)

        if edge_index_at.shape[1] == 0:
            return torch.tensor(0.0, device=device)

        a_indices = edge_index_at[0]  # (num_edges,)
        t_indices = edge_index_at[1]  # (num_edges,)
        num_a = metadata.get('num_a', int(a_indices.max().item()) + 1)

        # Squared Euclidean distance per edge
        coord_a = agent_coords[a_indices]  # (num_edges, 2)
        coord_t = target_coords[t_indices]  # (num_edges, 2)
        dist_sq = torch.sum((coord_a - coord_t) ** 2, dim=1)  # (num_edges,)

        # Per-agent z-score normalization
        epsilon = 1e-8

        # Mean distance per agent
        dist_sum = torch.zeros(num_a, device=device)
        dist_sum.scatter_add_(0, a_indices, dist_sq)
        edge_count = torch.zeros(num_a, device=device)
        edge_count.scatter_add_(0, a_indices, torch.ones_like(dist_sq))
        dist_mean = dist_sum / edge_count.clamp(min=1.0)

        # Standard deviation of distance per agent
        diff_sq = (dist_sq - dist_mean[a_indices]) ** 2
        var_sum = torch.zeros(num_a, device=device)
        var_sum.scatter_add_(0, a_indices, diff_sq)
        dist_std = torch.sqrt(var_sum / edge_count.clamp(min=1.0) + epsilon)

        # z-score normalized distance
        d_norm = (dist_sq - dist_mean[a_indices]) / (dist_std[a_indices] + epsilon)

        # Loss: mean_a [ Sum_t (w_at x d^2_norm) ]
        loss = torch.mean(edge_weights * d_norm)

        return loss
