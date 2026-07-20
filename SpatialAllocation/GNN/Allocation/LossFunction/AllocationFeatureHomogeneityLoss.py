import torch
from torch import nn

from SpatialAllocation.GNN.Allocation.LossFunction.AllocationLossRegistry import allocation_loss_registry
from SpatialAllocation.GNN.Layer.LossFunction.LossFunction import BaseLoss


@allocation_loss_registry.register(
    "allocation_feature_homogeneity",
    default_weight=0.5,
    description="Feature homogeneity loss: encourages each target's service region to be compact in feature space (the core value GNN adds over Voronoi)."
)
class AllocationFeatureHomogeneityLoss(BaseLoss):
    """
    Feature homogeneity loss L_homo: the core source of value GNN provides over Voronoi.

    Formula:
        centroid_t = Sum_a (w_at x f_a) / Sum_a (w_at)     # weighted feature centroid of target t
        L_homo = mean_t [ (1/deg_t) x Sum_a (w_at x ||f_a - centroid_t||^2) ]

    Design motivation:
        - Voronoi ignores agent features -> may assign agents with different landuse to the same target
        - This loss encourages each target's service region to be compact in feature space
        - Equivalent to the objective of weighted k-means in feature space
        - Creates tension with the distance loss -> GNN learns feature-aware spatial allocation

    Note:
        - Uses f_a.detach() to freeze feature gradients (only the weights are optimized, not the feature representations)

    metadata requires:
        - 'agent_features': (num_a, feat_dim) agent feature matrix
        - 'edge_index_at': (2, num_edges) agent->target edge index
        - 'num_targets': int, number of target nodes
    """

    def forward(self, edge_weights, edge_index, metadata):
        device = edge_weights.device

        agent_features = metadata.get('agent_features')
        edge_index_at = metadata.get('edge_index_at')
        num_targets = metadata.get('num_targets')

        if (agent_features is None or edge_index_at is None
                or num_targets is None):
            return torch.tensor(0.0, device=device)

        if num_targets == 0 or edge_index_at.shape[1] == 0:
            return torch.tensor(0.0, device=device)

        a_indices = edge_index_at[0]  # (num_edges,)
        t_indices = edge_index_at[1]  # (num_edges,)

        # Freeze feature gradients: only optimize the weights
        f_a = agent_features[a_indices].detach()  # (num_edges, feat_dim)
        feat_dim = f_a.shape[1]

        epsilon = 1e-8

        # Compute the weighted feature centroid for each target
        # centroid_t = Sum_a (w_at x f_a) / Sum_a (w_at)
        weighted_features = edge_weights.unsqueeze(1) * f_a  # (num_edges, feat_dim)
        centroid_sum = torch.zeros(num_targets, feat_dim, device=device)
        t_exp = t_indices.unsqueeze(1).expand_as(weighted_features)
        centroid_sum.scatter_add_(0, t_exp, weighted_features)

        weight_sum = torch.zeros(num_targets, device=device)
        weight_sum.scatter_add_(0, t_indices, edge_weights)

        centroids = centroid_sum / (weight_sum.unsqueeze(1) + epsilon)  # (num_targets, feat_dim)

        # Per-edge: squared distance between the agent feature and its target's centroid
        edge_centroids = centroids[t_indices]  # (num_edges, feat_dim)
        dist_sq = torch.sum((f_a - edge_centroids) ** 2, dim=1)  # (num_edges,)

        # Weighted squared distance
        weighted_dist_sq = edge_weights * dist_sq  # (num_edges,)

        # Aggregate by target
        target_loss = torch.zeros(num_targets, device=device)
        target_loss.scatter_add_(0, t_indices, weighted_dist_sq)

        # Number of edges (degree) per target
        edge_count = torch.zeros(num_targets, device=device)
        edge_count.scatter_add_(0, t_indices, torch.ones_like(edge_weights))

        # Normalize: divide each target by its degree
        valid_mask = edge_count > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        target_loss_normed = target_loss[valid_mask] / edge_count[valid_mask]

        # Average loss across all targets
        loss = target_loss_normed.mean()

        return loss
