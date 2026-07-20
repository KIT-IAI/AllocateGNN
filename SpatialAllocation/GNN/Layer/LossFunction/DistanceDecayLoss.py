import torch
from torch import nn

from SpatialAllocation.GNN.Layer.LossFunction.LossFunction import loss_registry, BaseLoss


@loss_registry.register(
    "distance_decay",
    default_weight=0.1,
    description="Distance decay prior loss: encourages agents farther from the source centroid to receive lower weights, encoding a spatial decay prior."
)
class DistanceDecayLoss(BaseLoss):
    """
    Distance decay prior loss L_distance_decay.

    For each source node, compute the distance from its associated agents to
    the source's weighted centroid, then minimize the dot product of the
    weights and the normalized distance (i.e. encourage lower weights for
    distant agents).

    Formula:
        centroid(s) = Sum_a (w_sa x coord(a)) / Sum_a w_sa   (weighted centroid)
        d_sa = ||coord(a) - centroid(s)||_2
        d_norm = (d_sa - mean(d_sa)) / std(d_sa)             (per-source z-score)
        L_decay = mean( w_sa x d_norm )

    Motivation:
        landuse_prediction_loss does not include any spatial coordinate
        information. Physical prior: agents farther from the source center
        typically contribute less to that source. This loss adds a spatial
        constraint dimension that is orthogonal to landuse composition
        matching.

    Complementarity with L_spatial_smooth:
        - L_smooth is a local constraint -- neighboring agents have similar weights
        - L_decay is a global constraint -- weights decay from center to edge
        - The two are orthogonal and do not conflict

    metadata must contain:
        - 'agent_coords': (num_a, 2) agent coordinates (EPSG:27700)
        - 'num_s': int, number of source nodes
    """

    def forward(self, edge_weights, edge_index, metadata):
        device = edge_weights.device

        agent_coords = metadata.get('agent_coords')
        num_s = metadata.get('num_s')

        if agent_coords is None or num_s is None or num_s == 0:
            return torch.tensor(0.0, device=device)

        s_indices = edge_index[0]  # (num_edges,)
        a_indices = edge_index[1]  # (num_edges,)

        # Get the agent coordinates corresponding to each edge
        edge_coords = agent_coords[a_indices]  # (num_edges, 2)

        epsilon = 1e-8

        # Compute the weighted centroid for each source
        weighted_coords = edge_weights.unsqueeze(1) * edge_coords  # (num_edges, 2)
        centroid_sum = torch.zeros(num_s, 2, device=device)
        s_exp = s_indices.unsqueeze(1).expand_as(weighted_coords)
        centroid_sum.scatter_add_(0, s_exp, weighted_coords)

        weight_sum = torch.zeros(num_s, device=device)
        weight_sum.scatter_add_(0, s_indices, edge_weights)

        centroids = centroid_sum / (weight_sum.unsqueeze(1) + epsilon)  # (num_s, 2)

        # Compute the Euclidean distance from each edge's agent to its source centroid
        edge_centroids = centroids[s_indices]  # (num_edges, 2)
        dist = torch.sqrt(
            torch.sum((edge_coords - edge_centroids) ** 2, dim=1) + epsilon
        )  # (num_edges,)

        # Per-source z-score normalization: makes distance magnitudes comparable across different sources
        dist_mean = torch.zeros(num_s, device=device)
        dist_mean.scatter_add_(0, s_indices, dist)
        edge_count = torch.zeros(num_s, device=device)
        edge_count.scatter_add_(0, s_indices, torch.ones_like(dist))
        dist_mean = dist_mean / edge_count.clamp(min=1.0)

        dist_sq_sum = torch.zeros(num_s, device=device)
        dist_sq_sum.scatter_add_(0, s_indices, (dist - dist_mean[s_indices]) ** 2)
        dist_std = torch.sqrt(dist_sq_sum / edge_count.clamp(min=1.0) + epsilon)

        # Normalized distance
        d_norm = (dist - dist_mean[s_indices]) / (dist_std[s_indices] + epsilon)

        # Loss: minimize w x d_norm -> distant agents get lower weights
        loss = torch.mean(edge_weights * d_norm)

        return loss
