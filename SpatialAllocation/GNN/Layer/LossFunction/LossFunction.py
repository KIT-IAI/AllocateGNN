import torch
from torch import nn
import torch.nn.functional as F

from SpatialAllocation.GNN.Layer.LossFunction.LossRegistry import LossRegistry

loss_registry = LossRegistry()


class BaseLoss(nn.Module):
    """
    Base class for loss functions; parent class of all custom loss functions.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss(reduction='mean')

    def forward(self, edge_weights, edge_index, metadata):
        return 0


@loss_registry.register("entropy_regularization", default_weight=1,
                        description="Encourages the edge weights for each source node to be as uniform as possible.")
class EntropyLoss(BaseLoss):

    def forward(self, edge_weights, edge_index, metadata):
        """
        Optimized version: vectorized entropy calculation.
        """
        device = edge_weights.device
        # Directly use the passed edge_index; the first row contains source node indices
        s_indices = edge_index[0]
        epsilon = 1e-8

        # Safe log computation
        safe_weights = torch.clamp(edge_weights, min=epsilon)
        log_weights = torch.log(safe_weights)

        # Compute the entropy contribution of each weight
        entropy_terms = edge_weights * log_weights

        # Use scatter_add to group-sum for each source
        # BUG FIX: dimension here should be the number of source nodes, not agent nodes
        num_s = metadata["num_s"]
        entropy_per_s = torch.zeros(num_s, device=device)
        entropy_per_s.scatter_add_(0, s_indices, -entropy_terms)

        # Compute the number of edges per source
        ones = torch.ones_like(edge_weights)
        edges_per_s = torch.zeros(num_s, device=device)
        edges_per_s.scatter_add_(0, s_indices, ones)

        # Only compute loss for sources that have edges
        valid_mask = edges_per_s > 0
        if valid_mask.sum() > 0:
            valid_entropy = entropy_per_s[valid_mask]
            valid_edges = edges_per_s[valid_mask]

            # Compute theoretical maximum entropy
            max_entropy = torch.log(valid_edges)

            # Compute total loss: we want to maximize entropy, i.e., minimize (max_entropy - current_entropy)
            entropy_loss = torch.sum(max_entropy - valid_entropy)
        else:
            entropy_loss = torch.tensor(0.0, device=device)

        return entropy_loss

@loss_registry.register("supervised_substation_demand_loss", default_weight=1.0,
                        description="Minimizes the MSE between predicted and actual demand at the substation level.")
class SupervisedSubstationDemandLoss(BaseLoss):
    """
    Supervised loss: minimizes the mean squared error between predicted and actual demand
    at the substation level. This loss function is generic and does not depend on
    the specific meaning of 'source' nodes.
    """
    def forward(self, edge_weights, edge_index, metadata):
        """
        Compute supervised loss.

        Args:
            edge_weights (torch.Tensor): Model output, edge weights.
            edge_index (torch.Tensor): Edge indices; edge_index[1] contains agent nodes.
            metadata (dict): Must contain:
                - 'agent_demand' (torch.Tensor): Base demand for each agent node.
                - 'agent_substation_map' (torch.Tensor): Index mapping from each agent node to its substation.
                - 'substation_y_true' (torch.Tensor): True demand for each substation (supervision signal).
                - 'num_substations' (int): Total number of substations in the batch.
        """
        a_indices = edge_index[1]  # Agent node index for each edge

        # Retrieve required tensors from metadata
        agent_demand = metadata['agent_demand']
        agent_substation_map = metadata['agent_substation_map']
        substation_y_true = metadata['substation_y_true']
        num_substations = metadata['num_substations']

        # 1. Compute the demand contribution of each edge (weight * corresponding agent node's base demand)
        predicted_edge_demand = edge_weights * agent_demand[a_indices]

        # 2. Find the substation index that each edge belongs to
        edge_substation_map = agent_substation_map[a_indices]

        # 3. Use scatter_add to aggregate all edges' demand contributions by their assigned substation
        predicted_substation_demand = torch.zeros(num_substations, device=edge_weights.device)
        predicted_substation_demand.scatter_add_(0, edge_substation_map, predicted_edge_demand)

        # 4. Compute the MAE loss between aggregated predicted demand and true demand
        loss = self.mae(predicted_substation_demand, substation_y_true)

        return loss

@loss_registry.register("feature_similarity_loss", default_weight=0.1,
                        description="Encourages agents connected to the same source to have similar features.")
class FeatureSimilarityLoss(BaseLoss):
    """
    Regularization loss: minimizes the variance of agent node features connected to the same source.
    This encourages the model to learn more homogeneous allocation clusters in feature space.
    """
    def forward(self, edge_weights, edge_index, metadata):
        """
        Compute feature similarity loss.

        Args:
            edge_weights (torch.Tensor): Model output, edge weights.
            edge_index (torch.Tensor): Edge indices.
            metadata (dict): Must contain:
                - 'agent_features' (torch.Tensor): Feature matrix of agent nodes.
        """
        s_indices = edge_index[0]
        a_indices = edge_index[1]
        agent_features = metadata['agent_features']

        if agent_features is None or agent_features.numel() == 0:
            return torch.tensor(0.0, device=edge_weights.device)

        # Get agent features for each edge
        edge_agent_features = agent_features[a_indices]

        # Compute weighted feature mean for each source cluster
        # weight * feature
        weighted_features = edge_weights.unsqueeze(1) * edge_agent_features
        # Use scatter_add to aggregate the weighted feature sum for each source
        sum_weighted_features = torch.zeros((metadata['num_s'], agent_features.shape[1]), device=edge_weights.device)
        s_indices_expanded = s_indices.unsqueeze(1).expand_as(weighted_features)
        sum_weighted_features.scatter_add_(0, s_indices_expanded, weighted_features)

        # Use scatter_add to aggregate the weight sum for each source (should theoretically be 1, but recomputed for numerical stability)
        sum_weights = torch.zeros(metadata['num_s'], device=edge_weights.device)
        sum_weights.scatter_add_(0, s_indices, edge_weights)

        # Compute weighted average features; +1e-8 to prevent division by zero
        mean_features_per_s = sum_weighted_features / (sum_weights.unsqueeze(1) + 1e-8)

        # Compute the weighted distance (variance) between each agent and its source cluster centroid
        # (feature - mean_feature_of_its_source)^2
        squared_diff = torch.sum((edge_agent_features - mean_features_per_s[s_indices])**2, dim=1)

        # loss = weight * (feature - mean_feature)^2
        weighted_squared_diff = edge_weights * squared_diff

        # Aggregate to get weighted variance per source
        variance_per_s = torch.zeros(metadata['num_s'], device=edge_weights.device)
        variance_per_s.scatter_add_(0, s_indices, weighted_squared_diff)

        # Return average variance across all source clusters
        # Only consider sources that have edges
        valid_mask = sum_weights > 0
        if valid_mask.sum() > 0:
            return variance_per_s[valid_mask].mean()
        else:
            return torch.tensor(0.0, device=edge_weights.device)


@loss_registry.register("feature_consistency_loss", default_weight=0.1,
                        description="Encourages agents with similar features under the same source to have similar weights.")
class FeatureConsistencyLoss(BaseLoss):
    """
    Feature consistency loss function.
    This loss encourages agent nodes that belong to the same source node and are similar
    in feature space to have similar allocation weights.
    """

    def forward(self, edge_weights, edge_index, metadata):
        s_indices = edge_index[0]
        a_indices = edge_index[1]
        features_a = metadata['agent_features']
        num_s = metadata["num_s"]
        device = edge_weights.device

        total_loss = torch.tensor(0.0, device=device)

        # This implementation uses loops for clarity. For ultimate GPU performance, it can be further vectorized.
        for s_idx in range(num_s):
            mask = (s_indices == s_idx)
            # If a source node connects to fewer than 2 agents, variance cannot be computed
            if mask.sum() <= 1:
                continue

            # Get features and weights of all agents connected to this source
            agents_for_s_features = features_a[a_indices[mask]]
            weights_for_s = edge_weights[mask]

            # 1. Compute the mean feature vector of all agents under this source
            mean_features_s = torch.mean(agents_for_s_features, dim=0)

            # 2. Compute cosine similarity between each agent's features and the mean features
            agents_for_s_features_norm = F.normalize(agents_for_s_features, p=2, dim=1)
            mean_features_s_norm = F.normalize(mean_features_s.unsqueeze(0), p=2, dim=1)

            # Higher similarity means the agent better represents the "typical" features of this source
            similarities = torch.mm(agents_for_s_features_norm, mean_features_s_norm.t()).squeeze()

            # 3. We want agents with typical features (high similarity) to have similar weights,
            # so we use similarity to weight the variance of weights and minimize it
            mean_weight_s = torch.mean(weights_for_s)

            variance = (weights_for_s - mean_weight_s) ** 2
            weighted_variance = similarities * variance

            total_loss += torch.mean(weighted_variance)

        return total_loss / num_s if num_s > 0 else total_loss

@loss_registry.register("landuse_prediction_loss", default_weight=1.0,
                        description="Computes loss between predicted and actual landuse ratios based on edge weights.")
class LandusePredictionLoss(BaseLoss):
    def forward(self, edge_weights, edge_index, metadata):
        """
        Predict landuse ratios based on edge weights and compute loss.

        Args:
            edge_weights: [num_edges] Predicted edge weights
            edge_index_mapping: Edge index mapping
            metadata: Metadata containing landuse_mapping_matrix and landuse_ratio
        """
        if 'landuse_mapping_matrix' not in metadata or 'landuse_ratio' not in metadata:
            print("Warning: Missing landuse mapping matrix or true ratio in metadata, returning zero loss")
            return torch.tensor(0.0, device=edge_weights.device)

        device = edge_weights.device
        mapping_matrix = metadata['landuse_mapping_matrix'].to(device)  # [num_edges, num_regions * num_landuse_types]
        true_landuse_ratio = metadata['landuse_ratio'].to(device)  # [num_regions, num_landuse_types]

        # Validate input dimensions
        if mapping_matrix.shape[0] != edge_weights.shape[0]:
            print(f"Warning: Mapping matrix rows ({mapping_matrix.shape[0]}) do not match edge weight count ({edge_weights.shape[0]})")
            return torch.tensor(0.0, device=device)

        if mapping_matrix.shape[1] == 0 or true_landuse_ratio.shape[1] == 0:
            # No landuse type information, return zero loss
            print("Warning: Mapping matrix or true landuse ratio has no type information, returning zero loss")
            return torch.tensor(0.0, device=device)

        num_regions = metadata['num_s']
        num_landuse_types = true_landuse_ratio.shape[1]

        # =================================================================
        # Step 1: Compute predicted landuse aggregation via edge weights and mapping matrix
        # =================================================================
        # Matrix multiplication for weighted aggregation:
        # edge_weights: [num_edges] @ mapping_matrix: [num_edges, num_regions * num_landuse_types]
        # -> [num_regions * num_landuse_types]
        #
        # Example: Region 0 connects to 3 agents, weights [0.5, 0.3, 0.2], types [residential, commercial, industrial]
        #          Region 1 connects to 2 agents, weights [0.7, 0.3], both residential
        #          After aggregation: [0.5, 0.3, 0.2, 1.0, 0.0, 0.0] (flattened format)
        predicted_flat = torch.matmul(edge_weights, mapping_matrix)

        # =================================================================
        # Step 2: Reshape to [num_regions, num_landuse_types] for easier processing
        # =================================================================
        # Reshape flattened result into a 2D matrix:
        # [[0.5, 0.3, 0.2],   # region 0: res=0.5, com=0.3, ind=0.2
        #  [1.0, 0.0, 0.0]]   # region 1: res=1.0, com=0.0, ind=0.0
        predicted_landuse_aggregated = predicted_flat.view(num_regions, num_landuse_types)

        # =================================================================
        # Step 3: Normalization - a critical step! Why can't we skip this?
        # =================================================================
        # Problem 1: Scale mismatch
        #   - predicted_landuse_aggregated may have arbitrary row sums (0.5, 1.7, 2.3, etc.)
        #   - true_landuse_ratio has row sums of 1.0 (standard probability distribution)
        #
        # Problem 2: Different number of agents per region
        #   - Region A has 10 agents -> larger aggregated values
        #   - Region B has 3 agents -> smaller aggregated values
        #   - Direct comparison would bias toward regions with more agents
        #
        # Problem 3: Imperfect weight constraints during training
        #   - A region's weight sum might be 0.95 or 1.05, not strictly 1.0
        #
        # Solution: Convert absolute values to relative proportions, ensuring predictions and
        # ground truth are in the same semantic space
        epsilon = 1e-8  # Prevent division by zero
        region_totals = torch.sum(predicted_landuse_aggregated, dim=1, keepdim=True) + epsilon
        predicted_landuse_ratio = predicted_landuse_aggregated / region_totals


        # Use KL divergence loss
        true_landuse_safe = torch.clamp(true_landuse_ratio, min=epsilon)
        predicted_landuse_safe = torch.clamp(predicted_landuse_ratio, min=epsilon)
        loss = F.kl_div(torch.log(predicted_landuse_safe), true_landuse_safe, reduction='batchmean')

        return loss