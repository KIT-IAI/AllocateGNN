import torch
from torch import nn
import torch.nn.functional as F

from SpatialAllocation.GNN.Layer.LossFunction.LossRegistry import LossRegistry

loss_registry = LossRegistry()


class BaseLoss(nn.Module):
    """
    Base class for loss functions; the parent class of all custom loss functions.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss(reduction='mean')
    # Modified the base class's forward signature
    def forward(self, edge_weights, edge_index, metadata):
        return 0


@loss_registry.register("entropy_regularization", default_weight=1,
                        description="Encourages the edge weights for each source node to be as uniform as possible.")
class EntropyLoss(BaseLoss):

    # Modified the forward signature and updated the internal logic
    def forward(self, edge_weights, edge_index, metadata):
        """
        Optimized version: vectorized entropy computation
        """
        device = edge_weights.device
        # Use the passed-in edge_index directly; the first row is the source node indices
        s_indices = edge_index[0]
        epsilon = 1e-8

        # Safe log computation
        safe_weights = torch.clamp(edge_weights, min=epsilon)
        log_weights = torch.log(safe_weights)

        # Compute the entropy contribution of each weight
        entropy_terms = edge_weights * log_weights

        # Use scatter_add to sum by source group
        # Bug fix: this dimension should be the number of source nodes, not the number of agent nodes
        num_s = metadata["num_s"]
        entropy_per_s = torch.zeros(num_s, device=device)
        entropy_per_s.scatter_add_(0, s_indices, -entropy_terms)

        # Compute the number of edges for each source
        ones = torch.ones_like(edge_weights)
        edges_per_s = torch.zeros(num_s, device=device)
        edges_per_s.scatter_add_(0, s_indices, ones)

        # Only compute the loss for sources that have edges
        valid_mask = edges_per_s > 0
        if valid_mask.sum() > 0:
            valid_entropy = entropy_per_s[valid_mask]
            valid_edges = edges_per_s[valid_mask]

            # Compute the theoretical maximum entropy
            max_entropy = torch.log(valid_edges)

            # Compute the total loss; we want to maximize entropy, i.e. minimize (max entropy - current entropy)
            entropy_loss = torch.sum(max_entropy - valid_entropy)
        else:
            entropy_loss = torch.tensor(0.0, device=device)

        return entropy_loss

@loss_registry.register("feature_similarity_loss", default_weight=0.1,
                        description="Encourages agents connected to the same source to have similar features.")
class FeatureSimilarityLoss(BaseLoss):
    """
    Regularization loss: minimizes the feature variance of agent nodes
    connected to the same source. This encourages the model to learn
    allocation clusters that are more homogeneous in feature space.
    """
    def forward(self, edge_weights, edge_index, metadata):
        """
        Compute the feature similarity loss.

        Args:
            edge_weights (torch.Tensor): the model's output, the edge weights.
            edge_index (torch.Tensor): the edge indices.
            metadata (dict): must contain:
                - 'agent_features' (torch.Tensor): the feature matrix of agent nodes.
        """
        s_indices = edge_index[0]
        a_indices = edge_index[1]
        agent_features = metadata['agent_features']

        if agent_features is None or agent_features.numel() == 0:
            return torch.tensor(0.0, device=edge_weights.device)

        # Get the agent features corresponding to each edge
        edge_agent_features = agent_features[a_indices]

        # Compute the weighted feature mean for each source cluster
        # weight * feature
        weighted_features = edge_weights.unsqueeze(1) * edge_agent_features
        # Use scatter_add to aggregate the weighted feature sum for each source
        sum_weighted_features = torch.zeros((metadata['num_s'], agent_features.shape[1]), device=edge_weights.device)
        s_indices_expanded = s_indices.unsqueeze(1).expand_as(weighted_features)
        sum_weighted_features.scatter_add_(0, s_indices_expanded, weighted_features)

        # Use scatter_add to aggregate the weight sum for each source (should theoretically be 1, but recomputed for numerical stability)
        sum_weights = torch.zeros(metadata['num_s'], device=edge_weights.device)
        sum_weights.scatter_add_(0, s_indices, edge_weights)

        # Compute the weighted average feature, +1e-8 to avoid division by zero
        mean_features_per_s = sum_weighted_features / (sum_weights.unsqueeze(1) + 1e-8)

        # Compute the weighted distance (variance) of each agent from its source cluster's center feature
        # (feature - mean_feature_of_its_source)^2
        squared_diff = torch.sum((edge_agent_features - mean_features_per_s[s_indices])**2, dim=1)

        # loss = weight * (feature - mean_feature)^2
        weighted_squared_diff = edge_weights * squared_diff

        # Aggregate to get the weighted variance for each source
        variance_per_s = torch.zeros(metadata['num_s'], device=edge_weights.device)
        variance_per_s.scatter_add_(0, s_indices, weighted_squared_diff)

        # Return the average variance across all source clusters
        # Only consider sources that have edges
        valid_mask = sum_weights > 0
        if valid_mask.sum() > 0:
            return variance_per_s[valid_mask].mean()
        else:
            return torch.tensor(0.0, device=edge_weights.device)


@loss_registry.register("feature_consistency_loss", default_weight=0.1,
                        description="Moore-adjacency pairwise: neighbors with similar features are pushed toward consistent weights, while neighbors with different features are allowed to diverge.")
class FeatureConsistencyLoss(BaseLoss):
    """
    Feature-aware consistency loss based on Moore adjacency.

    Formula:
        L = mean( sim(f_i, f_j) x (w_{s,i} - w_{s,j})^2 )
            for each Moore pair (i,j) where both agents connect to source s

    Semantics:
        - sim > 0 (similar features): penalizes weight differences -> pushes toward consistency
        - sim < 0 (dissimilar features): rewards weight differences -> allows divergence
        - sim ~= 0: neutral, no constraint

    Difference from SpatialSmoothLoss:
        - SpatialSmoothLoss: undifferentiated (w_i - w_j)^2, not source-specific
        - This loss: sim(f_i, f_j) x (w_i - w_j)^2, only for Moore pairs within the same source
    """

    def forward(self, edge_weights, edge_index, metadata):
        s_indices = edge_index[0]
        a_indices = edge_index[1]
        features_a = metadata['agent_features']
        num_s = metadata["num_s"]
        num_a = metadata.get('num_a')
        device = edge_weights.device

        agent_adj_pairs = metadata.get('agent_adj_pairs')
        if agent_adj_pairs is None or agent_adj_pairs.shape[1] == 0:
            return torch.tensor(0.0, device=device)

        if features_a is None or features_a.numel() == 0:
            return torch.tensor(0.0, device=device)

        if num_a is None:
            num_a = int(a_indices.max().item()) + 1

        adj_i = agent_adj_pairs[0]  # (P,)
        adj_j = agent_adj_pairs[1]  # (P,)

        # Precompute L2-normalized features for all agents
        features_norm = F.normalize(features_a, p=2, dim=1)

        # Cosine similarity of features for Moore pairs (computed once, shared across all sources)
        sim = (features_norm[adj_i] * features_norm[adj_j]).sum(dim=1)  # (P,)

        total_loss = torch.tensor(0.0, device=device)
        valid_sources = 0

        for s in range(num_s):
            s_mask = (s_indices == s)
            if s_mask.sum() <= 1:
                continue

            # The set of agents connected to this source and their weights
            s_agents = a_indices[s_mask]
            s_weights = edge_weights[s_mask]

            # agent_idx -> whether connected to this source
            agent_has_edge = torch.zeros(num_a, dtype=torch.bool, device=device)
            agent_has_edge[s_agents] = True

            # Filter: both agents in the Moore pair are connected to this source
            valid_pair = agent_has_edge[adj_i] & agent_has_edge[adj_j]
            if valid_pair.sum() == 0:
                continue

            # agent_idx -> weight under this source
            agent_weight = torch.zeros(num_a, device=device)
            agent_weight[s_agents] = s_weights

            w_i = agent_weight[adj_i[valid_pair]]
            w_j = agent_weight[adj_j[valid_pair]]
            sim_valid = sim[valid_pair]

            # Log-space: sim x (log w_i - log w_j)^2
            # Sensitive to weight ratio rather than absolute difference, avoiding numerical collapse to 1e-6 when N is large
            epsilon = 1e-8
            log_w_i = torch.log(w_i + epsilon)
            log_w_j = torch.log(w_j + epsilon)
            diff_sq = (log_w_i - log_w_j) ** 2
            total_loss = total_loss + (sim_valid * diff_sq).mean()
            valid_sources += 1

        return total_loss / valid_sources if valid_sources > 0 else total_loss

@loss_registry.register("landuse_prediction_loss", default_weight=1.0,
                        description="Computes loss between predicted and actual landuse ratios based on edge weights.")
class LandusePredictionLoss(BaseLoss):
    def forward(self, edge_weights, edge_index, metadata):
        """
        Predict land-use ratios from edge weights and compute the loss.

        Args:
            edge_weights: [num_edges] the predicted edge weights
            edge_index_mapping: the edge index mapping
            metadata: metadata containing landuse_mapping_matrix and landuse_ratio
        """
        if 'landuse_mapping_matrix' not in metadata or 'landuse_ratio' not in metadata:
            print("Warning: metadata is missing the land-use mapping matrix or ground-truth ratio information, returning zero loss")
            return torch.tensor(0.0, device=edge_weights.device)

        device = edge_weights.device
        mapping_matrix = metadata['landuse_mapping_matrix'].to(device)  # [num_edges, num_regions * num_landuse_types]
        true_landuse_ratio = metadata['landuse_ratio'].to(device)  # [num_regions, num_landuse_types]

        # Check whether the input data is on the correct device
        # Check whether the matrix dimensions match
        if mapping_matrix.shape[0] != edge_weights.shape[0]:
            print(f"Warning: mapping matrix row count ({mapping_matrix.shape[0]}) does not match the number of edge weights ({edge_weights.shape[0]})")
            return torch.tensor(0.0, device=device)

        if mapping_matrix.shape[1] == 0 or true_landuse_ratio.shape[1] == 0:
            # No land-use type information, return zero loss
            print("Warning: the mapping matrix or ground-truth land-use ratio has no type information, returning zero loss")
            return torch.tensor(0.0, device=device)

        num_regions = metadata['num_s']
        num_landuse_types = true_landuse_ratio.shape[1]

        # =================================================================
        # Step 1: Compute the predicted land-use aggregate values from edge weights and the mapping matrix
        # =================================================================
        # Weighted aggregation via matrix multiplication:
        # edge_weights: [num_edges] @ mapping_matrix: [num_edges, num_regions * num_landuse_types]
        # -> [num_regions * num_landuse_types]
        #
        # Example: Region 0 connects to 3 agents, weights [0.5, 0.3, 0.2], types [residential, commercial, industrial]
        #      Region 1 connects to 2 agents, weights [0.7, 0.3], both residential type
        #      After aggregation: [0.5, 0.3, 0.2, 1.0, 0.0, 0.0] (flattened format)
        predicted_flat = torch.matmul(edge_weights, mapping_matrix)

        # =================================================================
        # Step 2: Reshape to [num_regions, num_landuse_types] for easier processing
        # =================================================================
        # Reshape the flattened result into a 2D matrix:
        # [[0.5, 0.3, 0.2],   # region 0: res=0.5, com=0.3, ind=0.2
        #  [1.0, 0.0, 0.0]]   # region 1: res=1.0, com=0.0, ind=0.0
        predicted_landuse_aggregated = predicted_flat.view(num_regions, num_landuse_types)

        # =================================================================
        # Step 3: Normalization -- a critical step! Why can't this be skipped?
        # =================================================================
        # Problem 1: Mismatched numerical scales
        #   - predicted_landuse_aggregated can have an arbitrary row sum (0.5, 1.7, 2.3, etc.)
        #   - true_landuse_ratio has a row sum of 1.0 (a standard probability distribution)
        #
        # Problem 2: Differing agent counts across regions
        #   - Region A has 10 agents -> larger aggregate value
        #   - Region B has 3 agents -> smaller aggregate value
        #   - A direct comparison would bias toward regions with more agents
        #
        # Problem 3: Imperfect weight constraints during training
        #   - A region's weight sum might be 0.95 or 1.05 rather than exactly 1.0
        #
        # Solution: convert absolute values to relative proportions, ensuring the
        # predicted and ground-truth values live in the same semantic space
        epsilon = 1e-8  # avoid division by zero
        region_totals = torch.sum(predicted_landuse_aggregated, dim=1, keepdim=True) + epsilon
        predicted_landuse_ratio = predicted_landuse_aggregated / region_totals


        # Use KL-divergence loss
        true_landuse_safe = torch.clamp(true_landuse_ratio, min=epsilon)
        predicted_landuse_safe = torch.clamp(predicted_landuse_ratio, min=epsilon)
        loss = F.kl_div(torch.log(predicted_landuse_safe), true_landuse_safe, reduction='batchmean')

        return loss
