import torch
import torch.nn as nn
import torch.nn.functional as F
from SpatialAllocation.GNN.core.ModelConfig import ModelConfig
from torch_scatter import scatter_softmax

class DifferentiableEdgeWeighting(nn.Module):
    """
    Differentiable edge weight prediction: predicts weights for known edges,
    ensuring the total weight for each source node sums to 1.
    """

    def __init__(self, config: ModelConfig):
        super(DifferentiableEdgeWeighting, self).__init__()
        self.config = config
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(self.config.allocation_temperature_start)))

        # Define an MLP to learn the edge cost (or "relationship score").
        # Input dimension is the size of the two concatenated embedding vectors (embedding_dim * 2).
        # Output dimension is 1, representing a scalar cost value.
        self.gating_mlp = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()  # Use Sigmoid to scale the output to the (0, 1) range
        )

    def forward(self, embeddings_s, embeddings_a, edge_index_sa, gate_values=None):
        """
        Predict edge weights.

        Args:
            embeddings_s: Source node embeddings [num_s, embedding_dim]
            embeddings_a: Agent node embeddings [num_a, embedding_dim]
            edge_index_sa: Edge index [2, num_edges], first row is the source node index,
                            second row is the agent node index
            gate_values: Optional agent validity gate values [num_a], range (0, 1).
                         When provided, g_a is multiplied into the softmax numerator,
                         so that agents with g_a -> 0 get weights approaching 0.

        Returns:
            edge_weights: Edge weights [num_edges]
            edge_costs: Edge costs [num_edges]
        """
        # 1. Compute the embedding distance for each edge
        s_indices = edge_index_sa[0]  # Source node indices
        a_indices = edge_index_sa[1]  # Agent node indices

        # Gather the embedding vectors corresponding to each edge
        edge_embeddings_s = embeddings_s[s_indices]  # [num_edges, embedding_dim]
        edge_embeddings_a = embeddings_a[a_indices]  # [num_edges, embedding_dim]

        # Compute the edge cost (embedding distance)
        edge_costs = torch.norm(edge_embeddings_s - edge_embeddings_a, dim=1)  # [num_edges]
        concatenated_embeddings = torch.cat([edge_embeddings_s, edge_embeddings_a], dim=1)
        gate = self.gating_mlp(concatenated_embeddings).squeeze(-1)
        edge_costs = edge_costs * gate


        # 2. Normalize with softmax using the temperature parameter
        temperature = torch.exp(self.log_temperature)

        # Prepare the values to be softmaxed.
        # The values to be softmaxed. We negate the cost because softmax gives higher probability to larger values.
        values_for_softmax = -edge_costs / temperature

        # If gate values are provided, add log(g_a) to the softmax input.
        # Mathematically equivalent to: w_sa = g_a * exp(-cost/tau) / sum(g_a' * exp(-cost'/tau))
        if gate_values is not None:
            epsilon = 1e-8
            edge_gate = gate_values[a_indices]
            values_for_softmax = values_for_softmax + torch.log(edge_gate + epsilon)

        # 3. Use scatter_softmax to perform a grouped softmax over each source node's edges
        # Use scatter_softmax to perform a grouped softmax for each source node's edges.
        # src: the input tensor -> values_for_softmax
        # index: the index to group by -> s_indices
        # dim: the dimension to operate on
        edge_weights = scatter_softmax(values_for_softmax, s_indices, dim=0)

        return edge_weights, edge_costs
