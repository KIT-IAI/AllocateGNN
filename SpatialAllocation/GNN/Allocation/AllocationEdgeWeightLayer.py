import torch
import torch.nn as nn
from torch_scatter import scatter_softmax
from SpatialAllocation.GNN.Allocation.AllocationConfig import AllocationConfig


class AllocationEdgeWeighting(nn.Module):
    """
    Stage 2 edge weight prediction: predicts agent->target edge weights with per-agent softmax normalization.

    Input: agent embedding concatenated with target embedding
    Output: per-agent scatter_softmax weights (weights sum to 1 across each agent's k-NN targets)
    """

    def __init__(self, config: AllocationConfig):
        super(AllocationEdgeWeighting, self).__init__()
        self.config = config

        # Learnable temperature
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(config.allocation_temperature))
        )

        # MLP gate: input dimension is the concatenation of agent and target embeddings
        self.gating_mlp = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h_a, h_t, edge_index_at):
        """
        Predicts agent->target edge weights.

        Args:
            h_a: Agent embeddings [num_a, embedding_dim]
            h_t: Target embeddings [num_t, embedding_dim]
            edge_index_at: Edge index [2, num_edges], row 0 = agent index, row 1 = target index

        Returns:
            edge_weights: Edge weights [num_edges], per-agent normalized
            edge_costs: Edge costs [num_edges]
        """
        a_indices = edge_index_at[0]  # agent index
        t_indices = edge_index_at[1]  # target index

        # Gather the embedding vectors corresponding to each edge
        edge_h_a = h_a[a_indices]  # [num_edges, embedding_dim]
        edge_h_t = h_t[t_indices]  # [num_edges, embedding_dim]

        # Compute edge cost (L2 distance x MLP gate)
        edge_costs = torch.norm(edge_h_a - edge_h_t, dim=1)  # [num_edges]
        concatenated = torch.cat([edge_h_a, edge_h_t], dim=1)
        gate = self.gating_mlp(concatenated).squeeze(-1)
        edge_costs = edge_costs * gate

        # Apply the learnable temperature
        temperature = torch.exp(self.log_temperature)

        # scatter_softmax normalized per agent (weights sum to 1 for each agent)
        values_for_softmax = -edge_costs / temperature
        edge_weights = scatter_softmax(values_for_softmax, a_indices, dim=0)

        return edge_weights, edge_costs
