import torch
from torch import nn

from SpatialAllocation.GNN.Layer.LossFunction.LossFunction import loss_registry, BaseLoss


@loss_registry.register(
    "diversity",
    default_weight=1.0,
    description="Diversity regularization across agents (Jensen gap): encourages agent T_hat distributions to be diverse and peaked, preventing degenerate uniform distributions."
)
class DiversityLoss(BaseLoss):
    """
    Diversity regularization loss across agents L_div (Jensen gap formulation).

    Formula:
        L_div = mean_s [ mean_a(H(T_hat_a)) - H(mean_a(T_hat_a)) ]

    The Jensen gap is >= 0, with equality iff all agents within the same
    source have identical T_hat. Minimizing this loss encourages agent
    T_hat to be:
        - diverse (different agents have different T_hat -> H(mean) is large)
        - and peaked (each agent's distribution is concentrated -> H(individual) is small)

    Degenerate solution analysis:
        - Fully uniform degenerate solution T_hat=[1/K,...,1/K]: mean_H=log(K), H_mean=log(K), L_div=0 (not the global minimum)
        - Diverse, peaked solution: mean_H~=0, H_mean~=log(K), L_div~=-log(K) (global minimum)
        - Problem with the original formula L_div=-H(mean T_hat): the fully uniform degenerate solution
          yields -log(K) (the global minimum!), which is fixed by this formula.

    metadata must contain:
        - 'T_hat': (N_agent, K) probability distribution output by the projection head
    """

    def forward(self, edge_weights, edge_index, metadata):
        device = edge_weights.device

        if 'T_hat' not in metadata:
            return torch.tensor(0.0, device=device)

        T_hat = metadata['T_hat']  # (N_agent, K)
        num_s = metadata['num_s']
        K = T_hat.shape[1]

        s_indices = edge_index[0]
        a_indices = edge_index[1]

        # Get the projection head output for the agent on each edge
        edge_T_hat = T_hat[a_indices]  # (num_edges, K)

        # Compute the within-group mean of T_hat for each source
        sum_T = torch.zeros(num_s, K, device=device)
        s_expanded = s_indices.unsqueeze(1).expand_as(edge_T_hat)
        sum_T.scatter_add_(0, s_expanded, edge_T_hat)

        # Compute the number of edges for each source
        counts = torch.zeros(num_s, device=device)
        counts.scatter_add_(0, s_indices, torch.ones_like(s_indices, dtype=torch.float))

        # Avoid division by zero
        valid_mask = counts > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        mean_T = sum_T[valid_mask] / counts[valid_mask].unsqueeze(1)  # (num_valid_s, K)

        # Compute the Shannon entropy of the mean distribution: H(mean_a T_hat_a)
        epsilon = 1e-8
        safe_mean_T = torch.clamp(mean_T, min=epsilon)
        entropy_per_s = -torch.sum(safe_mean_T * torch.log(safe_mean_T), dim=1)  # (num_valid_s,)

        # Compute the individual entropy of each agent: H(T_hat_a) = -sum(p * log(p))
        safe_T_hat = torch.clamp(T_hat, min=epsilon)
        indiv_entropy_all = -torch.sum(safe_T_hat * torch.log(safe_T_hat), dim=1)  # (N_agent,)

        # Map individual entropy to edges via a_indices, then aggregate by source and average
        edge_ind_entropy = indiv_entropy_all[a_indices]  # (num_edges,)
        sum_ind_entropy = torch.zeros(num_s, device=device)
        sum_ind_entropy.scatter_add_(0, s_indices, edge_ind_entropy)
        mean_ind_entropy_per_s = sum_ind_entropy[valid_mask] / counts[valid_mask]  # (num_valid_s,)

        # Jensen gap: mean(H_individual) - H(mean_T)
        # - Degenerate uniform solution: mean_H = log(K), H_mean = log(K), gap = 0 (not optimal)
        # - Diverse, peaked solution: mean_H ~= 0, H_mean ~= log(K), gap = -log(K) (global minimum!)
        # Minimizing this loss -> encourages agent T_hat to be diverse and peaked (non-uniform)
        loss = torch.mean(mean_ind_entropy_per_s - entropy_per_s)

        return loss
