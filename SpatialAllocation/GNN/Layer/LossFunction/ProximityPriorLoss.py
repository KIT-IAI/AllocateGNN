import torch
from torch import nn

from SpatialAllocation.GNN.Layer.LossFunction.LossFunction import loss_registry, BaseLoss


@loss_registry.register(
    "proximity_prior",
    default_weight=0.1,
    description="Substation proximity prior: Forward KL(target || w), a soft constraint requiring w to cover high-proximity regions."
)
class ProximityPriorLoss(BaseLoss):
    """
    Substation proximity prior loss L_proximity_prior.

    For each source node, log-transform the agents' proximity scores and
    normalize them into a target distribution, then use forward KL
    divergence to constrain edge_weights to cover high-proximity regions.

    Formula:
        target(s,a) = log(1 + prox(a)) / Sum_{a' in s} log(1 + prox(a'))  (RCI agents only)
        L = mean_s [ Sum_a target(s,a) x log( target(s,a) / w(s,a) ) ]

    Design notes:
        - Forward KL(target || w): mode-covering, w must cover the high-value region of target
        - RCI mask: the proximity contribution of non-RCI agents is zeroed out
        - log(1+x): compresses the right-skewed distribution of raw values
        - Per-source mean: does not bias toward sources with more agents

    metadata must contain:
        - 'agent_proximity': (num_agents,) the proximity score of each agent
        - 'agent_rci_mask': (num_agents,) bool, RCI agent indicator (optional)
        - 'num_s': int, number of source nodes
    """

    def forward(self, edge_weights, edge_index, metadata):
        device = edge_weights.device

        agent_proximity = metadata.get('agent_proximity')
        rci_mask = metadata.get('agent_rci_mask')
        num_s = metadata.get('num_s')

        if agent_proximity is None or num_s is None or num_s == 0:
            return torch.tensor(0.0, device=device)

        s_indices = edge_index[0]  # (num_edges,)
        a_indices = edge_index[1]  # (num_edges,)

        # 1. edge-level proximity, RCI filtering + log transform
        edge_prox = agent_proximity[a_indices]
        if rci_mask is not None:
            edge_prox = edge_prox * rci_mask[a_indices].float()
        edge_prox = torch.log(1.0 + torch.clamp(edge_prox, min=0.0))
        edge_prox = torch.clamp(edge_prox, min=1e-8)

        # 2. per-source normalization: target = log(1+prox(a)) / Sum_{a' in s} log(1+prox(a'))
        prox_sum = torch.zeros(num_s, device=device)
        prox_sum.scatter_add_(0, s_indices, edge_prox)
        target = edge_prox / (prox_sum[s_indices] + 1e-8)

        # 3. Forward KL(target || w) = Sum target * log(target / w)
        t_safe = torch.clamp(target, min=1e-8)
        w_safe = torch.clamp(edge_weights, min=1e-8)
        kl_terms = t_safe * (torch.log(t_safe) - torch.log(w_safe))

        # 4. sum per source, then take the mean
        kl_per_s = torch.zeros(num_s, device=device)
        kl_per_s.scatter_add_(0, s_indices, kl_terms)
        valid = prox_sum > 1e-6
        if valid.sum() > 0:
            return kl_per_s[valid].mean()
        else:
            return torch.tensor(0.0, device=device)
