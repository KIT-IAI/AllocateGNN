import torch
from torch import nn

from SpatialAllocation.GNN.Layer.LossFunction.LossFunction import loss_registry, BaseLoss


@loss_registry.register(
    "ntl_prior",
    default_weight=0.1,
    description="NTL nighttime lights prior: Forward KL(target || w), a soft constraint requiring w to cover regions with high NTL values."
)
class NTLPriorLoss(BaseLoss):
    """
    Nighttime lights (NTL) prior loss L_ntl_prior.

    For each source node, log-transform the agents' NTL values and
    normalize them into a target distribution, then use forward KL
    divergence to constrain edge_weights to cover regions with high NTL
    values.

    Formula:
        target(s,a) = log(1 + ntl(a)) / Sum_{a' in s} log(1 + ntl(a'))   (RCI agents only)
        L = mean_s [ Sum_a target(s,a) x log( target(s,a) / w(s,a) ) ]

    Design notes:
        - Forward KL(target || w): mode-covering, w must cover the
          high-value region of target, but does not force w toward zero in
          low-value regions of target, avoiding excessive weight
          concentration when stacked with temperature softmax
        - RCI mask: the NTL contribution of non-RCI agents
          (lu_res+com+ind <= threshold) is zeroed out and excluded from target
        - log(1+x): compresses the right-skewed raw NTL values
          (range [0.34, 267]), preventing a small number of high-NTL agents
          from dominating
        - Per-source mean: does not bias toward sources with more agents

    metadata must contain:
        - 'agent_ntl': (num_agents,) the NTL value of each agent
        - 'agent_rci_mask': (num_agents,) bool, RCI agent indicator (optional)
        - 'num_s': int, number of source nodes
    """

    def forward(self, edge_weights, edge_index, metadata):
        device = edge_weights.device

        agent_ntl = metadata.get('agent_ntl')
        rci_mask = metadata.get('agent_rci_mask')
        num_s = metadata.get('num_s')

        if agent_ntl is None or num_s is None or num_s == 0:
            return torch.tensor(0.0, device=device)

        s_indices = edge_index[0]  # (num_edges,)
        a_indices = edge_index[1]  # (num_edges,)

        # 1. edge-level NTL, RCI filtering + log transform
        edge_ntl = agent_ntl[a_indices]
        if rci_mask is not None:
            edge_ntl = edge_ntl * rci_mask[a_indices].float()
        edge_ntl = torch.log(1.0 + torch.clamp(edge_ntl, min=0.0))
        edge_ntl = torch.clamp(edge_ntl, min=1e-8)

        # 2. per-source normalization: target = log(1+ntl(a)) / Sum_{a' in s} log(1+ntl(a'))
        ntl_sum = torch.zeros(num_s, device=device)
        ntl_sum.scatter_add_(0, s_indices, edge_ntl)
        target = edge_ntl / (ntl_sum[s_indices] + 1e-8)

        # 3. Forward KL(target || w) = Sum target * log(target / w)
        t_safe = torch.clamp(target, min=1e-8)
        w_safe = torch.clamp(edge_weights, min=1e-8)
        kl_terms = t_safe * (torch.log(t_safe) - torch.log(w_safe))

        # 4. sum per source, then take the mean
        kl_per_s = torch.zeros(num_s, device=device)
        kl_per_s.scatter_add_(0, s_indices, kl_terms)
        valid = ntl_sum > 1e-6
        if valid.sum() > 0:
            return kl_per_s[valid].mean()
        else:
            return torch.tensor(0.0, device=device)
