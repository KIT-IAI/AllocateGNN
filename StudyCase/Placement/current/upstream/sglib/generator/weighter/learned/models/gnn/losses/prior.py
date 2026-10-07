from __future__ import annotations

import torch

from .....priors import prior_registry
from .base import BaseLoss, loss_registry


class PriorKLLoss(BaseLoss):
    def __init__(self, prior_id: str):
        super().__init__()
        self.prior_id = prior_id
        self.spec = prior_registry.get(prior_id)

    @property
    def metadata_key(self) -> str:
        return "agent_ntl" if self.prior_id == "N" else "agent_proximity"

    def forward(self, edge_weights, edge_index, metadata):
        device = edge_weights.device
        values = metadata.get(self.metadata_key)
        rci_mask = metadata.get("agent_rci_mask")
        num_sources = metadata.get("num_s")
        if values is None or num_sources is None or num_sources == 0:
            raise ValueError(f"{self.prior_id} prior metadata is missing")
        source_index = edge_index[0]
        agent_index = edge_index[1]
        edge_prior = values[agent_index]
        if rci_mask is not None:
            edge_prior = edge_prior * rci_mask[agent_index].float()
        edge_prior = torch.clamp(torch.log1p(torch.clamp(edge_prior, min=0.0)), min=1e-8)
        prior_sum = torch.zeros(num_sources, device=device)
        prior_sum.scatter_add_(0, source_index, edge_prior)
        target = edge_prior / (prior_sum[source_index] + 1e-8)
        terms = torch.clamp(target, min=1e-8) * (
            torch.log(torch.clamp(target, min=1e-8))
            - torch.log(torch.clamp(edge_weights, min=1e-8))
        )
        per_source = torch.zeros(num_sources, device=device)
        per_source.scatter_add_(0, source_index, terms)
        valid = prior_sum > 1e-6
        return per_source[valid].mean() if valid.any() else torch.tensor(0.0, device=device)


@loss_registry.register("ntl_prior", default_weight=0.1, description="registered N prior KL")
class NTLPriorLoss(PriorKLLoss):
    def __init__(self):
        super().__init__("N")


@loss_registry.register("proximity_prior", default_weight=0.1, description="registered P prior KL")
class ProximityPriorLoss(PriorKLLoss):
    def __init__(self):
        super().__init__("P")


def prior_loss(prior_id: str) -> PriorKLLoss:
    return PriorKLLoss(prior_id)
