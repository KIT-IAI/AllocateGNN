import torch
from torch import nn

from SpatialAllocation.GNN.Layer.LossFunction.LossFunction import loss_registry, BaseLoss


@loss_registry.register(
    "gate",
    default_weight=0.01,
    description="Gate sparsification loss: encourages some agent gates to trend toward 0, competing with the reconstruction gradient to produce genuine selectivity. Uses a fixed weight and is excluded from the learnable uncertainty weighting system."
)
class GateLoss(BaseLoss):
    """
    Gate sparsification loss L_gate.

    Formula:
        L_gate = lambda_g x mean(g_a)

    The gradient is the constant lambda_g, gently pushing all g_a toward 0.
    The gradient from the reconstruction loss pulls up g_a for useful
    agents, creating competition:
    - Useful agents: reconstruction gradient > gate gradient -> g_a stays high
    - Useless agents: gate gradient dominates -> g_a trends toward 0
    This produces genuine selectivity (replacing the original -log(g)
    behavior, which pushed everything toward 1).

    metadata must contain:
        - 'gate_values': (N_agent,) agent validity score g_a in (0, 1)
        - 'gate_lambda': float, gate sparsification coefficient (0.05 recommended)
    """

    def forward(self, edge_weights, edge_index, metadata):
        device = edge_weights.device

        if 'gate_values' not in metadata:
            return torch.tensor(0.0, device=device)

        g = metadata['gate_values']  # (N_agent,)
        lambda_g = metadata.get('gate_lambda', 0.01)

        # L_gate = lambda_g x mean(g_a), sparsification regularization
        loss = lambda_g * torch.mean(g)

        return loss
