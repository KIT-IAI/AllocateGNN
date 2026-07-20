import torch
import torch.nn as nn


class AgentGating(nn.Module):
    """
    Learnable agent validity gating module.

    Predicts a validity score g_a ∈ (0, 1) for each agent, used in the
    first-layer weight computation to exclude zero-load regions (water
    bodies, wasteland, etc.).

    Formula: g_a = sigmoid(v^T z_a + b_v)

    The bias b_v is initialized to a positive value (default +2.0), so the
    initial g_a ≈ 0.88 — by default the model treats all agents as valid,
    and only excludes them when necessary.

    Args:
        d_z: Bottleneck embedding dimension (default 32)
        bias_init: Bias initialization value (default 2.0)
    """

    def __init__(self, d_z: int = 32, bias_init: float = 2.0):
        super().__init__()
        self.d_z = d_z

        # Linear layer: d_z -> 1
        self.gate_linear = nn.Linear(d_z, 1)

        # Initialize bias to a positive value (valid by default)
        with torch.no_grad():
            self.gate_linear.bias.fill_(bias_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: (N, d_z) bottleneck embedding

        Returns:
            g: (N,) validity score, range (0, 1)
        """
        return torch.sigmoid(self.gate_linear(z)).squeeze(-1)
