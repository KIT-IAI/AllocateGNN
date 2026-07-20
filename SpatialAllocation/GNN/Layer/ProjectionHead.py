import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralProjectionHead(nn.Module):
    """
    Spectral feature projection head: maps spectral index features into a
    bottleneck embedding space, then projects them to a K-dimensional
    probability distribution.

    Architecture: Linear(d_spec -> d_z) + ReLU + Linear(d_z -> K) + temperature softmax

    Args:
        d_spec: Spectral feature dimension (default 10, corresponding to
            5 indices x 2 statistics)
        d_z: Bottleneck embedding dimension (default 32)
        K: Projection output dimension (default 5, corresponding to the
            5 GVA industry classes in the UK dataset)
    """

    def __init__(self, d_spec: int = 10, d_z: int = 32, K: int = 5):
        super().__init__()
        self.d_spec = d_spec
        self.d_z = d_z
        self.K = K

        # Bottleneck encoding layer
        self.encoder = nn.Linear(d_spec, d_z)

        # Projection layer: from bottleneck space to K dimensions
        self.projector = nn.Linear(d_z, K)

        # Learnable temperature parameter (log space, ensures a positive value)
        self.log_tau_proj = nn.Parameter(torch.zeros(1))

    def forward(self, spectral_features: torch.Tensor):
        """
        Forward pass.

        Args:
            spectral_features: (N, d_spec) spectral features

        Returns:
            z: (N, d_z) bottleneck embedding
            T_hat: (N, K) probability distribution (softmax-normalized)
        """
        # Bottleneck embedding
        z = F.relu(self.encoder(spectral_features))

        # Project to K dimensions + temperature softmax
        logits = self.projector(z)
        tau = torch.exp(self.log_tau_proj)
        T_hat = F.softmax(logits / tau, dim=-1)

        return z, T_hat


class MacroReconstructionHead(nn.Module):
    """
    Macro attribute reconstruction head: maps the bottleneck embedding z_a
    into an M-dimensional macro attribute space.

    Supports three head types:
        - 'softmax': outputs a probability distribution (suitable for
          proportional attributes, e.g. GVA industry shares)
        - 'linear': unconstrained output (suitable for continuous-valued
          attributes)
        - 'mixed': the first K_softmax dimensions use softmax, the rest use
          a linear output

    Args:
        d_z: Bottleneck embedding dimension (default 32)
        M: Macro attribute dimension (default 5)
        head_type: Head type ('softmax', 'linear', 'mixed')
        K_softmax: Dimension of the softmax part in 'mixed' mode
            (defaults to the same value as M)
    """

    def __init__(self, d_z: int = 32, M: int = 5, head_type: str = 'softmax',
                 K_softmax: int = None):
        super().__init__()
        self.d_z = d_z
        self.M = M
        self.head_type = head_type
        self.K_softmax = K_softmax if K_softmax is not None else M

        self.linear = nn.Linear(d_z, M)

    def forward(self, z: torch.Tensor):
        """
        Forward pass.

        Args:
            z: (N, d_z) bottleneck embedding

        Returns:
            recon: (N, M) reconstructed macro attributes
        """
        raw = self.linear(z)

        if self.head_type == 'softmax':
            return F.softmax(raw, dim=-1)
        elif self.head_type == 'linear':
            return raw
        elif self.head_type == 'mixed':
            softmax_part = F.softmax(raw[:, :self.K_softmax], dim=-1)
            linear_part = raw[:, self.K_softmax:]
            return torch.cat([softmax_part, linear_part], dim=-1)
        else:
            raise ValueError(f"Unsupported head type: {self.head_type}. Options: 'softmax', 'linear', 'mixed'")
