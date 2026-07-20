import torch
from torch import nn

from SpatialAllocation.GNN.Layer.LossFunction.LossFunction import loss_registry, BaseLoss


@loss_registry.register(
    "reconstruction",
    default_weight=1.0,
    description="Macro-attribute reconstruction loss: aggregates agents' reconstructed attributes using w_sa as weights, then computes standardized MSE against the ground-truth regional attributes."
)
class ReconstructionLoss(BaseLoss):
    """
    Macro-attribute reconstruction loss L_recon.

    Formula:
        A_hat_s = sum_{a in N(s)} w_sa * phi(z_a)
        L_recon = (1/|S|) * sum_s (1/M) * sum_m ((A_hat_sm - A_sm) / sigma_m)^2

    where sigma_m is the standard deviation of the m-th attribute
    (precomputed at the dataset level), used to normalize attributes with
    different units/scales.

    metadata must contain:
        - 'recon_predictions': (N_agent,) or (N_agent, M) agent-level reconstructed attribute predictions
        - 'macro_attributes': (num_s, M) the ground-truth regional macro attributes
        - 'attribute_stds': (M,) the standard deviation of each attribute
    """

    def forward(self, edge_weights, edge_index, metadata):
        """
        Compute the macro-attribute reconstruction loss.

        Uses edge_weights (w_sa) to perform a weighted aggregation of the
        agents' reconstruction predictions, then computes the standardized
        MSE against the ground-truth regional-level macro attributes.
        """
        device = edge_weights.device

        # Check required metadata fields
        if 'recon_predictions' not in metadata or 'macro_attributes' not in metadata:
            return torch.tensor(0.0, device=device)

        recon_pred = metadata['recon_predictions']  # (N_agent, M)
        macro_attrs = metadata['macro_attributes'].to(device)  # (num_s, M)
        attr_stds = metadata.get('attribute_stds', None)

        num_s = metadata['num_s']
        M = macro_attrs.shape[1]

        s_indices = edge_index[0]  # source index
        a_indices = edge_index[1]  # agent index

        # Get the reconstruction prediction of the agent for each edge
        edge_recon = recon_pred[a_indices]  # (num_edges, M)

        # Aggregate to source level using w_sa as weights
        weighted_recon = edge_weights.unsqueeze(1) * edge_recon  # (num_edges, M)

        aggregated = torch.zeros(num_s, M, device=device)
        s_expanded = s_indices.unsqueeze(1).expand_as(weighted_recon)
        aggregated.scatter_add_(0, s_expanded, weighted_recon)

        # Standardized MSE
        if attr_stds is not None:
            attr_stds = attr_stds.to(device)
            # Avoid division by zero
            safe_stds = torch.clamp(attr_stds, min=1e-8)
            diff = (aggregated - macro_attrs) / safe_stds
        else:
            diff = aggregated - macro_attrs

        # Average MSE across attributes, then average over all sources
        loss = torch.mean(diff ** 2)

        return loss
