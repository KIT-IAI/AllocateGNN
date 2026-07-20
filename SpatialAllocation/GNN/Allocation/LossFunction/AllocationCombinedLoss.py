from torch import nn
import torch
# Import loss modules to trigger registration
import SpatialAllocation.GNN.Allocation.LossFunction.AllocationDistanceLoss  # noqa: F401
import SpatialAllocation.GNN.Allocation.LossFunction.AllocationFeatureHomogeneityLoss  # noqa: F401
from SpatialAllocation.GNN.Allocation.LossFunction.AllocationLossRegistry import allocation_loss_registry


class AllocationCombinedLoss(nn.Module):
    """
    Stage 2 combined loss: uses allocation_loss_registry (independent of Stage 1).
    Same structure as Stage 1's CombinedLoss, but with a separate registry.
    """

    def __init__(self, weights=None, learnable=False):
        super().__init__()

        self.registry = allocation_loss_registry
        self.learnable = learnable

        if weights is None:
            weights = {
                'allocation_distance': 1.0,
                'allocation_feature_homogeneity': 0.5,
            }
        self.weights = weights

        self.use_losses = list(self.weights.keys())

        if self.learnable:
            self.log_vars = nn.ParameterDict()
            for name in self.use_losses:
                if self.weights[name] > 0:
                    self.log_vars[name] = nn.Parameter(torch.zeros(1))
        else:
            self.log_vars = None

        # Create loss function instances
        self.loss_functions = {
            name: self.registry.get_loss(name)()
            for name in self.use_losses
            if self.weights[name] > 0
        }

    def forward(self, edge_weights, edge_index, metadata):
        losses = {
            name: loss_fn(edge_weights, edge_index, metadata)
            for name, loss_fn in self.loss_functions.items()
        }

        total_loss = 0

        if self.learnable:
            for name, loss_value in losses.items():
                if name in self.log_vars:
                    precision = torch.exp(-self.log_vars[name])
                    total_loss += precision * loss_value + self.log_vars[name]
                else:
                    weight = self.weights.get(name, 0)
                    total_loss += weight * loss_value
        else:
            for name, loss_value in losses.items():
                weight = self.weights.get(name, 0)
                total_loss += weight * loss_value

        return total_loss, losses
