from torch import nn
import torch
# Import all loss modules to trigger loss_registry registration
import SpatialAllocation.GNN.Layer.LossFunction.ReconstructionLoss  # noqa: F401
import SpatialAllocation.GNN.Layer.LossFunction.DiversityLoss  # noqa: F401
import SpatialAllocation.GNN.Layer.LossFunction.GateLoss  # noqa: F401
from SpatialAllocation.GNN.Layer.LossFunction.LossFunction import loss_registry


# Set of loss names that always use fixed weights (excluded from learnable uncertainty weighting even in learnable mode)
FIXED_WEIGHT_LOSSES = {'gate'}


class CombinedLoss(nn.Module):
    def __init__(self, weights=None, learnable=False):
        super().__init__()

        self.registry = loss_registry
        self.learnable = learnable

        # Default weights
        if weights is None:
            weights = {'entropy_regularization': 1.0}
        self.weights = weights

        # Determine which loss functions to use
        self.use_losses = list(self.weights.keys())

        if self.learnable:
            print("Using learnable weights for losses. Each loss will have a learnable log variance parameter.")
            self.log_vars = nn.ParameterDict()
            # Create a learnable log-variance parameter for each loss function (excluding fixed-weight losses)
            for name in self.use_losses:
                if self.weights[name] > 0 and name not in FIXED_WEIGHT_LOSSES:
                    self.log_vars[name] = nn.Parameter(torch.zeros(1))
        else:
            self.log_vars = None

        # Create loss function instances
        self.loss_functions = {
            name: self.registry.get_loss(name)()
            for name in self.use_losses
            if self.weights[name] > 0
        }

    # Forward method signature modified to remove the no-longer-needed edge_index_mapping parameter
    def forward(self, edge_weights, edge_index, metadata):
        # Compute each loss
        # Pass parameters correctly to each sub-loss function
        losses = {name: loss_fn(edge_weights, edge_index, metadata) for name, loss_fn in self.loss_functions.items()}

        total_loss = 0

        if self.learnable:
            # Use learnable weights (uncertainty weighting in multi-task learning)
            for name, loss_value in losses.items():
                if name in FIXED_WEIGHT_LOSSES:
                    # Fixed-weight loss (e.g. gate): always uses the preset weight
                    weight = self.weights.get(name, 0)
                    total_loss += weight * loss_value
                elif name in self.log_vars:
                    # Apply uncertainty weighting: loss / (2 * sigma^2) + log(sigma)
                    precision = torch.exp(-self.log_vars[name])
                    total_loss += precision * loss_value + self.log_vars[name]
        else:
            # Use fixed weights
            for name, loss_value in losses.items():
                weight = self.weights.get(name, 0)
                total_loss += weight * loss_value

        return total_loss, losses
