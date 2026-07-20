# Import all loss modules to trigger loss_registry registration
from SpatialAllocation.GNN.Layer.LossFunction.LossFunction import loss_registry
from SpatialAllocation.GNN.Layer.LossFunction.LossFunction import (
    EntropyLoss,
    FeatureSimilarityLoss,
    FeatureConsistencyLoss,
    LandusePredictionLoss,
)
from SpatialAllocation.GNN.Layer.LossFunction.ReconstructionLoss import ReconstructionLoss
from SpatialAllocation.GNN.Layer.LossFunction.DiversityLoss import DiversityLoss
from SpatialAllocation.GNN.Layer.LossFunction.GateLoss import GateLoss
from SpatialAllocation.GNN.Layer.LossFunction.SpatialSmoothLoss import SpatialSmoothLoss
from SpatialAllocation.GNN.Layer.LossFunction.WeightVarianceLoss import WeightVarianceLoss
from SpatialAllocation.GNN.Layer.LossFunction.DistanceDecayLoss import DistanceDecayLoss
from SpatialAllocation.GNN.Layer.LossFunction.NTLPriorLoss import NTLPriorLoss
from SpatialAllocation.GNN.Layer.LossFunction.ProximityPriorLoss import ProximityPriorLoss
from SpatialAllocation.GNN.Layer.LossFunction.CombinedLoss import CombinedLoss
