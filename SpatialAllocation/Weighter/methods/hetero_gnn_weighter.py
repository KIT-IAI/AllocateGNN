"""
Thin wrapper for heterogeneous GNN edge weights

Wraps the inference results of the GNN subsystem's EdgeWeightSolver into a WeightResult.
A pretrained model is required, so requires_fit is True when no trained model is loaded.
"""
import numpy as np
import geopandas as gpd
from typing import Optional

from .base import BaseWeighter, WeightResult
from .registry import weighter_registry


@weighter_registry.register(
    "hetero_gnn",
    description="Heterogeneous graph neural network edge weights (requires a pretrained model)",
)
class HeteroGNNWeighter(BaseWeighter):
    """
    GNN edge weight wrapper.

    Configuration parameters:
        model_path: pretrained model path (required)
        device: inference device (default "cpu")
        model_config: ModelConfig override parameters (optional)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._solver = None

    def fit(
        self,
        grid_gdf: gpd.GeoDataFrame,
        target_gdf: gpd.GeoDataFrame,
        source_gdf: Optional[gpd.GeoDataFrame] = None,
        **kwargs,
    ) -> None:
        """
        Load a pretrained model or run training.

        Additional kwargs:
            hetero_data: a constructed PyG HeteroData object
        """
        from SpatialAllocation.GNN.core.EdgeWeightSolver import EdgeWeightSolver
        from SpatialAllocation.GNN.core.ModelConfig import ModelConfig

        model_config_overrides = self.config.get("model_config", {})
        model_config = ModelConfig(**model_config_overrides)

        hetero_data = kwargs.get("hetero_data")
        if hetero_data is None:
            raise ValueError(
                "HeteroGNNWeighter.fit() requires the hetero_data parameter "
                "(a PyG HeteroData object)"
            )

        device = self.config.get("device", "cpu")
        self._solver = EdgeWeightSolver(model_config, device=device)

        model_path = self.config.get("model_path")
        if model_path:
            self._solver.load(model_path)
        else:
            # No pretrained model provided, training is required
            epochs = self.config.get("epochs", 100)
            self._solver.train([hetero_data], epochs=epochs)

    def compute(
        self,
        grid_gdf: gpd.GeoDataFrame,
        target_gdf: gpd.GeoDataFrame,
        source_gdf: Optional[gpd.GeoDataFrame] = None,
        **kwargs,
    ) -> WeightResult:
        """
        Compute edge weights using the trained GNN model.

        Additional kwargs:
            hetero_data: a constructed PyG HeteroData object
        """
        if self._solver is None:
            raise RuntimeError(
                "HeteroGNNWeighter is not initialized, call fit() first to load the model"
            )

        hetero_data = kwargs.get("hetero_data")
        if hetero_data is None:
            raise ValueError(
                "HeteroGNNWeighter.compute() requires the hetero_data parameter"
            )

        # Run inference to obtain edge weights
        edge_weights = self._solver.predict(hetero_data)

        # Aggregate edge weights into node weights (sum of outgoing edge weights for each agent node)
        if hasattr(edge_weights, 'numpy'):
            weights = edge_weights.detach().cpu().numpy()
        else:
            weights = np.asarray(edge_weights)

        return WeightResult(
            weights=weights,
            weight_column_name="gnn_weight",
            normalized=False,
            metadata={
                "method": "hetero_gnn",
                "model_path": self.config.get("model_path", "trained_in_memory"),
                "n_weights": len(weights),
            },
        )

    @property
    def requires_fit(self) -> bool:
        return self._solver is None
