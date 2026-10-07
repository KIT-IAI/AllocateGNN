"""
异构 GNN 边权重薄包装

将 GNN 子系统的 EdgeWeightSolver 推理结果包装为 WeightResult。
模式必须显式区分只读 checkpoint 推理与当前进程本地训练；本包装器从不提交 HPC 作业。
"""
import numpy as np
import geopandas as gpd
from typing import Optional
from torch_geometric.loader import DataLoader

from ...base import BaseWeighter, WeightResult
from ...registry import weighter_registry


@weighter_registry.register(
    "hetero_gnn",
    description="异构图神经网络边权重（需要预训练模型）",
)
class HeteroGNNWeighter(BaseWeighter):
    """
    GNN 边权重包装器。

    配置参数:
        mode: checkpoint_only（默认）或 local_train
        model_path: checkpoint_only 模式下的预训练模型路径（必需）
        device: 推理设备（默认 "cpu"）
        model_config: ModelConfig 覆盖参数（可选）
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.mode = self.config.get("mode", "checkpoint_only")
        if self.mode not in {"checkpoint_only", "local_train"}:
            raise ValueError("HeteroGNNWeighter.mode 必须是 checkpoint_only 或 local_train")
        model_path = self.config.get("model_path")
        if self.mode == "checkpoint_only" and not model_path:
            raise ValueError("checkpoint_only 模式必须显式提供 model_path")
        if self.mode == "local_train" and model_path:
            raise ValueError("local_train 模式不得同时提供 model_path；避免隐式恢复/重训")
        self._solver = None

    def fit(
        self,
        grid_gdf: gpd.GeoDataFrame,
        target_gdf: gpd.GeoDataFrame,
        source_gdf: Optional[gpd.GeoDataFrame] = None,
        **kwargs,
    ) -> None:
        """
        按显式 mode 加载预训练模型或在当前本地进程训练。

        额外 kwargs:
            hetero_data: 已构建的 PyG HeteroData 对象
        """
        from .gnn.solver import EdgeWeightSolver
        from .gnn.config import ModelConfig

        model_config_overrides = dict(self.config.get("model_config", {}))
        device = self.config.get("device", "cpu")
        model_config_overrides["device"] = device
        if "epochs" in self.config:
            model_config_overrides["epochs"] = int(self.config["epochs"])
        if self.mode == "checkpoint_only":
            model_config_overrides["save_path"] = self.config["model_path"]
        elif "save_path" not in model_config_overrides:
            raise ValueError("local_train mode requires model_config.save_path")
        model_config = ModelConfig(**model_config_overrides)

        hetero_data = kwargs.get("hetero_data")
        if hetero_data is None:
            raise ValueError(
                "HeteroGNNWeighter.fit() 需要 hetero_data 参数 "
                "（PyG HeteroData 对象）"
            )

        self._solver = EdgeWeightSolver(model_config)
        loader = DataLoader([hetero_data], batch_size=1, shuffle=False)
        objective_weights = self.config.get("objective_weights", {})
        if self.mode == "checkpoint_only":
            self._solver.init_model(loader, objective_weights)
            self._solver._load_checkpoint()
        else:
            self._solver.train_multi_graph(
                loader, objective_weights=objective_weights
            )

    def compute(
        self,
        grid_gdf: gpd.GeoDataFrame,
        target_gdf: gpd.GeoDataFrame,
        source_gdf: Optional[gpd.GeoDataFrame] = None,
        **kwargs,
    ) -> WeightResult:
        """
        使用已训练的 GNN 模型计算边权重。

        额外 kwargs:
            hetero_data: 已构建的 PyG HeteroData 对象
        """
        if self._solver is None:
            raise RuntimeError(
                "HeteroGNNWeighter 未初始化，请先调用 fit() 加载模型"
            )

        hetero_data = kwargs.get("hetero_data")
        if hetero_data is None:
            raise ValueError(
                "HeteroGNNWeighter.compute() 需要 hetero_data 参数"
            )

        edge_weights = self._solver.predict_edge_weights(hetero_data)
        weights = np.zeros(len(grid_gdf), dtype=float)
        indices = edge_weights["agent_original_idx"].to_numpy(dtype=int)
        if len(indices) and ((indices < 0).any() or (indices >= len(weights)).any()):
            raise ValueError("predicted agent_original_idx is outside grid_gdf")
        np.add.at(
            weights,
            indices,
            edge_weights["predicted_weight"].to_numpy(dtype=float),
        )

        return WeightResult(
            weights=weights,
            weight_column_name="gnn_weight",
            normalized=False,
            metadata={
                "method": "hetero_gnn",
                "mode": self.mode,
                "model_path": self.config.get("model_path", "trained_in_memory"),
                "n_weights": len(weights),
            },
        )

    @property
    def requires_fit(self) -> bool:
        return self._solver is None

