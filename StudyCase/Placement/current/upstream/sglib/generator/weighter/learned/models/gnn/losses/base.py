import torch
from torch import nn
import torch.nn.functional as F

from .registry import LossRegistry
from ..landuse import (
    INDEXED_LANDUSE_REPRESENTATION,
    LEGACY_DENSE_LANDUSE_REPRESENTATION,
)

loss_registry = LossRegistry()


class BaseLoss(nn.Module):
    """
    损失函数基类, 所有自定义损失函数的父类
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss(reduction='mean')
    # 修改基类的 forward 签名
    def forward(self, edge_weights, edge_index, metadata):
        return 0


@loss_registry.register("entropy_regularization", default_weight=1,
                        description="Encourages the edge weights for each source node to be as uniform as possible.")
class EntropyLoss(BaseLoss):

    # 修改 forward 签名并更新内部逻辑
    def forward(self, edge_weights, edge_index, metadata):
        """
        优化版本：向量化熵计算
        """
        device = edge_weights.device
        # 直接使用传入的 edge_index，第一行是 source 节点的索引
        s_indices = edge_index[0]
        epsilon = 1e-8

        # 安全的log计算
        safe_weights = torch.clamp(edge_weights, min=epsilon)
        log_weights = torch.log(safe_weights)

        # 计算每个权重的熵贡献
        entropy_terms = edge_weights * log_weights

        # 使用scatter_add对每个source进行分组求和
        # BUG修复：这里的维度应该是 source 节点的数量，而不是 agent 节点的数量
        num_s = metadata["num_s"]
        entropy_per_s = torch.zeros(num_s, device=device)
        entropy_per_s.scatter_add_(0, s_indices, -entropy_terms)

        # 计算每个source的边数
        ones = torch.ones_like(edge_weights)
        edges_per_s = torch.zeros(num_s, device=device)
        edges_per_s.scatter_add_(0, s_indices, ones)

        # 只对有边的source计算损失
        valid_mask = edges_per_s > 0
        if valid_mask.sum() > 0:
            valid_entropy = entropy_per_s[valid_mask]
            valid_edges = edges_per_s[valid_mask]

            # 计算理论最大熵
            max_entropy = torch.log(valid_edges)

            # 计算总损失，我们希望最大化熵，即最小化 (最大熵 - 当前熵)
            entropy_loss = torch.sum(max_entropy - valid_entropy)
        else:
            entropy_loss = torch.tensor(0.0, device=device)

        return entropy_loss

@loss_registry.register("feature_similarity_loss", default_weight=0.1,
                        description="Encourages agents connected to the same source to have similar features.")
class FeatureSimilarityLoss(BaseLoss):
    """
    正则化损失：最小化连接到同一源的代理节点特征的方差。
    这鼓励模型学习到特征上更同质的分配簇。
    """
    def forward(self, edge_weights, edge_index, metadata):
        """
        计算特征相似度损失

        Args:
            edge_weights (torch.Tensor): 模型的输出，边的权重。
            edge_index (torch.Tensor): 边的索引。
            metadata (dict): 必须包含:
                - 'agent_features' (torch.Tensor): agent节点的特征矩阵。
        """
        s_indices = edge_index[0]
        a_indices = edge_index[1]
        agent_features = metadata['agent_features']

        if agent_features is None or agent_features.numel() == 0:
            return torch.tensor(0.0, device=edge_weights.device)

        # 获取每条边对应的agent特征
        edge_agent_features = agent_features[a_indices]

        # 计算每个source簇的加权特征均值
        # weight * feature
        weighted_features = edge_weights.unsqueeze(1) * edge_agent_features
        # 使用scatter_add聚合每个source的加权特征总和
        sum_weighted_features = torch.zeros((metadata['num_s'], agent_features.shape[1]), device=edge_weights.device)
        s_indices_expanded = s_indices.unsqueeze(1).expand_as(weighted_features)
        sum_weighted_features.scatter_add_(0, s_indices_expanded, weighted_features)

        # 使用scatter_add聚合每个source的权重总和 (理论上应为1，但为了数值稳定性重算)
        sum_weights = torch.zeros(metadata['num_s'], device=edge_weights.device)
        sum_weights.scatter_add_(0, s_indices, edge_weights)

        # 计算加权平均特征, +1e-8 防止除以零
        mean_features_per_s = sum_weighted_features / (sum_weights.unsqueeze(1) + 1e-8)

        # 计算每个agent与其所属source簇中心特征的加权距离（方差）
        # (feature - mean_feature_of_its_source)^2
        squared_diff = torch.sum((edge_agent_features - mean_features_per_s[s_indices])**2, dim=1)

        # loss = weight * (feature - mean_feature)^2
        weighted_squared_diff = edge_weights * squared_diff

        # 聚合得到每个source的加权方差
        variance_per_s = torch.zeros(metadata['num_s'], device=edge_weights.device)
        variance_per_s.scatter_add_(0, s_indices, weighted_squared_diff)

        # 返回所有source簇的平均方差
        # 只考虑有边的source
        valid_mask = sum_weights > 0
        if valid_mask.sum() > 0:
            return variance_per_s[valid_mask].mean()
        else:
            return torch.tensor(0.0, device=edge_weights.device)


@loss_registry.register("feature_consistency_loss", default_weight=0.1,
                        description="Moore邻接pairwise: 特征相似的邻居权重趋近一致，特征不同的允许跳变。")
class FeatureConsistencyLoss(BaseLoss):
    """
    基于 Moore 邻接的特征感知一致性损失。

    公式:
        L = mean( sim(f_i, f_j) × (w_{s,i} - w_{s,j})² )
            for each Moore pair (i,j) where both agents connect to source s

    语义:
        - sim > 0（相似特征）：惩罚权重差异 → 推向一致
        - sim < 0（不同特征）：奖励权重差异 → 允许跳变
        - sim ≈ 0：中性，不约束

    与 014 已退役的 SpatialSmoothLoss 的区别:
        - 旧 SpatialSmoothLoss: 无差别 (w_i - w_j)²，不分 source
        - 本损失: sim(f_i, f_j) × (w_i - w_j)²，仅同 source 的 Moore 对
    """

    def forward(self, edge_weights, edge_index, metadata):
        s_indices = edge_index[0]
        a_indices = edge_index[1]
        features_a = metadata['agent_features']
        num_s = metadata["num_s"]
        num_a = metadata.get('num_a')
        device = edge_weights.device

        agent_adj_pairs = metadata.get('agent_adj_pairs')
        if agent_adj_pairs is None or agent_adj_pairs.shape[1] == 0:
            return torch.tensor(0.0, device=device)

        if features_a is None or features_a.numel() == 0:
            return torch.tensor(0.0, device=device)

        if num_a is None:
            num_a = int(a_indices.max().item()) + 1

        adj_i = agent_adj_pairs[0]  # (P,)
        adj_j = agent_adj_pairs[1]  # (P,)

        # 预计算所有 agent 的 L2 归一化特征
        features_norm = F.normalize(features_a, p=2, dim=1)

        # Moore 对的特征余弦相似度（只算一次，所有 source 共享）
        sim = (features_norm[adj_i] * features_norm[adj_j]).sum(dim=1)  # (P,)

        total_loss = torch.tensor(0.0, device=device)
        valid_sources = 0

        for s in range(num_s):
            s_mask = (s_indices == s)
            if s_mask.sum() <= 1:
                continue

            # 该 source 连接的 agent 集合及其权重
            s_agents = a_indices[s_mask]
            s_weights = edge_weights[s_mask]

            # agent_idx → 是否连接到此 source
            agent_has_edge = torch.zeros(num_a, dtype=torch.bool, device=device)
            agent_has_edge[s_agents] = True

            # 筛选：Moore 对的两个 agent 都连接到此 source
            valid_pair = agent_has_edge[adj_i] & agent_has_edge[adj_j]
            if valid_pair.sum() == 0:
                continue

            # agent_idx → 在此 source 下的权重
            agent_weight = torch.zeros(num_a, device=device)
            agent_weight[s_agents] = s_weights

            w_i = agent_weight[adj_i[valid_pair]]
            w_j = agent_weight[adj_j[valid_pair]]
            sim_valid = sim[valid_pair]

            # Log-space: sim × (log w_i - log w_j)²
            # 对权重比率敏感而非绝对差异，避免 N 大时数值崩塌到 1e-6
            epsilon = 1e-8
            log_w_i = torch.log(w_i + epsilon)
            log_w_j = torch.log(w_j + epsilon)
            diff_sq = (log_w_i - log_w_j) ** 2
            total_loss = total_loss + (sim_valid * diff_sq).mean()
            valid_sources += 1

        return total_loss / valid_sources if valid_sources > 0 else total_loss

@loss_registry.register("landuse_prediction_loss", default_weight=1.0,
                        description="Computes loss between predicted and actual landuse ratios based on edge weights.")
class LandusePredictionLoss(BaseLoss):
    def forward(self, edge_weights, edge_index, metadata):
        predicted_landuse_ratio, true_landuse_ratio = landuse_prediction_ratio(
            edge_weights, edge_index, metadata
        )
        epsilon = 1e-8
        true_landuse_safe = torch.clamp(true_landuse_ratio, min=epsilon)
        predicted_landuse_safe = torch.clamp(predicted_landuse_ratio, min=epsilon)
        return F.kl_div(
            torch.log(predicted_landuse_safe),
            true_landuse_safe,
            reduction='batchmean',
        )


def _single_representation(value):
    """Undo PyG's batch-size-one string collation without accepting ambiguity."""

    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("landuse supervision representation must be singular")
        value = value[0]
    return value


def landuse_prediction_ratio(edge_weights, edge_index, metadata):
    """Validate and aggregate dense-v2 or indexed-v3 land-use supervision.

    Dense metadata without an explicit representation remains valid so existing
    UK/AU v2 graph caches continue to load.  Indexed metadata is fail-closed and
    must declare its versioned representation.
    """

    if not torch.is_tensor(edge_weights) or edge_weights.ndim != 1:
        raise ValueError("edge_weights must be a one-dimensional tensor")
    if not edge_weights.dtype.is_floating_point:
        raise TypeError("edge_weights must have a floating dtype")
    if not bool(torch.isfinite(edge_weights).all()):
        raise ValueError("edge_weights must be finite")
    num_edges = int(edge_weights.numel())
    if num_edges == 0:
        raise ValueError("landuse supervision requires at least one edge")

    if not torch.is_tensor(edge_index) or edge_index.ndim != 2:
        raise ValueError("edge_index must be a two-dimensional tensor")
    if edge_index.shape != (2, num_edges):
        raise ValueError("edge_index length differs from edge_weights")
    if edge_index.dtype != torch.int64:
        raise TypeError("edge_index must use int64 indices")

    if "num_s" not in metadata:
        raise ValueError("landuse supervision metadata is missing num_s")
    num_s = metadata["num_s"]
    if isinstance(num_s, torch.Tensor):
        if num_s.numel() != 1:
            raise ValueError("landuse supervision num_s must be scalar")
        num_s = int(num_s.item())
    if isinstance(num_s, bool) or not isinstance(num_s, int) or num_s <= 0:
        raise ValueError("landuse supervision num_s must be a positive integer")
    source_indices = edge_index[0]
    if bool(((source_indices < 0) | (source_indices >= num_s)).any()):
        raise ValueError("edge_index contains a source index outside num_s")
    if bool((edge_index[1] < 0).any()):
        raise ValueError("edge_index contains a negative agent index")

    if "landuse_ratio" not in metadata:
        raise ValueError("landuse supervision metadata is missing landuse_ratio")
    true_landuse_ratio = metadata["landuse_ratio"]
    if not torch.is_tensor(true_landuse_ratio):
        raise TypeError("landuse_ratio must be a tensor")
    if true_landuse_ratio.ndim != 2 or true_landuse_ratio.shape[0] != num_s:
        raise ValueError("landuse_ratio shape must be [num_s, num_landuse_types]")
    if not true_landuse_ratio.dtype.is_floating_point:
        raise TypeError("landuse_ratio must have a floating dtype")
    num_landuse_types = int(true_landuse_ratio.shape[1])
    if num_landuse_types <= 0:
        raise ValueError("landuse_ratio must contain at least one landuse type")
    if not bool(torch.isfinite(true_landuse_ratio).all()):
        raise ValueError("landuse_ratio must be finite")
    if bool(((true_landuse_ratio < 0) | (true_landuse_ratio > 1)).any()):
        raise ValueError("landuse_ratio values must be within [0, 1]")
    ratio_sums = true_landuse_ratio.sum(dim=1)
    valid_ratio_sums = torch.isclose(
        ratio_sums, torch.zeros_like(ratio_sums), atol=1e-6, rtol=0
    ) | torch.isclose(
        ratio_sums, torch.ones_like(ratio_sums), atol=1e-6, rtol=0
    )
    if not bool(valid_ratio_sums.all()):
        raise ValueError("each landuse_ratio row must sum to zero or one")

    has_dense = "landuse_mapping_matrix" in metadata
    has_indexed = "landuse_flat_index" in metadata
    if has_dense == has_indexed:
        raise ValueError(
            "landuse supervision requires exactly one of "
            "landuse_mapping_matrix or landuse_flat_index"
        )
    representation = _single_representation(
        metadata.get("landuse_supervision_representation")
    )
    if representation is None:
        if not has_dense:
            raise ValueError(
                "indexed landuse supervision is missing representation metadata"
            )
        representation = LEGACY_DENSE_LANDUSE_REPRESENTATION

    flat_width = num_s * num_landuse_types
    if representation == LEGACY_DENSE_LANDUSE_REPRESENTATION:
        if not has_dense:
            raise ValueError("dense representation is missing landuse_mapping_matrix")
        mapping_matrix = metadata["landuse_mapping_matrix"]
        if not torch.is_tensor(mapping_matrix):
            raise TypeError("landuse_mapping_matrix must be a tensor")
        if mapping_matrix.ndim != 2 or mapping_matrix.shape != (
            num_edges,
            flat_width,
        ):
            raise ValueError(
                "landuse_mapping_matrix shape must be "
                "[num_edges, num_s * num_landuse_types]"
            )
        if not mapping_matrix.dtype.is_floating_point:
            raise TypeError("landuse_mapping_matrix must have a floating dtype")
        if not bool(torch.isfinite(mapping_matrix).all()):
            raise ValueError("landuse_mapping_matrix must be finite")
        if bool(((mapping_matrix < 0) | (mapping_matrix > 1)).any()):
            raise ValueError("landuse_mapping_matrix values must be within [0, 1]")
        row_sums = mapping_matrix.sum(dim=1)
        valid_rows = torch.isclose(
            row_sums, torch.zeros_like(row_sums), atol=1e-6, rtol=0
        ) | torch.isclose(
            row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=0
        )
        if not bool(valid_rows.all()):
            raise ValueError("dense mapping rows must sum to zero or one")
        predicted_flat = torch.matmul(
            edge_weights,
            mapping_matrix.to(
                device=edge_weights.device, dtype=edge_weights.dtype
            ),
        )
    elif representation == INDEXED_LANDUSE_REPRESENTATION:
        if not has_indexed:
            raise ValueError("indexed representation is missing landuse_flat_index")
        flat_index = metadata["landuse_flat_index"]
        if not torch.is_tensor(flat_index):
            raise TypeError("landuse_flat_index must be a tensor")
        if flat_index.dtype != torch.int64:
            raise TypeError("landuse_flat_index must use int64 indices")
        if flat_index.ndim != 1 or flat_index.numel() != num_edges:
            raise ValueError("landuse_flat_index length must equal num_edges")
        if bool(((flat_index < -1) | (flat_index >= flat_width)).any()):
            raise ValueError(
                "landuse_flat_index values must be -1 or within flattened range"
            )
        flat_index_device = flat_index.to(device=edge_weights.device)
        valid_device = flat_index_device >= 0
        if bool(valid_device.any()):
            encoded_sources = torch.div(
                flat_index_device[valid_device],
                num_landuse_types,
                rounding_mode="floor",
            )
            if not torch.equal(encoded_sources, source_indices[valid_device]):
                raise ValueError(
                    "landuse_flat_index source component differs from edge_index"
                )
        predicted_flat = edge_weights.new_zeros(flat_width)
        if bool(valid_device.any()):
            predicted_flat.scatter_add_(
                0,
                flat_index_device[valid_device],
                edge_weights[valid_device],
            )
    else:
        raise ValueError(
            f"unknown landuse supervision representation: {representation!r}"
        )

    predicted_landuse_aggregated = predicted_flat.view(
        num_s, num_landuse_types
    )
    epsilon = 1e-8
    region_totals = predicted_landuse_aggregated.sum(dim=1, keepdim=True) + epsilon
    predicted_landuse_ratio = predicted_landuse_aggregated / region_totals
    return (
        predicted_landuse_ratio,
        true_landuse_ratio.to(
            device=edge_weights.device, dtype=edge_weights.dtype
        ),
    )
