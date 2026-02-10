import torch
from torch import nn
import torch.nn.functional as F

from SpatialAllocation.GNN.Layer.LossFunction.LossRegistry import LossRegistry

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

@loss_registry.register("supervised_substation_demand_loss", default_weight=1.0,
                        description="Minimizes the MSE between predicted and actual demand at the substation level.")
class SupervisedSubstationDemandLoss(BaseLoss):
    """
    监督损失：在变电站级别，最小化预测需求与实际需求之间的均方误差。
    这个损失函数是通用的，不关心'source'节点的具体含义。
    """
    def forward(self, edge_weights, edge_index, metadata):
        """
        计算监督损失

        Args:
            edge_weights (torch.Tensor): 模型的输出，边的权重。
            edge_index (torch.Tensor): 边的索引, edge_index[1] 是代理节点。
            metadata (dict): 必须包含:
                - 'agent_demand' (torch.Tensor): 每个代理节点的基础需求。
                - 'agent_substation_map' (torch.Tensor): 每个代理节点到其所属变电站的索引映射。
                - 'substation_y_true' (torch.Tensor): 每个变电站的真实需求（监督信号）。
                - 'num_substations' (int): 批次中变电站的总数。
        """
        a_indices = edge_index[1]  # 每条边对应的代理节点（agent）索引

        # 从元数据中获取所需张量
        agent_demand = metadata['agent_demand']
        agent_substation_map = metadata['agent_substation_map']
        substation_y_true = metadata['substation_y_true']
        num_substations = metadata['num_substations']

        # 1. 计算每条边所代表的需求贡献值 (权重 * 对应代理节点的基础需求)
        predicted_edge_demand = edge_weights * agent_demand[a_indices]

        # 2. 找到每条边最终应归属的变电站索引
        edge_substation_map = agent_substation_map[a_indices]

        # 3. 使用 scatter_add 将所有边的需求贡献值，按照其归属的变电站进行聚合
        predicted_substation_demand = torch.zeros(num_substations, device=edge_weights.device)
        predicted_substation_demand.scatter_add_(0, edge_substation_map, predicted_edge_demand)

        # 4. 计算聚合后的预测需求与真实需求之间的MSE损失
        loss = self.mae(predicted_substation_demand, substation_y_true)

        return loss

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
                        description="Encourages agents with similar features under the same source to have similar weights.")
class FeatureConsistencyLoss(BaseLoss):
    """
    特征一致性损失函数。
    该损失鼓励归属于同一个source节点、且在特征空间中相似的agent节点拥有相近的分配权重。
    """

    def forward(self, edge_weights, edge_index, metadata):
        s_indices = edge_index[0]
        a_indices = edge_index[1]
        features_a = metadata['agent_features']
        num_s = metadata["num_s"]
        device = edge_weights.device

        total_loss = torch.tensor(0.0, device=device)

        # 此实现为清晰起见使用了循环。为了在GPU上获得极致性能，可以进一步向量化。
        for s_idx in range(num_s):
            mask = (s_indices == s_idx)
            # 如果一个source节点连接的agent数量少于2，则无法计算方差
            if mask.sum() <= 1:
                continue

            # 获取连接到此source的所有agent的特征和权重
            agents_for_s_features = features_a[a_indices[mask]]
            weights_for_s = edge_weights[mask]

            # 1. 计算此source下所有agent的平均特征向量
            mean_features_s = torch.mean(agents_for_s_features, dim=0)

            # 2. 计算每个agent特征与平均特征的余弦相似度
            agents_for_s_features_norm = F.normalize(agents_for_s_features, p=2, dim=1)
            mean_features_s_norm = F.normalize(mean_features_s.unsqueeze(0), p=2, dim=1)

            # 相似度越高，代表该agent越能代表这个source的“典型”特征
            similarities = torch.mm(agents_for_s_features_norm, mean_features_s_norm.t()).squeeze()

            # 3. 我们希望特征典型的agent（相似度高）的权重彼此接近
            # 因此，我们用相似度来加权计算权重的方差，并将其最小化
            mean_weight_s = torch.mean(weights_for_s)

            variance = (weights_for_s - mean_weight_s) ** 2
            weighted_variance = similarities * variance

            total_loss += torch.mean(weighted_variance)

        return total_loss / num_s if num_s > 0 else total_loss

@loss_registry.register("landuse_prediction_loss", default_weight=1.0,
                        description="Computes loss between predicted and actual landuse ratios based on edge weights.")
class LandusePredictionLoss(BaseLoss):
    def forward(self, edge_weights, edge_index, metadata):
        """
        基于边权重预测土地利用比例并计算损失

        Args:
            edge_weights: [num_edges] 预测的边权重
            edge_index_mapping: 边索引映射
            metadata: 包含landuse_mapping_matrix和landuse_ratio的元数据
        """
        if 'landuse_mapping_matrix' not in metadata or 'landuse_ratio' not in metadata:
            print("Warning: 元数据中缺少土地利用映射矩阵或真实比例信息，返回零损失")
            return torch.tensor(0.0, device=edge_weights.device)

        device = edge_weights.device
        mapping_matrix = metadata['landuse_mapping_matrix'].to(device)  # [num_edges, num_regions * num_landuse_types]
        true_landuse_ratio = metadata['landuse_ratio'].to(device)  # [num_regions, num_landuse_types]

        # 检查输入数据是否在正确的设备上
        # 检查矩阵维度是否匹配
        if mapping_matrix.shape[0] != edge_weights.shape[0]:
            print(f"Warning: 映射矩阵行数 ({mapping_matrix.shape[0]}) 与边权重数量 ({edge_weights.shape[0]}) 不匹配")
            return torch.tensor(0.0, device=device)

        if mapping_matrix.shape[1] == 0 or true_landuse_ratio.shape[1] == 0:
            # 没有土地利用类型信息，返回零损失
            print("Warning: 映射矩阵或真实土地利用比例没有类型信息，返回零损失")
            return torch.tensor(0.0, device=device)

        num_regions = metadata['num_s']
        num_landuse_types = true_landuse_ratio.shape[1]

        # =================================================================
        # 第一步: 通过边权重和映射矩阵计算预测的土地利用聚合值
        # =================================================================
        # 矩阵乘法进行加权聚合:
        # edge_weights: [num_edges] @ mapping_matrix: [num_edges, num_regions * num_landuse_types]
        # -> [num_regions * num_landuse_types]
        #
        # 例如: Region 0 连接3个agent，权重[0.5, 0.3, 0.2]，类型[residential, commercial, industrial]
        #      Region 1 连接2个agent，权重[0.7, 0.3]，都是residential类型
        #      聚合后: [0.5, 0.3, 0.2, 1.0, 0.0, 0.0] (flatten格式)
        predicted_flat = torch.matmul(edge_weights, mapping_matrix)

        # =================================================================
        # 第二步: 重塑为 [num_regions, num_landuse_types] 便于处理
        # =================================================================
        # 将扁平化结果重塑为二维矩阵:
        # [[0.5, 0.3, 0.2],   # region 0: res=0.5, com=0.3, ind=0.2
        #  [1.0, 0.0, 0.0]]   # region 1: res=1.0, com=0.0, ind=0.0
        predicted_landuse_aggregated = predicted_flat.view(num_regions, num_landuse_types)

        # =================================================================
        # 第三步: 归一化 - 关键步骤！为什么不能跳过这步？
        # =================================================================
        # 问题1: 数值尺度不匹配
        #   - predicted_landuse_aggregated 可能有任意的行和（0.5, 1.7, 2.3等）
        #   - true_landuse_ratio 每行和都是1.0（标准概率分布）
        #
        # 问题2: 不同region的agent数量差异
        #   - Region A有10个agent -> 聚合值较大
        #   - Region B有3个agent -> 聚合值较小
        #   - 直接比较会导致偏向agent多的region
        #
        # 问题3: 训练过程中权重约束不完美
        #   - 某个region权重和可能是0.95或1.05，而不是严格的1.0
        #
        # 解决方案: 将绝对值转换为相对比例，确保预测值和真实值在同一语义空间
        epsilon = 1e-8  # 防止除零
        region_totals = torch.sum(predicted_landuse_aggregated, dim=1, keepdim=True) + epsilon
        predicted_landuse_ratio = predicted_landuse_aggregated / region_totals


        # 使用KL散度损失
        true_landuse_safe = torch.clamp(true_landuse_ratio, min=epsilon)
        predicted_landuse_safe = torch.clamp(predicted_landuse_ratio, min=epsilon)
        loss = F.kl_div(torch.log(predicted_landuse_safe), true_landuse_safe, reduction='batchmean')

        return loss