import torch
import torch.nn as nn
import torch.nn.functional as F
from SpatialAllocation.GNN.core.ModelConfig import ModelConfig
from torch_scatter import scatter_softmax

class DifferentiableEdgeWeighting(nn.Module):
    """
    可微分边权重预测：预测已知边的权重，确保每个A点的总权重为1
    """

    def __init__(self, config: ModelConfig):
        super(DifferentiableEdgeWeighting, self).__init__()
        self.config = config
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(self.config.allocation_temperature_start)))

        # 定义一个MLP来学习边成本（或称为“关系得分”）
        # 输入维度是两个嵌入向量拼接后的大小 (embedding_dim * 2)
        # 输出维度为1，代表一个标量的成本值
        self.gating_mlp = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()  # 使用 Sigmoid 将输出缩放到 (0, 1) 范围
        )

    def forward(self, embeddings_s, embeddings_a, edge_index_sa):
        """
        预测边权重
        Args:
            embeddings_s: Source点嵌入 [num_s, embedding_dim]
            embeddings_a: Agent点嵌入 [num_a, embedding_dim]
            edge_index_sa: 边索引 [2, num_edges]，第一行是A点索引，第二行是B点索引
        Returns:
            edge_weights: 边权重 [num_edges]
            edge_costs: 边的成本 [num_edges]
        """
        # 1. 计算边的嵌入距离
        s_indices = edge_index_sa[0]  # Source点索引
        a_indices = edge_index_sa[1]  # Agent点索引

        # 获取边对应的嵌入向量
        edge_embeddings_s = embeddings_s[s_indices]  # [num_edges, embedding_dim]
        edge_embeddings_a = embeddings_a[a_indices]  # [num_edges, embedding_dim]

        # 计算边的成本（嵌入距离）
        edge_costs = torch.norm(edge_embeddings_s - edge_embeddings_a, dim=1)  # [num_edges]
        concatenated_embeddings = torch.cat([edge_embeddings_s, edge_embeddings_a], dim=1)
        gate = self.gating_mlp(concatenated_embeddings).squeeze(-1)
        edge_costs = edge_costs * gate


        # 2. 使用温度参数进行softmax归一化
        temperature = torch.exp(self.log_temperature)

        # 准备 softmax 的输入值
        # The values to be softmaxed. We negate the cost because softmax gives higher probability to larger values.
        values_for_softmax = -edge_costs / temperature

        # 3. 使用 scatter_softmax 对每个S点的边进行分组Softmax
        # Use scatter_softmax to perform a grouped softmax for each source node's edges.
        # src: 输入张量 (the input tensor) -> values_for_softmax
        # index: 分组依据的索引 (the index to group by) -> s_indices
        # dim: 操作的维度 (the dimension to operate on)
        edge_weights = scatter_softmax(values_for_softmax, s_indices, dim=0)

        return edge_weights, edge_costs