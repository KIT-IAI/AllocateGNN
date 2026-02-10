import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, HeteroConv, SAGEConv, GATConv,
    GINConv, HGTConv
)
from SpatialAllocation.GNN.core.ModelConfig import ModelConfig
from typing import Dict, List, Tuple


class GraphEncoder(nn.Module):
    """
    异构图编码器：使用HeteroConv学习多类型节点的低维表示
    目的: 为 'source' 和 'agent' 两种节点类型分别学习信息丰富的嵌入。
    工作原理:
        1. 使用一个 nn.ModuleDict 包含的线性层，将不同维度的输入特征统一到相同的隐藏维度。
        2. 使用多层 HeteroConv。在每一层，HeteroConv会为图中的每种边关系（例如 'source' -> 'agent'）调用一个指定的GNN层（如GCNConv）来进行消息传递和聚合。
        3. 经过多层传播，每个节点的最终嵌入不仅包含了邻域信息，还隐式地包含了邻居的类型信息。
    结果: 返回一个字典，包含每种节点类型对应的低维、稠密的嵌入向量。
    """

    def __init__(self, input_dims: Dict[str, int], config: ModelConfig, metadata: Tuple[List[str], List[Tuple[str, str, str]]]):
        super(GraphEncoder, self).__init__()
        self.config = config
        self.conv_type = self.config.conv_type.lower()

        # 验证conv_type
        supported_types = ['gcn', 'sage', 'gat', 'gin', 'hgt']
        if self.conv_type not in supported_types:
            raise ValueError(f"Unsupported conv_type: {self.conv_type}. Supported types: {supported_types}")

        # 1. 为每种节点类型创建一个初始的线性投影层，以统一维度
        self.lin_dict = nn.ModuleDict()
        for node_type, in_dim in input_dims.items():
            # 如果某个节点类型没有特征，我们创建一个可学习的嵌入来表示它
            if in_dim == 0:
                self.lin_dict[node_type] = nn.Embedding(1, config.hidden_dim)
            else:
                self.lin_dict[node_type] = nn.Linear(in_dim, config.hidden_dim)

        # 2. 创建异构图卷积层列表
        self.convs = nn.ModuleList()
        # HGTConv 需要一个不同的初始化和前向传播路径
        if self.conv_type == 'hgt':
            for _ in range(config.num_layers):
                # HGTConv需要图的元数据来进行初始化
                conv = HGTConv(-1, config.hidden_dim, metadata, heads=self.config.gat_heads)
                self.convs.append(conv)
        else:
            # 对于其他类型的卷积，使用HeteroConv包装
            for _ in range(config.num_layers):
                # MODIFICATION 1: 为每种边类型创建独立的卷积实例
                conv_dict = {}
                for edge_type in metadata[1]:
                    # edge_type 本身就是元组，例如 ('source', 'connects_to', 'agent')
                    if self.conv_type == 'gcn':
                        # 直接使用元组 edge_type 作为键
                        conv_dict[edge_type] = GCNConv(-1, config.hidden_dim, add_self_loops=False)
                    elif self.conv_type == 'sage':
                        conv_dict[edge_type] = SAGEConv(-1, config.hidden_dim)
                    elif self.conv_type == 'gat':
                        heads = self.config.gat_heads
                        if config.hidden_dim % heads != 0:
                            raise ValueError(
                                f"For GAT, hidden_dim ({config.hidden_dim}) must be divisible by heads ({heads}).")
                        out_channels = config.hidden_dim // heads
                        conv_dict[edge_type] = GATConv(-1, out_channels, heads=heads, add_self_loops=False)
                    elif self.conv_type == 'gin':
                        mlp = nn.Sequential(
                            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
                            nn.ReLU(),
                            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                        )
                        conv_dict[edge_type] = GINConv(nn=mlp, train_eps=True)

                # HeteroConv 接收以元组为键的字典，并会自动管理其中的模块
                conv = HeteroConv(conv_dict, aggr='sum')
                self.convs.append(conv)

        # 3. 创建一个最终的线性层，用于输出期望的 embedding_dim
        self.out_lin = nn.Linear(config.hidden_dim, config.embedding_dim)

        # 4. 可学习的缩放因子 g (与之前相同)
        self.g = nn.Parameter(torch.ones(config.embedding_dim))
        self.dropout = nn.Dropout(0.1)

        # 5. 为每种节点类型创建一个 LayerNorm 层
        self.norm_dict = nn.ModuleDict()
        for node_type in input_dims.keys():
            self.norm_dict[node_type] = nn.LayerNorm(config.hidden_dim)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[str, torch.Tensor]) -> Dict[
        str, torch.Tensor]:
        """
        前向传播函数，接收并返回字典格式的数据
        """
        # 1. 应用初始的线性变换或嵌入
        for node_type, x in x_dict.items():
            if x.size(1) == 0:  # 如果没有输入特征
                # 创建一个全零的索引张量来查询嵌入
                idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                x_dict[node_type] = self.lin_dict[node_type](idx)
            else:
                x_dict[node_type] = self.lin_dict[node_type](x)

        # 2. 通过多层异构卷积
        for i, conv in enumerate(self.convs):
            # 缓存输入以用于残差连接
            x_input_dict = x_dict
            x_dict = conv(x_dict, edge_index_dict)

            # 对每种节点类型应用激活、归一化和残差连接
            for node_type, x_out in x_dict.items():
                x = F.relu(x_out)
                x = self.norm_dict[node_type](x)

                # 添加残差连接 (确保输入字典中有对应的节点类型)
                if node_type in x_input_dict:
                    x = x + x_input_dict[node_type]

                if i < len(self.convs) - 1:
                    x = self.dropout(x)
                x_dict[node_type] = x


        # 3. 应用最终的输出线性层
        for node_type in x_dict.keys():
            x_dict[node_type] = self.out_lin(x_dict[node_type])

        # 4. L2归一化和缩放 (与之前相同, 但对字典中的每个张量操作)
        for node_type in x_dict.keys():
            x_dict[node_type] = F.normalize(x_dict[node_type], p=2, dim=1)
            x_dict[node_type] = self.g * x_dict[node_type]

        return x_dict