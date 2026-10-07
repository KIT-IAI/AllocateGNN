from torch import nn
import torch
from .base import loss_registry


# 当前正式 loss 均可使用固定权重或不确定性权重。
FIXED_WEIGHT_LOSSES = set()


class CombinedLoss(nn.Module):
    def __init__(self, weights=None, learnable=False):
        super().__init__()

        self.registry = loss_registry
        self.learnable = learnable

        # 默认权重
        if weights is None:
            weights = {'entropy_regularization': 1.0}
        self.weights = weights

        # 确定要使用的损失函数
        self.use_losses = list(self.weights.keys())

        if self.learnable:
            print("Using learnable weights for losses. Each loss will have a learnable log variance parameter.")
            self.log_vars = nn.ParameterDict()
            # 为每个损失函数创建可学习的对数方差参数（排除固定权重损失）
            for name in self.use_losses:
                if self.weights[name] > 0 and name not in FIXED_WEIGHT_LOSSES:
                    self.log_vars[name] = nn.Parameter(torch.zeros(1))
        else:
            self.log_vars = None

        # 创建损失函数实例
        self.loss_functions = {
            name: self.registry.get_loss(name)()
            for name in self.use_losses
            if self.weights[name] > 0
        }

    # 修改 forward 方法的签名，移除不再需要的 edge_index_mapping
    def forward(self, edge_weights, edge_index, metadata):
        # 计算每个损失
        # 将参数正确传递给每个子损失函数
        losses = {name: loss_fn(edge_weights, edge_index, metadata) for name, loss_fn in self.loss_functions.items()}

        total_loss = 0

        if self.learnable:
            # 使用可学习权重（多任务学习中的不确定性权重）
            for name, loss_value in losses.items():
                if name in FIXED_WEIGHT_LOSSES:
                    # 固定权重损失（如 gate）：始终使用预设权重
                    weight = self.weights.get(name, 0)
                    total_loss += weight * loss_value
                elif name in self.log_vars:
                    # 使用不确定性加权：loss / (2 * sigma^2) + log(sigma)
                    precision = torch.exp(-self.log_vars[name])
                    total_loss += precision * loss_value + self.log_vars[name]
        else:
            # 使用固定权重
            for name, loss_value in losses.items():
                weight = self.weights.get(name, 0)
                total_loss += weight * loss_value

        return total_loss, losses


