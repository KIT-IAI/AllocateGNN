import torch
import torch.nn as nn


class AgentGating(nn.Module):
    """
    可学习的 Agent 有效性门控模块。

    对每个 agent 预测一个有效性得分 g_a ∈ (0, 1)，用于在第一层权重计算中
    排除零负荷区域（水体、荒地等）。

    公式: g_a = sigmoid(v^T z_a + b_v)

    偏置 b_v 初始化为正值（默认 +2.0），使初始 g_a ≈ 0.88，
    让模型默认认为所有 agent 有效，只在必要时排除。

    参数:
        d_z: 瓶颈嵌入维度（默认 32）
        bias_init: 偏置初始化值（默认 2.0）
    """

    def __init__(self, d_z: int = 32, bias_init: float = 2.0):
        super().__init__()
        self.d_z = d_z

        # 线性层：d_z → 1
        self.gate_linear = nn.Linear(d_z, 1)

        # 初始化偏置为正值（默认有效）
        with torch.no_grad():
            self.gate_linear.bias.fill_(bias_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            z: (N, d_z) 瓶颈嵌入

        Returns:
            g: (N,) 有效性得分，值域 (0, 1)
        """
        return torch.sigmoid(self.gate_linear(z)).squeeze(-1)


