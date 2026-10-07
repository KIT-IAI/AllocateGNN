import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralProjectionHead(nn.Module):
    """
    光谱特征投影头：将光谱指数特征映射到瓶颈嵌入空间，再投影到 K 维概率分布。

    架构: Linear(d_spec → d_z) + ReLU + Linear(d_z → K) + temperature softmax

    参数:
        d_spec: 光谱特征维度（默认 10，对应 5 指数 × 2 统计量）
        d_z: 瓶颈嵌入维度（默认 32）
        K: 投影输出维度（默认 5，对应英国数据集的 5 类 GVA 行业）
    """

    def __init__(self, d_spec: int = 10, d_z: int = 32, K: int = 5):
        super().__init__()
        self.d_spec = d_spec
        self.d_z = d_z
        self.K = K

        # 瓶颈编码层
        self.encoder = nn.Linear(d_spec, d_z)

        # 投影层：从瓶颈空间到 K 维
        self.projector = nn.Linear(d_z, K)

        # 可学习温度参数（log 空间，确保正值）
        self.log_tau_proj = nn.Parameter(torch.zeros(1))

    def forward(self, spectral_features: torch.Tensor):
        """
        前向传播。

        Args:
            spectral_features: (N, d_spec) 光谱特征

        Returns:
            z: (N, d_z) 瓶颈嵌入
            T_hat: (N, K) 概率分布（softmax 归一化）
        """
        # 瓶颈嵌入
        z = F.relu(self.encoder(spectral_features))

        # 投影到 K 维 + temperature softmax
        logits = self.projector(z)
        tau = torch.exp(self.log_tau_proj)
        T_hat = F.softmax(logits / tau, dim=-1)

        return z, T_hat


class MacroReconstructionHead(nn.Module):
    """
    宏观属性重建头：将瓶颈嵌入 z_a 映射到 M 维宏观属性空间。

    支持三种头类型:
        - 'softmax': 输出概率分布（适用于比例类属性，如 GVA 行业占比）
        - 'linear': 无约束输出（适用于连续值属性）
        - 'mixed': 前 K_softmax 维用 softmax，其余用线性

    参数:
        d_z: 瓶颈嵌入维度（默认 32）
        M: 宏观属性维度（默认 5）
        head_type: 头类型（'softmax', 'linear', 'mixed'）
        K_softmax: mixed 模式下 softmax 部分的维度（默认与 M 相同）
    """

    def __init__(self, d_z: int = 32, M: int = 5, head_type: str = 'softmax',
                 K_softmax: int = None):
        super().__init__()
        self.d_z = d_z
        self.M = M
        self.head_type = head_type
        self.K_softmax = K_softmax if K_softmax is not None else M

        self.linear = nn.Linear(d_z, M)

    def forward(self, z: torch.Tensor):
        """
        前向传播。

        Args:
            z: (N, d_z) 瓶颈嵌入

        Returns:
            recon: (N, M) 重建的宏观属性
        """
        raw = self.linear(z)

        if self.head_type == 'softmax':
            return F.softmax(raw, dim=-1)
        elif self.head_type == 'linear':
            return raw
        elif self.head_type == 'mixed':
            softmax_part = F.softmax(raw[:, :self.K_softmax], dim=-1)
            linear_part = raw[:, self.K_softmax:]
            return torch.cat([softmax_part, linear_part], dim=-1)
        else:
            raise ValueError(f"不支持的头类型: {self.head_type}，可选: 'softmax', 'linear', 'mixed'")


