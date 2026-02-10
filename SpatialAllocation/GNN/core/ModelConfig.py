from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable


@dataclass
class ModelConfig:
    """模型配置参数"""
    # GraphEncoder 参数
    hidden_dim: int = 128
    embedding_dim: int = 64
    num_layers: int = 3
    conv_type: str = 'sage'  # 可选: 'gcn', 'sage', 'gat', 'gin', 'transformer', 'gated', 'hgt', 'rgcn', 'pna', 'edge', 'graph', 'gmm', 'appnp', 'sg'
    gat_heads: int = 4  # GAT头数，适用于GATConv

    # Differentiableallocation 参数
    allocation_temperature_start: float = 2.0  # 初始温度（较高）

    # 训练参数
    learning_rate: float = 0.001
    weight_decay: float = 1e-4  # 添加权重衰减
    epochs: int = 300
    clip_grad_norm: Optional[float] = None  # 是否使用梯度裁剪
    use_scheduler: bool = True  # 是否使用余弦退火学习率调度器
    warmup_epochs: Optional[int] = int(epochs*0.1)  # 预热阶段的epoch数，如果为None则不使用预热
    warmup_start_factor: float = 0.1  # 预热阶段的学习率起始倍率
    warmup_end_factor: float = 1.0  # 预热阶段的学习率结束倍率
    decay_epochs: Optional[int] = int(epochs*0.1)
    decay_start_factor: float = 1.0  # 衰减阶段的学习率起始倍率
    decay_end_factor: float = 0.01
    cosine_epochs: Optional[int] = int(epochs*0.8)  # 周期长度，如果为None则使用总epoch数
    cosine_eta_min: float = 1e-6  # 最小学习率

    # 损失函数
    learnable: bool = True  # 是否使用可学习的损失函数

    # 设备参数
    device: Optional[str] = None  # None表示自动选择
    debug: bool = True
    save_path: str = 'best_model.pth'