from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    # GraphEncoder parameters
    hidden_dim: int = 128
    embedding_dim: int = 64
    num_layers: int = 3
    conv_type: str = 'sage'  # Options: 'gcn', 'sage', 'gat', 'gin', 'transformer', 'gated', 'hgt', 'rgcn', 'pna', 'edge', 'graph', 'gmm', 'appnp', 'sg'
    gat_heads: int = 4  # Number of GAT heads, applies to GATConv

    # Differentiable allocation parameters
    allocation_temperature_start: float = 2.0  # Initial temperature (relatively high)

    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-4  # Add weight decay
    epochs: int = 300
    clip_grad_norm: Optional[float] = None  # Whether to use gradient clipping
    use_scheduler: bool = True  # Whether to use a cosine annealing LR scheduler
    warmup_epochs: Optional[int] = int(epochs*0.1)  # Number of warmup epochs; if None, no warmup is used
    warmup_start_factor: float = 0.1  # Starting LR multiplier during warmup
    warmup_end_factor: float = 1.0  # Ending LR multiplier during warmup
    decay_epochs: Optional[int] = int(epochs*0.1)
    decay_start_factor: float = 1.0  # Starting LR multiplier during decay
    decay_end_factor: float = 0.01
    cosine_epochs: Optional[int] = int(epochs*0.8)  # Cycle length; if None, uses the total epoch count
    cosine_eta_min: float = 1e-6  # Minimum learning rate

    # Loss function
    learnable: bool = True  # Whether to use learnable loss weighting

    # Phase A: spectral features + projection head (dims/hyperparams; enabling is inferred from objective_weights)
    spectral_feature_dim: int = 10  # Spectral feature dimension (10=mean+std, 20=extended)
    projection_bottleneck_dim: int = 32  # Projection head bottleneck dimension d_z
    macro_attribute_dim: int = 5  # Macro attribute dimension M (UK = 5 GVA industry classes)
    recon_head_type: str = 'softmax'  # Reconstruction head type: 'softmax', 'linear', 'mixed'

    # Phase A: agent gating (enabling is inferred from whether objective_weights contains 'gate')
    gate_lambda: float = 0.01  # Gate regularization coefficient lambda_g
    gate_bias_init: float = 2.0  # Gate bias initial value (default g_a ~ 0.88)

    # Device parameters
    device: Optional[str] = None  # None means auto-select
    debug: bool = True
    save_path: str = 'best_model.pth'