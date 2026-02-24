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
    gat_heads: int = 4  # Number of GAT heads, used for GATConv

    # Differentiable allocation parameters
    allocation_temperature_start: float = 2.0  # Initial temperature (relatively high)

    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-4  # Weight decay
    epochs: int = 300
    clip_grad_norm: Optional[float] = None  # Whether to use gradient clipping
    use_scheduler: bool = True  # Whether to use cosine annealing learning rate scheduler
    warmup_epochs: Optional[int] = int(epochs*0.1)  # Number of warmup epochs; None disables warmup
    warmup_start_factor: float = 0.1  # Learning rate start multiplier for warmup phase
    warmup_end_factor: float = 1.0  # Learning rate end multiplier for warmup phase
    decay_epochs: Optional[int] = int(epochs*0.1)
    decay_start_factor: float = 1.0  # Learning rate start multiplier for decay phase
    decay_end_factor: float = 0.01
    cosine_epochs: Optional[int] = int(epochs*0.8)  # Period length; None uses total epoch count
    cosine_eta_min: float = 1e-6  # Minimum learning rate

    # Loss function
    learnable: bool = True  # Whether to use learnable loss function

    # Device parameters
    device: Optional[str] = None  # None means automatic selection
    debug: bool = True
    save_path: str = 'best_model.pth'