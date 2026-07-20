from dataclasses import dataclass
from typing import Optional


@dataclass
class AllocationConfig:
    """Stage 2 allocation system configuration parameters"""

    # Graph structure
    k_nearest_targets: int = 5          # number of agent->target k-NN edges

    # Encoder
    hidden_dim: int = 64
    embedding_dim: int = 32
    num_layers: int = 2
    conv_type: str = 'hgt'
    gat_heads: int = 4

    # Weight layer
    allocation_temperature: float = 0.1

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    clip_grad_norm: Optional[float] = None
    use_scheduler: bool = True
    warmup_epochs: int = 20
    decay_epochs: int = 20
    cosine_epochs: int = 160
    cosine_eta_min: float = 1e-5
    learnable: bool = True              # whether to use learnable loss function weights

    # Feature control
    use_weight_feature: bool = True     # whether to include W_sa as an agent feature

    # Device/save
    device: Optional[str] = None
    debug: bool = True
    save_path: str = 'allocation_model.pth'
