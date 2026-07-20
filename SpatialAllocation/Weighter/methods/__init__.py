"""
Weighter methods subpackage — imports all weighter modules to trigger registration
"""
from .registry import weighter_registry, WeighterRegistry
from .base import BaseWeighter, WeightResult

# Import concrete weighter modules to trigger the @weighter_registry.register decorator
from . import uniform_weighter  # noqa: F401
from . import gpm_weighter  # noqa: F401
from . import hetero_gnn_weighter  # noqa: F401

__all__ = ["weighter_registry", "WeighterRegistry", "BaseWeighter", "WeightResult"]
