"""
Correctors subpackage — imports all corrector modules to trigger registration
"""
from .registry import corrector_registry, CorrectorRegistry
from .base import BaseCorrector, CorrectorResult

# Import concrete corrector modules to trigger the @corrector_registry.register decorator
from . import wc_corrector       # noqa: F401
from . import ntl_corrector      # noqa: F401
from . import proximity_corrector  # noqa: F401

__all__ = ["corrector_registry", "CorrectorRegistry", "BaseCorrector", "CorrectorResult"]
