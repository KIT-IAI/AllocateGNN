from .agent_gating import AgentGating
from .edge_weight import DifferentiableEdgeWeighting
from .graph_encoder import GraphEncoder
from .positional_encoding import sinusoidal_pe
from .projection import MacroReconstructionHead, SpectralProjectionHead

__all__ = [
    "AgentGating",
    "DifferentiableEdgeWeighting",
    "GraphEncoder",
    "MacroReconstructionHead",
    "SpectralProjectionHead",
    "sinusoidal_pe",
]


def __getattr__(name):
    if name == "TemperatureAnnealer":
        raise AttributeError(
            "TemperatureAnnealer retired in 014; use "
            "DifferentiableEdgeWeighting.log_temperature"
        )
    raise AttributeError(name)


