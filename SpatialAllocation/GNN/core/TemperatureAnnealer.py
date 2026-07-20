"""
Temperature annealer: controls the temperature decay of the second-layer
edge weights.

Supports two scheduling modes:
  - exponential: tau(epoch) = max(tau_init * alpha^epoch, min_tau)
  - cosine: holds tau_init during the warmup phase, then decays via cosine
    schedule to min_tau

A plain Python class (not an nn.Module); it does not participate in
gradient computation.
"""
import math


class TemperatureAnnealer:
    """
    Temperature annealer used to control the temperature parameter of
    SecondLayerEdgeWeighting.

    Args:
        tau_init: Initial temperature tau_0
        alpha: Decay rate alpha in (0, 1); the temperature is multiplied by
            alpha every epoch (exponential mode)
        min_tau: Lower bound on temperature, to prevent numerical
            instability from an excessively low temperature
        schedule: Scheduling mode, 'exponential' or 'cosine'
        warmup_epochs: Warmup epochs under cosine mode (temperature stays high)
        anneal_epochs: Total cosine annealing epochs (including warmup)
    """

    def __init__(
        self,
        tau_init: float = 1.0,
        alpha: float = 0.95,
        min_tau: float = 0.01,
        schedule: str = 'exponential',
        warmup_epochs: int = 20,
        anneal_epochs: int = 100,
    ):
        self.tau_init = tau_init
        self.alpha = alpha
        self.min_tau = min_tau
        self.schedule = schedule
        self.warmup_epochs = warmup_epochs
        self.anneal_epochs = anneal_epochs

    def get_temperature(self, epoch: int) -> float:
        """
        Compute the temperature value for a given epoch.

        Args:
            epoch: Current epoch number (starting at 0)

        Returns:
            Temperature value tau(epoch)
        """
        if self.schedule == 'cosine':
            # Hold a high temperature during the warmup phase
            if epoch < self.warmup_epochs:
                return self.tau_init
            # Cosine decay phase
            progress = (epoch - self.warmup_epochs) / max(self.anneal_epochs - self.warmup_epochs, 1)
            progress = min(progress, 1.0)
            return self.min_tau + 0.5 * (self.tau_init - self.min_tau) * (1 + math.cos(math.pi * progress))
        else:
            # exponential (default, kept for backward compatibility)
            tau = self.tau_init * (self.alpha ** epoch)
            return max(tau, self.min_tau)

    def state_dict(self) -> dict:
        """Serialize the annealer state for checkpoint saving."""
        return {
            'tau_init': self.tau_init,
            'alpha': self.alpha,
            'min_tau': self.min_tau,
            'schedule': self.schedule,
            'warmup_epochs': self.warmup_epochs,
            'anneal_epochs': self.anneal_epochs,
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore the annealer state from a checkpoint."""
        self.tau_init = d['tau_init']
        self.alpha = d['alpha']
        self.min_tau = d['min_tau']
        self.schedule = d.get('schedule', 'exponential')
        self.warmup_epochs = d.get('warmup_epochs', 20)
        self.anneal_epochs = d.get('anneal_epochs', 100)
