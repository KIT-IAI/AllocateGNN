"""
Corrector registry singleton — modeled after the FetcherRegistry pattern
"""


class CorrectorRegistry:
    """
    Demand corrector registry, used to manage and register different Corrector plugins.
    New Correctors are registered via the @corrector_registry.register(name) decorator.
    """

    def __init__(self):
        self._correctors = {}
        self._descriptions = {}

    def register(self, name, description=""):
        """Decorator factory: registers a Corrector class"""
        def wrapper(cls):
            if name in self._correctors:
                raise ValueError(f"Corrector '{name}' is already registered, cannot register again")
            self._correctors[name] = cls
            self._descriptions[name] = description
            return cls
        return wrapper

    def get_corrector(self, name):
        """Get a Corrector class by name"""
        return self._correctors.get(name)

    def create(self, name, config=None):
        """Factory method: instantiates and returns a Corrector"""
        cls = self._correctors.get(name)
        if cls is None:
            raise ValueError(
                f"Corrector '{name}' is not registered. "
                f"Available: {list(self._correctors.keys())}"
            )
        return cls(config or {})

    def get_available_correctors(self):
        """Enumerate all registered Correctors"""
        return {
            name: {"description": self._descriptions.get(name, "")}
            for name in self._correctors
        }


# Global singleton
corrector_registry = CorrectorRegistry()
