"""
Weighter registry singleton — follows the ExtractorRegistry pattern
"""


class WeighterRegistry:
    """
    Registry for weight calculators, used to manage and register different Weighter plugins.
    New Weighters are registered via the @weighter_registry.register(name) decorator.
    """

    def __init__(self):
        self._weighters = {}
        self._descriptions = {}

    def register(self, name, description=""):
        """Decorator factory: register a Weighter class"""
        def wrapper(cls):
            if name in self._weighters:
                raise ValueError(f"Weighter '{name}' is already registered and cannot be registered again")
            self._weighters[name] = cls
            self._descriptions[name] = description
            return cls
        return wrapper

    def get_weighter(self, name):
        """Get a Weighter class by name"""
        return self._weighters.get(name)

    def get_available_weighters(self):
        """Enumerate all registered Weighters"""
        return {
            name: {"description": self._descriptions.get(name, "")}
            for name in self._weighters
        }

    def create(self, name, config=None):
        """
        Convenience factory: instantiate a Weighter by name.

        Args:
            name: registered name
            config: configuration dictionary passed to Weighter.__init__
        """
        cls = self._weighters.get(name)
        if cls is None:
            available = list(self._weighters.keys())
            raise KeyError(f"Weighter '{name}' is not registered, available: {available}")
        return cls(config or {})


# Global singleton
weighter_registry = WeighterRegistry()
