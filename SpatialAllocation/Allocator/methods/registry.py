"""
Allocator registry singleton — follows the ExtractorRegistry pattern
"""


class AllocatorRegistry:
    """
    Registry for spatial allocators, used to manage and register different Allocator plugins.
    New Allocators are registered via the @allocator_registry.register(name) decorator.
    """

    def __init__(self):
        self._allocators = {}
        self._descriptions = {}

    def register(self, name, description=""):
        """Decorator factory: register an Allocator class"""
        def wrapper(cls):
            if name in self._allocators:
                raise ValueError(f"Allocator '{name}' is already registered and cannot be registered again")
            self._allocators[name] = cls
            self._descriptions[name] = description
            return cls
        return wrapper

    def get_allocator(self, name):
        """Get an Allocator class by name"""
        return self._allocators.get(name)

    def get_available_allocators(self):
        """Enumerate all registered Allocators"""
        return {
            name: {"description": self._descriptions.get(name, "")}
            for name in self._allocators
        }

    def create(self, name, config=None):
        """
        Convenience factory: instantiate an Allocator by name.

        Args:
            name: registered name
            config: configuration dictionary passed to Allocator.__init__
        """
        cls = self._allocators.get(name)
        if cls is None:
            available = list(self._allocators.keys())
            raise KeyError(f"Allocator '{name}' is not registered, available: {available}")
        return cls(config or {})


# Global singleton
allocator_registry = AllocatorRegistry()
