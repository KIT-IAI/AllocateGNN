"""
Fetcher registry singleton - modeled after the LossRegistry pattern
"""


class FetcherRegistry:
    """
    Data fetcher registry, used to manage and register different Fetcher plugins.
    New Fetchers are registered via the @fetcher_registry.register(name) decorator.
    """

    def __init__(self):
        self._fetchers = {}
        self._descriptions = {}

    def register(self, name, description=""):
        """Decorator factory: register a Fetcher class"""
        def wrapper(cls):
            if name in self._fetchers:
                raise ValueError(f"Fetcher '{name}' is already registered and cannot be registered again")
            self._fetchers[name] = cls
            self._descriptions[name] = description
            return cls
        return wrapper

    def get_fetcher(self, name):
        """Get a Fetcher class by name"""
        return self._fetchers.get(name)

    def get_available_fetchers(self):
        """Enumerate all registered Fetchers"""
        return {
            name: {"description": self._descriptions.get(name, "")}
            for name in self._fetchers
        }


# Global singleton
fetcher_registry = FetcherRegistry()
