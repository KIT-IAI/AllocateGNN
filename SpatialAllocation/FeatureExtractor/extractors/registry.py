"""
Extractor registry singleton — modeled after the LossRegistry pattern
"""


class ExtractorRegistry:
    """
    Feature extractor registry, used to manage and register different Extractor plugins.
    New Extractors are registered via the @extractor_registry.register(name) decorator.
    """

    def __init__(self):
        self._extractors = {}
        self._descriptions = {}

    def register(self, name, description=""):
        """Decorator factory: registers an Extractor class"""
        def wrapper(cls):
            if name in self._extractors:
                raise ValueError(f"Extractor '{name}' is already registered, cannot register again")
            self._extractors[name] = cls
            self._descriptions[name] = description
            return cls
        return wrapper

    def get_extractor(self, name):
        """Get an Extractor class by name"""
        return self._extractors.get(name)

    def get_available_extractors(self):
        """Enumerate all registered Extractors"""
        return {
            name: {"description": self._descriptions.get(name, "")}
            for name in self._extractors
        }


# Global singleton
extractor_registry = ExtractorRegistry()
