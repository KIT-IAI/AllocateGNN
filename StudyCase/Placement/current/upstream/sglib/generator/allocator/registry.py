"""
Allocator 注册表单例 — 仿 ExtractorRegistry 模式
"""

import re
import unicodedata


class AllocatorRegistry:
    """
    空间分配器注册表，用于管理和注册不同的 Allocator 插件。
    通过装饰器 @allocator_registry.register(name) 注册新的 Allocator。
    """

    def __init__(self):
        self._allocators = {}
        self._descriptions = {}

    _RETIRED_ALLOCATORS = frozenset()
    _IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*$")

    @classmethod
    def _canonicalize_identifier(cls, name):
        """Return one safe registry key for every accepted spelling."""

        if not isinstance(name, str):
            raise TypeError("Allocator identifier must be a string")
        canonical = unicodedata.normalize("NFKC", name).strip().casefold()
        if not canonical or cls._IDENTIFIER_RE.fullmatch(canonical) is None:
            raise ValueError(
                f"Invalid allocator identifier {name!r}; expected an ASCII slug "
                "starting with a letter and containing only letters, digits, '_' or '-'"
            )
        return canonical

    def register(self, name, description=""):
        """装饰器工厂：注册一个 Allocator 类"""
        canonical = self._canonicalize_identifier(name)

        def wrapper(cls):
            if canonical in self._allocators:
                raise ValueError(f"Allocator '{canonical}' 已注册，不可重复注册")
            self._allocators[canonical] = cls
            self._descriptions[canonical] = description
            return cls
        return wrapper

    def get_allocator(self, name):
        """按名称获取 Allocator 类"""
        canonical = self._canonicalize_identifier(name)
        return self._allocators.get(canonical)

    def get_available_allocators(self):
        """枚举所有已注册的 Allocator"""
        return {
            name: {"description": self._descriptions.get(name, "")}
            for name in self._allocators
        }

    def create(self, name, config=None):
        """
        便捷工厂：按名称实例化 Allocator。

        参数:
            name: 注册名称
            config: 传给 Allocator.__init__ 的配置字典
        """
        canonical = self._canonicalize_identifier(name)
        cls = self._allocators.get(canonical)
        if cls is None:
            available = list(self._allocators.keys())
            raise KeyError(f"Allocator '{canonical}' 未注册，可用: {available}")
        return cls(config or {})


# 全局单例
allocator_registry = AllocatorRegistry()


