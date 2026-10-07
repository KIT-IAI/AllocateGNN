"""
Fetcher 注册表单例 — 仿 LossRegistry 模式
"""


class FetcherRegistry:
    """
    数据获取器注册表，用于管理和注册不同的 Fetcher 插件。
    通过装饰器 @fetcher_registry.register(name) 注册新的 Fetcher。
    """

    def __init__(self):
        self._fetchers = {}
        self._descriptions = {}

    def register(self, name, description=""):
        """装饰器工厂：注册一个 Fetcher 类"""
        def wrapper(cls):
            if name in self._fetchers:
                raise ValueError(f"Fetcher '{name}' 已注册，不可重复注册")
            self._fetchers[name] = cls
            self._descriptions[name] = description
            return cls
        return wrapper

    def get_fetcher(self, name):
        """按名称获取 Fetcher 类"""
        return self._fetchers.get(name)

    def get_available_fetchers(self):
        """枚举所有已注册的 Fetcher"""
        return {
            name: {"description": self._descriptions.get(name, "")}
            for name in self._fetchers
        }


# 全局单例
fetcher_registry = FetcherRegistry()
