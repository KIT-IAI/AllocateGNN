"""
Weighter 注册表单例 — 仿 ExtractorRegistry 模式
"""


class WeighterRegistry:
    """
    权重计算器注册表，用于管理和注册不同的 Weighter 插件。
    通过装饰器 @weighter_registry.register(name) 注册新的 Weighter。
    """

    def __init__(self):
        self._weighters = {}
        self._descriptions = {}

    def register(self, name, description=""):
        """装饰器工厂：注册一个 Weighter 类"""
        def wrapper(cls):
            if name in self._weighters:
                raise ValueError(f"Weighter '{name}' 已注册，不可重复注册")
            self._weighters[name] = cls
            self._descriptions[name] = description
            return cls
        return wrapper

    def get_weighter(self, name):
        """按名称获取 Weighter 类"""
        return self._weighters.get(name)

    def get_available_weighters(self):
        """枚举所有已注册的 Weighter"""
        return {
            name: {"description": self._descriptions.get(name, "")}
            for name in self._weighters
        }

    def create(self, name, config=None):
        """
        便捷工厂：按名称实例化 Weighter。

        参数:
            name: 注册名称
            config: 传给 Weighter.__init__ 的配置字典
        """
        cls = self._weighters.get(name)
        if cls is None:
            available = list(self._weighters.keys())
            raise KeyError(f"Weighter '{name}' 未注册，可用: {available}")
        return cls(config or {})


# 全局单例
weighter_registry = WeighterRegistry()


