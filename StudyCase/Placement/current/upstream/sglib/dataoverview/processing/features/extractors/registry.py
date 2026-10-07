"""
Extractor 注册表单例 — 仿 LossRegistry 模式
"""


class ExtractorRegistry:
    """
    特征提取器注册表，用于管理和注册不同的 Extractor 插件。
    通过装饰器 @extractor_registry.register(name) 注册新的 Extractor。
    """

    def __init__(self):
        self._extractors = {}
        self._descriptions = {}

    def register(self, name, description=""):
        """装饰器工厂：注册一个 Extractor 类"""
        def wrapper(cls):
            if name in self._extractors:
                raise ValueError(f"Extractor '{name}' 已注册，不可重复注册")
            self._extractors[name] = cls
            self._descriptions[name] = description
            return cls
        return wrapper

    def get_extractor(self, name):
        """按名称获取 Extractor 类"""
        return self._extractors.get(name)

    def get_available_extractors(self):
        """枚举所有已注册的 Extractor"""
        return {
            name: {"description": self._descriptions.get(name, "")}
            for name in self._extractors
        }


# 全局单例
extractor_registry = ExtractorRegistry()
