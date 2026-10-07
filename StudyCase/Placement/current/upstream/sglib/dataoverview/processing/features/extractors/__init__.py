"""
Extractors 子包 — 导入所有 extractor 模块触发注册
"""
from .registry import extractor_registry, ExtractorRegistry
from .base import BaseExtractor, ExtractorResult

# 导入具体 extractor 模块，触发 @extractor_registry.register 装饰器
# 随着新增 extractor，在此添加 import
from . import landuse_extractor  # noqa: F401
from . import ntl_extractor  # noqa: F401
from . import built_surface_extractor  # noqa: F401

__all__ = ["extractor_registry", "ExtractorRegistry", "BaseExtractor", "ExtractorResult"]
