"""
Fetchers 子包 — 导入所有 fetcher 模块触发注册
"""
from .registry import fetcher_registry, FetcherRegistry
from .base import BaseFetcher

# 导入具体 fetcher 模块，触发 @fetcher_registry.register 装饰器
# 随着新增 fetcher，在此添加 import
from . import osm_fetcher  # noqa: F401
from . import ntl_fetcher  # noqa: F401
from . import ghsl_built_s_fetcher  # noqa: F401

__all__ = ["fetcher_registry", "FetcherRegistry", "BaseFetcher"]
