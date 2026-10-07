"""
FeatureExtractor 模块化特征组件

仿 LossFunction 的 Registry 插件模式:
- fetchers/ — 数据获取层（GEE, OSM 等）
- extractors/ — 特征计算层
案例级编排位于 ``casestudy/1_DataOverview``；共享执行 API 位于
``sglib.dataoverview``。
"""

# 导入活动子包（触发 fetcher/extractor 注册）
from . import fetchers  # noqa: F401
from . import extractors  # noqa: F401

# 导出公共 API
from .grid_generator import calculate_step_size, generate_base_grid
from .fetchers import fetcher_registry, FetcherRegistry, BaseFetcher
from .extractors import extractor_registry, ExtractorRegistry, BaseExtractor, ExtractorResult
from .cuz_support import (
    CuzSupport,
    CuzSupportError,
    LANDUSE_CATEGORIES,
    OSM_LANDUSE_COLUMNS,
    SCHEMA_VERSION as CUZ_SCHEMA_VERSION,
    atomic_save_npy,
    atomic_savez,
    build_cuz_support,
    deterministic_built_surface_field,
    load_cuz_support,
    save_cuz_support,
    validate_cuz_grid_crs,
)

__all__ = [
    "calculate_step_size",
    "generate_base_grid",
    "fetcher_registry",
    "FetcherRegistry",
    "BaseFetcher",
    "extractor_registry",
    "ExtractorRegistry",
    "BaseExtractor",
    "ExtractorResult",
    "CuzSupport",
    "CuzSupportError",
    "LANDUSE_CATEGORIES",
    "OSM_LANDUSE_COLUMNS",
    "CUZ_SCHEMA_VERSION",
    "atomic_save_npy",
    "atomic_savez",
    "build_cuz_support",
    "deterministic_built_surface_field",
    "load_cuz_support",
    "save_cuz_support",
    "validate_cuz_grid_crs",
]


_RETIRED_CORRECTOR_EXPORTS = {
    "corrector_registry", "CorrectorRegistry", "BaseCorrector", "CorrectorResult"
}


def __getattr__(name):
    if name in _RETIRED_CORRECTOR_EXPORTS:
        raise AttributeError(
            f"FeatureExtractor.{name} retired in 014; use "
            "sglib.generator.weighter.correction"
        )
    raise AttributeError(name)
