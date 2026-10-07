"""
BaseFetcher 基类 — 所有 Fetcher 插件的统一接口
"""
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseFetcher(ABC):
    """
    数据获取器基类。
    每个 Fetcher 负责从特定数据源（GEE、OSM 等）下载原始数据到本地，
    并管理缓存（文件已存在则跳过下载）。
    """

    def __init__(self, config: dict):
        """
        参数:
            config: JSON 配置中对应 fetcher 的配置块
        """
        self.config = config

    @abstractmethod
    def fetch(self, region_bounds: Any, cache_dir: str) -> str:
        """
        下载原始数据到本地，返回文件路径。
        如果缓存文件已存在，直接返回路径（跳过下载）。

        参数:
            region_bounds: 区域边界（Shapely geometry 或 (minx, miny, maxx, maxy) 元组，EPSG:4326）
            cache_dir: 缓存根目录

        返回:
            本地文件路径（GeoTIFF / pickle 等）
        """
        raise NotImplementedError

    def get_cache_key(self, region_bounds: Any) -> str:
        """
        根据区域边界和配置生成缓存文件名哈希。

        参数:
            region_bounds: 区域边界

        返回:
            MD5 哈希字符串（前 12 位）
        """
        # 将 bounds 规范化为元组
        if hasattr(region_bounds, 'bounds'):
            bounds_tuple = region_bounds.bounds
        else:
            bounds_tuple = tuple(region_bounds)

        key_str = f"{bounds_tuple}_{self.config}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
