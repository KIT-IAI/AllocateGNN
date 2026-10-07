"""解析一次性采集的本地 NTL GeoTIFF。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseFetcher
from .registry import fetcher_registry


@fetcher_registry.register("ntl", description="本地 VIIRS 夜间灯光 GeoTIFF")
class NtlFetcher(BaseFetcher):
    """运行时只返回 registry 指定的本地 GeoTIFF，不访问网络。"""

    def __init__(self, config: dict):
        super().__init__(config)
        value = config.get("path")
        if not isinstance(value, str) or not value:
            raise ValueError("NTL fetcher 需要非空 path")
        self.path = Path(value)
        self.output_crs = config.get("output_crs")

    def fetch(self, region_bounds: Any = None, cache_dir: str = "") -> str:
        path = self.path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"离线 NTL GeoTIFF 不存在: {path}")
        return str(path)
