"""D3 统一非 Z、严格 k 与不足支持的边界。"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from sglib.experiment.planning_pool import build_pool, solve_strict

pytestmark = pytest.mark.consume


def test_insufficient_non_z_support_is_not_silently_clamped():
    grid = gpd.GeoDataFrame({"zero_mask": [False] * 20}, geometry=[Point(i, 0) for i in range(20)], crs="EPSG:4326")
    result, record = build_pool(grid, 11, "EPSG:27700")
    assert result is None and record["status"] == "INELIGIBLE_BY_DESIGN"
    assert record["requested_M"] == 300
    with pytest.raises(ValueError, match="禁止缩小"):
        solve_strict(np.zeros((3, 2)), np.zeros((2, 2)), np.ones(3), 3, {})
