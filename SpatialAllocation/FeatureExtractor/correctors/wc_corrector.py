"""
WorldCover non-built-up land penalty corrector
"""
import numpy as np

from .base import BaseCorrector, CorrectorResult
from .registry import corrector_registry


@corrector_registry.register("wc", description="WorldCover non-built-up land penalty")
class WcCorrector(BaseCorrector):
    """
    WC correction: base_demand × (1 - wc_others_ratio), renormalized per-ITL3.

    scores = wc_others_ratio (0~1)
    factor = 1 - scores
    """

    def correct(self, grid_gdf, region_sub, base_demand_col, scores, corrected_col):
        wc_factor = 1 - scores

        grid_gdf[corrected_col] = 0.0
        region_info = region_sub.set_index('ITL3')
        metadata = {}

        for itl3, group in grid_gdf.groupby('ITL3'):
            if itl3 not in region_info.index:
                continue

            total_demand = region_info.loc[itl3, 'Demand (MVA)']
            idx = group.index

            base = grid_gdf.loc[idx, base_demand_col].values
            raw = base * wc_factor[idx]
            raw_sum = raw.sum()

            if raw_sum > 0:
                grid_gdf.loc[idx, corrected_col] = total_demand * raw / raw_sum
            else:
                grid_gdf.loc[idx, corrected_col] = total_demand / len(group)

            metadata[itl3] = {
                'factor_mean': float(np.mean(wc_factor[idx])),
                'factor_range': [float(wc_factor[idx].min()),
                                 float(wc_factor[idx].max())],
            }

        return CorrectorResult(corrected_col=corrected_col, metadata=metadata)
