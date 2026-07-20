"""
NTL nighttime lights log-ratio corrector
"""
import numpy as np

from .base import BaseCorrector, CorrectorResult
from .registry import corrector_registry


@corrector_registry.register("ntl", description="NTL nighttime lights log-ratio correction")
class NtlCorrector(BaseCorrector):
    """
    NTL correction: base_demand × ntl_factor, renormalized per-ITL3.

    ntl_factor = log(1 + ntl + eps) / log(1 + ntl_median_rci)
    eps = percentile_p(ntl[RCI and > 0], per ITL3)
    median is computed only over RCI grid points

    Config
    ------
    rci_threshold : float, default 0.5
    epsilon_percentile : int, default 5
    """

    def correct(self, grid_gdf, region_sub, base_demand_col, scores, corrected_col):
        rci_threshold = self.config.get('rci_threshold', 0.5)
        epsilon_pct = self.config.get('epsilon_percentile', 5)

        # RCI mask
        rci_sum = (grid_gdf['lu_residential_prop'].values
                   + grid_gdf['lu_commercial_prop'].values
                   + grid_gdf['lu_industrial_prop'].values)
        rci_mask = rci_sum > rci_threshold

        grid_gdf[corrected_col] = 0.0
        region_info = region_sub.set_index('ITL3')
        metadata = {}

        for itl3, group in grid_gdf.groupby('ITL3'):
            if itl3 not in region_info.index:
                continue

            total_demand = region_info.loc[itl3, 'Demand (MVA)']
            idx = group.index
            ntl_group = scores[idx]
            rci_group = rci_mask[idx]

            # eps: percentile over RCI and nonzero NTL values
            rci_nonzero_ntl = ntl_group[rci_group & (ntl_group > 0)]
            if len(rci_nonzero_ntl) > 0:
                epsilon = np.percentile(rci_nonzero_ntl, epsilon_pct)
            else:
                nonzero_ntl = ntl_group[ntl_group > 0]
                epsilon = (np.percentile(nonzero_ntl, epsilon_pct)
                           if len(nonzero_ntl) > 0 else 0.1)

            # median: computed only over RCI grid points
            rci_ntl = ntl_group[rci_group]
            if len(rci_ntl) > 0:
                ntl_median = np.median(rci_ntl)
            else:
                ntl_median = np.median(ntl_group)
            if ntl_median <= 0:
                ntl_median = epsilon

            # ntl_factor = log(1 + ntl + eps) / log(1 + ntl_median)
            ntl_factor = np.log(1 + ntl_group + epsilon) / np.log(1 + ntl_median)

            base = grid_gdf.loc[idx, base_demand_col].values
            raw = base * ntl_factor
            raw_sum = raw.sum()

            if raw_sum > 0:
                grid_gdf.loc[idx, corrected_col] = total_demand * raw / raw_sum
            else:
                grid_gdf.loc[idx, corrected_col] = total_demand / len(group)

            metadata[itl3] = {
                'epsilon': float(epsilon),
                'median_rci': float(ntl_median),
                'rci_count': int(rci_group.sum()),
                'total_count': len(group),
            }

        return CorrectorResult(corrected_col=corrected_col, metadata=metadata)
