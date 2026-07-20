"""
Substation proximity log-ratio corrector
"""
import numpy as np
from scipy.spatial.distance import cdist

from .base import BaseCorrector, CorrectorResult
from .registry import corrector_registry


@corrector_registry.register("proximity", description="Substation proximity log-ratio correction")
class ProximityCorrector(BaseCorrector):
    """
    Proximity correction: base_demand × prox_factor, renormalized per-ITL3.

    prox_factor = log(1 + proximity) / log(1 + prox_median_rci)
    median is computed only over RCI grid points.

    Config
    ------
    rci_threshold : float, default 0.5
    """

    def correct(self, grid_gdf, region_sub, base_demand_col, scores, corrected_col):
        rci_threshold = self.config.get('rci_threshold', 0.5)

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
            prox_group = scores[idx]
            rci_group = rci_mask[idx]

            # median: computed only over RCI grid points
            rci_prox = prox_group[rci_group]
            if len(rci_prox) > 0:
                prox_median = np.median(rci_prox)
            else:
                prox_median = np.median(prox_group)
            if prox_median <= 0:
                prox_median = 1e-6

            # prox_factor = log(1 + proximity) / log(1 + prox_median)
            prox_factor = np.log(1 + prox_group) / np.log(1 + prox_median)

            base = grid_gdf.loc[idx, base_demand_col].values
            raw = base * prox_factor
            raw_sum = raw.sum()

            if raw_sum > 0:
                grid_gdf.loc[idx, corrected_col] = total_demand * raw / raw_sum
            else:
                grid_gdf.loc[idx, corrected_col] = total_demand / len(group)

            metadata[itl3] = {
                'median_rci': float(prox_median),
                'factor_range': [float(prox_factor.min()),
                                 float(prox_factor.max())],
                'rci_count': int(rci_group.sum()),
                'total_count': len(group),
            }

        return CorrectorResult(corrected_col=corrected_col, metadata=metadata)

    @staticmethod
    def compute_scores(grid_gdf, subs_gdf, gamma=1.0,
                       target_crs='EPSG:27700', clamp_km=0.01):
        """
        Compute the substation proximity score for each grid point.

        proximity_q = Σ_{t∈T} d(q,t)^{-γ}

        Parameters
        ----------
        grid_gdf : grid GeoDataFrame
        subs_gdf : substation GeoDataFrame
        gamma : distance decay exponent
        target_crs : projected coordinate system (must be metric)
        clamp_km : minimum distance (km), prevents numerical overflow

        Returns
        -------
        proximity : (N,) ndarray
        """
        grid_proj = grid_gdf.to_crs(target_crs)
        subs_proj = subs_gdf.to_crs(target_crs)

        grid_coords = np.column_stack([grid_proj.geometry.x.values,
                                       grid_proj.geometry.y.values])
        subs_coords = np.column_stack([subs_proj.geometry.x.values,
                                       subs_proj.geometry.y.values])

        # Euclidean distance (N_grid × N_subs), converted to km
        dist_km = cdist(grid_coords, subs_coords, metric='euclidean') / 1000.0
        dist_km = np.maximum(dist_km, clamp_km)

        # Inverse-distance weighted sum
        proximity = np.sum(dist_km ** (-gamma), axis=1)
        return proximity
