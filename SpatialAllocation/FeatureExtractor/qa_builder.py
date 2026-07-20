"""
QaBuilder - q_a vector construction + soft mapping matrix

Builds the q_a (N, K) vector from the WorldCover aggregated ratios + OSM
landuse split, along with the landuse_mapping_matrix and landuse_ratio used
for the GNN loss computation.

Configuration-driven: all sector definitions, mapping rules, and thresholds
are read from JSON.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .extractors.base import ExtractorResult

logger = logging.getLogger(__name__)


class QaBuilder:
    """
    q_a vector builder.

    Config fields (config["qa_builder"]):
        buildup_threshold: built_up ratio threshold, above which an R/C/I split is performed
        buildup_split_source: split data source ("landuse")
        buildup_split_categories: category list used to split built_up
        buildup_default_split: default split ratios (used when no landuse data is available)
        sector_names: output sector name list, determines the q_a column order
        sector_mapping: mapping rules for each sector
    """

    def __init__(self, config: dict):
        """
        Args:
            config: the qa_builder configuration block
        """
        self.config = config
        self.buildup_threshold = config.get("buildup_threshold", 0.1)
        self.buildup_split_source = config.get("buildup_split_source", "landuse")
        self.buildup_split_categories = config.get(
            "buildup_split_categories", ["residential", "commercial", "industrial"]
        )
        self.buildup_default_split = config.get(
            "buildup_default_split", [0.333, 0.333, 0.334]
        )
        self.sector_names = config.get(
            "sector_names",
            ["residential", "commercial", "industrial", "agricultural", "others"],
        )
        self.sector_mapping = config.get("sector_mapping", {})

    def build(
        self,
        worldcover_result: ExtractorResult,
        landuse_result: Optional[ExtractorResult],
        grid_gdf: Any,
        step_size_m: float,
    ) -> np.ndarray:
        """
        Builds the q_a (N, K) vector.

        Algorithm:
        1. Get the aggregated group ratios from worldcover_result.numerical_columns
           (e.g. wc_built_up_ratio, wc_agricultural_ratio, wc_others_ratio)
        2. For each sector, assemble according to the sector_mapping rules:
           - "from" specifies the source group (e.g. "built_up")
           - "split_by" specifies the sub-split basis (e.g. "residential")
           - "absorb_residual" indicates that this sector absorbs the unallocated residual
        3. For points where built_up > threshold, split into R/C/I using the landuse ratios
        4. Row-normalize to ensure the values sum to 1

        Args:
            worldcover_result: the output of WorldCoverExtractor
            landuse_result: the output of LanduseExtractor (optional, used for the built_up split)
            grid_gdf: the grid point GeoDataFrame
            step_size_m: the grid step size

        Returns:
            q_a: (N, K) ndarray, K = len(sector_names)
        """
        n_points = len(grid_gdf)
        k = len(self.sector_names)
        q_a = np.zeros((n_points, k), dtype=np.float64)

        # Get the aggregated worldcover group ratios
        wc_cols = worldcover_result.numerical_columns

        # Get the landuse ratios (if available)
        lu_cols = landuse_result.numerical_columns if landuse_result else {}

        for s, sector in enumerate(self.sector_names):
            mapping = self.sector_mapping.get(sector, {})
            from_group = mapping.get("from")
            split_by = mapping.get("split_by")
            absorb_residual = mapping.get("absorb_residual", False)

            if from_group is None:
                continue

            # Get the source group ratio
            wc_key = f"wc_{from_group}_ratio"
            group_ratio = wc_cols.get(wc_key)
            if group_ratio is None:
                logger.warning(f"Sector '{sector}': WorldCover group '{wc_key}' not found")
                continue

            if split_by is not None:
                # Needs to be split using landuse data
                q_a[:, s] = self._split_by_landuse(
                    group_ratio, split_by, lu_cols, n_points
                )
            else:
                # Use the group ratio directly
                q_a[:, s] = group_ratio

        # Row normalization
        row_sums = q_a.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        q_a = q_a / row_sums

        return q_a

    def _split_by_landuse(
        self,
        group_ratio: np.ndarray,
        split_category: str,
        lu_cols: Dict[str, np.ndarray],
        n_points: int,
    ) -> np.ndarray:
        """
        Splits a WorldCover group using the landuse ratios.

        For points where group_ratio > threshold:
        - Get the landuse ratios for each subcategory
        - Compute the share of split_category among all subcategories
        - Multiply by group_ratio to get this sector's value

        For points where group_ratio <= threshold:
        - Use the corresponding default ratio from buildup_default_split
        """
        result = np.zeros(n_points, dtype=np.float64)

        # Collect the ratios for all split_categories
        split_cats = self.buildup_split_categories
        default_split = self.buildup_default_split

        # Find the index of the current split_category in the list
        if split_category not in split_cats:
            logger.warning(f"Split category '{split_category}' is not in buildup_split_categories")
            return result

        cat_idx = split_cats.index(split_category)

        # Get the landuse ratio for each subcategory
        cat_proportions = []
        for cat in split_cats:
            lu_key = f"lu_{cat}_prop"
            if lu_key in lu_cols:
                cat_proportions.append(lu_cols[lu_key])
            else:
                cat_proportions.append(np.zeros(n_points))

        cat_proportions = np.stack(cat_proportions, axis=1)  # (N, len(split_cats))

        # Compute per point
        for i in range(n_points):
            if group_ratio[i] <= self.buildup_threshold:
                # Use the default split
                result[i] = group_ratio[i] * default_split[cat_idx]
            else:
                # Split using the landuse ratios
                total_lu = cat_proportions[i].sum()
                if total_lu > 0:
                    split_fraction = cat_proportions[i, cat_idx] / total_lu
                else:
                    split_fraction = default_split[cat_idx]
                result[i] = group_ratio[i] * split_fraction

        return result

    def build_soft_mapping_matrix(
        self,
        qa_matrix: np.ndarray,
        gdf_s: Any,
        s_to_a_edges: Any,
        gdf_a: Any,
    ) -> Dict[str, Any]:
        """
        Builds the soft mapping matrix and target ratios needed for the GNN loss function.

        Args:
            qa_matrix: (N_a, K) q_a vector
            gdf_s: the source region GeoDataFrame (must contain sector ratio columns)
            s_to_a_edges: (2, E) edge index, [0]=source, [1]=agent
            gdf_a: the agent GeoDataFrame

        Returns:
            Dict:
            - landuse_mapping_matrix: (E, N_s x K) soft mapping matrix
            - landuse_ratio: (N_s, K) source region target ratios
            - sector_names: the sector name list
        """
        num_s = len(gdf_s)
        k = len(self.sector_names)

        if isinstance(s_to_a_edges, torch.Tensor):
            edge_array = s_to_a_edges.numpy()
        else:
            edge_array = np.asarray(s_to_a_edges)

        num_edges = edge_array.shape[1]
        source_indices = edge_array[0]
        agent_indices = edge_array[1]

        # === Build landuse_mapping_matrix (E, N_s x K) ===
        # Each edge's q_a vector serves as the soft weight
        mapping_matrix = np.zeros((num_edges, num_s * k), dtype=np.float64)

        for e in range(num_edges):
            s_idx = source_indices[e]
            a_idx = agent_indices[e]

            if a_idx < len(qa_matrix):
                qa_vec = qa_matrix[a_idx]  # (K,)
                for j in range(k):
                    col_idx = s_idx * k + j
                    mapping_matrix[e, col_idx] = qa_vec[j]

        # === Build landuse_ratio (N_s, K) ===
        # Read from the percent columns of gdf_s, or aggregate from q_a
        landuse_ratio = np.zeros((num_s, k), dtype=np.float64)

        # Try to read the existing percent columns from gdf_s
        percent_suffix = "_percent"
        has_percent_cols = all(
            f"{name}{percent_suffix}" in gdf_s.columns
            for name in self.sector_names
        )

        if has_percent_cols:
            for j, name in enumerate(self.sector_names):
                landuse_ratio[:, j] = gdf_s[f"{name}{percent_suffix}"].values
        else:
            # Aggregate from q_a by source
            for e in range(num_edges):
                s_idx = source_indices[e]
                a_idx = agent_indices[e]
                if a_idx < len(qa_matrix):
                    landuse_ratio[s_idx] += qa_matrix[a_idx]

        # Row normalization
        row_sums = landuse_ratio.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        landuse_ratio = landuse_ratio / row_sums

        return {
            "landuse_mapping_matrix": torch.tensor(mapping_matrix, dtype=torch.float32),
            "landuse_ratio": torch.tensor(landuse_ratio, dtype=torch.float32),
            "sector_names": self.sector_names,
        }
