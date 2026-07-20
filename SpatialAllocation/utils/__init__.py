from .color import assign_color
from .ImageFetcher import (
    fetch_satellite_image,
    batch_fetch_images,
    generate_image_index,
    download_region_mosaic,
    extract_patch_from_mosaic,
    extract_patch_from_open_dataset,
)
from .VisualFeatureExtractor import (
    FeatureExtractorConfig,
    SatelliteImageDataset,
    load_feature_extractor,
    extract_features,
    validate_features,
    compute_semantic_similarity,
    apply_pca_reduction
)
from .NetworkDistance import (
    NetworkDistanceConfig,
    download_road_network,
    map_points_to_network,
    compute_sparse_distance_matrix,
    save_distance_results,
    load_distance_results,
    validate_connectivity,
    validate_distance_logic,
    compute_and_save_distances,
    get_network_connectivity_stats
)
from .SpectralIndexCalculator import (
    validate_mosaic,
    compute_spectral_indices,
    aggregate_spectral_patch,
    extract_spectral_features_from_mosaic,
    batch_extract_from_mosaic,
)