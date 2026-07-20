"""
Visual feature extraction module

Uses pretrained CNN models to extract visual feature vectors from
satellite imagery.
Supports the ResNet-50 and EfficientNet-B0 models.

Author: AllocateGNN Team
Date: 2026-01-22
"""

import logging
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration class
# =============================================================================

@dataclass
class FeatureExtractorConfig:
    """
    Visual feature extractor configuration.

    Attributes:
        model_name: Pretrained model name ('resnet50' or 'efficientnet_b0')
        batch_size: Batch size
        num_workers: Number of DataLoader worker processes (defaults to 0 on Windows)
        image_size: Input image size (default 224)
        device: Compute device (None means auto-select)
        enable_pca: Whether to enable PCA dimensionality reduction
        pca_variance_ratio: PCA retained-variance ratio (default 0.95)
    """
    model_name: str = 'resnet50'
    batch_size: int = 32
    num_workers: int = field(default_factory=lambda: _get_default_num_workers())
    image_size: int = 224
    device: Optional[str] = None
    enable_pca: bool = False
    pca_variance_ratio: float = 0.95


def _get_default_num_workers() -> int:
    """
    Return the default number of worker processes based on the platform.

    On Windows, uses num_workers=0 to avoid multiprocessing issues.
    """
    import os
    if platform.system() == 'Windows':
        return 0
    else:
        return min(4, os.cpu_count() or 1)


def _get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Get the compute device, with support for automatic selection.

    Args:
        preferred: Preferred device; None means auto-select

    Returns:
        torch.device: The compute device
    """
    if preferred is not None:
        return torch.device(preferred)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("GPU unavailable; using CPU (may be slower)")

    return device


# =============================================================================
# Model loading
# =============================================================================

def load_feature_extractor(
    model_name: str = 'resnet50',
    device: Optional[str] = None
) -> Tuple[nn.Module, int]:
    """
    Load a pretrained feature extraction model.

    Supported models:
        - 'resnet50': ResNet-50, outputs 2048-dimensional features
        - 'efficientnet_b0': EfficientNet-B0, outputs 1280-dimensional features

    Args:
        model_name: Model name
        device: Target device; None means auto-select

    Returns:
        Tuple[nn.Module, int]: (model instance, feature dimension)

    Raises:
        ValueError: Unsupported model name
    """
    from torchvision import models

    device = _get_device(device)

    # Load the pretrained model based on the model name
    if model_name == 'resnet50':
        feature_dim = 2048
        try:
            # New API in torchvision >= 0.13
            from torchvision.models import ResNet50_Weights
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            logger.info("Using ResNet50_Weights.IMAGENET1K_V2 weights")
        except ImportError:
            # Fall back to the legacy API
            model = models.resnet50(pretrained=True)
            logger.info("Loaded ResNet-50 pretrained weights via the legacy API")

        # Remove the final fully connected classification layer
        model.fc = nn.Identity()

    elif model_name == 'efficientnet_b0':
        feature_dim = 1280
        try:
            # New API in torchvision >= 0.13
            from torchvision.models import EfficientNet_B0_Weights
            model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            logger.info("Using EfficientNet_B0_Weights.IMAGENET1K_V1 weights")
        except ImportError:
            # Fall back to the legacy API
            model = models.efficientnet_b0(pretrained=True)
            logger.info("Loaded EfficientNet-B0 pretrained weights via the legacy API")

        # Remove the classifier
        model.classifier = nn.Identity()

    else:
        raise ValueError(
            f"Unsupported model name: {model_name}. "
            f"Supported models: 'resnet50', 'efficientnet_b0'"
        )

    # Set to evaluation mode
    model.eval()

    # Move to the target device
    model = model.to(device)

    logger.info(f"Successfully loaded {model_name}, feature dimension: {feature_dim}")

    return model, feature_dim


# =============================================================================
# Dataset class
# =============================================================================

class SatelliteImageDataset(Dataset):
    """
    Satellite imagery dataset.

    Loads PNG-format satellite images from the specified directory and
    applies standard ImageNet preprocessing.

    Args:
        image_dir: Image directory path
        index_csv_path: Index CSV file path (optional; scans automatically if None)
        image_size: Output image size (default 224)

    Attributes:
        grid_ids: List of grid_id values corresponding to the images, order preserved
        image_paths: List of image file paths
    """

    # ImageNet normalization parameters
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        image_dir: str,
        index_csv_path: Optional[str] = None,
        image_size: int = 224
    ) -> None:
        """
        Initialize the dataset.

        Args:
            image_dir: Image directory path
            index_csv_path: Index CSV file path (optional)
            image_size: Output image size
        """
        self.image_dir = Path(image_dir)
        self.image_size = image_size

        # Build the preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.IMAGENET_MEAN,
                std=self.IMAGENET_STD
            )
        ])

        # Load the image list
        self.grid_ids: List[str] = []
        self.image_paths: List[Path] = []
        self._load_image_list(index_csv_path)

        logger.info(f"Dataset initialization complete, {len(self)} images total")

    def _load_image_list(self, index_csv_path: Optional[str]) -> None:
        """
        Load the image list.

        Prefers the index CSV file if available, otherwise scans the directory.
        """
        # Try to use the index file
        if index_csv_path is None:
            # By default, look for image_index.csv under image_dir
            default_index = self.image_dir / 'image_index.csv'
            if default_index.exists():
                index_csv_path = str(default_index)
                logger.info(f"Found the default index file: {index_csv_path}")

        if index_csv_path and Path(index_csv_path).exists():
            # Use the index file
            logger.info(f"Loading from the index file: {index_csv_path}")
            index_df = pd.read_csv(index_csv_path)

            if 'grid_id' not in index_df.columns:
                raise ValueError("The index file must contain a 'grid_id' column")

            for _, row in index_df.iterrows():
                grid_id = str(row['grid_id'])

                # Build the image path
                if 'image_file_path' in index_df.columns and pd.notna(row['image_file_path']):
                    # Use the path from the index
                    img_path = self.image_dir / row['image_file_path']
                else:
                    # Default naming convention
                    img_path = self.image_dir / f"{grid_id}.png"

                if img_path.exists():
                    self.grid_ids.append(grid_id)
                    self.image_paths.append(img_path)
                else:
                    logger.warning(f"Image file does not exist: {img_path}")
        else:
            # Fallback: scan all PNG files in the directory
            logger.info(f"Scanning directory: {self.image_dir}")
            png_files = sorted(self.image_dir.glob('*.png'))

            for img_path in png_files:
                # Extract grid_id from the file name (strip the extension)
                grid_id = img_path.stem
                self.grid_ids.append(grid_id)
                self.image_paths.append(img_path)

            if not self.grid_ids:
                logger.warning(f"No PNG files found in the directory: {self.image_dir}")

    def __len__(self) -> int:
        """Return the dataset size."""
        return len(self.grid_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple[torch.Tensor, str]: (preprocessed image tensor, grid_id)
        """
        grid_id = self.grid_ids[idx]
        img_path = self.image_paths[idx]

        try:
            # Load and transform the image
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor, grid_id

        except Exception as e:
            logger.warning(f"Failed to load image {grid_id} ({img_path}): {e}")
            # Return an all-zero placeholder
            placeholder = torch.zeros(3, self.image_size, self.image_size)
            return placeholder, grid_id


# =============================================================================
# Numerical validation
# =============================================================================

def validate_features(
    features: np.ndarray,
    grid_ids: List[str]
) -> Dict[str, Any]:
    """
    Check the numerical stability of the feature matrix.

    Checks performed:
        - NaN value detection
        - Inf value detection
        - All-zero vector detection

    Args:
        features: Feature matrix of shape [N, D]
        grid_ids: Corresponding list of grid_id values

    Returns:
        Dict[str, Any]: Validation result, containing:
            - 'is_valid': Whether all checks passed
            - 'nan_indices': List of sample indices containing NaN
            - 'inf_indices': List of sample indices containing Inf
            - 'zero_indices': List of sample indices that are all-zero vectors
            - 'problematic_grid_ids': List of grid_ids with problems
    """
    result = {
        'is_valid': True,
        'nan_indices': [],
        'inf_indices': [],
        'zero_indices': [],
        'problematic_grid_ids': []
    }

    # Check for NaN
    nan_mask = np.any(np.isnan(features), axis=1)
    nan_indices = np.where(nan_mask)[0].tolist()
    if nan_indices:
        result['nan_indices'] = nan_indices
        result['is_valid'] = False
        logger.warning(f"Found {len(nan_indices)} samples containing NaN values")

    # Check for Inf
    inf_mask = np.any(np.isinf(features), axis=1)
    inf_indices = np.where(inf_mask)[0].tolist()
    if inf_indices:
        result['inf_indices'] = inf_indices
        result['is_valid'] = False
        logger.warning(f"Found {len(inf_indices)} samples containing Inf values")

    # Check for all-zero vectors
    zero_mask = np.all(features == 0, axis=1)
    zero_indices = np.where(zero_mask)[0].tolist()
    if zero_indices:
        result['zero_indices'] = zero_indices
        result['is_valid'] = False
        logger.warning(f"Found {len(zero_indices)} all-zero vectors")

    # Collect the problematic grid_ids
    problematic_indices = set(nan_indices + inf_indices + zero_indices)
    result['problematic_grid_ids'] = [grid_ids[i] for i in problematic_indices]

    if result['is_valid']:
        logger.info("Numerical stability check passed")

    return result


# =============================================================================
# Semantic consistency validation
# =============================================================================

def compute_semantic_similarity(
    features: np.ndarray,
    grid_ids: List[str],
    similar_pairs: List[Tuple[str, str]],
    different_pairs: List[Tuple[str, str]]
) -> Dict[str, Any]:
    """
    Compute the cosine similarity between specified sample pairs.

    Used to validate the semantic consistency of feature extraction.

    Args:
        features: Feature matrix of shape [N, D]
        grid_ids: Corresponding list of grid_id values
        similar_pairs: List of grid_id pairs expected to be similar
        different_pairs: List of grid_id pairs expected to be different

    Returns:
        Dict[str, Any]: Similarity statistics, containing:
            - 'similar_mean': Average similarity of the similar pairs
            - 'different_mean': Average similarity of the different pairs
            - 'similar_scores': Individual similarity of each pair
            - 'different_scores': Individual similarity of each pair
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Create a mapping from grid_id to index
    grid_id_to_idx = {gid: i for i, gid in enumerate(grid_ids)}

    def compute_pair_similarity(pair: Tuple[str, str]) -> Optional[float]:
        """Compute the cosine similarity of a pair of samples."""
        gid1, gid2 = pair
        if gid1 not in grid_id_to_idx or gid2 not in grid_id_to_idx:
            logger.warning(f"grid_id not found: {gid1} or {gid2}")
            return None

        idx1 = grid_id_to_idx[gid1]
        idx2 = grid_id_to_idx[gid2]

        vec1 = features[idx1:idx1+1]
        vec2 = features[idx2:idx2+1]

        sim = cosine_similarity(vec1, vec2)[0, 0]
        return float(sim)

    # Compute the similarity for the similar pairs
    similar_scores = []
    for pair in similar_pairs:
        sim = compute_pair_similarity(pair)
        if sim is not None:
            similar_scores.append(sim)

    # Compute the similarity for the different pairs
    different_scores = []
    for pair in different_pairs:
        sim = compute_pair_similarity(pair)
        if sim is not None:
            different_scores.append(sim)

    result = {
        'similar_mean': np.mean(similar_scores) if similar_scores else None,
        'different_mean': np.mean(different_scores) if different_scores else None,
        'similar_scores': similar_scores,
        'different_scores': different_scores
    }

    logger.info(f"Average similarity of similar pairs: {result['similar_mean']:.4f}" if result['similar_mean'] else "No similar-pair data")
    logger.info(f"Average similarity of different pairs: {result['different_mean']:.4f}" if result['different_mean'] else "No different-pair data")

    return result


# =============================================================================
# PCA dimensionality reduction
# =============================================================================

def apply_pca_reduction(
    features: np.ndarray,
    variance_ratio: float = 0.95,
    output_path: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply PCA dimensionality reduction to the features.

    Args:
        features: Original feature matrix [N, D]
        variance_ratio: Target retained-variance ratio
        output_path: Save path for the reduced features (optional)

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - The reduced feature matrix [N, D']
            - PCA analysis result, containing:
                - 'original_dim': Original dimension
                - 'reduced_dim': Reduced dimension
                - 'explained_variance_ratio': Cumulative explained variance
    """
    from sklearn.decomposition import PCA

    original_dim = features.shape[1]
    logger.info(f"Starting PCA dimensionality reduction, original dimension: {original_dim}, target variance retention: {variance_ratio}")

    # Fit with PCA
    pca = PCA(n_components=variance_ratio, svd_solver='full')
    reduced_features = pca.fit_transform(features)

    reduced_dim = reduced_features.shape[1]
    cumulative_variance = np.sum(pca.explained_variance_ratio_)

    logger.info(f"Dimensionality reduction complete: {original_dim} -> {reduced_dim} dims")
    logger.info(f"Cumulative explained variance: {cumulative_variance:.4f}")

    # Optionally save
    if output_path:
        np.save(output_path, reduced_features)
        logger.info(f"Reduced features saved to: {output_path}")

    result = {
        'original_dim': original_dim,
        'reduced_dim': reduced_dim,
        'explained_variance_ratio': cumulative_variance,
        'variance_per_component': pca.explained_variance_ratio_.tolist()
    }

    return reduced_features, result


# =============================================================================
# Batch feature extraction
# =============================================================================

def extract_features(
    config: FeatureExtractorConfig,
    image_dir: str,
    output_path: str,
    index_csv_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Batch-extract visual features and save them.

    Args:
        config: Feature extractor configuration
        image_dir: Image directory path
        output_path: Output .npy file path
        index_csv_path: Index CSV file path (optional)

    Returns:
        Dict[str, Any]: Extraction result, containing:
            - 'num_images': Number of images processed
            - 'feature_dim': Feature dimension
            - 'embeddings_path': Embeddings file path
            - 'grid_ids_path': grid_id order file path
            - 'warnings': List of warning messages

    Side effects:
        - Saves visual_embeddings.npy to output_path
        - Saves grid_id_order.npy to the same directory
    """
    warnings_list: List[str] = []

    # Get the device
    device = _get_device(config.device)

    # Load the model
    logger.info(f"Loading model: {config.model_name}")
    model, feature_dim = load_feature_extractor(config.model_name, str(device))

    # Create the dataset
    logger.info(f"Creating the dataset: {image_dir}")
    dataset = SatelliteImageDataset(
        image_dir=image_dir,
        index_csv_path=index_csv_path,
        image_size=config.image_size
    )

    if len(dataset) == 0:
        raise ValueError(f"The dataset is empty; please check the directory: {image_dir}")

    # Create the data loader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,  # Preserve order
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Batch-extract the features
    logger.info(f"Starting feature extraction, {len(dataset)} images total")
    all_features: List[np.ndarray] = []
    all_grid_ids: List[str] = []

    with torch.no_grad():
        for batch_idx, (batch_images, batch_ids) in enumerate(dataloader):
            # Move to the device
            batch_images = batch_images.to(device)

            # Forward pass
            features = model(batch_images)

            # Immediately move back to CPU to free GPU memory
            features = features.cpu().numpy()

            all_features.append(features)
            all_grid_ids.extend(batch_ids)

            # Explicitly free GPU memory
            del batch_images

            # Progress display
            progress = (batch_idx + 1) * config.batch_size
            if progress % (len(dataset) // 10 + 1) < config.batch_size:
                logger.info(f"Progress: {min(progress, len(dataset))}/{len(dataset)}")

    # Concatenate all the features
    features_matrix = np.vstack(all_features)
    logger.info(f"Feature matrix shape: {features_matrix.shape}")

    # Numerical stability check
    validation_result = validate_features(features_matrix, all_grid_ids)
    if not validation_result['is_valid']:
        warning_msg = f"Numerical stability check failed, problematic samples: {validation_result['problematic_grid_ids']}"
        warnings_list.append(warning_msg)

    # Optional PCA dimensionality reduction
    if config.enable_pca:
        features_matrix, pca_result = apply_pca_reduction(
            features_matrix,
            variance_ratio=config.pca_variance_ratio
        )
        feature_dim = pca_result['reduced_dim']
        logger.info(f"Dimension after PCA reduction: {feature_dim}")

    # Ensure the output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the feature matrix
    np.save(output_path, features_matrix)
    logger.info(f"Feature matrix saved to: {output_path}")

    # Save the grid_id order
    grid_ids_path = output_path.parent / 'grid_id_order.npy'
    np.save(grid_ids_path, np.array(all_grid_ids))
    logger.info(f"Grid ID order saved to: {grid_ids_path}")

    # Build the return result
    result = {
        'num_images': len(all_grid_ids),
        'feature_dim': feature_dim,
        'embeddings_path': str(output_path),
        'grid_ids_path': str(grid_ids_path),
        'warnings': warnings_list,
        'validation': validation_result
    }

    return result
