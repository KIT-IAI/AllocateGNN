import warnings

import torch


def sinusoidal_pe(coords: torch.Tensor, L: int = 16, scale_factor: float = 100000.0) -> torch.Tensor:
    """
    Sinusoidal positional encoding: encodes projected coordinates into a
    positional vector.

    Supports two kinds of coordinate input:
    1. Raw projected coordinates (EPSG:27700/3857, in meters) -> uses the
       given scale_factor
    2. Pre-normalized coordinates (e.g. from StandardScaler, range roughly
       [-10, 10]) -> auto-detected, scale_factor is set to 1.0

    Formula:
        omega_l = 1 / 10000^(2l / d_pe),  l = 1, ..., L
        PE(x, y) = [sin(omega_1*x), cos(omega_1*x), sin(omega_1*y), cos(omega_1*y), ...,
                    sin(omega_L*x), cos(omega_L*x), sin(omega_L*y), cos(omega_L*y)]
        Output dimension d_pe = 4L

    Args:
        coords: (N, 2) coordinate tensor
        L: number of frequencies, output dimension = 4L (default 16 -> 64 dims)
        scale_factor: coordinate normalization factor, default 100000 (100 km)

    Returns:
        (N, 4L) positional encoding tensor, value range [-1, 1]
    """
    # Detect the coordinate range to distinguish: lat/lon / pre-normalized / projected coordinates
    coord_max = coords.abs().max().item()
    if coord_max <= 180.0:
        # Further distinguish: lat/lon vs. pre-normalized coordinates
        coord_std = coords.std().item()
        if coord_std < 5.0 and coord_max < 10.0:
            # Pre-normalized coordinates (StandardScaler output, std ~1, max ~a few std devs)
            warnings.warn(
                f"Detected pre-normalized coordinates (max={coord_max:.2f}, std={coord_std:.2f}); "
                f"automatically adjusting scale_factor from {scale_factor} to 1.0.",
                stacklevel=2,
            )
            scale_factor = 1.0
        else:
            # Genuine lat/lon coordinates
            warnings.warn(
                f"Input coordinates fall within [-180, 180] (max={coord_max:.2f}), "
                f"suspected to be lat/lon coordinates. A projected CRS "
                f"(EPSG:27700, meters) is recommended. "
                f"Continuing with scale_factor=1.0 for now.",
                stacklevel=2,
            )
            scale_factor = 1.0

    device = coords.device

    # Normalize coordinates
    xy = coords / scale_factor  # (N, 2)
    x = xy[:, 0]  # (N,)
    y = xy[:, 1]  # (N,)

    # Compute frequencies omega_l = 1 / 10000^(2l / d_pe), l=1,...,L
    # d_pe = 4L
    l_idx = torch.arange(1, L + 1, dtype=torch.float32, device=device)  # (L,)
    omega = 1.0 / (10000.0 ** (2.0 * l_idx / (4.0 * L)))  # (L,)

    # (N, L)
    x_angles = x.unsqueeze(1) * omega.unsqueeze(0)
    y_angles = y.unsqueeze(1) * omega.unsqueeze(0)

    # Concatenate [sin(omega*x), cos(omega*x), sin(omega*y), cos(omega*y)] -> (N, 4L)
    pe = torch.cat([
        torch.sin(x_angles),
        torch.cos(x_angles),
        torch.sin(y_angles),
        torch.cos(y_angles),
    ], dim=1)

    return pe
