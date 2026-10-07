import warnings

import torch


def sinusoidal_pe(coords: torch.Tensor, L: int = 16, scale_factor: float = 100000.0) -> torch.Tensor:
    """
    正弦余弦位置编码：将投影坐标编码为位置向量。

    支持两种坐标输入：
    1. 原始投影坐标（EPSG:27700/3857，单位米）→ 使用指定 scale_factor
    2. 预归一化坐标（StandardScaler 等，范围约 [-10, 10]）→ 自动检测，scale_factor 设为 1.0

    公式:
        ω_l = 1 / 10000^(2l / d_pe),  l = 1, ..., L
        PE(x, y) = [sin(ω_1·x), cos(ω_1·x), sin(ω_1·y), cos(ω_1·y), ...,
                    sin(ω_L·x), cos(ω_L·x), sin(ω_L·y), cos(ω_L·y)]
        输出维度 d_pe = 4L

    Args:
        coords: (N, 2) 坐标张量
        L: 频率数量，输出维度 = 4L（默认 16 → 64 维）
        scale_factor: 坐标归一化因子，默认 100000（100 km）

    Returns:
        (N, 4L) 位置编码张量，值域 [-1, 1]
    """
    # 检测坐标范围，区分：经纬度 / 预归一化 / 投影坐标
    coord_max = coords.abs().max().item()
    if coord_max <= 180.0:
        # 进一步区分：经纬度 vs 预归一化坐标
        coord_std = coords.std().item()
        if coord_std < 5.0 and coord_max < 10.0:
            # 预归一化坐标（StandardScaler 输出，std≈1，max≈几个标准差）
            warnings.warn(
                f"检测到预归一化坐标（max={coord_max:.2f}, std={coord_std:.2f}），"
                f"自动将 scale_factor 从 {scale_factor} 调整为 1.0。",
                stacklevel=2,
            )
            scale_factor = 1.0
        else:
            # 真正的经纬度坐标
            warnings.warn(
                f"输入坐标范围在 [-180, 180] 内（max={coord_max:.2f}），"
                f"疑似经纬度坐标。建议使用投影坐标系（EPSG:27700，单位米）。"
                f"当前使用 scale_factor=1.0 继续执行。",
                stacklevel=2,
            )
            scale_factor = 1.0

    device = coords.device

    # 坐标归一化
    xy = coords / scale_factor  # (N, 2)
    x = xy[:, 0]  # (N,)
    y = xy[:, 1]  # (N,)

    # 计算频率 ω_l = 1 / 10000^(2l / d_pe), l=1,...,L
    # d_pe = 4L
    l_idx = torch.arange(1, L + 1, dtype=torch.float32, device=device)  # (L,)
    omega = 1.0 / (10000.0 ** (2.0 * l_idx / (4.0 * L)))  # (L,)

    # (N, L)
    x_angles = x.unsqueeze(1) * omega.unsqueeze(0)
    y_angles = y.unsqueeze(1) * omega.unsqueeze(0)

    # 拼接 [sin(ω·x), cos(ω·x), sin(ω·y), cos(ω·y)]  → (N, 4L)
    pe = torch.cat([
        torch.sin(x_angles),
        torch.cos(x_angles),
        torch.sin(y_angles),
        torch.cos(y_angles),
    ], dim=1)

    return pe


