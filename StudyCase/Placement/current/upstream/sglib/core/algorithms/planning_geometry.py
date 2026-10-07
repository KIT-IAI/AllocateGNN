# -*- coding: utf-8 -*-
"""
球面几何工具(无国别假设)
==========================

约定:经纬度一律 WGS84 度数;距离一律 km。
haversine 的单一居所——选址/定容/MILP 消费方一律从此导入,禁止二次实现。
"""
from __future__ import annotations

import numpy as np

EARTH_R_KM = 6371.0


def to_ecef_km(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """经纬度转球面直角坐标(km),供 KD-tree 做欧氏近似。"""
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ]) * EARTH_R_KM


def chord_km(radius_km: float) -> float:
    """大圆距离 → 弦长,保证 KD-tree 的半径语义与 haversine 一致。"""
    return 2 * EARTH_R_KM * np.sin(radius_km / (2 * EARTH_R_KM))


def haversine_distance_matrix(coords_a: np.ndarray,
                              coords_b: np.ndarray) -> np.ndarray:
    """
    向量化 haversine 距离矩阵。

    参数:
        coords_a: (N_a, 2) lon/lat 度数
        coords_b: (N_b, 2) lon/lat 度数

    返回:
        距离矩阵 (N_a, N_b),单位 km
    """
    lon_a = np.radians(coords_a[:, 0:1])
    lat_a = np.radians(coords_a[:, 1:2])
    lon_b = np.radians(coords_b[:, 0:1].T)
    lat_b = np.radians(coords_b[:, 1:2].T)

    dlat = lat_b - lat_a
    dlon = lon_b - lon_a
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    return EARTH_R_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def haversine_vector(lon1: np.ndarray, lat1: np.ndarray,
                     lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """逐元素 haversine 距离(km),输入为度数。"""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_R_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
