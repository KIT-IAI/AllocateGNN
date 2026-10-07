# -*- coding: utf-8 -*-
"""解耦对照场与当前 planning 诊断的完整构造合同。

底层数值路径来自 ``allocategnn-vibe`` 的 UK ``run_criterion_v2``：
``PERM-R``/``PERM-3R`` 与 SMOOTH 共用冻结主随机流，后补的
``PERM-R/3`` 使用独立 ``seed + int(radius_km)`` 随机流，避免新增臂改变
下一半径的既有随机数消费。坐标一律为平面 km（ECEF 前两维亦可）。

    PERM    块内随机置换 —— 毁逐点排序、保块级总量
    SMOOTH  低频对数正态乘子 —— 改变局部总量，也能改变逐点排序
"""
from __future__ import annotations

import numpy as np


DEFAULT_SMOOTH_STRENGTHS = (0.5, 1.0, 2.0)
DEFAULT_WAVELENGTH_FACTOR = 5.0


def _as_field(w: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(w, dtype=float)
    points = np.asarray(xy, dtype=float)
    if field.ndim != 1 or points.shape != (len(field), 2):
        raise ValueError("w must be (n,) and xy must be (n, 2)")
    if len(field) == 0 or not np.isfinite(field).all() or (field < 0).any():
        raise ValueError("control field must be non-empty, finite and non-negative")
    if not np.isfinite(points).all():
        raise ValueError("control coordinates must be finite")
    if float(field.sum()) <= 0:
        raise ValueError("control field must have positive total mass")
    return field, points


def preserve_mass(values: np.ndarray, total: float) -> np.ndarray:
    """Scale a non-negative control field to one exact declared total."""

    field = np.asarray(values, dtype=float)
    current = float(field.sum())
    if not np.isfinite(field).all() or (field < 0).any() or current <= 0:
        raise ValueError("control field cannot be mass-normalised")
    if not np.isfinite(total) or total <= 0:
        raise ValueError("declared control-field mass must be positive and finite")
    return field * (float(total) / current)


def block_permute(w: np.ndarray, xy: np.ndarray, block_km: float,
                  rng: np.random.RandomState) -> np.ndarray:
    """块内随机置换:块内和精确不变,块内逐点分布被打乱。"""
    w, xy = _as_field(w, xy)
    if not np.isfinite(block_km) or block_km <= 0:
        raise ValueError("block_km must be positive and finite")
    bid = (np.floor(xy[:, 0] / block_km).astype(np.int64) * 1_000_003
           + np.floor(xy[:, 1] / block_km).astype(np.int64))
    out = w.copy()
    order = np.argsort(bid, kind="stable")
    bs = bid[order]
    starts = np.flatnonzero(np.r_[True, bs[1:] != bs[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(bs)]):
        idx = order[a:b]
        out[idx] = w[rng.permutation(idx)]
    return out


def smooth_multiplier(xy: np.ndarray, wavelength_km: float, strength: float,
                      rng: np.random.RandomState, n_modes: int = 4) -> np.ndarray:
    """低频平滑乘子:若干长波正弦叠加后取指数,保证正值且空间平滑。"""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("xy must be a finite (n, 2) array")
    if not np.isfinite(wavelength_km) or wavelength_km <= 0:
        raise ValueError("wavelength_km must be positive and finite")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and non-negative")
    if not isinstance(n_modes, int) or isinstance(n_modes, bool) or n_modes <= 0:
        raise ValueError("n_modes must be a positive integer")
    f = 2 * np.pi / wavelength_km
    z = np.zeros(len(xy))
    for _ in range(n_modes):
        th = rng.uniform(0, 2 * np.pi)
        ph = rng.uniform(0, 2 * np.pi)
        z += np.sin(f * (xy[:, 0] * np.cos(th) + xy[:, 1] * np.sin(th)) + ph)
    z /= np.sqrt(n_modes)
    return np.exp(strength * z)


def build_control_arms(
    w: np.ndarray,
    xy: np.ndarray,
    radius_km: float,
    *,
    seed: int = 42,
    smooth_strengths: tuple[float, ...] = DEFAULT_SMOOTH_STRENGTHS,
    wavelength_factor: float = DEFAULT_WAVELENGTH_FACTOR,
    n_modes: int = 4,
) -> dict[str, np.ndarray]:
    """Build the complete source-compatible PERM/SMOOTH diagnostic family.

    All returned fields preserve the input total.  They are diagnostic controls,
    never L11 candidates and never inputs to the facility candidate pool.
    """

    field, points = _as_field(w, xy)
    if not np.isfinite(radius_km) or radius_km <= 0:
        raise ValueError("radius_km must be positive and finite")
    if not np.isfinite(wavelength_factor) or wavelength_factor <= 0:
        raise ValueError("wavelength_factor must be positive and finite")
    total = float(field.sum())
    rng = np.random.RandomState(int(seed))
    arms: dict[str, np.ndarray] = {}
    for name, block in (("PERM-R", radius_km), ("PERM-3R", 3.0 * radius_km)):
        arms[name] = preserve_mass(block_permute(field, points, block, rng), total)
    for strength in smooth_strengths:
        multiplier = smooth_multiplier(
            points,
            wavelength_factor * radius_km,
            float(strength),
            rng,
            n_modes=n_modes,
        )
        arms[f"SMOOTH-{float(strength):g}"] = preserve_mass(field * multiplier, total)
    extra_rng = np.random.RandomState(int(seed) + int(radius_km))
    arms["PERM-R/3"] = preserve_mass(
        block_permute(field, points, radius_km / 3.0, extra_rng), total
    )
    return arms
