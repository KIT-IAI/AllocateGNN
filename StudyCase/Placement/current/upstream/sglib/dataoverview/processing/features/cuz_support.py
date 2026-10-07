"""构建并读写 OSM covered/unknown/zero 支撑数组。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "osm_cuz_schema_v2"
LANDUSE_CATEGORIES = (
    "residential",
    "commercial",
    "industrial",
    "agricultural",
    "others",
    "unknown",
)
OSM_LANDUSE_COLUMNS = tuple(f"lu_{name}_prop" for name in LANDUSE_CATEGORIES[:5])

_ARTIFACT_KEYS = {
    "schema_version",
    "landuse_categories",
    "osm_landuse",
    "built_fraction",
    "features",
    "covered_mask",
    "unknown_mask",
    "zero_mask",
    "covered_full_indices",
    "unknown_full_indices",
    "zero_full_indices",
    "source_keys",
    "source_key_order",
    "source_index",
    "source_built_covered",
    "source_built_unknown",
    "source_unknown_share",
    "source_capacity_positive",
}


class CuzSupportError(ValueError):
    """输入或 C/U/Z 数组不符合数值契约。"""


def validate_cuz_grid_crs(
    grid_crs: str,
    osm_support_crs: str,
    built_support_crs: str,
) -> str:
    """确保网格、OSM 与 GHSL 使用同一个米制投影。"""

    values = (grid_crs, osm_support_crs, built_support_crs)
    if any(not isinstance(value, str) or not value for value in values):
        raise CuzSupportError("grid/OSM/GHSL support CRS 必须是非空字符串")
    if osm_support_crs != grid_crs or built_support_crs != grid_crs:
        raise CuzSupportError(
            "C/U/Z support CRS 不一致: "
            f"grid={grid_crs}, OSM={osm_support_crs}, GHSL={built_support_crs}"
        )
    try:
        from pyproj import CRS

        parsed = CRS.from_user_input(grid_crs)
    except Exception as error:  # noqa: BLE001 - 统一为输入契约错误
        raise CuzSupportError(f"无效的 C/U/Z grid CRS: {grid_crs}") from error
    if not parsed.is_projected or parsed.axis_info[0].unit_name.lower() not in {
        "metre",
        "meter",
    }:
        raise CuzSupportError("C/U/Z grid CRS 必须是米制投影")
    return grid_crs


@dataclass(frozen=True)
class CuzSupport:
    """一个区域的 C/U/Z 分区及其源区索引。"""

    osm_landuse: np.ndarray
    built_fraction: np.ndarray
    features: np.ndarray
    covered_mask: np.ndarray
    unknown_mask: np.ndarray
    zero_mask: np.ndarray
    covered_full_indices: np.ndarray
    unknown_full_indices: np.ndarray
    zero_full_indices: np.ndarray
    source_keys: np.ndarray
    source_key_order: np.ndarray
    source_index: np.ndarray
    source_built_covered: np.ndarray
    source_built_unknown: np.ndarray
    source_unknown_share: np.ndarray
    source_capacity_positive: np.ndarray


def atomic_save_npy(path: Path | str, value: np.ndarray) -> Path:
    """原子写入一个 NumPy 数组。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.part")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def atomic_savez(path: Path | str, **arrays: np.ndarray) -> Path:
    """原子写入一个 NPZ 文件。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.part")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _normalise_source_keys(source_keys: Sequence[object], n_cells: int) -> np.ndarray:
    if len(source_keys) != n_cells:
        raise CuzSupportError(
            f"source_keys 长度 {len(source_keys)} 与网格数 {n_cells} 不一致"
        )
    normalised: list[str] = []
    for idx, key in enumerate(source_keys):
        if key is None or (isinstance(key, (float, np.floating)) and np.isnan(key)):
            raise CuzSupportError(f"source_keys[{idx}] 缺失")
        text = str(key)
        if not text:
            raise CuzSupportError(f"source_keys[{idx}] 为空")
        normalised.append(text)
    return np.asarray(normalised, dtype=np.str_)


def _source_order_and_index(source_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order: list[str] = []
    lookup: dict[str, int] = {}
    source_index = np.empty(len(source_keys), dtype=np.int64)
    for cell_idx, key in enumerate(source_keys.tolist()):
        if key not in lookup:
            lookup[key] = len(order)
            order.append(key)
        source_index[cell_idx] = lookup[key]
    return np.asarray(order, dtype=np.str_), source_index


def build_cuz_support(
    osm_landuse: np.ndarray,
    built_fraction: np.ndarray,
    source_keys: Sequence[object],
) -> CuzSupport:
    """构建固定六通道 C/U/Z 表示，不做归一化、填补或国别分支。"""

    raw_landuse = np.asarray(osm_landuse)
    raw_built = np.asarray(built_fraction)
    if raw_landuse.dtype != np.dtype("float64"):
        raise CuzSupportError(f"osm_landuse 必须是 float64，实际为 {raw_landuse.dtype}")
    if raw_built.dtype != np.dtype("float64"):
        raise CuzSupportError(f"built_fraction 必须是 float64，实际为 {raw_built.dtype}")
    landuse = np.ascontiguousarray(raw_landuse)
    built = np.ascontiguousarray(raw_built)
    if landuse.ndim != 2 or landuse.shape[1] != 5:
        raise CuzSupportError(f"osm_landuse 必须是 (N, 5)，实际为 {landuse.shape}")
    if built.shape != (landuse.shape[0],):
        raise CuzSupportError(
            f"built_fraction 必须是 ({landuse.shape[0]},)，实际为 {built.shape}"
        )
    if not np.isfinite(landuse).all() or (landuse < 0).any():
        raise CuzSupportError("osm_landuse 必须有限且非负")
    if not np.isfinite(built).all() or (built < 0).any() or (built > 1).any():
        raise CuzSupportError("built_fraction 必须有限且位于 [0, 1]")

    keys = _normalise_source_keys(source_keys, len(landuse))
    source_order, source_index = _source_order_and_index(keys)

    osm_sum = np.sum(landuse, axis=1, dtype=np.float64)
    covered = osm_sum > 0.0
    unknown = (osm_sum == 0.0) & (built > 0.0)
    zero = (osm_sum == 0.0) & (built == 0.0)
    if np.any((covered.astype(np.int8) + unknown + zero) != 1):
        raise CuzSupportError("covered/unknown/zero 不互斥完备")

    features = np.zeros((len(landuse), 6), dtype=np.float64)
    features[:, :5] = landuse
    features[unknown, 5] = 1.0
    if not np.array_equal(features[:, :5], landuse):
        raise CuzSupportError("构建六通道时 OSM 土地利用值发生变化")
    if np.any(features[zero] != 0.0):
        raise CuzSupportError("zero 单元必须保持六通道全零")

    n_sources = len(source_order)
    built_covered = np.zeros(n_sources, dtype=np.float64)
    built_unknown = np.zeros(n_sources, dtype=np.float64)
    np.add.at(built_covered, source_index[covered], built[covered])
    np.add.at(built_unknown, source_index[unknown], built[unknown])
    total_capacity = built_covered + built_unknown
    capacity_positive = total_capacity > 0.0
    unknown_share = np.zeros(n_sources, dtype=np.float64)
    unknown_share[capacity_positive] = (
        built_unknown[capacity_positive] / total_capacity[capacity_positive]
    )

    return CuzSupport(
        osm_landuse=landuse,
        built_fraction=built,
        features=features,
        covered_mask=covered,
        unknown_mask=unknown,
        zero_mask=zero,
        covered_full_indices=np.flatnonzero(covered).astype(np.int64),
        unknown_full_indices=np.flatnonzero(unknown).astype(np.int64),
        zero_full_indices=np.flatnonzero(zero).astype(np.int64),
        source_keys=keys,
        source_key_order=source_order,
        source_index=source_index,
        source_built_covered=built_covered,
        source_built_unknown=built_unknown,
        source_unknown_share=unknown_share,
        source_capacity_positive=capacity_positive,
    )


def _artifact_payload(support: CuzSupport) -> dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "landuse_categories": np.asarray(LANDUSE_CATEGORIES, dtype=np.str_),
        "osm_landuse": support.osm_landuse,
        "built_fraction": support.built_fraction,
        "features": support.features,
        "covered_mask": support.covered_mask,
        "unknown_mask": support.unknown_mask,
        "zero_mask": support.zero_mask,
        "covered_full_indices": support.covered_full_indices,
        "unknown_full_indices": support.unknown_full_indices,
        "zero_full_indices": support.zero_full_indices,
        "source_keys": support.source_keys,
        "source_key_order": support.source_key_order,
        "source_index": support.source_index,
        "source_built_covered": support.source_built_covered,
        "source_built_unknown": support.source_built_unknown,
        "source_unknown_share": support.source_unknown_share,
        "source_capacity_positive": support.source_capacity_positive,
    }


def save_cuz_support(path: Path | str, support: CuzSupport, *, overwrite: bool = False) -> Path:
    """原子保存 C/U/Z 产物；非覆盖模式下已有文件直接复用。"""

    output = Path(path)
    if output.exists() and not overwrite:
        return output
    return atomic_savez(output, **_artifact_payload(support))


def load_cuz_support(path: Path | str) -> CuzSupport:
    """加载 C/U/Z 产物，并用数值定义复算派生字段。"""

    artifact = Path(path)
    with np.load(artifact, allow_pickle=False) as loaded:
        keys = set(loaded.files)
        if keys != _ARTIFACT_KEYS:
            raise CuzSupportError(
                f"产物字段不符合 {SCHEMA_VERSION}: "
                f"missing={sorted(_ARTIFACT_KEYS - keys)}, extra={sorted(keys - _ARTIFACT_KEYS)}"
            )
        if str(loaded["schema_version"].item()) != SCHEMA_VERSION:
            raise CuzSupportError("未知的 C/U/Z schema version")
        categories = tuple(str(x) for x in loaded["landuse_categories"].tolist())
        if categories != LANDUSE_CATEGORIES:
            raise CuzSupportError(f"土地利用类别顺序不一致: {categories}")
        rebuilt = build_cuz_support(
            loaded["osm_landuse"],
            loaded["built_fraction"],
            loaded["source_keys"],
        )
        for name in (
            "features",
            "covered_mask",
            "unknown_mask",
            "zero_mask",
            "covered_full_indices",
            "unknown_full_indices",
            "zero_full_indices",
            "source_key_order",
            "source_index",
            "source_built_covered",
            "source_built_unknown",
            "source_unknown_share",
            "source_capacity_positive",
        ):
            if not np.array_equal(getattr(rebuilt, name), loaded[name]):
                raise CuzSupportError(f"产物字段 {name!r} 与 C/U/Z 数值定义不一致")
    return rebuilt


def deterministic_built_surface_field(
    support: CuzSupport,
    demand_by_source: Mapping[str, float],
) -> np.ndarray:
    """按 Built-S 容量在 C/U 单元分配源区需求，Z 始终为零。"""

    expected = support.source_key_order.tolist()
    actual = {str(key) for key in demand_by_source}
    if actual != set(expected):
        raise CuzSupportError(
            f"demand source keys 不一致: missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )

    field = np.zeros(len(support.built_fraction), dtype=np.float64)
    for source_pos, source_key in enumerate(expected):
        demand = float(demand_by_source[source_key])
        if not np.isfinite(demand) or demand < 0.0:
            raise CuzSupportError(f"source {source_key}: demand 必须有限且非负")
        cells = support.source_index == source_pos
        if demand == 0.0:
            continue
        a = support.source_built_covered[source_pos]
        b = support.source_built_unknown[source_pos]
        if not np.isfinite(a + b) or a + b <= 0.0:
            raise CuzSupportError(f"source {source_key}: 正需求没有 C/U Built-S 容量")
        allocatable = cells & (support.covered_mask | support.unknown_mask)
        field[allocatable] = demand * support.built_fraction[allocatable] / (a + b)
        tolerance = (
            np.finfo(np.float64).eps
            * max(int(np.count_nonzero(cells)), 1)
            * max(demand, 1.0)
            * 128.0
        )
        error = abs(float(np.sum(field[cells], dtype=np.float64)) - demand)
        if error > tolerance:
            raise CuzSupportError(
                f"source {source_key}: conservation error {error} > {tolerance}"
            )
    if not np.isfinite(field).all() or (field < 0.0).any():
        raise CuzSupportError("allocated field 包含非有限值或负值")
    if np.any(field[support.zero_mask] != 0.0):
        raise CuzSupportError("zero 单元收到非零需求")
    return field
