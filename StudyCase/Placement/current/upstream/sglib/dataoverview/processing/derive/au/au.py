"""AU 站表与区域边界派生函数（由 ``sglib.dataoverview`` DAG 驱动）。


    data/datasets/2_derived/au/asgs_nsw_sa2_sa3_sa4_gda2020.gpkg   —— NSW SA2/SA3/SA4 边界子集
    data/datasets/2_derived/au/station_table_fy2008.csv            —— 站表三年窗
    data/datasets/2_derived/au/station_table_fy2009.csv
    data/datasets/2_derived/au/station_table_fy2010.csv
    data/datasets/2_derived/au/station_points_fy2009.geojson       —— 主分析年站点空间分布

1. a3 交叉核对从"不一致即硬失败"降为**警告**——完全重新下载视角下,fresh 用户拿到的
   现行位置图层会自然偏离论文快照,不应拦截(a3/a4 为论文冻结登记件,随库分发于
   `au_registry/`,供口径核对与 PV 分层列);
2. 不再生成 a6 报告/JSON;汇总改为控制台打印。

已知警示:Ausgrid 负荷为未经校验的原始 SCADA/计量数据,部分站
MW 由 Amps 按假定功率因数折算;主供大客户站因 Rule 5.13A(g) 保密不公开;报告年
May-to-May;站级求和 ≈ 供区配电量的子集口径。

"""

from __future__ import annotations

import io
import json
import re
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path, PurePosixPath

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

from ... import source_au as au_download
from . import au_matching as M

from ...config import CountryPipelineContext


REGISTRY_DIR = Path(__file__).parent / "au_registry"

# 历史窗口与清洗口径的依据见 data/README.md「历史口径说明」。

# 三年窗:主分析 FY2009,邻年 FY2008/FY2010
YEARS: list[int] = [2008, 2009, 2010]
MAIN_YEAR = 2009

# —— 清洗规则常量 ——
MISSING_THRESHOLD = 0.10          # 站-年缺测率 > 10% → cleaned_out
EXPECTED_UNIT = "MW"              # 单位断言:只接受 MW(出现其他单位硬失败,不静默换算)
META_COLS = ("year", "zone substation", "date", "unit")

# 米制 CRS:GDA2020 / MGA zone 56(Ausgrid 足迹 150.5–152.3°E 全落 zone 56)
METRIC_CRS = "EPSG:7856"
_TRANSFORM_4326_TO_7856 = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)

OSM_BYNAME_CACHE_REL = "osm/overpass_byname.json"
OSM_SLEEP_SECONDS = 3.0

# 归因状态枚举(互斥单标签)
STATUS_MATCHED = "matched"
STATUS_MATCHED_OSM = "matched_osm"
STATUS_CLEANED_OUT = "cleaned_out"
STATUS_EXCLUDED_RETIRED = "excluded_retired"
STATUS_EXCLUDED_UNLOCATED = "excluded_unlocated"
ALL_STATUSES = (STATUS_MATCHED, STATUS_MATCHED_OSM, STATUS_CLEANED_OUT,
                STATUS_EXCLUDED_RETIRED, STATUS_EXCLUDED_UNLOCATED)

OUTPUTS = ["asgs_nsw_sa2_sa3_sa4_gda2020.gpkg",
           "station_table_fy2008.csv", "station_table_fy2009.csv",
           "station_table_fy2010.csv", "station_points_fy2009.geojson"]


# ---------------------------------------------------------------------------
# ASGS NSW 子集 GPKG
# ---------------------------------------------------------------------------

def convert_asgs_nsw_gpkg(context: CountryPipelineContext, force: bool = False) -> Path:
    """三层 SHP 裁 NSW 子集并合写单一 GPKG(幂等)。"""
    gpkg_path = context.derived_root / "asgs_nsw_sa2_sa3_sa4_gda2020.gpkg"
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    if gpkg_path.exists() and gpkg_path.stat().st_size > 0 and not force:
        print(f"[跳过] NSW 子集 GPKG 已存在:{gpkg_path.name}")
        return gpkg_path
    for level in ("SA2", "SA3", "SA4"):
        zip_path = context.raw_root / "asgs_ed3" / f"{level}_2021_AUST_SHP_GDA2020.zip"
        if not zip_path.is_file():
            raise FileNotFoundError(f"缺 {zip_path.name}(先运行 2_AU notebook 第一步下载)")
        gdf = gpd.read_file(f"zip://{zip_path}")
        nsw = gdf[gdf["STE_CODE21"] == "1"].copy()  # NSW = state code 1
        nsw = nsw[nsw.geometry.notna()]  # 剔除无几何行(离岸/未分配占位)
        nsw.to_file(gpkg_path, layer=f"{level.lower()}_2021_nsw", driver="GPKG")
        print(f"[转换] {level} NSW 子集 {len(nsw)} 面 -> layer {level.lower()}_2021_nsw")
    return gpkg_path


# ---------------------------------------------------------------------------
# 负荷 zip 解析与清洗
# ---------------------------------------------------------------------------

def fy_window(fy: int) -> tuple[date, date]:
    """财年窗口:FY{Y} = (Y-1) 年 5 月 1 日 ~ Y 年 4 月 30 日(官方 May-to-May)。"""
    return date(fy - 1, 5, 1), date(fy, 4, 30)


def parse_dates(series: pd.Series) -> list[date]:
    """两代日期格式自动识别:01MAY2008(%d%b%Y)与 01/05/2023(%d/%m/%Y)。"""
    sample = str(series.iloc[0])
    fmt = "%d/%m/%Y" if "/" in sample else "%d%b%Y"
    return list(pd.to_datetime(series, format=fmt).dt.date)


def strip_base_name(station: str) -> str:
    """剥电压 token 得基名(与匹配工具同口径)。"""
    return re.sub(r"\s+", " ", M.VOLTAGE_TOKEN_RE.sub("", station)).strip()


def parse_year(context: CountryPipelineContext, fy: int) -> dict:
    """解析一个财年 zip:逐 CSV 建全窗格矩阵,按基名合并变体,产出站级统计。"""
    zp = context.raw_root / "ausgrid_zs" / f"ausgrid_zs_fy{fy}.zip"
    if not zp.exists():
        raise FileNotFoundError(f"{zp} 不存在(先运行 2_AU notebook 第一步下载)")

    win_start, win_end = fy_window(fy)
    n_days = (win_end - win_start).days + 1
    full_index = [win_start + timedelta(days=i) for i in range(n_days)]

    csv_all: list[str] = []
    empty_csvs: list[str] = []
    unit_values: set[str] = set()
    slots_set: set[int] = set()
    dup_dates_total = 0
    variant_frames: dict[str, pd.DataFrame] = {}

    with zipfile.ZipFile(zp) as zf:
        for member in zf.namelist():
            if member.endswith("/") or not member.lower().endswith(".csv"):
                continue
            stem = PurePosixPath(member).stem
            station = M.FY_SUFFIX_RE.sub("", stem.strip())
            csv_all.append(station)

            with zf.open(member) as fh:
                df = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8-sig"))
            norm_cols = [c.strip().lower() for c in df.columns]
            meta_idx = {c: i for i, c in enumerate(norm_cols) if c in META_COLS}
            time_cols = [df.columns[i] for i in range(len(df.columns))
                         if i not in meta_idx.values()]
            slots_set.add(len(time_cols))

            if "unit" in meta_idx:
                unit_values.update(
                    df.iloc[:, meta_idx["unit"]].astype(str).str.strip().unique())

            values = df[time_cols].apply(pd.to_numeric, errors="coerce")
            if "date" in meta_idx and len(df) > 0:
                dates = parse_dates(df.iloc[:, meta_idx["date"]])
            else:
                dates = []
            if len(dates) != len(values):
                raise ValueError(f"FY{fy} {station}:日期列与数据行数不一致")

            values.index = pd.Index(dates)
            dup_mask = values.index.duplicated(keep="first")
            dup_dates_total += int(dup_mask.sum())
            values = values[~dup_mask]
            # 限窗 + 全窗格 reindex(整日缺失 → NaN 行)
            in_window = [d for d in values.index if win_start <= d <= win_end]
            values = values.loc[in_window]
            values.columns = range(values.shape[1])
            mat = values.reindex(full_index)

            if int(mat.notna().sum().sum()) == 0:
                empty_csvs.append(station)  # 占位空站:登记后剔除,不进基名池
                continue
            variant_frames[station] = mat

    # —— 单位断言:只接受 MW,出现其他单位立即硬失败(禁静默换算) ——
    if unit_values - {EXPECTED_UNIT}:
        raise AssertionError(
            f"FY{fy} 单位列出现非 MW 取值:{sorted(unit_values)}——"
            f"不做静默换算,需人工核查后扩展规则")

    # —— 按基名合并变体 ——
    grouped: dict[str, list[str]] = {}
    for station in variant_frames:
        grouped.setdefault(strip_base_name(station), []).append(station)

    bases: dict[str, dict] = {}
    for base, variants in sorted(grouped.items()):
        slots_counts = {variant_frames[v].shape[1] for v in variants}
        if len(slots_counts) > 1:
            raise ValueError(f"FY{fy} {base}:变体时段列数不一致 {slots_counts}")
        slots = slots_counts.pop()
        n_total = n_days * slots

        variant_missing = {
            v: 1.0 - int(variant_frames[v].notna().sum().sum()) / n_total
            for v in variants}
        merge_mode = "single"
        used = sorted(variants)
        dropped: list[str] = []
        if len(variants) > 1:
            valid_count = sum(variant_frames[v].notna().astype(int) for v in variants)
            n_any = int((valid_count >= 1).sum().sum())
            n_overlap = int((valid_count >= 2).sum().sum())
            overlap_share = n_overlap / n_any if n_any else 0.0
            merge_mode = "handover" if overlap_share < 0.05 else "parallel"
            if merge_mode == "parallel":
                ok_variants = sorted(v for v, m in variant_missing.items()
                                     if m <= MISSING_THRESHOLD)
                used = ok_variants if ok_variants else sorted(variants)
                dropped = sorted(set(variants) - set(used))

        if merge_mode == "handover":
            # 切换合并:时段值 = 各变体有效值之和(缺测视 0),全变体皆缺才记缺测
            total = sum(variant_frames[v].fillna(0.0) for v in used)
            cnt = sum(variant_frames[v].notna().astype(int) for v in used)
            combined = total.where(cnt > 0)
        else:
            combined = variant_frames[used[0]]
            for v in used[1:]:
                combined = combined + variant_frames[v]  # NaN 传播 = 任一并行变体缺测记缺测

        n_valid = int(combined.notna().sum().sum())
        missing_rate = 1.0 - n_valid / n_total
        slot_hours = 24.0 / slots
        energy_valid_mwh = float(combined.sum().sum()) * slot_hours  # skipna 默认
        energy_gwh = (energy_valid_mwh * n_total / n_valid / 1000.0
                      if n_valid else float("nan"))
        peak_mw = float(combined.max().max()) if n_valid else float("nan")
        bases[base] = {
            "variants": sorted(variants),
            "used_variants": used,
            "dropped_variants": dropped,
            "merge_mode": merge_mode,
            "variant_missing": {v: round(m, 6) for v, m in variant_missing.items()},
            "slots": slots,
            "n_total": n_total,
            "n_valid": n_valid,
            "missing_rate": missing_rate,
            "energy_gwh": energy_gwh,
            "peak_mw": peak_mw,
            "n_neg_cells": int((combined < 0).sum().sum()),
        }

    return {
        "fy": fy,
        "n_csv_all": len(csv_all),
        "empty_csvs": sorted(empty_csvs),
        "n_empty_placeholder": len(empty_csvs),
        "n_with_data_csv": len(csv_all) - len(empty_csvs),
        "unit_values": sorted(unit_values),
        "slots_set": sorted(slots_set),
        "dup_dates_total": dup_dates_total,
        "bases": bases,
    }


# ---------------------------------------------------------------------------
# 匹配与 OSM 兜底
# ---------------------------------------------------------------------------

def build_layer_index(context: CountryPipelineContext) -> tuple[dict[str, str], dict[str, tuple[float, float]]]:
    """位置图层索引:归一名 -> 图层站名;图层站名 -> (lon, lat)(WGS84)。"""
    geo = au_download.land_locations_layer(context.raw_root)  # 存在即读,缺失才 fresh 查询
    norm2name: dict[str, str] = {}
    name2coord: dict[str, tuple[float, float]] = {}
    for feat in geo["features"]:
        name = feat["properties"]["substation"]
        lon, lat = feat["geometry"]["coordinates"][:2]
        name2coord[name] = (float(lon), float(lat))
        norm2name[M.normalise(name)] = name
    return norm2name, name2coord


def match_bases(bases: dict[str, dict], layer_norm2name: dict[str, str]) -> dict[str, dict]:
    """基名 ↔ 图层匹配。"""
    out: dict[str, dict] = {}
    layer_norms = list(layer_norm2name)
    for base in sorted(bases):
        norm = M.normalise(base)
        if norm in layer_norm2name:
            out[base] = {"match_type": "exact", "layer_name": layer_norm2name[norm],
                         "similarity": 1.0}
            continue
        sims = sorted(((M.similarity(norm, ln), ln) for ln in layer_norms), reverse=True)
        best_sim, best_norm = sims[0]
        second_sim = sims[1][0] if len(sims) > 1 else 0.0
        if best_sim >= M.FUZZY_THRESHOLD and (best_sim - second_sim) >= M.AMBIGUITY_GAP:
            out[base] = {"match_type": "fuzzy", "layer_name": layer_norm2name[best_norm],
                         "similarity": round(best_sim, 4)}
        else:
            out[base] = {"match_type": "none", "layer_name": None,
                         "similarity": round(best_sim, 4),
                         "best_candidate": layer_norm2name[best_norm]}
    return out


def crosscheck_a3(match_2009: dict[str, dict]) -> bool:
    """FY2009 匹配结果与论文冻结登记 a3_matching.json 交叉核对。

    """
    a3 = json.loads((REGISTRY_DIR / "a3_matching.json").read_text(encoding="utf-8"))
    a3_fy = a3["fy2009"]
    n_exact = sum(1 for m in match_2009.values() if m["match_type"] == "exact")
    n_fuzzy = sum(1 for m in match_2009.values() if m["match_type"] == "fuzzy")
    n_none = sum(1 for m in match_2009.values() if m["match_type"] == "none")
    unmatched_here = sorted(b for b, m in match_2009.items() if m["match_type"] == "none")
    unmatched_a3 = sorted(r["load_base"] for r in a3_fy["unmatched"])
    consistent = (a3_fy["n_exact"] == n_exact and a3_fy["n_fuzzy"] == n_fuzzy
                  and a3_fy["n_unmatched"] == n_none and unmatched_here == unmatched_a3)
    if consistent:
        print(f"[核对] FY{MAIN_YEAR} 匹配与论文登记 a3 一致"
              f"(exact {n_exact} / fuzzy {n_fuzzy} / none {n_none})")
    else:
        print(f"[警告] FY{MAIN_YEAR} 匹配与论文登记 a3 不一致"
              f"(本次 exact {n_exact}/fuzzy {n_fuzzy}/none {n_none},"
              f"a3 登记 {a3_fy['n_exact']}/{a3_fy['n_fuzzy']}/{a3_fy['n_unmatched']})——"
              f"位置图层可能为现行版本而非论文快照;流程照常")
    return consistent


def _element_coord(el: dict) -> tuple[float, float] | None:
    if "lat" in el and "lon" in el:
        return float(el["lon"]), float(el["lat"])
    center = el.get("center")
    if center:
        return float(center["lon"]), float(center["lat"])
    return None


def _candidate_names(el: dict) -> list[str]:
    tags = el.get("tags", {})
    return [tags[k] for k in ("name", "official_name", "alt_name", "short_name")
            if tags.get(k)]


def _match_osm_candidates(base: str, elements: list[dict], method: str) -> dict | None:
    """在 OSM 要素列表中为基名找唯一命中。"""
    norm = M.normalise(base)
    scored: list[tuple[float, str, dict]] = []
    for el in elements:
        if _element_coord(el) is None:
            continue
        for cand in _candidate_names(el):
            scored.append((M.similarity(norm, M.normalise(cand)), cand, el))
    if not scored:
        return None
    scored.sort(key=lambda t: -t[0])
    best_sim, best_name, best_el = scored[0]
    best_key = (best_el["type"], best_el["id"])
    second_sim = next(
        (s for s, _n, el in scored[1:] if (el["type"], el["id"]) != best_key), 0.0)
    if best_sim >= M.FUZZY_THRESHOLD and (best_sim - second_sim) >= M.AMBIGUITY_GAP:
        lon, lat = _element_coord(best_el)
        return {"osm_type": best_el["type"], "osm_id": best_el["id"],
                "osm_name": best_name, "similarity": round(best_sim, 4),
                "lon": lon, "lat": lat, "method": method}
    return None


def osm_fallback(context: CountryPipelineContext, unmatched_bases: list[str]) -> dict:
    """未匹配基名的 OSM 兜底定位。"""
    bbox_dump = au_download.fetch_osm_substation_dump(
        au_download.OSM_BBOX, au_download.OSM_BBOX_RELPATH, "足迹 bbox ", target=context.raw_root)
    named = [el for el in bbox_dump.get("elements", []) if _candidate_names(el)]

    byname_path = context.raw_root / OSM_BYNAME_CACHE_REL
    byname_cache: dict[str, list] = {}
    if byname_path.is_file() and byname_path.stat().st_size > 0:
        byname_cache = json.loads(byname_path.read_text(encoding="utf-8"))

    first_pass: dict[str, dict | None] = {
        base: _match_osm_candidates(base, named, method="bbox_dump")
        for base in sorted(unmatched_bases)}
    need_query = [b for b, h in first_pass.items() if h is None and b not in byname_cache]

    if need_query:
        ext_dump = au_download.fetch_osm_substation_dump(
            au_download.OSM_BBOX_EXT, au_download.OSM_BBOX_EXT_RELPATH, "扩展供区 bbox ", target=context.raw_root)
        ext_named = [el for el in ext_dump.get("elements", []) if _candidate_names(el)]
        for base in need_query:
            norm = M.normalise(base)
            byname_cache[base] = [
                el for el in ext_named
                if any(M.similarity(norm, M.normalise(c)) >= 0.70
                       for c in _candidate_names(el))]
        tmp = byname_path.parent / (byname_path.name + ".part")
        tmp.write_text(json.dumps(byname_cache, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(byname_path)
        time.sleep(OSM_SLEEP_SECONDS)  # 礼貌间隔(仅实际构建缓存时)

    hits: dict[str, dict] = {}
    misses: list[str] = []
    for base in sorted(unmatched_bases):
        hit = first_pass[base]
        if hit is None:
            hit = _match_osm_candidates(base, byname_cache.get(base, []),
                                        method="byname_ext_bbox")
        if hit is not None:
            hits[base] = hit
        else:
            misses.append(base)
    print(f"[OSM] 兜底命中 {len(hits)} / {len(unmatched_bases)}")
    return {"hits": hits, "misses": sorted(misses)}


# ---------------------------------------------------------------------------
# 归因 + 站表
# ---------------------------------------------------------------------------

def load_fy2024_base_norms(context: CountryPipelineContext) -> set[str]:
    """FY2024 负荷侧基名(归一化)集合——退役证据用(2024 负荷仍在 = 未退役)。"""
    zp = context.raw_root / "ausgrid_zs" / "ausgrid_zs_fy2024.zip"
    if not zp.exists():
        raise FileNotFoundError(f"{zp} 不存在——退役归因需要 FY2024 负荷站名集")
    norms: set[str] = set()
    with zipfile.ZipFile(zp) as zf:
        for member in zf.namelist():
            if member.endswith("/") or not member.lower().endswith(".csv"):
                continue
            stem = PurePosixPath(member).stem
            station = M.FY_SUFFIX_RE.sub("", stem.strip())
            norms.add(M.normalise(strip_base_name(station)))
    return norms


def load_pv_levels(context: CountryPipelineContext) -> dict[str, str]:
    """FY2009 逐站 PV 分层(变体名 -> level),供站表附注列。

    唯一合法来源 = **再生件** datasets/2_derived/au/a4_pv_screening.json
    (R9 解冻 #6,2026-08-11:生成链补迁 au_pv.screen_pv,再生 vs 论文登记件
    全 payload 逐位 EXACT;au_registry/a4 自此降级纯对照,代码不再消费)。
    缺失时现场再生(幂等,输入 = ausgrid_zs raw)。
    """
    from . import au_pv
    a4_path = au_pv.screen_pv(context)   # 已在位则跳过,缺失则从 raw 再生
    a4 = json.loads(a4_path.read_text(encoding="utf-8"))
    return {s["station"]: s["level"] for s in a4["fy2009"]["stations"]}


_PV_ORDER = {"suspected": 2, "mild": 1, "clean": 0}


def worst_pv_level(variants: list[str], pv_map: dict[str, str]) -> str:
    levels = [pv_map[v] for v in variants if v in pv_map]
    if not levels:
        return ""
    return max(levels, key=lambda lv: _PV_ORDER.get(lv, -1))


def attribute_year(year_data: dict, match: dict[str, dict],
                   name2coord: dict[str, tuple[float, float]],
                   osm_hits: dict[str, dict], fy2024_norms: set[str],
                   pv_map: dict[str, str]) -> pd.DataFrame:
    """构建一个财年的站表。"""
    rows: list[dict] = []
    is_main = year_data["fy"] == MAIN_YEAR
    for base, stats in sorted(year_data["bases"].items()):
        m = match[base]
        lon = lat = x = y = None
        matched_name = ""
        match_source = "none"
        note_parts: list[str] = []
        if stats["merge_mode"] == "handover":
            note_parts.append(
                "切换合并（变体有效时段重叠 <5%）：按'至少一变体有效'合并——"
                + "；".join(f"{v}（缺测 {stats['variant_missing'][v]:.2%}）"
                            for v in stats["variants"]))
        if stats["dropped_variants"]:
            note_parts.append(
                "变体级清洗：剔除缺测超限变体 "
                + "；".join(f"{v}（{stats['variant_missing'][v]:.2%}）"
                            for v in stats["dropped_variants"])
                + f"，聚合仅用 {'；'.join(stats['used_variants'])}")

        if m["match_type"] in ("exact", "fuzzy"):
            matched_name = m["layer_name"]
            lon, lat = name2coord[matched_name]
            match_source = "layer"
            if m["match_type"] == "fuzzy":
                note_parts.append(f"编辑距离兜底匹配（相似度 {m['similarity']}）")
        elif base in osm_hits:
            hit = osm_hits[base]
            matched_name = hit["osm_name"]
            lon, lat = hit["lon"], hit["lat"]
            match_source = "osm"
            note_parts.append(
                f"OSM 兜底：{hit['osm_type']}/{hit['osm_id']}"
                f"（相似度 {hit['similarity']}，{hit['method']}）")

        # —— 互斥归因标签(清洗仅作用于已定位的站) ——
        if match_source == "none":
            in_fy2024 = M.normalise(base) in fy2024_norms
            if in_fy2024:
                status = STATUS_EXCLUDED_UNLOCATED
                note_parts.append(
                    "无坐标但 FY2024 负荷仍在档——退役证据不足，登记为未定位")
            else:
                status = STATUS_EXCLUDED_RETIRED
                note_parts.append(
                    f"退役证据：2024 位置图层无该站（最优候选 "
                    f"{m.get('best_candidate', '')}，相似度 {m['similarity']}），"
                    f"FY2024 负荷数据亦无该站，OSM 兜底未命中")
        elif stats["missing_rate"] > MISSING_THRESHOLD:
            status = STATUS_CLEANED_OUT
            note_parts.append(
                f"缺测率 {stats['missing_rate']:.2%} > {MISSING_THRESHOLD:.0%}，"
                f"剔出可用集（行保留，外推电量仅供参考）")
        else:
            status = STATUS_MATCHED if match_source == "layer" else STATUS_MATCHED_OSM

        if lon is not None:
            x, y = _TRANSFORM_4326_TO_7856.transform(lon, lat)

        rows.append({
            "station": base,
            "variants": "；".join(stats["variants"]),
            "n_variants": len(stats["variants"]),
            "status": status,
            "match_source": match_source,
            "matched_name": matched_name,
            "lon_wgs84": round(lon, 6) if lon is not None else None,
            "lat_wgs84": round(lat, 6) if lat is not None else None,
            "x_epsg7856": round(x, 2) if x is not None else None,
            "y_epsg7856": round(y, 2) if y is not None else None,
            "energy_gwh": round(stats["energy_gwh"], 4),
            "peak_mw": round(stats["peak_mw"], 3),
            "missing_rate": round(stats["missing_rate"], 6),
            "n_valid_slots": stats["n_valid"],
            "n_total_slots": stats["n_total"],
            "n_neg_cells": stats["n_neg_cells"],
            "pv_level_fy2009": worst_pv_level(stats["variants"], pv_map) if is_main else "",
            "attribution_note": "；".join(note_parts),
        })
    return pd.DataFrame(rows)


def export_geojson(table: pd.DataFrame, path: Path) -> int:
    """导出主分析年站点空间分布 GeoJSON。"""
    features = []
    for _, r in table.iterrows():
        if r["lon_wgs84"] is None or pd.isna(r["lon_wgs84"]):
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [r["lon_wgs84"], r["lat_wgs84"]]},
            "properties": {
                "station": r["station"], "status": r["status"],
                "match_source": r["match_source"], "matched_name": r["matched_name"],
                "energy_gwh": r["energy_gwh"], "peak_mw": r["peak_mw"],
                "missing_rate": r["missing_rate"],
            },
        })
    payload = {"type": "FeatureCollection",
               "crs_note": "坐标 WGS84 (EPSG:4326)；米制换算列见站表 CSV (EPSG:7856)",
               "features": features}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return len(features)


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------

def derive_au(context: CountryPipelineContext, force: bool = False) -> list[Path]:
    """派生 AU 站表、ASGS 子集及统一 runner 所需区域层。"""
    outs = [context.derived_root / n for n in OUTPUTS]
    if not force and all(p.is_file() for p in outs):
        print(f"派生产物已存在,跳过(force=True 可重算):{', '.join(OUTPUTS)}")
        from .au_regions import derive_au_regions

        return [*outs, *derive_au_regions(context, force=False)]

    convert_asgs_nsw_gpkg(context, force=force)

    year_data: dict[int, dict] = {}
    for fy in YEARS:
        print(f"[解析] FY{fy} …")
        year_data[fy] = parse_year(context, fy)
        print(f"  CSV {year_data[fy]['n_csv_all']},空站 {year_data[fy]['n_empty_placeholder']},"
              f"有数据基名 {len(year_data[fy]['bases'])},单位 {year_data[fy]['unit_values']}")

    layer_norm2name, name2coord = build_layer_index(context)
    matches = {fy: match_bases(year_data[fy]["bases"], layer_norm2name) for fy in YEARS}
    crosscheck_a3(matches[MAIN_YEAR])

    unmatched_union = sorted({
        b for fy in YEARS for b, m in matches[fy].items() if m["match_type"] == "none"})
    print(f"[OSM] 三年未匹配基名并集:{len(unmatched_union)} 个")
    osm_result = osm_fallback(context, unmatched_union)

    fy2024_norms = load_fy2024_base_norms(context)
    pv_map = load_pv_levels(context)
    context.derived_root.mkdir(parents=True, exist_ok=True)
    for fy in YEARS:
        table = attribute_year(year_data[fy], matches[fy], name2coord,
                               osm_result["hits"], fy2024_norms, pv_map)
        out_csv = context.derived_root / f"station_table_fy{fy}.csv"
        table.to_csv(out_csv, index=False, encoding="utf-8-sig")
        usable = int(table["status"].isin([STATUS_MATCHED, STATUS_MATCHED_OSM]).sum())
        print(f"[站表] {out_csv.name}:{len(table)} 行,usable {usable},"
              f"电量合计 {table['energy_gwh'].sum() / 1000.0:.3f} TWh")
        if fy == MAIN_YEAR:
            n_feat = export_geojson(table, context.derived_root / "station_points_fy2009.geojson")
            print(f"[导出] station_points_fy2009.geojson:{n_feat} 点")
    from .au_regions import derive_au_regions

    return [*outs, *derive_au_regions(context, force=force)]
