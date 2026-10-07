"""DE(Börde)区域与变电站派生函数（由 ``sglib.dataoverview`` DAG 驱动）。

+ `fix_rebuild_gva_xlsx.py` 的 GVA xlsx 重建,产出:

    data/datasets/2_derived/de/nama_10r_3gva.xlsx      —— Eurostat GVA 按 001 期望布局重建(中间件)
    data/datasets/2_derived/de/source_regions.gpkg     —— 34 Gemeinden + 人口 + GVA percent + 面积占比
    data/datasets/2_derived/de/substations.gpkg        —— 13 座 UW + 负荷 + Gemeinde 归属

Berlin/Hannover/Leipzig/Magdeburg)、人口填补策略、NACE→四扇区映射(对齐 UK SIC07 分类)、
零负荷筛除、Gemeinde 归属(within + 最近邻兜底)逐 cell 对应源 notebook。
**登记的适配**(一处):xlsx 重建的 2022 参照核对从"失败即拒绝写出"降为
**警告**——完全重新下载视角下 Eurostat 修订数据不应拦截;参照值仍打印供核对。

与 UK 案例的关键差异:单 NUTS3 区域(DEE07);GVA percent 为
NUTS3 整体值、全部 Gemeinden 共享;CRS 面积计算用 EPSG:25832(UTM32N)。

"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ...config import CountryPipelineContext


# GVA 重建与源区域口径的依据见 data/README.md「历史口径说明」。

BOERDE_NUTS3 = "DEE07"
GERMANY_POP_2022 = 84_359_000  # 德国总人口 ~84M

# —— GVA xlsx 重建 ——
YEARS_GVA = list(range(2015, 2025))
GEO_LABELS = {"DE": "Germany", "DEE07": "Börde"}
# 原始导出的 sheet 顺序(Eurostat NACE_R2 维度协议序);001 只用 Sheet 1/2/3/5/6/9/13
SHEET_NACE = [
    ("Sheet 1", "TOTAL", "Total - all NACE activities"),
    ("Sheet 2", "A", "Agriculture, forestry and fishing"),
    ("Sheet 3", "B-E", "Industry (except construction)"),
    ("Sheet 4", "C", "Manufacturing"),
    ("Sheet 5", "F", "Construction"),
    ("Sheet 6", "G-J", "Wholesale and retail trade; transport; accommodation and food service activities; information and communication"),
    ("Sheet 7", "G-I", "Wholesale and retail trade; transport; accommodation and food service activities"),
    ("Sheet 8", "J", "Information and communication"),
    ("Sheet 9", "K-N", "Financial and insurance activities; real estate activities; professional, scientific and technical activities; administrative and support service activities"),
    ("Sheet 10", "K", "Financial and insurance activities"),
    ("Sheet 11", "L", "Real estate activities"),
    ("Sheet 12", "M_N", "Professional, scientific and technical activities; administrative and support service activities"),
    ("Sheet 13", "O-U", "Public administration and defence; compulsory social security; education; human health and social work activities; arts, entertainment and recreation, other service activities"),
    ("Sheet 14", "O-Q", "Public administration and defence; compulsory social security; education; human health and social work activities"),
    ("Sheet 15", "R-U", "Arts, entertainment and recreation; other service activities; activities of household and extra-territorial organizations and bodies"),
]
GVA_ANCHORS = {
    ("TOTAL", "DEE07"): 5684.20, ("TOTAL", "DE"): 3591874,
    ("A", "DEE07"): 315.00,      ("A", "DE"): 39680,
    ("B-E", "DEE07"): 1997.94,   ("B-E", "DE"): 845760,
    ("F", "DEE07"): 285.95,      ("F", "DE"): 173942,
    ("G-J", "DEE07"): 1410.44,   ("G-J", "DE"): 810102,
    ("K-N", "DEE07"): 710.46,    ("K-N", "DE"): 910759,
    ("O-U", "DEE07"): 964.33,    ("O-U", "DE"): 811631,
}

OUTPUTS = ["nama_10r_3gva.xlsx", "source_regions.gpkg", "substations.gpkg"]


def _require(context: CountryPipelineContext, rel: str) -> Path:
    p = context.raw_root / rel
    if not p.is_file():
        raise FileNotFoundError(
            f"缺文件:{p}\n  → 请先运行 python -m sglib.dataoverview --country de")
    return p


# ---------------------------------------------------------------------------
# GVA xlsx 重建
# ---------------------------------------------------------------------------

def _build_sheet_frame(df: pd.DataFrame, nace: str, label: str) -> pd.DataFrame:
    """按原始导出布局构建单个 sheet 的 DataFrame。"""
    n_cols = 1 + 2 * len(YEARS_GVA)  # TIME + (年, flag) × 10 = 21 列

    def meta(label0, label1=None):
        row = [np.nan] * n_cols
        row[0] = label0
        if label1 is not None:
            row[1] = label1
        return row

    rows = [
        meta("Data rebuilt from Eurostat SDMX API (fix_rebuild_gva_xlsx.py)"),
        meta("Dataset:", "Gross value added at basic prices by NUTS 3 region [nama_10r_3gva]"),
        meta("Source:", "ESTAT dissemination API, A.CP_MEUR"),
        meta("Time frequency", "Annual"),
        meta("Unit of measure", "Current prices, million euro"),
        meta("Statistical classification of economic activities in the European Community (NACE Rev. 2)", label),
        [np.nan] * n_cols,
        [np.nan] * n_cols,
    ]
    header = [np.nan] * n_cols
    header[0] = "TIME"
    for k, y in enumerate(YEARS_GVA):
        header[1 + 2 * k] = y
    rows.append(header)
    rows.append(meta("GEO (Labels)"))

    sub = df[df["nace_r2"] == nace]
    for geo in ("DE", "DEE07"):
        row = [np.nan] * n_cols
        row[0] = GEO_LABELS[geo]
        g = sub[sub["geo"] == geo]
        for k, y in enumerate(YEARS_GVA):
            rec = g[g["TIME_PERIOD"] == y]
            if len(rec) == 1 and pd.notna(rec["OBS_VALUE"].iloc[0]):
                row[1 + 2 * k] = float(rec["OBS_VALUE"].iloc[0])
                flag = rec["OBS_FLAG"].iloc[0]
                if pd.notna(flag):
                    row[2 + 2 * k] = str(flag)
            else:
                row[1 + 2 * k] = ":"
        rows.append(row)
    return pd.DataFrame(rows)


def rebuild_gva_xlsx(context: CountryPipelineContext, force: bool = False) -> Path:
    """SDMX CSV → 001 期望布局的 xlsx;2022 参照值比对仅打印提示(适配登记见 docstring)。"""
    out = context.derived_root / "nama_10r_3gva.xlsx"
    if out.is_file() and out.stat().st_size > 0 and not force:
        print(f"[跳过] {out.name} 已存在")
        return out
    df = pd.read_csv(_require(context, "eurostat/nama_10r_3gva_sdmx.csv"))

    n_diff = 0
    for (nace, geo), expected in GVA_ANCHORS.items():
        rec = df[(df["nace_r2"] == nace) & (df["geo"] == geo) & (df["TIME_PERIOD"] == 2022)]
        got = float(rec["OBS_VALUE"].iloc[0]) if len(rec) == 1 else None
        if got is None or abs(got - expected) >= 0.01:
            n_diff += 1
            print(f"  △ GVA {nace} {geo}:本次 {got} ≠ 论文 {expected}(Eurostat 可能已修订;不拦截)")
    if n_diff == 0:
        print("  ✓ GVA 2022 七组参照值与论文一致")

    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame([["Rebuilt from Eurostat SDMX API — see fetcher.derive.de"]]).to_excel(
            writer, sheet_name="Summary", header=False, index=False)
        for sheet_name, nace, label in SHEET_NACE:
            _build_sheet_frame(df, nace, label).to_excel(
                writer, sheet_name=sheet_name, header=False, index=False)
    print(f"[重建] {out.name}")
    return out


def _parse_gva_sheet(xlsx_path: Path, sheet_name: str,
                     region_name: str = "Börde", year_col_idx: int = 15):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    region_val = germany_val = None
    for _, row in df.iterrows():
        geo_label = str(row.iloc[0])
        if geo_label == "Germany":
            val = row.iloc[year_col_idx]
            germany_val = float(val) if str(val) != ":" and pd.notna(val) else None
        if region_name in geo_label:
            val = row.iloc[year_col_idx]
            region_val = float(val) if str(val) != ":" and pd.notna(val) else None
    return region_val, germany_val


# ---------------------------------------------------------------------------
# Börde 区域与变电站
# ---------------------------------------------------------------------------

def derive_de(context: CountryPipelineContext, force: bool = False) -> list[Path]:
    """派生 DE 三件产物;幂等(全部产物已存在且非 force 则跳过)。"""
    outs = [context.derived_root / n for n in OUTPUTS]
    if not force and all(p.is_file() for p in outs):
        print(f"派生产物已存在,跳过(force=True 可重算):{', '.join(OUTPUTS)}")
        return outs

    gva_xlsx = rebuild_gva_xlsx(context, force=force)

    # —— 1. Gemeinden(source 区域):质心 within Kreis 过滤 GeoServer 脏数据 ——
    gemeinden_raw = gpd.read_file(_require(context, "geoserver/Postgres_v_lkb_vg250_gem_sql.geojson"))
    kreis = gpd.read_file(_require(context, "geoserver/Postgres_v_lkb_vg250_krs_sql.geojson"))
    kreis_pop = kreis.iloc[0]["Einwohner"]

    gemeinden_raw["_centroid"] = gemeinden_raw.geometry.centroid
    centroids_gdf = gemeinden_raw.set_geometry("_centroid")
    within_mask = centroids_gdf.within(kreis.geometry.iloc[0])
    removed = gemeinden_raw.loc[~within_mask, "Name"].tolist()
    gemeinden = gemeinden_raw[within_mask].drop(columns=["_centroid"]).copy()
    print(f"Gemeinden:{len(gemeinden_raw)} → {len(gemeinden)}(移除不在 Börde 内:{removed})")

    # —— 2. 人口:缺失者按面积比例从 Kreis 剩余人口分配;已知超总量则填 0 ——
    gem_proj = gemeinden.to_crs("EPSG:25832")
    gemeinden["area_m2"] = gem_proj.geometry.area
    known_mask = gemeinden["Einwohner"].notna()
    known_pop = gemeinden.loc[known_mask, "Einwohner"].sum()
    if known_pop >= kreis_pop:
        print(f"已知人口({known_pop:,.0f})≥ Kreis 总人口({kreis_pop:,.0f}):缺失填 0")
        gemeinden["Einwohner"] = gemeinden["Einwohner"].fillna(0)
    else:
        remaining_pop = kreis_pop - known_pop
        missing_area = gemeinden.loc[~known_mask, "area_m2"].sum()
        gemeinden.loc[~known_mask, "Einwohner"] = (
            gemeinden.loc[~known_mask, "area_m2"] / missing_area * remaining_pop)
        print(f"按面积分配剩余人口 {remaining_pop:,.0f} 到 {(~known_mask).sum()} 个 Gemeinden")
    gemeinden["population"] = gemeinden["Einwohner"]
    gemeinden["residential_percent"] = gemeinden["population"] / GERMANY_POP_2022

    # —— 3. GVA(NACE → 四扇区,对齐 UK SIC07 分类;NUTS3 整体值全体共享) ——
    g = {key: _parse_gva_sheet(gva_xlsx, sheet)
         for key, sheet in [("total", "Sheet 1"), ("agriculture", "Sheet 2"),
                            ("industry", "Sheet 3"), ("construction", "Sheet 5"),
                            ("retail", "Sheet 6"), ("finance", "Sheet 9")]}
    agricultural_gva = g["agriculture"][0]
    industrial_gva = g["industry"][0] + g["construction"][0]
    commercial_gva = g["retail"][0] + g["finance"][0]
    others_gva = g["total"][0] - agricultural_gva - industrial_gva - commercial_gva
    agricultural_gva_de = g["agriculture"][1]
    industrial_gva_de = g["industry"][1] + g["construction"][1]
    commercial_gva_de = g["retail"][1] + g["finance"][1]
    others_gva_de = g["total"][1] - agricultural_gva_de - industrial_gva_de - commercial_gva_de

    gemeinden["agricultural_gva"] = agricultural_gva
    gemeinden["industrial_gva"] = industrial_gva
    gemeinden["commercial_gva"] = commercial_gva
    gemeinden["others_gva"] = others_gva
    gemeinden["agricultural_percent"] = agricultural_gva / agricultural_gva_de
    gemeinden["industrial_percent"] = industrial_gva / industrial_gva_de
    gemeinden["commercial_percent"] = commercial_gva / commercial_gva_de
    gemeinden["others_percent"] = others_gva / others_gva_de

    # —— 4. 面积占比 + NUTS3 标识 ——
    gemeinden["area_percent"] = gemeinden["area_m2"] / gemeinden["area_m2"].sum()
    gemeinden["NUTS3"] = BOERDE_NUTS3

    # —— 5. 变电站:负荷图层裁 Börde、强转数值、筛零负荷、Gemeinde 归属 ——
    uw_last_all = gpd.read_file(_require(context, "geoserver/Postgres_v_epa_uw_last.geojson"))
    uw_last_boerde = gpd.sjoin(uw_last_all, kreis[["geometry"]], how="inner",
                               predicate="within").drop(columns=["index_right"])
    uw_last_boerde["Last aktuell in MW"] = pd.to_numeric(
        uw_last_boerde["Last aktuell in MW"], errors="coerce").fillna(0.0)
    print(f"Börde 内含负荷 UW:{len(uw_last_boerde)},"
          f"非零 {(uw_last_boerde['Last aktuell in MW'] > 0).sum()},"
          f"总负荷 {uw_last_boerde['Last aktuell in MW'].sum():.1f} MW")

    substations = uw_last_boerde[["Kennzeichen", "Name", "Last aktuell in MW", "geometry"]].copy()
    substations = substations.rename(columns={"Last aktuell in MW": "p_mw"})
    substations = substations[substations["p_mw"] > 0].copy()

    subs_with_gem = gpd.sjoin(substations, gemeinden[["Name", "geometry"]],
                              how="left", predicate="within")
    substations["Gemeinde"] = subs_with_gem["Name_right"].values
    substations["NUTS3"] = BOERDE_NUTS3
    for idx in substations[substations["Gemeinde"].isna()].index:
        pt = substations.loc[idx, "geometry"]
        nearest_idx = gemeinden.geometry.distance(pt).idxmin()  # 边界外站最近邻兜底
        substations.loc[idx, "Gemeinde"] = gemeinden.loc[nearest_idx, "Name"]

    # —— 6. 保存 ——
    source_cols = ["geometry", "Name", "NUTS3", "population", "residential_percent",
                   "agricultural_gva", "industrial_gva", "commercial_gva", "others_gva",
                   "agricultural_percent", "industrial_percent", "commercial_percent",
                   "others_percent", "area_m2", "area_percent"]
    source_regions = gemeinden[source_cols].copy().reset_index(drop=True)
    substations_out = substations[["Kennzeichen", "Name", "p_mw", "Gemeinde",
                                   "NUTS3", "geometry"]].copy().reset_index(drop=True)

    context.derived_root.mkdir(parents=True, exist_ok=True)
    source_regions.to_file(context.derived_root / "source_regions.gpkg", driver="GPKG")
    substations_out.to_file(context.derived_root / "substations.gpkg", driver="GPKG")
    print(f"已写出:source_regions.gpkg({len(source_regions)} Gemeinden)、"
          f"substations.gpkg({len(substations_out)} UW,"
          f"合计 {substations_out.p_mw.sum():.1f} MW)")
    return outs


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
