"""从 AU 001 派生件构建 34 个 SA3 与 12 个研究区域。"""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xlrd


from ...config import CountryPipelineContext


CENSUS_MEMBERS = [
    "2011 Census BCP Statistical Areas Level 3 for NSW/NSW/"
    f"2011Census_B43{part}_NSW_SA3_short.csv"
    for part in ("A", "B", "C", "D")
]

USABLE_STATUSES = {"matched", "matched_osm"}
SECTOR_GROUPS = ("industrial", "commercial", "agricultural", "others")
ANZSIC_ABBR_TO_GROUP = {
    "Ag_For_Fshg": "agricultural",
    "Mining": "agricultural",
    "Manufact": "industrial",
    "El_Gas_Wt_Waste": "industrial",
    "Constru": "industrial",
    "WhlesaleTde": "commercial",
    "RetTde": "commercial",
    "Accom_food": "commercial",
    "Trans_post_wrehsg": "commercial",
    "Info_media_teleco": "commercial",
    "Fin_Insur": "commercial",
    "RtnHir_REst": "commercial",
    "Pro_scien_tec": "others",
    "Admin_supp": "others",
    "Public_admin_sfty": "others",
    "Educ_trng": "others",
    "HlthCare_SocAs": "others",
    "Art_recn": "others",
    "Oth_scs": "others",
    "ID_NS": "others",
}


def _usable_stations(context: CountryPipelineContext) -> gpd.GeoDataFrame:
    stations = pd.read_csv((context.derived_root / "station_table_fy2009.csv"), encoding="utf-8-sig")
    usable = stations.loc[stations["status"].isin(USABLE_STATUSES)].copy()
    if len(usable) != 143:
        raise RuntimeError(f"FY2009 usable station 数从 143 变为 {len(usable)}")
    return gpd.GeoDataFrame(
        usable,
        geometry=gpd.points_from_xy(usable["lon_wgs84"], usable["lat_wgs84"]),
        crs="EPSG:4326",
    )


def _footprint(context: CountryPipelineContext, stations: gpd.GeoDataFrame):
    sa3 = gpd.read_file((context.derived_root / "asgs_nsw_sa2_sa3_sa4_gda2020.gpkg"), layer="sa3_2021_nsw").to_crs("EPSG:4326")
    joined = gpd.sjoin(stations, sa3, how="left", predicate="within")
    if joined["SA3_CODE21"].isna().any():
        raise RuntimeError("可用站存在无法归入 NSW SA3 的坐标")
    codes = sorted(joined["SA3_CODE21"].unique())
    footprint = sa3.loc[sa3["SA3_CODE21"].isin(codes)].copy()
    if len(footprint) != 34 or footprint["SA4_CODE21"].nunique() != 12:
        raise RuntimeError("AU 研究范围不再是 34 SA3 / 12 SA4")
    return footprint, joined


def _population(context: CountryPipelineContext, codes: list[str]) -> pd.Series:
    import openpyxl

    workbook = openpyxl.load_workbook((context.raw_root / "abs_population/32180DS0003_2001-25.xlsx"), read_only=True)
    sheet = workbook["Table 2"]
    rows = list(sheet.iter_rows(values_only=True))
    header_idx = next(i for i, row in enumerate(rows) if row and row[0] == "S/T code")
    header = list(rows[header_idx])
    years = list(rows[header_idx - 1])
    code_col = header.index("SA3 code")
    year_col = next(
        i for i, value in enumerate(years) if str(value).strip() in {"2009", "2009.0"}
    )
    data = {}
    for row in rows[header_idx + 1 :]:
        if not row or row[code_col] is None:
            continue
        code = str(row[code_col]).strip()
        if code in codes:
            data[code] = float(row[year_col])
    if set(data) != set(codes):
        raise RuntimeError(f"人口表缺少 SA3: {sorted(set(codes) - set(data))}")
    return pd.Series(data, name="population_erp_2009")


def _income(context: CountryPipelineContext, footprint: gpd.GeoDataFrame) -> pd.DataFrame:
    workbook = xlrd.open_workbook((context.raw_root / "abs_income/6524055002do004_200506201011.xls"))
    sheet = workbook.sheet_by_name("Table_8")
    rows = []
    for row_idx in range(7, sheet.nrows):
        value = sheet.cell_value(row_idx, 0)
        if isinstance(value, float) and 10000 <= value < 20000:
            rows.append(
                {
                    "sa3_code": str(int(value)),
                    "sa3_name_2011": str(sheet.cell_value(row_idx, 1)).strip(),
                    "income_earners_2009": sheet.cell_value(row_idx, 95),
                    "income_total_aud_2009": sheet.cell_value(row_idx, 101),
                    "income_mean_aud_2009": sheet.cell_value(row_idx, 107),
                }
            )
    income = pd.DataFrame(rows).set_index("sa3_code")
    names = footprint.set_index("SA3_CODE21")["SA3_NAME21"].str.strip()
    missing = [code for code in names.index if code not in income.index]
    mismatched = [
        code
        for code in names.index
        if code in income.index and income.loc[code, "sa3_name_2011"] != names[code]
    ]
    if missing or mismatched:
        raise RuntimeError(
            f"2011→2021 SA3 稳定性失败: missing={missing}, names={mismatched}"
        )
    output = income.loc[list(names.index)].copy()
    for column in (
        "income_earners_2009",
        "income_total_aud_2009",
        "income_mean_aud_2009",
    ):
        output[column] = pd.to_numeric(output[column], errors="raise")
    return output


def _industry(context: CountryPipelineContext, codes: list[str]) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile((context.raw_root / "census2011/2011_BCP_SA3_for_NSW_short-header.zip")) as archive:
        for member in CENSUS_MEMBERS:
            with archive.open(member) as stream:
                frames.append(pd.read_csv(stream).set_index("region_id"))
    table = pd.concat(frames, axis=1)
    table.index = table.index.astype(str)
    output = pd.DataFrame(index=table.index)
    for group in SECTOR_GROUPS:
        output[f"emp_2011_{group}"] = 0.0
    grand_total = None
    for column in (
        name for name in table.columns if name.startswith("P_") and name.endswith("_Tot")
    ):
        abbreviation = column[len("P_") : -len("_Tot")]
        if abbreviation == "Tot":
            grand_total = column
            continue
        group = ANZSIC_ABBR_TO_GROUP.get(abbreviation)
        if group is None:
            raise RuntimeError(f"未知 ANZSIC 行业缩写: {abbreviation}")
        output[f"emp_2011_{group}"] += table[column]
    output["emp_2011_total"] = output[
        [f"emp_2011_{group}" for group in SECTOR_GROUPS]
    ].sum(axis=1)
    if grand_total is None or not np.array_equal(output["emp_2011_total"], table[grand_total]):
        raise RuntimeError("AU 行业就业分组不守恒")
    result = output.loc[codes].copy()
    for group in SECTOR_GROUPS:
        result[f"emp_share_{group}"] = (
            result[f"emp_2011_{group}"] / result["emp_2011_total"]
        )
    return result


def derive_au_regions(context: CountryPipelineContext, force: bool = False) -> list[Path]:
    """构建 source/evaluation regions 与 SA3 属性表。"""

    outputs = [
        context.derived_root / "regions_sa3.gpkg",
        context.derived_root / "regions_sa4.gpkg",
        context.derived_root / "region_attributes.csv",
    ]
    if not force and all(path.is_file() for path in outputs):
        return outputs
    stations = _usable_stations(context)
    footprint, joined = _footprint(context, stations)
    codes = sorted(footprint["SA3_CODE21"].astype(str))
    demand = joined.groupby("SA3_CODE21").agg(
        n_stations_usable=("station", "count"),
        demand_peak_mw=("peak_mw", "sum"),
        demand_energy_gwh=("energy_gwh", "sum"),
    )
    population = _population(context, codes)
    income = _income(context, footprint)
    industry = _industry(context, codes)
    attrs = footprint.set_index("SA3_CODE21")[[
        "SA3_NAME21",
        "SA4_CODE21",
        "SA4_NAME21",
    ]].copy()
    attrs.columns = ["sa3_name", "sa4_code", "sa4_name"]
    attrs["loc_key"] = attrs["sa4_name"].str.replace(r"[^0-9A-Za-z]+", "_", regex=True).str.strip("_")
    attrs = attrs.join([demand, population, income.drop(columns=["sa3_name_2011"]), industry])
    attrs["residential_percent"] = (
        attrs["population_erp_2009"] / attrs["population_erp_2009"].sum()
    )
    for group in SECTOR_GROUPS:
        value = attrs["income_total_aud_2009"] * attrs[f"emp_share_{group}"]
        attrs[f"sector_value_{group}"] = value
        attrs[f"{group}_percent"] = value / value.sum()
    percent_columns = ["residential_percent", *[f"{group}_percent" for group in SECTOR_GROUPS]]
    if not np.allclose(attrs[percent_columns].sum(axis=0), 1.0):
        raise RuntimeError("AU region percent 未归一化")

    attrs_out = attrs.reset_index().rename(columns={"SA3_CODE21": "sa3_code"})
    attrs_out.to_csv(outputs[2], index=False, encoding="utf-8-sig")
    sa3 = footprint.merge(attrs_out, left_on="SA3_CODE21", right_on="sa3_code")
    sa3 = sa3[[
        "sa3_code",
        "sa3_name",
        "sa4_code",
        "sa4_name",
        "loc_key",
        "n_stations_usable",
        "demand_peak_mw",
        "demand_energy_gwh",
        "population_erp_2009",
        "income_total_aud_2009",
        *percent_columns,
        "geometry",
    ]].copy()
    sa3["SA3"] = sa3["sa3_code"]
    sa3["SA4"] = sa3["sa4_code"]
    for path in outputs[:2]:
        path.unlink(missing_ok=True)
    sa3.to_file(outputs[0], layer="regions_sa3", driver="GPKG")
    sa4 = sa3.dissolve(
        by="sa4_code",
        aggfunc={
            "sa4_name": "first",
            "loc_key": "first",
            "n_stations_usable": "sum",
            "demand_peak_mw": "sum",
            "demand_energy_gwh": "sum",
            "population_erp_2009": "sum",
            "sa3_code": "count",
        },
    ).rename(columns={"sa3_code": "n_sa3"}).reset_index()
    sa4.to_file(outputs[1], layer="regions_sa4", driver="GPKG")
    print(f"AU regions: {len(sa3)} SA3 / {len(sa4)} SA4")
    return outputs
