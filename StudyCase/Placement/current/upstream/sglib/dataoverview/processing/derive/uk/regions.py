from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ...config import CountryPipelineContext

from ..common import atomic_geofile


LOOKUP = (
    "Local_Authority_District_(April_2021)_to_LAU1_to_ITL3_to_ITL2_to_ITL1_"
    "(January_2021)_Lookup_in_United_Kingdom.csv"
)
SIC07_CLASSIFICATION = {
    "industrial_gva": ["C (10-33)", "DE (35-39)", "F (41-43)"],
    "commercial_gva": ["G (45-47)", "H (49-53)", "I (55-56)", "J (58-63)", "K (64-66)", "L (68)"],
    "agricultural_gva": ["AB (1-9)"],
    "others_gva": ["M (69-75)", "N (77-82)", "O (84)", "P (85)", "Q (86-88)", "R (90-93)", "S (94-96)", "T (97-98)"],
}


def build(context: CountryPipelineContext) -> gpd.GeoDataFrame:
    archive = context.raw_root / "HDRah-Data-PS-GB-1f63a32"
    lad_path = archive / "Geo_data" / "LAD_DEC_2022_UK_BFC.shp"
    required = [lad_path, context.raw_root / LOOKUP, context.raw_root / "mye23tablesew.xlsx", context.raw_root / "regionalgrossvalueaddedbalancedbyindustryandallitlregions.xlsx"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"UK region inputs missing: {missing}")
    lad = gpd.read_file(lad_path, encoding="ISO-8859-1").to_crs("EPSG:4326")[["LAD22CD", "LAD22NM", "geometry"]]
    mapping = pd.read_csv(context.raw_root / LOOKUP)[["LAD21CD", "LAD21NM", "ITL321CD", "ITL221CD"]]
    lookup = mapping.drop_duplicates("LAD21CD", keep="first").set_index("LAD21CD")
    lad["ITL3"] = lad["LAD22CD"].map(lookup["ITL321CD"])
    lad["ITL2"] = lad["LAD22CD"].map(lookup["ITL221CD"])
    lad = lad.rename(columns={"LAD22CD": "LAD", "LAD22NM": "LADName"})
    population = pd.read_excel(context.raw_root / "mye23tablesew.xlsx", sheet_name="MYE5", skiprows=7)
    population = population[["Code", "Estimated Population mid-2023"]].set_index("Code")
    lad["population"] = lad["LAD"].map(population["Estimated Population mid-2023"])
    remove = set(lad.loc[lad["population"].isna(), "ITL2"])
    filtered = lad.loc[~lad["ITL2"].isin(remove)].copy()
    attributes = filtered.drop(columns="geometry").groupby("ITL3").agg({"population": "sum", "ITL2": "first"})
    regions = filtered[["ITL3", "geometry"]].dissolve(by="ITL3").join(attributes)
    regions["residential_percent"] = regions["population"] / regions["population"].sum()
    raw_gva = pd.read_excel(context.raw_root / "regionalgrossvalueaddedbalancedbyindustryandallitlregions.xlsx", sheet_name="Table 3c", skiprows=1)
    pivot = raw_gva.pivot_table(index="ITL code", columns="SIC07 code", values="2022", aggfunc="sum")
    gva = pd.DataFrame(index=pivot.index)
    for category, codes in SIC07_CLASSIFICATION.items():
        gva[category] = pivot[[code for code in codes if code in pivot]].sum(axis=1)
        gva[category.replace("_gva", "_percent")] = gva[category] / gva[category].sum()
    regions = regions.merge(gva, left_index=True, right_index=True, how="left").reset_index()
    metric = regions.to_crs("EPSG:27700")
    regions["area"] = metric.geometry.area.to_numpy()
    regions["area_percent"] = regions["area"] / regions.groupby("ITL2")["area"].transform("sum")
    regions["area_crs"] = "EPSG:27700"
    return regions.to_crs("EPSG:4326")


def run(context: CountryPipelineContext, *, force: bool = False) -> Path:
    output = (context.derived_root / "bplus") / "regions.gpkg"
    if output.is_file() and not force:
        existing = gpd.read_file(output, rows=1)
        if "area_crs" in existing:
            return output
    return atomic_geofile(build(context), output)
