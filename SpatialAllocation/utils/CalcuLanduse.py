from shapely.ops import unary_union
from shapely.geometry import Point, Polygon
import geopandas as gpd
from .GetOsmData import get_osm_data
import pandas as pd

def fetch_landuse_data(grid_gdf, polygons, step_size_m):
    """
    获取土地利用数据并计算每个网格单元的土地利用类型占比。

    参数:
    - polygons: GeoDataFrame，包含边界多边形。
    - step_size_m: 网格步长（米）。

    返回:
    - landuse_gdf: GeoDataFrame，包含土地利用数据。
    """
    # Assuming landuse_gdf is defined globally or passed as an argument
    boundary_polygon = polygons.copy().to_crs("EPSG:4326")
    boundary_polygon = unary_union(boundary_polygon["geometry"])
    landuse_gdf = get_osm_data(boundary_polygon.bounds)
    landuse_gdf = landuse_gdf.to_crs(epsg=4326)
    landuse_slim = landuse_gdf[['landuse', 'geometry']]
    grid_gdf = gpd.sjoin(grid_gdf.to_crs("EPSG:4326"), landuse_slim, how="left", predicate="within") # only for points geometry
    grid_gdf = grid_gdf[~grid_gdf.index.duplicated(keep='first')]
    # grid_gdf['landuse'] = grid_gdf['landuse'].fillna('others')
    grid_gdf['landuse'] = grid_gdf['landuse'].apply(get_category)

    return grid_gdf, [], {"landuse": grid_gdf["landuse"].unique().tolist()}

def calculate_landuse_proportions(grid_gdf, polygons, step_size_m):
    """
    计算每个网格单元的土地利用类型占比。

    参数:
    - grid_gdf: GeoDataFrame，包含网格几何对象。
    - landuse_gdf: GeoDataFrame，包含土地利用数据。
    - step_size_m: 网格步长（米）。

    返回:
    - grid_gdf: 更新后的GeoDataFrame，包含土地利用类型占比列。
    """
    # 将网格点转换为网格单元（正方形多边形）
    # 4a. 将网格点转换为网格单元（正方形多边形）
    half_step = step_size_m / 2
    grid_gdf['geometry'] = grid_gdf['geometry'].apply(
        lambda pt: Polygon([
            (pt.x - half_step, pt.y - half_step),
            (pt.x + half_step, pt.y - half_step),
            (pt.x + half_step, pt.y + half_step),
            (pt.x - half_step, pt.y + half_step)
        ])
    )
    # 为每个网格单元添加唯一ID，方便后续聚合
    grid_gdf['grid_id'] = grid_gdf.index

    # 4b. 获取OSM土地利用数据
    # 边界需要转换为EPSG:4326以匹配OSM API的常用坐标系
    boundary_polygon_4326 = unary_union(polygons.to_crs("EPSG:4326")["geometry"])
    landuse_gdf = get_osm_data(boundary_polygon_4326.bounds)

    # 确保landuse_gdf有内容
    if landuse_gdf.empty:
        raise ValueError("Cannot fetch landuse data. The returned GeoDataFrame is empty.")

    # 4c. 准备土地利用数据：选择所需列并转换CRS以进行空间计算
    landuse_gdf = landuse_gdf[['landuse', 'geometry']].dropna(subset=['landuse'])
    landuse_gdf['landuse'] = landuse_gdf['landuse'].apply(get_category)
    landuse_gdf_m = landuse_gdf.to_crs("EPSG:3857")

    # [FIX] 过滤土地利用数据，只保留多边形（Polygon/MultiPolygon）类型
    # OSM数据常常混合了点、线、面等多种几何类型，而overlay操作需要单一类型的数据
    landuse_gdf_m = landuse_gdf_m[landuse_gdf_m.geom_type.isin(['Polygon', 'MultiPolygon'])]

    # 4d. 计算网格单元与土地利用多边形的交集
    # 增加判断，如果过滤后landuse_gdf_m为空，则直接返回grid_gdf
    if landuse_gdf_m.empty:
        raise ValueError("Cannot calculate landuse proportions. The returned GeoDataFrame is empty.")

    intersection_gdf = gpd.overlay(grid_gdf, landuse_gdf_m, how='intersection')

    # 4e. 计算交集面积并聚合
    intersection_gdf['intersection_area'] = intersection_gdf.geometry.area

    # 使用pivot_table来聚合每个网格单元中每种土地利用类型的总面积
    landuse_areas = intersection_gdf.pivot_table(
        index='grid_id',
        columns='landuse',
        values='intersection_area',
        aggfunc='sum'
    ).fillna(0)

    # 4f. 计算占比
    cell_area = step_size_m ** 2
    landuse_proportions = landuse_areas / cell_area
    # 为了方便阅读，可以给列名加上前缀
    landuse_proportions.columns = [f'lu_{col}_prop' for col in landuse_proportions.columns]

    # 4g. 将计算出的占比合并回原始网格GeoDataFrame
    grid_gdf = grid_gdf.merge(landuse_proportions, on='grid_id', how='left').fillna(0)

    return grid_gdf, landuse_proportions.columns, {}


def get_category(landuse_value):

    landuse_categories = {
        "industrial": ["industrial", "quarry", "construction", "landfill", "yard", "storage", "demolition", "concrete",
                       "logistics", "power", "logging", "water_wellfield", "maintenance_work"],
        "commercial": ["depot", "commercial", "retail", "hospitality", "tourism", "commerce"],
        "agricultural": ["farm", "animal_keeping", "meadow", "allotments", "plant_nursery", "farmland", "farmyard",
                         "orchard", "greenhouse_horticulture", "vineyard", "apiary", "animal_enclosure",
                         "community_food_growing", "aquaculture", "paddock"],
        "residential": ["residential", "retail;residential", "driveway"],
        "others": ["military", "recreation_ground", "brownfield", "cemetery", "grass", "village_green", "railway",
                   "forest",
                   "conservation", "religious", "observatory", "garages", "traffic_island", "education", "basin",
                   "healthcare", "nursery", "flowerbed", "churchyard", "platform", "civic", "sport", "mixed", "road",
                   "greenfield", "teaching_area", "arboretum", "fire_station", "governmental", "runway", "university",
                   "shrubs", "institutional", "greenery", "courtyard", "civic_admin", "unclear", "planting", "leisure",
                   "trees", "playing_fields", "transport", "shelter", "fairground", "winter_sports", "piste", "exercise_area",
                   "highway", "reservoir", "garden", "proposed", "Insect reserve"]}

    if pd.isna(landuse_value):
        return 'others'

    for category, types in landuse_categories.items():
        if landuse_value in types:
            return category

    print(f"Landuse value '{landuse_value}' not found in predefined categories. Assigning to 'others'.")
    return 'others'
