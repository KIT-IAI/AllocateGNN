from shapely.ops import unary_union
from shapely.geometry import Point, Polygon
import geopandas as gpd
from .GetOsmData import get_osm_data
import pandas as pd

def fetch_landuse_data(grid_gdf, polygons, step_size_m):
    """
    Fetch landuse data and calculate the primary landuse type for each grid cell.

    Parameters:
    - polygons: GeoDataFrame containing boundary polygons.
    - step_size_m: Grid step size in meters.

    Returns:
    - landuse_gdf: GeoDataFrame containing landuse data.
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
    Calculate the proportion of each landuse type for each grid cell.

    Parameters:
    - grid_gdf: GeoDataFrame containing grid geometry objects.
    - landuse_gdf: GeoDataFrame containing landuse data.
    - step_size_m: Grid step size in meters.

    Returns:
    - grid_gdf: Updated GeoDataFrame containing landuse type proportion columns.
    """
    # 4a. Convert grid points to grid cells (square polygons)
    half_step = step_size_m / 2
    grid_gdf['geometry'] = grid_gdf['geometry'].apply(
        lambda pt: Polygon([
            (pt.x - half_step, pt.y - half_step),
            (pt.x + half_step, pt.y - half_step),
            (pt.x + half_step, pt.y + half_step),
            (pt.x - half_step, pt.y + half_step)
        ])
    )
    # Add a unique ID to each grid cell for subsequent aggregation
    grid_gdf['grid_id'] = grid_gdf.index

    # 4b. Fetch OSM landuse data
    # Boundary needs to be converted to EPSG:4326 to match the common OSM API coordinate system
    boundary_polygon_4326 = unary_union(polygons.to_crs("EPSG:4326")["geometry"])
    landuse_gdf = get_osm_data(boundary_polygon_4326.bounds)

    # Ensure landuse_gdf has content
    if landuse_gdf.empty:
        raise ValueError("Cannot fetch landuse data. The returned GeoDataFrame is empty.")

    # 4c. Prepare landuse data: select required columns and convert CRS for spatial computation
    landuse_gdf = landuse_gdf[['landuse', 'geometry']].dropna(subset=['landuse'])
    landuse_gdf['landuse'] = landuse_gdf['landuse'].apply(get_category)
    landuse_gdf_m = landuse_gdf.to_crs("EPSG:3857")

    # [FIX] Filter landuse data to keep only Polygon/MultiPolygon types.
    # OSM data often mixes multiple geometry types (points, lines, polygons),
    # while overlay operations require single-type data.
    landuse_gdf_m = landuse_gdf_m[landuse_gdf_m.geom_type.isin(['Polygon', 'MultiPolygon'])]

    # 4d. Compute intersection of grid cells and landuse polygons
    # If filtered landuse_gdf_m is empty, return grid_gdf directly
    if landuse_gdf_m.empty:
        raise ValueError("Cannot calculate landuse proportions. The returned GeoDataFrame is empty.")

    intersection_gdf = gpd.overlay(grid_gdf, landuse_gdf_m, how='intersection')

    # 4e. Compute intersection areas and aggregate
    intersection_gdf['intersection_area'] = intersection_gdf.geometry.area

    # Use pivot_table to aggregate the total area of each landuse type per grid cell
    landuse_areas = intersection_gdf.pivot_table(
        index='grid_id',
        columns='landuse',
        values='intersection_area',
        aggfunc='sum'
    ).fillna(0)

    # 4f. Calculate proportions
    cell_area = step_size_m ** 2
    landuse_proportions = landuse_areas / cell_area
    # Add prefix to column names for readability
    landuse_proportions.columns = [f'lu_{col}_prop' for col in landuse_proportions.columns]

    # 4g. Merge calculated proportions back to the original grid GeoDataFrame
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