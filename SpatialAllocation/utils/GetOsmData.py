import osmnx as ox
import pickle
import os

def get_osm_data(bounds, path = None, tags=None):

    if tags is None:
        osm_tags = {'landuse':True}
    else:
        osm_tags = tags


    # 获取边界值
    minx, miny, maxx, maxy = bounds


    landuse_gdf = ox.features.features_from_bbox(
        bbox=(minx, miny, maxx, maxy),  # (west, south, east, north)
        tags=osm_tags
    )

    if landuse_gdf.empty:
        print("在指定区域未找到landuse数据")

    if path is not None:
        with open(path, "wb") as f:
            pickle.dump(landuse_gdf, f)

    return landuse_gdf
