import osmnx as ox
import pickle
import os

def get_osm_data(bounds, path = None, tags=None):

    if tags is None:
        tags = {'landuse':True}
    # 获取边界值
    minx, miny, maxx, maxy = bounds

    # 按照新的顺序重新组织边界值: (left, bottom, right, top)
    # landuse_gdf = ox.features.features_from_bbox(
    #     north=maxy,  # top
    #     south=miny,  # bottom
    #     east=maxx,  # right
    #     west=minx,  # left
    #     tags=tags
    # )
    landuse_gdf = ox.features.features_from_bbox(
        bbox=(minx, miny, maxx, maxy),  # (west, south, east, north)
        tags=tags
    )

    if landuse_gdf.empty:
        print("在指定区域未找到landuse数据")

    if path is not None:
        with open(path, "wb") as f:
            pickle.dump(landuse_gdf, f)

    return landuse_gdf
