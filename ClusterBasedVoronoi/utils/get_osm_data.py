import osmnx as ox
import pickle
import os

def get_osm_data(bounds, path = None, tags=None):

    if tags is None:
        tags = {'landuse':True}
    # Get boundary values
    minx, miny, maxx, maxy = bounds

    # Reorganize boundary values in new order: (left, bottom, right, top)
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
        print("No landuse data found in the specified area")

    if path is not None:
        with open(path, "wb") as f:
            pickle.dump(landuse_gdf, f)

    return landuse_gdf