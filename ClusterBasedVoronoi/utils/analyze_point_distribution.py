import geopandas as gpd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


# 假设 gdf 是你的包含点数据的 GeoDataFrame
def analyze_point_distribution(gdf):
    # 确保是点几何类型
    if not all(geom.geom_type == 'Point' for geom in gdf.geometry):
        print("非点几何类型，请只包含点几何")
        return

    # 提取坐标
    coords = np.array([(geom.x, geom.y) for geom in gdf.geometry])

    # 使用KDTree计算最近邻
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)  # k=2因为每个点到自己的距离为0
    mean_distance = np.mean(distances[:, 1])  # 第二列是到最近邻的距离

    # 计算研究区域面积
    area = gdf.total_bounds[2] - gdf.total_bounds[0] * (gdf.total_bounds[3] - gdf.total_bounds[1])

    # 计算点密度
    density = len(gdf) / area

    # 计算期望的平均最近邻距离（对于随机分布）
    expected_mean_distance = 0.5 / np.sqrt(density)

    # 计算最近邻指数
    nn_index = mean_distance / expected_mean_distance

    print(f"观测到的平均最近邻距离: {mean_distance}")
    print(f"期望的平均最近邻距离（随机分布）: {expected_mean_distance}")
    print(f"最近邻指数: {nn_index}")

    # 解释结果
    if nn_index < 1:
        print("点分布呈聚集模式")
    elif nn_index > 1:
        print("点分布呈均匀模式")
    else:
        print("点分布接近随机模式")

    return nn_index