import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
import matplotlib.pyplot as plt


def calculate_adjacency_matrix(gdf):
    """计算GeoDataFrame中多边形的邻接矩阵"""
    n = len(gdf)
    adjacency_matrix = np.zeros((n, n), dtype=bool)

    # 为提高效率，先创建空间索引
    sindex = gdf.sindex

    for i, geom in enumerate(gdf.geometry):
        # 获取潜在的相邻多边形
        possible_matches_index = list(sindex.intersection(geom.bounds))
        possible_matches = gdf.iloc[possible_matches_index]

        # 筛选出真正相邻的多边形
        for j, possible_match in enumerate(possible_matches_index):
            if i != possible_match:  # 不计算自身
                if geom.touches(possible_matches.iloc[j].geometry):
                    adjacency_matrix[i, possible_match] = True

    return adjacency_matrix


def find_merge_pairs(adjacency_matrix):
    """
    找出最佳的两两合并对

    参数:
    adjacency_matrix: 邻接矩阵

    返回:
    merge_pairs: 要合并的区域对列表 [(i1,j1), (i2,j2), ...]
    """
    n = len(adjacency_matrix)
    remaining = set(range(n))
    pairs = []

    # 对每个区域，找到一个相邻区域进行配对
    while len(remaining) >= 2:
        found_pair = False

        # 从剩余区域中取第一个
        if not remaining:
            break

        i = min(remaining)
        remaining.remove(i)

        # 查找i的所有相邻区域
        neighbors = [j for j in remaining if adjacency_matrix[i, j]]

        if neighbors:
            # 选择第一个相邻区域配对
            j = neighbors[0]
            pairs.append((i, j))
            remaining.remove(j)
            found_pair = True

        # 如果没有找到相邻区域，尝试与任何剩余区域配对（即使不相邻）
        if not found_pair and remaining:
            j = min(remaining)
            pairs.append((i, j))
            remaining.remove(j)

    # 如果剩余奇数个区域，最后一个单独保留
    if remaining:
        pairs.append((list(remaining)[0],))

    return pairs


def merge_by_pairs(gdf, id_column='id', attribute_operations=None):
    """
    两两合并区域

    参数:
    gdf: 包含区域的GeoDataFrame
    id_column: 区域ID列名
    attribute_operations: 字典，键为列名，值为合并操作('mean', 'sum', 'min', 'max', 'first', 'last', 'count')

    返回:
    合并后的新GeoDataFrame
    """
    if len(gdf) <= 1:
        return gdf.copy()

    # 计算邻接矩阵
    adjacency_matrix = calculate_adjacency_matrix(gdf)

    # 找出要合并的区域对
    merge_pairs = find_merge_pairs(adjacency_matrix)

    # 创建新的合并区域
    new_geometries = []
    new_ids = []

    # 为每个属性准备数据
    attribute_data = {}
    if attribute_operations:
        for col, operation in attribute_operations.items():
            if col in gdf.columns and col != 'geometry' and col != id_column:
                attribute_data[col] = []

    for i, pair in enumerate(merge_pairs):
        if len(pair) == 2:
            # 合并两个区域
            i1, i2 = pair
            geom1 = gdf.iloc[i1].geometry
            geom2 = gdf.iloc[i2].geometry
            merged_geom = unary_union([geom1, geom2])

            # 处理属性
            if attribute_operations:
                for col, operation in attribute_operations.items():
                    if col in gdf.columns and col != 'geometry' and col != id_column:
                        values = [gdf.iloc[i1][col], gdf.iloc[i2][col]]

                        # 执行指定的操作
                        if operation == 'mean':
                            attribute_data[col].append(np.mean(values))
                        elif operation == 'sum':
                            attribute_data[col].append(sum(values))
                        elif operation == 'min':
                            attribute_data[col].append(min(values))
                        elif operation == 'max':
                            attribute_data[col].append(max(values))
                        elif operation == 'first':
                            attribute_data[col].append(values[0])
                        elif operation == 'last':
                            attribute_data[col].append(values[1])
                        elif operation == 'count':
                            attribute_data[col].append(len(values))
                        else:  # 默认为平均值
                            attribute_data[col].append(np.mean(values))
        else:
            # 单个区域保持不变
            i1 = pair[0]
            merged_geom = gdf.iloc[i1].geometry

            # 处理属性
            if attribute_operations:
                for col, operation in attribute_operations.items():
                    if col in gdf.columns and col != 'geometry' and col != id_column:
                        attribute_data[col].append(gdf.iloc[i1][col])

        new_geometries.append(merged_geom)
        new_ids.append(i + 1)  # 从1开始的新ID

    # 创建数据字典
    data_dict = {
        id_column: new_ids,
        'geometry': new_geometries
    }

    # 添加属性数据
    for col, values in attribute_data.items():
        data_dict[col] = values

    # 创建新的GeoDataFrame
    new_gdf = gpd.GeoDataFrame(data_dict, crs=gdf.crs)

    return new_gdf


def create_hierarchical_regions(gdf, target_region_count, id_column='id', attribute_operations=None):
    """
    创建分层区域合并，通过两两合并的方式

    参数:
    gdf: 原始GeoDataFrame
    target_region_count: 最终目标区域数量（例如1）
    id_column: 区域ID列名
    attribute_operations: 字典，键为列名，值为合并操作('mean', 'sum', 'min', 'max', 'first', 'last', 'count')
        例如: {'population': 'sum', 'income': 'mean', 'area_code': 'first'}

    返回:
    list: 含有原始GDF和各个层级合并结果的GeoDataFrame列表，
          按照区域数量从多到少排序
    """
    # 确保输入的GDF有连续的ID，从1开始
    original_gdf = gdf.copy()
    original_gdf[id_column] = range(1, len(original_gdf) + 1)

    # 创建结果列表，首先添加原始GDF
    results_list = [original_gdf]

    # 当前工作的GDF
    current_gdf = original_gdf.copy()

    # 持续合并直到达到或低于目标数量
    while len(current_gdf) > target_region_count:
        # 两两合并，传递属性操作参数
        current_gdf = merge_by_pairs(current_gdf, id_column, attribute_operations)

        # 如果没有进一步减少（例如只剩一个区域），则停止
        if len(current_gdf) == len(results_list[-1]):
            break

        # 添加到结果列表
        results_list.append(current_gdf.copy())

        # 如果已经达到目标数量，停止合并
        if len(current_gdf) <= target_region_count:
            break

    return results_list