import time
import pandas as pd
from torch_geometric.data import Data
import torch
import numpy as np
import geopandas as gpd
from typing import Dict, Any, Optional, List
from torch_geometric.data import HeteroData
from .layers.positional_encoding import sinusoidal_pe
from .landuse import (
    INDEXED_LANDUSE_REPRESENTATION,
    LANDUSE_REPRESENTATIONS,
    LEGACY_DENSE_LANDUSE_REPRESENTATION,
)


def _build_source_agent_graph(gdf_s: gpd.GeoDataFrame, gdf_a: gpd.GeoDataFrame,
                              relation_column: str) -> list:
    """
    基于预先计算的从属关系构建区域-点图。

    Args:
        gdf_s (gpd.GeoDataFrame): 区域数据（polygon的centroid）
        gdf_a (gpd.GeoDataFrame): 点数据
        relation_column (str): 表示从属关系的共同索引列名
        include_region_connections (bool): 是否包含区域间的连接（可选）

    Returns:
        list: 图的边列表
    """
    graph_edges = []

    # 验证索引列是否存在
    if relation_column not in gdf_s.columns:
        raise ValueError(f"索引列 '{relation_column}' 不存在于 gdf_s 中")
    if relation_column not in gdf_a.columns:
        raise ValueError(f"索引列 '{relation_column}' 不存在于 gdf_a 中")

    # 创建索引映射
    # gdf_s的索引映射：region_id -> node_index_in_graph
    region_to_node = {region_id: i for i, region_id in enumerate(gdf_s[relation_column])}

    # gdf_a的索引映射：region_id -> [point_indices_in_graph]
    point_groups = {}
    for i, region_id in enumerate(gdf_a[relation_column]):
        if region_id not in point_groups:
            point_groups[region_id] = []
        point_groups[region_id].append(len(gdf_s) + i)  # 点节点的索引从len(gdf_s)开始

    # 1. 构建区域-点连接（基于从属关系）
    for region_id, region_node_idx in region_to_node.items():
        if region_id in point_groups:
            for point_node_idx in point_groups[region_id]:
                # 双向连接：区域 <-> 点
                graph_edges.append((region_node_idx, point_node_idx))
                graph_edges.append((point_node_idx, region_node_idx))

    return graph_edges


def preprocess_features(
        gdf: gpd.GeoDataFrame,
        numerical_col_names_all: Optional[List[str]] = None,
        categorical_col_members_all: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    自动或根据预定义的架构预处理DataFrame中的特征，将类别文字转换为数值格式。

    Args:
        gdf (gpd.GeoDataFrame): 包含混合类型特征的输入GeoDataFrame。
        numerical_col_names_all (Optional[List[str]]):
            一个包含所有期望的数值特征列名的列表。如果提供，将使用此列表代替自动检测。
        categorical_col_members_all (Optional[Dict[str, List[str]]]):
            一个字典，键是类别特征的名称，值是该特征所有可能的类别成员列表。
            如果提供，将使用此字典进行独热编码，确保特征矩阵的一致性。

    Returns:
        Dict[str, Any]: 包含以下键值对的字典：
            - 'final_features': pd.DataFrame，只包含纯数值特征的DataFrame。
            - 'mapping': Dict，特征名到列范围的映射 {feature_name: (start_col, end_col, feature_type)}。
    """
    # 1. 排除 'geometry' 列
    features_gdf = gdf.drop(columns='geometry', errors='ignore')

    # 初始化映射字典和特征列表
    mapping = {}
    feature_components = []
    current_col_idx = 0

    # =============================================================================
    # 模式选择：自动检测 vs. 预定义架构
    # =============================================================================

    # 模式一：自动检测 (如果两个新参数都未提供)
    if numerical_col_names_all is None and categorical_col_members_all is None:
        print("模式：自动检测特征...")
        # 自动识别数值列和类别列
        numerical_cols = features_gdf.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = features_gdf.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"找到 {len(numerical_cols)} 个数值列: {numerical_cols}")
        print(f"找到 {len(categorical_cols)} 个类别列: {categorical_cols}")

        # 处理数值列
        if numerical_cols:
            numerical_df = features_gdf[numerical_cols].copy()
            feature_components.append(numerical_df)
            for col in numerical_cols:
                mapping[col] = (current_col_idx, current_col_idx + 1, 'numerical')
                current_col_idx += 1

        # 处理类别列
        if categorical_cols:
            one_hot_encoded = pd.get_dummies(features_gdf[categorical_cols], prefix=categorical_cols, dtype=float)
            feature_components.append(one_hot_encoded)
            for col in categorical_cols:
                prefix = f"{col}_"
                one_hot_cols = [c for c in one_hot_encoded.columns if c.startswith(prefix)]
                num_categories = len(one_hot_cols)
                if num_categories > 0:
                    mapping[col] = (current_col_idx, current_col_idx + num_categories, 'categorical')
                    current_col_idx += num_categories

    # 模式二：使用预定义的架构
    else:
        print("模式：使用预定义的特征架构...")

        # 处理数值列
        if numerical_col_names_all:
            print(f"根据架构处理 {len(numerical_col_names_all)} 个数值特征...")
            # 创建一个符合完整架构的DataFrame
            numerical_df = pd.DataFrame(index=features_gdf.index, columns=numerical_col_names_all)

            # 找到输入gdf中实际存在的列
            cols_to_fill = [col for col in numerical_col_names_all if col in features_gdf.columns]

            # 从输入gdf中填充数据
            if cols_to_fill:
                numerical_df[cols_to_fill] = features_gdf[cols_to_fill]

            # 将所有缺失值（即gdf中不存在的列）填充为0
            numerical_df = numerical_df.fillna(0).astype(float)

            feature_components.append(numerical_df)
            for col in numerical_col_names_all:
                mapping[col] = (current_col_idx, current_col_idx + 1, 'numerical')
                current_col_idx += 1

        # 处理类别列
        if categorical_col_members_all:
            print(f"根据架构处理 {len(categorical_col_members_all)} 个类别特征...")
            all_categorical_dfs = []
            for feature_name, all_categories in categorical_col_members_all.items():
                # 检查原始类别列是否存在于输入gdf中
                if feature_name in features_gdf.columns:
                    # 使用pd.Categorical来确保所有类别都被考虑到，即使它们在当前数据中未出现
                    cat_series = pd.Categorical(features_gdf[feature_name], categories=all_categories)
                    one_hot_df = pd.get_dummies(cat_series, prefix=feature_name, dtype=float)
                else:
                    # 如果原始列完全不存在，则创建全为0的虚拟列
                    one_hot_cols = [f"{feature_name}_{cat}" for cat in all_categories]
                    one_hot_df = pd.DataFrame(0, index=features_gdf.index, columns=one_hot_cols, dtype=float)

                all_categorical_dfs.append(one_hot_df)

                num_categories = len(one_hot_df.columns)
                if num_categories > 0:
                    mapping[feature_name] = (current_col_idx, current_col_idx + num_categories, 'categorical')
                    current_col_idx += num_categories

            if all_categorical_dfs:
                feature_components.append(pd.concat(all_categorical_dfs, axis=1))

    # =============================================================================
    # 合并所有特征并返回
    # =============================================================================
    if feature_components:
        final_features = pd.concat(feature_components, axis=1)
    else:
        final_features = pd.DataFrame(index=features_gdf.index)
        print("警告：未找到任何特征列，返回空的特征矩阵。")

    print(f"处理完成。最终特征维度: {final_features.shape}")
    print(f"特征映射: {mapping}")
    print("-" * 30)

    return {
        'final_features': final_features,
        'mapping': mapping
    }


def _build_landuse_matrices(
        gdf_s: gpd.GeoDataFrame,
        gdf_a: gpd.GeoDataFrame,
        s_to_a_edges: torch.Tensor,
        landuse_categorical_col: str = 'landuse',
        landuse_percent_suffix: str = '_percent',
        representation: str = LEGACY_DENSE_LANDUSE_REPRESENTATION,
) -> Dict[str, Any]:
    """
    为 LandusePredictionLoss 构建 landuse_mapping_matrix 和 landuse_ratio。
    V2: 根据绝对值百分比列进行验证和归一化。
    """
    if representation not in LANDUSE_REPRESENTATIONS:
        raise ValueError(
            f"unknown landuse supervision representation: {representation!r}"
        )
    print(f"开始构建土地利用监督 ({representation})...")

    # 1. 验证输入数据
    if landuse_categorical_col not in gdf_a.columns:
        print(f"警告: gdf_a 中缺少 '{landuse_categorical_col}' 列。跳过土地利用矩阵构建。")
        return {}

    # 2. 验证 gdf_s 中是否存在所有必需的土地利用比例列
    # 从 agent 数据中提取所有出现过的土地利用类型
    agent_landuse_types = gdf_a[landuse_categorical_col].dropna().unique()

    # 构建预期的列名
    expected_percent_cols = [f"{lu_type}{landuse_percent_suffix}" for lu_type in agent_landuse_types]

    # 检查这些列是否存在于 source 数据中
    missing_cols = [col for col in expected_percent_cols if col not in gdf_s.columns]
    if missing_cols:
        print(f"警告: gdf_s 中缺少以下必需的土地利用比例列: {missing_cols}。跳过土地利用矩阵构建。")
        return {}

    print(f"验证通过: gdf_s 中包含所有 {len(expected_percent_cols)} 个必需的土地利用比例列。")

    # 3. 构建 landuse_ratio (真实比例)
    # 按照字母顺序对类别排序，以保证处理顺序的确定性
    landuse_categories = sorted(list(agent_landuse_types))
    sorted_percent_cols = [f"{cat}{landuse_percent_suffix}" for cat in landuse_categories]

    # 提取绝对值
    absolute_values = gdf_s[sorted_percent_cols].values

    # 逐行归一化
    row_sums = absolute_values.sum(axis=1, keepdims=True)
    # 防止除以零：如果某一行总和为0，归一化后仍为0
    row_sums[row_sums == 0] = 1

    normalized_ratios = absolute_values / row_sums
    landuse_ratio = torch.tensor(normalized_ratios, dtype=torch.float32)

    # 4. 构建 landuse_mapping_matrix
    num_s = len(gdf_s)
    num_edges = s_to_a_edges.shape[1]
    num_landuse_types = len(landuse_categories)

    # 将 landuse 文本类别转换为索引
    category_to_idx = {cat: i for i, cat in enumerate(landuse_categories)}
    agent_landuse_indices = gdf_a[landuse_categorical_col].map(category_to_idx).fillna(-1).astype(int)

    # 获取每条边对应的 source 和 agent 索引
    source_indices = s_to_a_edges[0].numpy()
    agent_indices = s_to_a_edges[1].numpy()

    # 获取每条边对应的 agent 的土地利用类型索引
    # 使用 .iloc 确保通过位置索引访问，避免索引对齐问题
    edge_landuse_indices = agent_landuse_indices.iloc[agent_indices].values

    flat_index = torch.full((num_edges,), -1, dtype=torch.int64)
    for i in range(num_edges):
        s_idx = int(source_indices[i])
        lu_idx = int(edge_landuse_indices[i])
        if not 0 <= s_idx < num_s:
            raise ValueError(f"source edge index out of range at edge {i}: {s_idx}")
        if not -1 <= lu_idx < num_landuse_types:
            raise ValueError(f"landuse category index out of range at edge {i}: {lu_idx}")
        if lu_idx != -1:
            flat_index[i] = s_idx * num_landuse_types + lu_idx

    if representation == INDEXED_LANDUSE_REPRESENTATION:
        print("土地利用 indexed 监督构建完成。")
        return {
            'landuse_flat_index': flat_index,
            'landuse_ratio': landuse_ratio,
            'landuse_supervision_representation': representation,
        }

    # Keep the historical dense layout byte-for-byte compatible by default.
    mapping_matrix = torch.zeros(
        (num_edges, num_s * num_landuse_types), dtype=torch.float32
    )
    valid = flat_index >= 0
    if bool(valid.any()):
        rows = torch.arange(num_edges, dtype=torch.int64)[valid]
        mapping_matrix[rows, flat_index[valid]] = 1.0
    print("土地利用 dense 监督矩阵构建完成。")
    return {
        'landuse_mapping_matrix': mapping_matrix,
        'landuse_ratio': landuse_ratio,
    }


def build_agent_adjacency(
        coords: np.ndarray,
        threshold: float = None,
) -> torch.Tensor:
    """
    构建 agent 间邻接边对：基于空间邻近性将距离小于阈值的 agent 配对。

    用于 SemanticLoss 中计算相邻 agent 的光谱相似度。

    Args:
        coords: (N_a, 2) agent 投影坐标（单位米）
        threshold: 距离阈值。如果未指定，自动计算为 median(最近邻距离) × 1.5

    Returns:
        (2, num_pairs) 对称边对张量（双向：(i,j) 和 (j,i) 均包含）
    """
    from scipy.spatial import KDTree

    N_a = len(coords)
    if N_a < 2:
        return torch.zeros((2, 0), dtype=torch.long)

    tree = KDTree(coords)

    # 自动阈值：median(最近邻距离) × 1.5
    if threshold is None:
        nn_dist, _ = tree.query(coords, k=2)  # k=2: 最近邻（排除自身）
        median_nn_dist = np.median(nn_dist[:, 1])
        threshold = median_nn_dist * 1.5

    # 查找阈值内的所有邻居
    neighbors = tree.query_ball_point(coords, r=threshold)

    src_list = []
    dst_list = []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if i != j:  # 排除自环
                src_list.append(i)
                dst_list.append(j)

    if len(src_list) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = torch.tensor(
        np.stack([src_list, dst_list], axis=0), dtype=torch.long
    )
    return edge_index


def build_grid_adjacency(
        coords: np.ndarray,
        mode: str = 'neumann',
        step_size: float = None,
        tolerance: float = 0.1,
) -> torch.Tensor:
    """
    基于规则网格拓扑构建 agent 间邻接边。

    支持两种邻域模式:
    - Von Neumann (4-连通): 仅上下左右正交邻居
    - Moore (8-连通): 正交 + 对角线邻居

    Args:
        coords: (N, 2) agent 坐标（任意坐标系）
        mode: 'neumann' (4邻域) 或 'moore' (8邻域)
        step_size: 网格步长，None 时从 median(最近邻距离) 自动推断
        tolerance: 距离匹配容差比例，默认 0.1 (10%)

    Returns:
        (2, num_edges) 双向对称边索引
    """
    from sglib.core.algorithms.grid_adjacency import build_grid_adjacency_indices

    return torch.as_tensor(
        build_grid_adjacency_indices(
            coords,
            mode=mode,
            step_size=step_size,
            tolerance=tolerance,
        ),
        dtype=torch.long,
    )


def prepare_hetero_graph_from_processed(
        gdf_s: gpd.GeoDataFrame,
        gdf_a: gpd.GeoDataFrame,
        processed_features_s: Dict[str, Any],
        processed_features_a: Dict[str, Any],
        relation_column: str,
        spectral_features: Optional[np.ndarray] = None,
        spectral_column_names: Optional[List[str]] = None,
        macro_attributes: Optional[np.ndarray] = None,
        macro_column_names: Optional[List[str]] = None,
        attribute_stds: Optional[np.ndarray] = None,
        agent_connectivity: Optional[str] = None,
        representation: str = LEGACY_DENSE_LANDUSE_REPRESENTATION,
) -> HeteroData:
    print("开始构建异构图 (HeteroData) 对象...")

    # 1. 初始化一个空的 HeteroData 对象
    data = HeteroData()

    # 2. 处理 'source' 节点
    num_s = len(gdf_s)
    coords_s = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf_s.geometry])
    features_s_df = processed_features_s['final_features']
    features_s_np = features_s_df.values if not features_s_df.empty else np.empty((num_s, 0))

    # 将 source 节点的特征和坐标存入 'source' 节点仓库
    data['source'].x = torch.tensor(features_s_np, dtype=torch.float32)
    data['source'].coords = torch.tensor(coords_s, dtype=torch.float32)
    # num_nodes 会被自动推断

    # 3. 处理 'agent' 节点
    num_a = len(gdf_a)
    coords_a = np.array([[point.x, point.y] for point in gdf_a.geometry])
    features_a_df = processed_features_a['final_features']
    features_a_np = features_a_df.values if not features_a_df.empty else np.empty((num_a, 0))

    # 如果提供了光谱特征数组，拼接到 agent 特征
    if spectral_features is not None:
        if spectral_features.shape[0] != num_a:
            raise ValueError(
                f"光谱特征行数 ({spectral_features.shape[0]}) 与 agent 数量 ({num_a}) 不匹配"
            )
        # 拼接原有特征和光谱特征
        if features_a_np.shape[1] > 0:
            features_a_np = np.concatenate([features_a_np, spectral_features], axis=1)
        else:
            features_a_np = spectral_features
        spec_dim = spectral_features.shape[1]
        if spectral_column_names is not None:
            print(f"已拼接光谱特征 ({spec_dim} 维): {spectral_column_names}，agent 总特征维度: {features_a_np.shape[1]}")
        else:
            print(f"已拼接光谱特征 ({spec_dim} 维)，agent 总特征维度: {features_a_np.shape[1]}")

    # 将 agent 节点的特征和坐标存入 'agent' 节点仓库
    data['agent'].x = torch.tensor(features_a_np, dtype=torch.float32)
    data['agent'].coords = torch.tensor(coords_a, dtype=torch.float32)

    # 存储宏观属性（用于 L_recon 损失）
    if macro_attributes is not None:
        data.macro_attributes = torch.tensor(macro_attributes, dtype=torch.float32)
        macro_dim = macro_attributes.shape[1]
        if macro_column_names is not None:
            print(f"已添加宏观属性 ({macro_dim} 维): {macro_column_names}")
        else:
            print(f"已添加宏观属性 ({macro_dim} 维)")
    if attribute_stds is not None:
        data.attribute_stds = torch.tensor(attribute_stds, dtype=torch.float32)

    print(f"Source 节点: {data['source'].num_nodes} 个, 特征维度: {data['source'].num_features}")
    print(f"Agent 节点: {data['agent'].num_nodes} 个, 特征维度: {data['agent'].num_features}")

    # 4. 构建边关系 (最核心的变化)
    # 首先，像原来一样获取全局索引的边列表
    global_edges = _build_source_agent_graph(gdf_s, gdf_a, relation_column)

    # 然后，将全局索引转换为特定边类型的局部索引
    s_to_a_edges = []
    a_to_s_edges = []
    for u, v in global_edges:
        if u < num_s and v >= num_s:  # 这是一条 Source -> Agent 的边
            local_u = u  # Source 节点的索引是局部的
            local_v = v - num_s  # Agent 节点的索引需要减去偏移量，使其变为局部
            s_to_a_edges.append([local_u, local_v])
        elif u >= num_s and v < num_s:  # 这是一条 Agent -> Source 的边
            local_u = u - num_s  # Agent 节点的索引是局部的
            local_v = v  # Source 节点的索引是局部的
            a_to_s_edges.append([local_u, local_v])

    s_to_a_edges_tensor = None
    # 将处理好的边列表存入对应的边仓库
    # 我们定义两种关系：'connects_to' 和它的反向 'rev_connects_to'
    if s_to_a_edges:
        s_to_a_edges_tensor = torch.tensor(s_to_a_edges,dtype=torch.long).t().contiguous()
        data['source', 'connects_to', 'agent'].edge_index = s_to_a_edges_tensor
    if a_to_s_edges:
        data['agent', 'rev_connects_to', 'source'].edge_index = torch.tensor(a_to_s_edges,
                                                                             dtype=torch.long).t().contiguous()

    print(f"Source->Agent 边: {data['source', 'connects_to', 'agent'].num_edges} 条")
    print(f"Agent->Source 边: {data['agent', 'rev_connects_to', 'source'].num_edges} 条")

    # 5. !! 调用新版函数 !!: 构建并添加土地利用监督所需的矩阵
    if s_to_a_edges_tensor is not None:
        landuse_data = _build_landuse_matrices(
            gdf_s,
            gdf_a,
            s_to_a_edges_tensor,
            representation=representation,
        )
        if landuse_data:
            data.landuse_ratio = landuse_data['landuse_ratio']
            if representation == INDEXED_LANDUSE_REPRESENTATION:
                data.landuse_flat_index = landuse_data['landuse_flat_index']
                data.landuse_supervision_representation = representation
                print("已将 landuse_flat_index 和 landuse_ratio 添加到图对象。")
            else:
                data.landuse_mapping_matrix = landuse_data['landuse_mapping_matrix']
                print("已将 landuse_mapping_matrix 和 landuse_ratio 添加到图对象。")

    # 6. 附加其他元数据
    # 这些可以作为图级别的全局属性
    data.feature_mapping_s = processed_features_s['mapping']
    data.feature_mapping_a = processed_features_a['mapping']
    # 您也可以存储原始的 GeoDataFrame，但要注意内存占用
    # data.source_gdf = gdf_s
    # data.agent_gdf = gdf_a
    # !! 新增 !!: 添加源节点的真实需求作为监督目标
    if 'Demand (MVA)' in gdf_s.columns:
        source_demand_true = gdf_s['Demand (MVA)'].values
        data['source'].y = torch.tensor(source_demand_true, dtype=torch.float32)
        print(f"Source 节点的监督目标 'y' 已添加。")

    if 'Demand (MVA)' in gdf_a.columns:
        agent_demand = gdf_a['Demand (MVA)'].values
        data['agent'].demand = torch.tensor(agent_demand, dtype=torch.float32)
        print(f"Agent 节点的基础需求 'demand' 已添加。")

    data.source_index_map = pd.Series(gdf_s.index.values)
    data.agent_index_map = pd.Series(gdf_a.index.values)
    print("Source 和 Agent 的原始索引映射 'source_index_map' 和 'agent_index_map' 已添加。")

    # 6.5 Agent-Agent 网格邻接边（可选）
    if agent_connectivity is not None:
        agent_near_edges = build_grid_adjacency(coords_a, mode=agent_connectivity)
        if agent_near_edges.shape[1] > 0:
            data['agent', 'near', 'agent'].edge_index = agent_near_edges
            # 同时设置 agent_adj_pairs（FeatureConsistencyLoss 等损失函数使用）
            data.agent_adj_pairs = agent_near_edges
            print(f"Agent-Agent 邻接边 ('{agent_connectivity}'): {agent_near_edges.shape[1]} 条")
            print(f"agent_adj_pairs 已设置: {agent_near_edges.shape[1]} 对")
        else:
            print(f"警告: agent_connectivity='{agent_connectivity}' 但未找到任何邻接边")
    else:
        # 无网格连接时，用 KDTree 距离阈值构建邻接对（损失函数仍可使用）
        agent_adj_pairs = build_agent_adjacency(coords_a)
        data.agent_adj_pairs = agent_adj_pairs
        print(f"agent_adj_pairs (KDTree): {agent_adj_pairs.shape[1]} 对")

    print("\n" + "=" * 50)
    print("异构图 (HeteroData) 对象构建完成！")
    print("=" * 50 + "\n")

    return data

