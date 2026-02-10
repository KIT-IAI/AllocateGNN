"""
Author: Xuanhao Mu
Email: xuanhao.mu@kit.edu
Description: This module contains utility functions for creating weights.
"""
import numpy as np


def create_weights(df, method="equal", **kwargs):
    weight_normalize = kwargs.get("weight_normalize", False)
    if method == "equal":
        return np.array([1.] * df.shape[0])
    elif method in ["dense_rank", "inverse_dense_rank", "rank", "inverse_rank"]:
        return create_rank_weights(df, method, weight_normalize=weight_normalize)
    elif method == "count":
        return create_count_weights(df, method, weight_normalize=weight_normalize)


def create_rank_weights(df, method, cluster_label_col='cluster_label', rank_col='rank', weight_normalize=False):
    """
    Adds a rank column to the DataFrame based on the size of clusters.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the cluster data.
    cluster_label_col (str, optional): The name of the column containing cluster labels. Defaults to 'cluster_label'.
    rank_col (str, optional): The name of the column to store the rank. Defaults to 'rank'.

    Returns:
    pandas.DataFrame: The DataFrame with the added rank column.
    """
    # Count the number of occurrences of each cluster label
    cluster_counts = df[cluster_label_col].value_counts()

    # Rank the clusters based on their size, with the smallest cluster getting the highest rank
    if method == 'dense_rank':
        cluster_ranks = cluster_counts.rank(method='dense', ascending=True)
    elif method == 'inverse_dense_rank':
        cluster_ranks = cluster_counts.rank(method='dense', ascending=False)
    elif method == 'rank':
        cluster_ranks = cluster_counts.rank(method='min', ascending=True)
    elif method == 'inverse_rank':
        cluster_ranks = cluster_counts.rank(method='min', ascending=False)

    # Map the cluster labels to their corresponding ranks
    if weight_normalize:
        # # Apply logarithmic scaling (plus 1 to avoid log(0))
        # log_scaled_ranks = np.log(cluster_ranks + 1)
        #
        # # Normalize the log ranks to avoid 0 values, you can scale to [0.01, 1] range if needed
        # df[rank_col] = df[cluster_label_col].map(log_scaled_ranks)
        # Convert to numpy array for vectorized operations
        # Convert to numpy array for easier operations
        ranks_array = np.array(cluster_ranks)

        # 应用一个初始的归一化，可以先使用min-max或softmax
        # 这里我们先用min-max作为起点
        min_rank = np.min(ranks_array)
        max_rank = np.max(ranks_array)

        # 检查所有值是否相同
        if max_rank == min_rank:
            normalized_ranks = [0.5 for _ in ranks_array]
        else:
            # 首先应用min-max归一化到[0,1]范围
            initial_norm = (ranks_array - min_rank) / (max_rank - min_rank)

            # 然后调整范围确保最大值/最小值不超过10倍
            # 我们将范围映射到[0.1, 1]，这样最大比例为10
            normalized_ranks = 0.1 + 0.1 * initial_norm
            normalized_ranks = normalized_ranks.tolist()

        # Map cluster labels to normalized ranks
        df[rank_col] = df[cluster_label_col].map(lambda x: normalized_ranks[x])
    else:
        df[rank_col] = df[cluster_label_col].map(cluster_ranks)

    return df[rank_col].values

def create_count_weights(df, method, cluster_label_col='cluster_label', rank_col='rank', weight_normalize=False):
    cluster_counts = df[cluster_label_col].value_counts()

    # Map the cluster labels to their corresponding ranks
    if weight_normalize:
        # # Apply logarithmic scaling (plus 1 to avoid log(0))
        # log_scaled_ranks = np.log(cluster_ranks + 1)
        #
        # # Normalize the log ranks to avoid 0 values, you can scale to [0.01, 1] range if needed
        # df[rank_col] = df[cluster_label_col].map(log_scaled_ranks)
        # Convert to numpy array for vectorized operations
        # Convert to numpy array for easier operations
        ranks_array = np.array(cluster_counts)

        # 应用一个初始的归一化，可以先使用min-max或softmax
        # 这里我们先用min-max作为起点
        min_rank = np.min(ranks_array)
        max_rank = np.max(ranks_array)

        # 检查所有值是否相同
        if max_rank == min_rank:
            normalized_ranks = [1 for _ in ranks_array]
        else:
            # 首先应用min-max归一化到[0,1]范围
            initial_norm = (ranks_array - min_rank) / (max_rank - min_rank)

            # 然后调整范围确保最大值/最小值不超过10倍
            # 我们将范围映射到[0.1, 1]，这样最大比例为10
            normalized_ranks = 1 + 1 * initial_norm
            normalized_ranks = normalized_ranks.tolist()

        # Map cluster labels to normalized ranks
        df[rank_col] = df[cluster_label_col].map(lambda x: normalized_ranks[x])
    else:
        df[rank_col] = df[cluster_label_col].map(cluster_counts)

    return df[rank_col].values
