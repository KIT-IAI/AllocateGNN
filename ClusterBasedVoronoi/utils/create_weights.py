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

        # Apply an initial normalization; start with min-max
        min_rank = np.min(ranks_array)
        max_rank = np.max(ranks_array)

        # Check if all values are the same
        if max_rank == min_rank:
            normalized_ranks = [0.5 for _ in ranks_array]
        else:
            # First apply min-max normalization to [0, 1] range
            initial_norm = (ranks_array - min_rank) / (max_rank - min_rank)

            # Then adjust the range to ensure max/min ratio does not exceed 10x
            # Map the range to [0.1, 1] so the maximum ratio is 10
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

        # Apply an initial normalization; start with min-max
        min_rank = np.min(ranks_array)
        max_rank = np.max(ranks_array)

        # Check if all values are the same
        if max_rank == min_rank:
            normalized_ranks = [1 for _ in ranks_array]
        else:
            # First apply min-max normalization to [0, 1] range
            initial_norm = (ranks_array - min_rank) / (max_rank - min_rank)

            # Then adjust the range to ensure max/min ratio does not exceed 10x
            # Map the range to [0.1, 1] so the maximum ratio is 10
            normalized_ranks = 1 + 1 * initial_norm
            normalized_ranks = normalized_ranks.tolist()

        # Map cluster labels to normalized ranks
        df[rank_col] = df[cluster_label_col].map(lambda x: normalized_ranks[x])
    else:
        df[rank_col] = df[cluster_label_col].map(cluster_counts)

    return df[rank_col].values