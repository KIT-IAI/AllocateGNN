color = {
    0: "#556b2f",   # darkolivegreen
    1: "#b22222",   # firebrick
    2: "#3cb371",   # mediumseagreen
    3: "#008080",   # teal
    4: "#9acd32",   # yellowgreen
    5: "#00008b",   # darkblue
    6: "#ff4500",   # orangered
    7: "#ffa500",   # orange
    8: "#daa520",   # goldenrod
    9: "#7fff00",   # chartreuse
    10: "#00ff7f",  # springgreen
    11: "#4169e1",  # royalblue
    12: "#a020f0",  # purple
    13: "#ffd700",  # gold
    14: "#32cd32",  # limegreen
    15: "#d8bfd8",  # thistle
    16: "#ff00ff",  # fuchsia
    17: "#db7093",  # palevioletred
    18: "#f0e68c",  # khaki
    19: "#4682b4",  # steelblue
    20: "#ffa07a",  # lightsalmon
    21: "#ee82ee",  # violet
    22: "#ff6347",  # tomato
    23: "#ff7f50",  # coral
    24: "#e9967a",  # darksalmon
    25: "#ff8c00",  # darkorange
    26: "#a0522d",  # sienna
    27: "#00bfff",  # deepskyblue
    28: "#0000ff",  # blue
    29: "#1e90ff",  # dodgerblue
    30: "#6a5acd",  # slateblue
    31: "#00ffff",  # aqua
}


def assign_color(cluster_gdf, single_element_color="#000000"):
    unique_labels = cluster_gdf['cluster_label'].unique()
    cluster_counts = cluster_gdf['cluster_label'].value_counts().to_dict()

    # Create color mapping for multi-element clusters
    multi_element_clusters = [label for label in unique_labels if cluster_counts[label] > 1]
    color_mapping = {}

    # Assign colors to multi-element clusters sequentially from color_dict
    for i, cluster_label in enumerate(multi_element_clusters):
        color_index = i % len(color)  # Ensure we wrap around if there are more clusters than colors
        color_mapping[cluster_label] = color[color_index]

    # Create a new column for color based on cluster size
    def assign_color(cluster_label):
        # If the cluster has only one element, use blue
        if cluster_counts[cluster_label] == 1:
            return single_element_color
        else:
            # Otherwise, use the color from our sequential mapping
            return color_mapping[cluster_label]

    # Apply the color assignment function
    cluster_gdf['color'] = cluster_gdf['cluster_label'].apply(assign_color)
    return cluster_gdf