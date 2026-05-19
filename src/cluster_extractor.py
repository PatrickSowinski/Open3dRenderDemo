import logging

import numpy as np
import open3d as o3d


class ClusterExtractor:
    def __init__(self, verbose: bool = True):
        self.logger = logging.getLogger("ClusterExtractor")
        self.verbose = verbose

    # returns a list of labels for each point in the input pointcloud
    # points with the same label belong to the same cluster
    def extract_clusters(
        self,
        pointcloud: o3d.geometry.PointCloud,
        max_distance_to_neighbors: float = 0.15,
        min_points_around_core_point: int = 10,
        min_points_in_cluster: int = 150,
    ) -> np.ndarray:
        labels = np.array(
            pointcloud.cluster_dbscan(
                eps=max_distance_to_neighbors, min_points=min_points_around_core_point, print_progress=False
            )
        )
        unique_labels, label_count = np.unique(labels, return_counts=True)
        if self.verbose:
            self.logger.info(
                f"Found {len(unique_labels) - 1} initial clusters (excluding noise)"
                f" in pointcloud with {len(pointcloud.points)} points."
            )
            self.logger.info(
                "The number of points in each cluster is: "
                + ", ".join(f"{i}: {count}" for i, count in zip(unique_labels, label_count))
            )
        # filter out labels below count limit
        tiny_clusters = unique_labels[label_count < min_points_in_cluster]
        labels[np.isin(labels, tiny_clusters)] = -1
        if self.verbose:
            self.logger.info(
                f"Keeping {len(np.unique(labels)) - 1} clusters after removing those"
                f" with less than {min_points_in_cluster} points."
            )
        return labels
