import numpy as np
import open3d as o3d


class ClusterExtractor:
    # returns a list of labels for each point in the input pointcloud
    # points with the same label belong to the same cluster
    def extract_clusters(
        self,
        pointcloud: o3d.geometry.PointCloud,
        max_distance_to_neighbors: float = 0.02,
        min_points_around_core_point: int = 10,
    ) -> np.ndarray:
        labels = np.array(
            pointcloud.cluster_dbscan(
                eps=max_distance_to_neighbors, min_points=min_points_around_core_point, print_progress=True
            )
        )
        return labels
