import numpy as np
import open3d as o3d


class ClusterExtractor:
    # returns a list of labels for each point in the input pointcloud
    # points with the same label belong to the same cluster
    def extract_clusters(self, pointcloud: o3d.geometry.PointCloud) -> np.ndarray:
        pass
