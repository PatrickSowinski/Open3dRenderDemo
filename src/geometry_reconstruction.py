from typing import Optional

import numpy as np
import open3d as o3d


class GeometryReconstruction:
    def __init__(self):
        self.full_cloud: Optional[o3d.geometry.PointCloud] = None
        self.downsampled_cloud: Optional[o3d.geometry.PointCloud] = None
        # We are just storing the cluster labels as an array the size of the pointcloud.
        # This means that pointcloud copies need to be created for visualization with colors when needed.
        # (Unless we would be ok with overwriting the original colors of the pointcloud)
        # The alternative would be to store colored copies of cluster pointclouds,
        # which would save computation effort for rendering, but require more memory for storing the results.
        self.downsampled_cluster_labels: Optional[np.ndarray] = None

    # Note: A Python property is computed whenever its value is accessed
    @property
    def full_cloud_has_normals(self) -> bool:
        return self.full_cloud is not None and self.full_cloud.has_normals()

    @property
    def downsampled_has_normals(self) -> bool:
        return self.downsampled_cloud is not None and self.downsampled_cloud.has_normals()

    # Delete full cloud to save space if needed
    def delete_full_cloud(self) -> None:
        self.full_cloud = None
        # The underlying memory will be freed if this was the last reference to the point cloud.
        # (Python uses reference counting for memory management)
        # If some other variables still refer to it, the memory will be freed when those variables go out of scope.

    # Delete downsampled cloud if needed
    def delete_downsampled_cloud(self) -> None:
        self.downsampled_cloud = None
        self.downsampled_cluster_labels = None
        # (Similar memory management note as in delete_full_cloud)
