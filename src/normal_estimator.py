import logging

import open3d as o3d


class NormalEstimator:
    def __init__(self, verbose: bool = True):
        self.logger = logging.getLogger("NormalEstimator")
        self.verbose = verbose

    def estimate_normals(
        self,
        pointcloud: o3d.geometry.PointCloud,
        search_param: o3d.geometry.KDTreeSearchParam = None,
        orient_normals_neighbors: int = 30,
    ) -> None:
        """
        Estimate normals for the input pointcloud.

        Args:
            pointcloud: The input pointcloud for which to estimate normals.
            search_param: Optional search parameter for normal estimation.
                If None, a default KDTreeSearchParamHybrid with radius=0.1 and max_nn=30 is used.

        Returns:
            None. The estimated normals are stored in pointcloud.normals.
        """
        # Use a default search_param if none is provided
        search_param = search_param or o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        # Use Open3D's built-in normal estimation function, which modifies pointcloud.normals
        pointcloud.estimate_normals(search_param=search_param)

        # Note: Below, orient normals is commented out to save computation time.
        #  It did not seem to improve the clustering significantly.
        # # Orient normals to face in a consistent direction (inward/outward)
        # pointcloud.orient_normals_consistent_tangent_plane(orient_normals_neighbors)

        if self.verbose:
            self.logger.info(f"Estimated normals for {len(pointcloud.points)} points in pointcloud.")
