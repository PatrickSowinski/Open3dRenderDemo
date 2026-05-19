import logging

import open3d as o3d


class NormalEstimator:
    def __init__(self, verbose: bool = True):
        self.logger = logging.getLogger("NormalEstimator")
        self.verbose = verbose

    def estimate_normals(
        self, pointcloud: o3d.geometry.PointCloud, search_param: o3d.geometry.KDTreeSearchParam = None
    ) -> None:
        # Use a default search_param if none is provided
        search_param = search_param or o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        pointcloud.estimate_normals(search_param=search_param)
        if self.verbose:
            self.logger.info(f"Estimated normals for {len(pointcloud.points)} points in pointcloud.")
