import open3d as o3d


class NormalEstimator:
    def __init__(self, search_param: o3d.geometry.KDTreeSearchParam = None):
        # Use a default search param if none specified
        self.search_param = search_param or o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)

    def estimate_normals(self, pointcloud: o3d.geometry.PointCloud) -> None:
        pointcloud.estimate_normals(search_param=self.search_param)
