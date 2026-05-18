import open3d as o3d


class Preprocessor:
    def downsample(self, pointcloud: o3d.geometry.PointCloud, voxel_size: float = 0.05) -> o3d.geometry.PointCloud:
        return pointcloud.voxel_down_sample(voxel_size=voxel_size)
