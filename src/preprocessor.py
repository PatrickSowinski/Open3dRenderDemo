import logging

import open3d as o3d


class Preprocessor:
    def __init__(self, verbose: bool = True):
        self.logger = logging.getLogger("Preprocessor")
        self.verbose = verbose

    def downsample(self, pointcloud: o3d.geometry.PointCloud, voxel_size: float = 0.1) -> o3d.geometry.PointCloud:
        downsampled = pointcloud.voxel_down_sample(voxel_size=voxel_size)
        if self.verbose:
            self.logger.info(
                f"Downsampled pointcloud from {len(pointcloud.points)} to {len(downsampled.points)} points."
            )
        return downsampled
