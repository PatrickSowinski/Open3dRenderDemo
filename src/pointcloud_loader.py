from pathlib import Path

import open3d as o3d


class PointcloudLoader:
    def load_from_path(self, path: Path) -> o3d.geometry.PointCloud:
        return o3d.io.read_point_cloud(str(path))

    def load_eagle_example(self) -> o3d.geometry.PointCloud:
        return o3d.data.EaglePointCloud()
