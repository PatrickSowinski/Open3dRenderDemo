import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from geometry_reconstruction import GeometryReconstruction


@dataclass
class VisualizationParams:
    front: List[float]
    lookat: List[float]
    up: List[float]
    zoom: float

    # provide a standard view for the eagle dataset
    @classmethod
    def eagle_front(cls):
        return cls(
            front=[-0.97604822713972006, -0.07040202601844868, 0.20584803382570194],
            lookat=[0.56815877868502129, 3.2540328971914581, 1.1290062986978375],
            up=[0.091901599279143362, -0.99105079298882548, 0.096811268797366179],
            zoom=0.62839660644531226,
        )


class Renderer:
    def __init__(self, verbose: bool = True):
        self.logger = logging.getLogger("Renderer")
        self.verbose = verbose

    @staticmethod
    def create_visualizer(show: bool, window_name: str) -> o3d.visualization.Visualizer:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=show, window_name=window_name)
        return vis

    @staticmethod
    def render_visualizer(
        visualizer: o3d.visualization.Visualizer,
        vis_params: VisualizationParams,
        show: bool,
        save_path: Optional[Path] = None,
    ):
        ctr = visualizer.get_view_control()
        ctr.set_front(vis_params.front)
        ctr.set_lookat(vis_params.lookat)
        ctr.set_up(vis_params.up)
        ctr.set_zoom(vis_params.zoom)
        visualizer.poll_events()
        visualizer.update_renderer()
        # Reuse the same visualization for showing and saving to avoid redundant rendering
        if save_path is not None:
            visualizer.capture_screen_image(str(save_path))
        if show:
            # keep window open until user closes it
            visualizer.run()
        visualizer.destroy_window()

    def render_pointcloud(
        self,
        pointcloud: o3d.geometry.PointCloud,
        vis_params: VisualizationParams,
        window_name: str = "",
        show: bool = True,
        save_path: Optional[Path] = None,
    ):
        vis = self.create_visualizer(show=show, window_name=window_name)
        vis.add_geometry(pointcloud)
        self.render_visualizer(vis, vis_params, show, save_path)

    def render_normals(
        self,
        pointcloud: o3d.geometry.PointCloud,
        vis_params: VisualizationParams,
        window_name: str = "",
        show: bool = True,
        save_path: Optional[Path] = None,
    ):
        vis = self.create_visualizer(show=show, window_name=window_name)
        vis.add_geometry(pointcloud)
        render_option = vis.get_render_option()
        render_option.point_show_normal = True
        self.render_visualizer(vis, vis_params, show, save_path)

    def render_segmentation(
        self,
        geometry: GeometryReconstruction,
        vis_params: VisualizationParams,
        window_name: str = "",
        show: bool = True,
        save_path: Optional[Path] = None,
    ):
        if geometry.downsampled_cloud is None:
            self.logger.error("No full cloud available for rendering segmentation!")
            return
        if geometry.downsampled_cluster_labels is None:
            self.logger.error("No cluster labels available for rendering segmentation!")
            return

        # create colors for each cluster (reusing the method from open3d docs here, since it seems elegant)
        max_label = geometry.downsampled_cluster_labels.max()
        colors = plt.get_cmap("tab20")(geometry.downsampled_cluster_labels / (max_label if max_label > 0 else 1))
        colors[geometry.downsampled_cluster_labels < 0] = 0  # -1 label = noise

        # plot each cluster
        all_clusters = []
        for label in np.unique(geometry.downsampled_cluster_labels):
            if label < 0:
                continue  # skip noise
            cluster_indices = np.where(geometry.downsampled_cluster_labels == label)[0]
            # Note: select_by_index creates a copy, so we can modify it without affecting the original point cloud
            # (e.g. for coloring)
            cluster_points = geometry.downsampled_cloud.select_by_index(cluster_indices)
            cluster_points.paint_uniform_color(colors[label, :3])
            all_clusters.append(cluster_points)
            vis = self.create_visualizer(
                show=show, window_name=f"{window_name} - Cluster {label + 1} / {max_label + 1}"
            )
            vis.add_geometry(cluster_points)
            if save_path is not None:
                suffix = save_path.suffix
                save_path = save_path.with_suffix("_" + str(label) + suffix)
            self.render_visualizer(
                vis, vis_params, show, None
            )  # we will save the combined plot at the end, so no need to save here

        # Plot all clusters together
        vis = self.create_visualizer(show=show, window_name=f"{window_name} - All Clusters")
        for cluster in all_clusters:
            vis.add_geometry(cluster)
        if save_path is not None:
            suffix = save_path.suffix
            save_path = save_path.with_suffix("_all" + suffix)
        self.render_visualizer(vis, vis_params, show, save_path)
