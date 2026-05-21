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
    # These params are compatible with o3d.visualization.Visualizer.get_view_control()
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

    def render_visualizer(
        self,
        visualizer: o3d.visualization.Visualizer,
        vis_params: VisualizationParams,
        show: bool,
        save_path: Optional[Path] = None,
    ):
        # set view parameters
        ctr = visualizer.get_view_control()
        ctr.set_front(vis_params.front)
        ctr.set_lookat(vis_params.lookat)
        ctr.set_up(vis_params.up)
        ctr.set_zoom(vis_params.zoom)
        # boilerplate to make visualizer load properly
        visualizer.poll_events()
        visualizer.update_renderer()
        # Reuse the same visualization for showing and saving to avoid redundant rendering
        if save_path is not None:
            self.logger.info(f"Saving visualization to {save_path}...")
            # create folder if needed
            save_path.parent.mkdir(exist_ok=True, parents=True)
            if save_path.exists():
                self.logger.warning(f"File {save_path} already exists and will be overwritten!")
            # save the image file
            visualizer.capture_screen_image(str(save_path))
        if show:
            # keep window open until user closes it
            visualizer.run()
        # cleanup
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
        self.render_visualizer(visualizer=vis, vis_params=vis_params, show=show, save_path=save_path)

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
        # activate normal showing in visualization
        render_option = vis.get_render_option()
        render_option.point_show_normal = True
        self.render_visualizer(visualizer=vis, vis_params=vis_params, show=show, save_path=save_path)

    @staticmethod
    def create_cluster_color_map(cluster_labels: np.ndarray) -> dict[int, np.ndarray]:
        unique_labels = np.unique(cluster_labels)
        # ignore noise label (-1)
        cluster_labels = unique_labels[unique_labels >= 0]
        n_clusters = len(cluster_labels)
        if n_clusters == 0:
            return {}

        cmap = plt.get_cmap("tab20")
        colors = cmap(np.linspace(0.0, 1.0, n_clusters))[:, :3]
        return {label: colors[idx] for idx, label in enumerate(cluster_labels)}

    def render_segmentation(
        self,
        geometry: GeometryReconstruction,
        vis_params: VisualizationParams,
        window_name: str = "",
        show: bool = True,
        save_path: Optional[Path] = None,
    ):
        # check that necessary data is available
        if geometry.downsampled_cloud is None:
            self.logger.error("No full cloud available for rendering segmentation!")
            return
        if geometry.downsampled_cluster_labels is None:
            self.logger.error("No cluster labels available for rendering segmentation!")
            return

        # create distinct colors for clusters
        label_color_map = self.create_cluster_color_map(geometry.downsampled_cluster_labels)
        if not label_color_map:
            self.logger.warning("No clusters found for segmentation rendering.")
            return
        # use black for noise points (label -1)
        label_color_map[-1] = np.array([0.0, 0.0, 0.0])

        # plot each cluster
        all_clusters = []
        for label in sorted(label_color_map):
            cluster_indices = np.where(geometry.downsampled_cluster_labels == label)[0]
            if len(cluster_indices) == 0:
                continue

            # Note: select_by_index creates a copy, so we can modify it without affecting the original point cloud
            # (e.g. for coloring)
            cluster_points = geometry.downsampled_cloud.select_by_index(cluster_indices)
            cluster_points.paint_uniform_color(label_color_map[label])
            all_clusters.append(cluster_points)

            if label == -1:
                # only store noise cluster for full render later on
                # don't render the noise cluster individually
                continue

            # visualize single cluster
            vis = self.create_visualizer(
                show=show, window_name=f"{window_name} - Cluster {label + 1} / {len(label_color_map) - 1}"
            )
            vis.add_geometry(cluster_points)
            cluster_save_path = None
            if save_path is not None:
                cluster_save_path = save_path.with_stem(save_path.stem + "_" + str(label))
            self.render_visualizer(visualizer=vis, vis_params=vis_params, show=show, save_path=cluster_save_path)

        # Plot all clusters together (incl. noise)
        vis = self.create_visualizer(show=show, window_name=f"{window_name} - All Clusters")
        for cluster in all_clusters:
            vis.add_geometry(cluster)
        if save_path is not None:
            save_path = save_path.with_stem(save_path.stem + "_all")
        self.render_visualizer(visualizer=vis, vis_params=vis_params, show=show, save_path=save_path)
