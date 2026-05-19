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

    def render_pointcloud(
        self,
        pointcloud: o3d.geometry.PointCloud,
        vis_params: VisualizationParams,
        show: bool = True,
        save_path: Optional[Path] = None,
    ):
        # Reuse the same visualization for showing and saving to avoid redundant rendering
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=show)
        vis.add_geometry(pointcloud)
        ctr = vis.get_view_control()
        ctr.set_front(vis_params.front)
        ctr.set_lookat(vis_params.lookat)
        ctr.set_up(vis_params.up)
        ctr.set_zoom(vis_params.zoom)
        vis.poll_events()
        vis.update_renderer()
        if save_path is not None:
            vis.capture_screen_image(str(save_path))
        if show:
            # keep the window open until user closes it
            vis.run()
        vis.destroy_window()

    # TODO: remove duplication for rendering
    def render_normals(
        self,
        pointcloud: o3d.geometry.PointCloud,
        vis_params: VisualizationParams,
        show: bool = True,
        save_path: Optional[Path] = None,
    ):
        # Reuse the same visualization for showing and saving to avoid redundant rendering
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=show)
        vis.add_geometry(pointcloud)
        ctr = vis.get_view_control()
        ctr.set_front(vis_params.front)
        ctr.set_lookat(vis_params.lookat)
        ctr.set_up(vis_params.up)
        ctr.set_zoom(vis_params.zoom)
        render_option = vis.get_render_option()
        render_option.point_show_normal = True
        vis.poll_events()
        vis.update_renderer()
        if save_path is not None:
            vis.capture_screen_image(str(save_path))
        if show:
            # keep the window open until user closes it
            vis.run()
        vis.destroy_window()

    def render_segmentation(
        self,
        geometry: GeometryReconstruction,
        vis_params: VisualizationParams,
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
        for label in np.unique(geometry.downsampled_cluster_labels):
            if label < 0:
                continue  # skip noise
            cluster_indices = np.where(geometry.downsampled_cluster_labels == label)[0]
            # Note: select_by_index creates a copy, so we can modify it without affecting the original point cloud
            # (e.g. for coloring)
            cluster_points = geometry.downsampled_cloud.select_by_index(cluster_indices)
            cluster_points.paint_uniform_color(colors[label, :3])
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=show)
            vis.add_geometry(cluster_points)
            ctr = vis.get_view_control()
            ctr.set_front(vis_params.front)
            ctr.set_lookat(vis_params.lookat)
            ctr.set_up(vis_params.up)
            ctr.set_zoom(vis_params.zoom)
            vis.poll_events()
            vis.update_renderer()
            # if save_path is not None:
            #     cluster_save_path = save_path.parent / f"{save_path.stem}_cluster_{label}{save_path.suffix}"
            #     vis.capture_screen_image(str(cluster_save_path))
            if show:
                # keep the window open until user closes it
                vis.run()
            vis.destroy_window()

        # # Reuse the same visualization for showing and saving to avoid redundant rendering
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(visible=show)
        # ctr = vis.get_view_control()
        # ctr.set_front(vis_params.front)
        # ctr.set_lookat(vis_params.lookat)
        # ctr.set_up(vis_params.up)
        # ctr.set_zoom(vis_params.zoom)
        # vis.poll_events()
        # vis.update_renderer()
        # if save_path is not None:
        #     vis.capture_screen_image(str(save_path))
        # if show:
        #     # keep the window open until user closes it
        #     vis.run()
        # vis.destroy_window()
