from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import open3d as o3d


@dataclass
class VisualizationParams:
    front: List[float]
    lookat: List[float]
    up: List[float]
    zoom: float


class Renderer:
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
