import argparse
import logging
from pathlib import Path

from cluster_extractor import ClusterExtractor
from geometry_reconstruction import GeometryReconstruction
from normal_estimator import NormalEstimator
from pointcloud_loader import PointcloudLoader
from preprocessor import Preprocessor
from renderer import Renderer, VisualizationParams

# configure logging globally for this session
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")


class PipelineController:
    def __init__(self):
        self.logger = logging.getLogger("PipelineController")
        self.logger.info("Initializing PipelineController with default components")
        self.pointcloud_loader = PointcloudLoader()
        self.preprocessor = Preprocessor()
        self.normal_estimator = NormalEstimator()
        self.cluster_extractor = ClusterExtractor()

        self.renderer = Renderer()
        self.eagle_viz_params = VisualizationParams.eagle_front()

    def run_example_pipeline(self, show_plots: bool = True, save_plots: bool = False):
        self.logger.info("Running example pipeline...")
        self.logger.info(f"show_plots={show_plots}, save_plots={save_plots}")
        # init an empty GeometryReconstruction where we can link related data for a pointcloud
        geometry: GeometryReconstruction = GeometryReconstruction()

        save_path = None
        if save_plots:
            save_path = Path("pipeline_results")
            save_path.mkdir(exist_ok=True)
            downsampled_path = save_path / "downsampled_cloud.png"
            if downsampled_path.exists():
                self.logger.warning(f"Overwriting existing file: {downsampled_path}")
            normals_path = save_path / "estimated_normals.png"
            if normals_path.exists():
                self.logger.warning(f"Overwriting existing file: {normals_path}")
            clusters_path = save_path / "extracted_clusters.png"
            if clusters_path.exists():
                self.logger.warning(f"Overwriting existing file: {clusters_path}")

        self.logger.info("Loading eagle point cloud...")
        geometry.full_cloud = self.pointcloud_loader.load_eagle_example()

        self.logger.info("Downsampling the point cloud...")
        geometry.downsampled_cloud = self.preprocessor.downsample(geometry.full_cloud, voxel_size=0.1)
        # render the downsampled cloud
        self.renderer.render_pointcloud(
            pointcloud=geometry.downsampled_cloud,
            vis_params=self.eagle_viz_params,
            show=show_plots,
            save_path=downsampled_path if save_plots else None,
        )

        self.logger.info("Estimating normals for downsampled cloud...")
        self.normal_estimator.estimate_normals(geometry.downsampled_cloud)
        # render the estimated normals
        self.renderer.render_normals(
            pointcloud=geometry.downsampled_cloud,
            vis_params=self.eagle_viz_params,
            show=show_plots,
            save_path=normals_path if save_plots else None,
        )

        self.logger.info("Extracting clusters from downsampled cloud...")
        geometry.downsampled_cluster_labels = self.cluster_extractor.extract_clusters(geometry.downsampled_cloud)
        # render the clusters
        self.renderer.render_segmentation(
            geometry=geometry,
            vis_params=self.eagle_viz_params,
            show=show_plots,
            save_path=clusters_path if save_plots else None,
        )

        self.logger.info("Finished running example pipeline.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_show", action="store_true", help="Whether to show the visualizations interactively.")
    parser.add_argument("--save_plots", action="store_true", help="Whether to save the visualizations as images.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    controller = PipelineController()
    controller.run_example_pipeline(show_plots=not args.no_show, save_plots=args.save_plots)
