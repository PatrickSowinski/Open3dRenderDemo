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

    def run_example_pipeline(
        self, show_plots: bool = True, save_plots: bool = False, save_path: Path = Path("pipeline_results")
    ):
        self.logger.info("Running example pipeline...")
        self.logger.info(f"show_plots={show_plots}, save_plots={save_plots}, save_path={save_path}")

        # init an empty GeometryReconstruction where we can link related data for a pointcloud
        geometry: GeometryReconstruction = GeometryReconstruction()

        # load example pointcloud
        self.logger.info("Loading eagle point cloud...")
        geometry.full_cloud = self.pointcloud_loader.load_eagle_example()

        # downsample
        self.logger.info("Downsampling the point cloud...")
        geometry.downsampled_cloud = self.preprocessor.downsample(geometry.full_cloud, voxel_size=0.1)
        # render the downsampled cloud
        self.renderer.render_pointcloud(
            pointcloud=geometry.downsampled_cloud,
            vis_params=self.eagle_viz_params,
            window_name="Downsampled pointcloud",
            show=show_plots,
            save_path=save_path / "downsampled_cloud.png" if save_plots else None,
        )

        # estimate normals
        self.logger.info("Estimating normals for downsampled cloud...")
        self.normal_estimator.estimate_normals(geometry.downsampled_cloud)
        # render the estimated normals
        self.renderer.render_normals(
            pointcloud=geometry.downsampled_cloud,
            vis_params=self.eagle_viz_params,
            window_name="Downsampled pointcloud with normals",
            show=show_plots,
            save_path=save_path / "estimated_normals.png" if save_plots else None,
        )

        # extract clusters
        self.logger.info("Extracting clusters from downsampled cloud...")
        geometry.downsampled_cluster_labels = self.cluster_extractor.extract_clusters(geometry.downsampled_cloud)
        # render the clusters
        self.renderer.render_segmentation(
            geometry=geometry,
            vis_params=self.eagle_viz_params,
            window_name="Segmented pointcloud",
            show=show_plots,
            save_path=save_path / "extracted_clusters.png" if save_plots else None,
        )

        self.logger.info("Finished running example pipeline.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_show", action="store_true", help="Do not show the visualizations interactively.")
    parser.add_argument("--save_plots", action="store_true", help="Save the visualizations as images.")
    parser.add_argument("--update_readme_images", action="store_true", help="Use the saved images for the Readme")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    controller = PipelineController()
    if args.update_readme_images:
        # ignore the save_plots arg if update_readme_images is True
        controller.run_example_pipeline(show_plots=not args.no_show, save_plots=True, save_path=Path("readme_images"))
    else:
        controller.run_example_pipeline(show_plots=not args.no_show, save_plots=args.save_plots)
