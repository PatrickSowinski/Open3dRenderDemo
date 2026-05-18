import logging
from pathlib import Path

from cluster_extractor import ClusterExtractor
from geometry_reconstruction import GeometryReconstruction
from normal_estimator import NormalEstimator
from pointcloud_loader import PointcloudLoader
from preprocessor import Preprocessor
from renderer import Renderer, VisualizationParams

# configure logging globally for this session
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class PipelineController:
    def __init__(self):
        self.logger = logging.getLogger("PipelineController")
        self.logger.info("Initializing PipelineController with default components")
        self.pointcloud_loader = PointcloudLoader()
        self.preprocessor = Preprocessor()
        self.normal_estimator = NormalEstimator()
        self.cluster_extractor = ClusterExtractor()
        self.renderer = Renderer()

        self.eagle_viz_params = VisualizationParams(
            front=[-0.97604822713972006, -0.07040202601844868, 0.20584803382570194],
            lookat=[0.56815877868502129, 3.2540328971914581, 1.1290062986978375],
            up=[0.091901599279143362, -0.99105079298882548, 0.096811268797366179],
            zoom=0.62839660644531226,
        )

    def run_example_pipeline(self, show_plots: bool = True, save_plots: bool = False):
        self.logger.info("Running example pipeline...")
        geometry: GeometryReconstruction = GeometryReconstruction()

        save_path = None
        if save_plots:
            save_path = Path("pipeline_results")
            save_path.mkdir(exist_ok=True)
            downsampled_path = save_path / "downsampled_cloud.png"
            if downsampled_path.exists():
                self.logger.warning(f"Overwriting existing file: {downsampled_path}")

        self.logger.info("Loading eagle point cloud...")
        geometry.full_cloud = self.pointcloud_loader.load_eagle_example()

        self.logger.info("Downsampling the point cloud...")
        geometry.downsampled_cloud = self.preprocessor.downsample(geometry.full_cloud)
        # render the downsampled cloud
        self.renderer.render_pointcloud(
            geometry.downsampled_cloud,
            vis_params=self.eagle_viz_params,
            show=show_plots,
            save_path=downsampled_path if save_plots else None,
        )

        self.logger.info("Estimating normals for downsampled cloud...")
        self.normal_estimator.estimate_normals(geometry.downsampled_cloud)

        self.logger.info("Extracting clusters from downsampled cloud...")
        geometry.downsampled_cluster_labels = self.cluster_extractor.extract_clusters(geometry.downsampled_cloud)

        self.logger.info("Finished running example pipeline.")


if __name__ == "__main__":
    controller = PipelineController()
    controller.run_example_pipeline()
