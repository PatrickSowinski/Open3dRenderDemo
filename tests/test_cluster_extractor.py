import numpy as np


def test_cluster_extraction_euclidean(eagle_pointcloud, sample_cluster_extractor, sample_preprocessor):
    # Arrange
    # Downsample the eagle point cloud to speed up the test
    downsampled_eagle = sample_preprocessor.downsample(eagle_pointcloud, voxel_size=0.1)

    # Act
    labels: np.ndarray = sample_cluster_extractor.extract_clusters_euclidean(
        downsampled_eagle, max_distance_to_neighbors=0.15, min_points_around_core_point=10, min_points_in_cluster=150
    )
    unique_labels = np.unique(labels)

    # Assert
    assert labels.shape == (len(downsampled_eagle.points),)
    assert -1 in unique_labels  # Expecting some noise points labeled as -1
    assert unique_labels.size > 1  # Expecting at least one non-noise cluster


def test_cluster_extraction_k_means(eagle_pointcloud, sample_cluster_extractor, sample_preprocessor):
    # Arrange
    # Downsample the eagle point cloud to speed up the test
    downsampled_eagle = sample_preprocessor.downsample(eagle_pointcloud, voxel_size=0.1)
    num_clusters = 4

    # Act
    labels: np.ndarray = sample_cluster_extractor.extract_clusters_k_means(downsampled_eagle, num_clusters=num_clusters)
    unique_labels = np.unique(labels)

    # Assert
    assert labels.shape == (len(downsampled_eagle.points),)
    assert unique_labels.size == num_clusters  # Expecting the specified number of clusters
