import numpy as np


def test_cluster_extraction_eagle(eagle_pointcloud, sample_cluster_extractor):
    # Act
    labels: np.ndarray = sample_cluster_extractor.extract_clusters(eagle_pointcloud)
    unique_labels = np.unique(labels)

    # Assert
    assert labels.shape == (len(eagle_pointcloud.points),)
    assert -1 in unique_labels  # Expecting some noise points labeled as -1
    assert unique_labels.size > 2  # Expecting multiple real clusters in the eagle example
