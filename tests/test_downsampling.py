import pytest

from preprocessor import Preprocessor


@pytest.fixture(scope="session")
def sample_preprocessor():
    return Preprocessor()


def test_downsampling(eagle_pointcloud, sample_preprocessor):
    # Act
    downsampled_cloud = sample_preprocessor.downsample(eagle_pointcloud)

    # Assert
    assert len(downsampled_cloud.points) < len(eagle_pointcloud.points)
