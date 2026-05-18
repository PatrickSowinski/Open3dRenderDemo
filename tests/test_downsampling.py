import pytest


def test_downsampling_default_voxel_size(eagle_pointcloud, sample_preprocessor):
    # Act
    downsampled_cloud = sample_preprocessor.downsample(eagle_pointcloud)

    # Assert
    assert len(downsampled_cloud.points) < len(eagle_pointcloud.points)


@pytest.mark.parametrize(
    "voxel_size",
    [
        0.1,
        0.2,
    ],
)
def test_downsampling(eagle_pointcloud, sample_preprocessor, voxel_size):
    # Act
    downsampled_cloud = sample_preprocessor.downsample(eagle_pointcloud, voxel_size=voxel_size)

    # Assert
    assert len(downsampled_cloud.points) < len(eagle_pointcloud.points)
