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
def test_downsampling_with_voxel_size(eagle_pointcloud, sample_preprocessor, voxel_size):
    # Act
    downsampled_cloud = sample_preprocessor.downsample(eagle_pointcloud, voxel_size=voxel_size)

    # Assert
    assert len(downsampled_cloud.points) < len(eagle_pointcloud.points)


def test_downsampling_bigger_voxel_smaller_cloud(eagle_pointcloud, sample_preprocessor):
    # Act
    downsampled_cloud_1 = sample_preprocessor.downsample(eagle_pointcloud, voxel_size=0.1)
    downsampled_cloud_2 = sample_preprocessor.downsample(eagle_pointcloud, voxel_size=0.2)

    # Assert
    assert len(downsampled_cloud_2.points) < len(downsampled_cloud_1.points)
