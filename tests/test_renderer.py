import numpy as np

from renderer import Renderer


def test_cluster_color_map():
    # Arrange
    labels = np.array([0, 0, 2, 0, 5, 2, -1, -1], dtype=int)

    # Act
    color_map = Renderer.create_cluster_color_map(labels)

    # Assert
    assert set(color_map.keys()) == {0, 2, 5}
    assert len(color_map) == 3

    assert np.all(color_map[0] >= 0.0) and np.all(color_map[0] <= 1.0)
    assert np.all(color_map[2] >= 0.0) and np.all(color_map[2] <= 1.0)
    assert np.all(color_map[5] >= 0.0) and np.all(color_map[5] <= 1.0)

    assert not np.allclose(color_map[0], color_map[2])
    assert not np.allclose(color_map[0], color_map[5])
    assert not np.allclose(color_map[2], color_map[5])
