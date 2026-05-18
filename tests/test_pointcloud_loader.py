# Extra test to make sure that the eagle data is loaded correctly for the test fixture
# (Otherwise most other tests would fail)
def test_eagle_fixture(eagle_pointcloud):
    # Assert
    assert len(eagle_pointcloud.points) > 0
