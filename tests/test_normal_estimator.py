def test_normal_estimation(eagle_pointcloud, sample_normal_estimator):
    # Act
    sample_normal_estimator.estimate_normals(eagle_pointcloud)

    # Assert
    assert len(eagle_pointcloud.normals) == len(eagle_pointcloud.points)
    assert eagle_pointcloud.has_normals()
    # Note: Since eagle_pointcloud is a session-scope fixture,
    # normal estimation can only be tested once per session with this design.
    # To test multiple methods, the fixture or assertion would need to change.
