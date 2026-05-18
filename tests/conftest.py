import pytest

from pointcloud_loader import PointcloudLoader

# conftest.py is autodiscovered by pytest to load fixtures
# we define shared fixtures here to avoid duplication across test files

# use session scope, so we can reuse the objects for all tests in the session


@pytest.fixture(scope="session")
def sample_loader():
    return PointcloudLoader()


@pytest.fixture(scope="session")
def eagle_pointcloud(sample_loader):
    return sample_loader.load_eagle_example()
