import pytest

from cluster_extractor import ClusterExtractor
from normal_estimator import NormalEstimator
from pointcloud_loader import PointcloudLoader
from preprocessor import Preprocessor

# conftest.py is autodiscovered by pytest to load fixtures
# we define shared fixtures here to avoid duplication across test files

# we use session scope, so we can reuse the objects for all tests in the session


@pytest.fixture(scope="session")
def sample_loader():
    return PointcloudLoader()


@pytest.fixture(scope="session")
def eagle_pointcloud(sample_loader):
    return sample_loader.load_eagle_example()


@pytest.fixture(scope="session")
def sample_preprocessor():
    return Preprocessor()


@pytest.fixture(scope="session")
def sample_normal_estimator():
    return NormalEstimator()


@pytest.fixture(scope="session")
def sample_cluster_extractor():
    return ClusterExtractor()
