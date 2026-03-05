"""
Unit tests for Distance class
"""

from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial import cdist

from cristal.commons.distance import IMPLEMENTED_DISTANCE, Distance


class TestDistance:
    """Test the Distance class functionality"""

    def test_distance_initialization(self):
        """Test Distance initialization with valid parameters"""
        # Test valid distance metrics
        for metric in IMPLEMENTED_DISTANCE.__args__:
            distance = Distance(metric=metric)
            assert distance.metric == metric

        # Test invalid metric
        with pytest.raises(AssertionError):
            Distance(metric="invalid_metric")  # type: ignore

    def test_cdist_euclidean(self):
        """Test Euclidean distance computation"""
        # Create a mock backend
        backend = Mock()
        backend.norm2D.return_value = np.array([0.0, 1.0])
        backend.clip.return_value = np.array([[0.0, 1.0], [1.0, 0.0]])
        backend.fill_diagonal.return_value = np.array([[0.0, 1.0], [1.0, 0.0]])

        distance = Distance(metric="euclidean")
        distance.backend = backend

        # Test with simple input
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = distance._cdist_euclidean(X, X)
        np.testing.assert_array_almost_equal(result, cdist(X, X, metric="sqeuclidean"))

    def test_compute_covariance_matrix(self):
        """Test covariance matrix computation"""
        # Create a mock backend
        backend = Mock()
        backend.mean.return_value = np.array([0.5, 0.5])
        backend.cholesky.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])

        distance = Distance(metric="mahalanobis")
        distance.backend = backend

        # Test with simple input
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = distance._compute_covariance_matrix(X)
        np.testing.assert_array_almost_equal(result, np.cov(X))

    def test_cdist_mahalanobis(self):
        """Test Mahalanobis distance computation"""
        # Create a mock backend
        backend = Mock()
        backend.norm2D.return_value = np.array([0.0, 1.0])
        backend.clip.return_value = np.array([[0.0, 1.0], [1.0, 0.0]])
        backend.fill_diagonal.return_value = np.array([[0.0, 1.0], [1.0, 0.0]])
        backend.mean.return_value = np.array([0.5, 0.5])
        backend.cholesky.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
        backend.solve_triangular.return_value = np.array([[0.0, 0.0], [1.0, 1.0]])

        distance = Distance(metric="mahalanobis")
        distance.backend = backend

        # Test with simple input
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = distance.cdist(X, X)
        np.testing.assert_array_almost_equal(result, cdist(X, X, metric="mahalanobis"))
