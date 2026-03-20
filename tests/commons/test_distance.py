"""
Unit tests for Distance class
"""

import unittest

import numpy as np
from scipy.spatial.distance import cdist

from cristal.backend import NumpyBackend
from cristal.commons.distance import IMPLEMENTED_DISTANCE, Distance


class TestDistance(unittest.TestCase):
    """Test the Distance class functionality"""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.rand(10, 5)
        self.Y = np.random.rand(15, 5)

    def test_distance_initialization(self):
        """Test Distance initialization with valid parameters"""
        # Test valid distance metrics
        for metric in IMPLEMENTED_DISTANCE.__args__:
            distance = Distance(metric=metric)
            self.assertEqual(distance.metric, metric)

            # Test no backend bound
            self.assertRaises(ValueError, distance, self.X)
            self.assertRaises(ValueError, distance.cdist, self.X)
            self.assertRaises(ValueError, distance._compute_covariance_matrix, self.X)
            self.assertRaises(ValueError, distance._cdist_euclidean, self.X)

        # Test invalid metric
        self.assertRaises(ValueError, Distance, metric="invalid_metric")

    def test_cdist_euclidean(self):
        """Test Euclidean distance computation"""
        backend = NumpyBackend()

        distance = Distance(metric="euclidean")
        distance.backend = backend

        # Compute Distance results (test)
        result_X = distance(self.X)
        result_X_X = distance(self.X, self.X)
        result_X_Y = distance(self.X, self.Y)
        result_Y_X = distance(self.Y, self.X)
        result_X_1 = distance(self.X, p=1)
        result_X_3 = distance(self.X, p=3)

        # Compute cdist results (ground truth)
        cdist_X = cdist(self.X, self.X, metric="sqeuclidean")
        cdist_X_eucl = cdist(self.X, self.X, metric="euclidean")
        cdist_X_Y = cdist(self.X, self.Y, metric="sqeuclidean")

        # Test euclidean square on X and X
        np.testing.assert_array_almost_equal(result_X, cdist_X, decimal=8, err_msg="X sqeuclidean")
        np.testing.assert_array_almost_equal(result_X_X, cdist_X, decimal=8, err_msg="X X sqeuclidean")

        # Test euclidean square on X and Y
        np.testing.assert_array_almost_equal(result_X_Y, cdist_X_Y, decimal=8, err_msg="X Y sqeuclidean")
        np.testing.assert_array_almost_equal(result_Y_X, cdist_X_Y.T, decimal=8, err_msg="Y X sqeuclidean")

        # Test euclidean^p on X
        np.testing.assert_array_almost_equal(result_X_1, cdist_X_eucl, decimal=8, err_msg="X euclidean")
        np.testing.assert_array_almost_equal(result_X_3, cdist_X_eucl**3, decimal=8, err_msg="X euclidean^3")

    def test_compute_covariance_matrix(self):
        """Test covariance matrix computation"""
        backend = NumpyBackend()

        distance = Distance(metric="mahalanobis")
        distance.backend = backend

        # Compute Distance results (test)
        result_X = distance._compute_covariance_matrix(self.X)
        result_Y = distance._compute_covariance_matrix(self.Y)

        # Test on X
        np.testing.assert_array_almost_equal(result_X, np.cov(self.X, rowvar=False), decimal=8, err_msg="cov X")

        # Test on Y
        np.testing.assert_array_almost_equal(result_Y, np.cov(self.Y, rowvar=False), decimal=8, err_msg="cov Y")

    def test_cdist_mahalanobis(self):
        """Test Mahalanobis distance computation"""
        backend = NumpyBackend()

        distance = Distance(metric="mahalanobis")
        distance.backend = backend

        # Compute Distance results (test)
        result_X = distance(self.X)
        result_X_X = distance(self.X, self.X)
        result_X_Y = distance(self.X, self.Y)
        result_Y_X = distance(self.Y, self.X)
        result_X_1 = distance(self.X, p=1)
        result_X_3 = distance(self.X, p=3)

        # Compute cdist results (ground truth)
        cdist_X = cdist(self.X, self.X, metric="mahalanobis")
        cdist_X_squared = cdist(self.X, self.X, metric="mahalanobis") ** 2
        cdist_X_Y_squared = cdist(self.X, self.Y, metric="mahalanobis") ** 2

        # Test euclidean square on X and X
        np.testing.assert_array_almost_equal(result_X, cdist_X_squared, decimal=6, err_msg="X mahalanobis")
        np.testing.assert_array_almost_equal(result_X_X, cdist_X_squared, decimal=8, err_msg="X X mahalanobis")

        # Test euclidean square on X and Y
        np.testing.assert_array_almost_equal(result_X_Y, cdist_X_Y_squared, decimal=8, err_msg="X Y mahalanobis")
        np.testing.assert_array_almost_equal(result_Y_X, cdist_X_Y_squared.T, decimal=8, err_msg="Y X mahalanobis")

        # Test euclidean^p on X
        np.testing.assert_array_almost_equal(result_X_1, cdist_X, decimal=7, err_msg="X mahalanobis^1")
        np.testing.assert_array_almost_equal(result_X_3, cdist_X**3, decimal=8, err_msg="X mahalanobis^3")
