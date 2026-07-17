"""
Unit tests for MinMaxScaler
"""

import unittest

import numpy as np

from cristal.backend import NumpyBackend
from cristal.preprocessing.scalers import MinMaxScaler


class TestMinMaxScaler(unittest.TestCase):
    """Test the MinMaxScaler functionality"""

    def setUp(self):
        np.random.seed(42)
        self.backend = NumpyBackend()

        # Test data
        self.data_2d = np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0], [5.0, 6.0, 30.0]])  # shape (3, 3)
        self.data_3d = np.random.rand(10, 5, 3)  # shape (10, 5, 3)

        # Expected min/max for 2D data
        self.expected_min_2d = np.array([1.0, 2.0, 10.0])
        self.expected_max_2d = np.array([5.0, 6.0, 30.0])

        # Data with constant column (to test division by zero handling)
        self.data_with_constant = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])  # First column is constant

    def test_initialization(self):
        """Test MinMaxScaler initialization with valid and invalid parameters"""
        # Test default initialization
        scaler = MinMaxScaler()
        self.assertEqual(scaler.feature_range, (-1, 1))
        self.assertIsNone(scaler.min_)
        self.assertIsNone(scaler.max_)
        self.assertIsNone(scaler.backend)

        # Test valid custom feature_range
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.assertEqual(scaler.feature_range, (0, 1))

        # Test invalid feature_range (not 2D tuple)
        with self.assertRaises(ValueError):
            MinMaxScaler(feature_range=(0, 1, 2))

        with self.assertRaises(ValueError):
            MinMaxScaler(feature_range=(1,))

        # Test invalid feature_range (min > max)
        with self.assertRaises(ValueError):
            MinMaxScaler(feature_range=(1, 0))

    def test_fit_no_backend(self):
        """Test that fit raises ValueError when backend is not bound"""
        scaler = MinMaxScaler()
        with self.assertRaises(ValueError):
            scaler.fit(self.data_2d)

    def test_fit_2d(self):
        """Test fit with 2D data"""
        scaler = MinMaxScaler()
        scaler.backend = self.backend

        scaler.fit(self.data_2d)

        self.assertEqual(scaler.n_features, 3)
        self.assertIsNone(scaler.n_channel)

        np.testing.assert_almost_equal(scaler.min_, self.expected_min_2d)
        np.testing.assert_almost_equal(scaler.max_, self.expected_max_2d)

    def test_fit_3d(self):
        """Test fit with 3D data"""
        scaler = MinMaxScaler()
        scaler.backend = self.backend

        scaler.fit(self.data_3d)

        self.assertEqual(scaler.n_features, 5)
        self.assertEqual(scaler.n_channel, 3)
        self.assertEqual(scaler.min_.shape, (3,))
        self.assertEqual(scaler.max_.shape, (3,))

    def test_fit_invalid_dimensions(self):
        """Test fit with invalid input dimensions"""
        scaler = MinMaxScaler()
        scaler.backend = self.backend

        # 1D data should raise error
        with self.assertRaises(ValueError):
            scaler.fit(np.array([1.0, 2.0, 3.0]))

        # 4D data should raise error
        with self.assertRaises(ValueError):
            scaler.fit(np.random.rand(2, 2, 2, 2))

    def test_transform_not_fitted(self):
        """Test that transform raises ValueError when scaler is not fitted"""
        scaler = MinMaxScaler()
        scaler.backend = self.backend

        with self.assertRaises(ValueError):
            scaler.transform(self.data_2d)

    def test_transform_2d(self):
        """Test transform with 2D data"""
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.backend = self.backend
        scaler.fit(self.data_2d)

        result = scaler.transform(self.data_2d)

        # Check output shape
        self.assertEqual(result.shape, self.data_2d.shape)

        # Check that values are in the feature_range
        self.assertTrue(np.all(result >= -1))
        self.assertTrue(np.all(result <= 1))

        # Check specific values
        # For column 0: min=1, max=5, range=(-1,1)
        # Value 1 -> -1, Value 5 -> 1
        # (1-1)/(5-1) * 2 + (-1) = -1
        # (5-1)/(5-1) * 2 + (-1) = 1
        np.testing.assert_almost_equal(result[0, 0], -1)
        np.testing.assert_almost_equal(result[2, 0], 1)

    def test_transform_3d(self):
        """Test transform with 3D data"""
        scaler = MinMaxScaler()
        scaler.backend = self.backend
        scaler.fit(self.data_3d)

        result = scaler.transform(self.data_3d)

        # Check output shape
        self.assertEqual(result.shape, self.data_3d.shape)

        # Check that values are in the default feature_range
        self.assertTrue(np.all(result >= -1))
        self.assertTrue(np.all(result <= 1))

    def test_transform_invalid_shape(self):
        """Test transform with invalid input shape for both 2D and 3D data"""
        scaler = MinMaxScaler()
        scaler.backend = self.backend

        # Test 2D: fitted with 3 features
        scaler.fit(self.data_2d)  # shape (3, 3) -> n_features=3, n_channel=None

        # Wrong number of features for 2D
        wrong_shape_data_2d = np.random.rand(5, 4)
        with self.assertRaises(ValueError):
            scaler.transform(wrong_shape_data_2d)

        # Test 3D: fitted with (10, 5, 3) -> n_features=5, n_channel=3
        scaler_3d = MinMaxScaler()
        scaler_3d.backend = self.backend
        scaler_3d.fit(self.data_3d)

        # Wrong number of features for 3D (window_size)
        wrong_features_3d = np.random.rand(8, 4, 3)  # 4 features instead of 5
        with self.assertRaises(ValueError):
            scaler_3d.transform(wrong_features_3d)

        # Wrong number of channels for 3D
        wrong_channels_3d = np.random.rand(8, 5, 2)  # 2 channels instead of 3
        with self.assertRaises(ValueError):
            scaler_3d.transform(wrong_channels_3d)

    def test_inverse_transform_not_fitted(self):
        """Test that inverse_transform raises ValueError when scaler is not fitted"""
        scaler = MinMaxScaler()
        scaler.backend = self.backend

        with self.assertRaises(ValueError):
            scaler.inverse_transform(self.data_2d)

    def test_inverse_transform_invalid_shape(self):
        """Test inverse_transform with invalid input shape for both 2D and 3D data"""
        scaler = MinMaxScaler()
        scaler.backend = self.backend

        # Test 2D: fitted with 3 features
        scaler.fit(self.data_2d)  # shape (3, 3) -> n_features=3, n_channel=None
        scaled_2d = scaler.transform(self.data_2d)

        # Wrong number of features for 2D
        wrong_shape_data_2d = np.random.rand(5, 4)
        with self.assertRaises(ValueError):
            scaler.inverse_transform(wrong_shape_data_2d)

        # Test 3D: fitted with (10, 5, 3) -> n_features=5, n_channel=3
        scaler_3d = MinMaxScaler()
        scaler_3d.backend = self.backend
        scaler_3d.fit(self.data_3d)
        scaled_3d = scaler_3d.transform(self.data_3d)

        # Wrong number of features for 3D (window_size)
        wrong_features_3d = np.random.rand(8, 4, 3)  # 4 features instead of 5
        with self.assertRaises(ValueError):
            scaler_3d.inverse_transform(wrong_features_3d)

        # Wrong number of channels for 3D
        wrong_channels_3d = np.random.rand(8, 5, 2)  # 2 channels instead of 3
        with self.assertRaises(ValueError):
            scaler_3d.inverse_transform(wrong_channels_3d)

    def test_inverse_transform_2d(self):
        """Test inverse_transform with 2D data"""
        scaler = MinMaxScaler()
        scaler.backend = self.backend
        scaler.fit(self.data_2d)

        scaled = scaler.transform(self.data_2d)
        result = scaler.inverse_transform(scaled)

        np.testing.assert_almost_equal(result, self.data_2d)

    def test_inverse_transform_3d(self):
        """Test inverse_transform with 3D data"""
        scaler = MinMaxScaler()
        scaler.backend = self.backend
        scaler.fit(self.data_3d)

        scaled = scaler.transform(self.data_3d)
        result = scaler.inverse_transform(scaled)

        np.testing.assert_almost_equal(result, self.data_3d)

    def test_roundtrip_2d(self):
        """Test fit_transform + inverse_transform roundtrip with 2D data"""
        scaler = MinMaxScaler()
        scaler.backend = self.backend

        result = scaler.fit_transform(self.data_2d)
        reconstructed = scaler.inverse_transform(result)

        np.testing.assert_almost_equal(reconstructed, self.data_2d)

    def test_roundtrip_3d(self):
        """Test fit_transform + inverse_transform roundtrip with 3D data"""
        scaler = MinMaxScaler()
        scaler.backend = self.backend

        result = scaler.fit_transform(self.data_3d)
        reconstructed = scaler.inverse_transform(result)

        np.testing.assert_almost_equal(reconstructed, self.data_3d)

    def test_custom_feature_range(self):
        """Test with custom feature_range (0, 1)"""
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.backend = self.backend
        scaler.fit(self.data_2d)

        result = scaler.transform(self.data_2d)

        # Check that values are in [0, 1]
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))

        # Check specific values for column 0
        np.testing.assert_almost_equal(result[0, 0], 0.0)  # min value
        np.testing.assert_almost_equal(result[2, 0], 1.0)  # max value

    def test_constant_column_handling(self):
        """Test that constant columns (scale=0) are handled without division by zero"""
        scaler = MinMaxScaler()
        scaler.backend = self.backend

        # Should not raise error
        scaler.fit(self.data_with_constant)
        result = scaler.transform(self.data_with_constant)
        reconstructed = scaler.inverse_transform(result)

        np.testing.assert_almost_equal(reconstructed, self.data_with_constant)

    def test_different_batch_sizes(self):
        """Test that transform works with different batch sizes than fit"""
        scaler = MinMaxScaler()
        scaler.backend = self.backend

        # Fit with 3 samples
        scaler.fit(self.data_2d)

        # Transform with different number of samples
        new_data = np.array([[2.0, 3.0, 15.0], [4.0, 5.0, 25.0]])

        result = scaler.transform(new_data)
        self.assertEqual(result.shape, (2, 3))

        # Check that min/max are still from fit
        self.assertTrue(np.all(result >= -1))
        self.assertTrue(np.all(result <= 1))

    def test_single_feature(self):
        """Test with single feature column"""
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        scaler = MinMaxScaler()
        scaler.backend = self.backend

        result = scaler.fit_transform(data)
        reconstructed = scaler.inverse_transform(result)

        np.testing.assert_almost_equal(reconstructed, data)
