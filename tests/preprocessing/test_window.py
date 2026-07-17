"""
Unit tests for Windowizer functionality
"""

import unittest

import numpy as np

from cristal.backend import NumpyBackend
from cristal.preprocessing.windowing import Windowizer


class TestWindowizer(unittest.TestCase):
    """Test the Windowizer functionality"""

    def setUp(self):
        np.random.seed(42)
        self.backend = NumpyBackend()

        # Simple data for exact verification
        self.simple_signal_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Data for roundtrip tests
        self.signal_1d = np.random.rand(100)
        self.signal_2d = np.random.rand(100, 3)

        # Window parameters
        self.window_size = 10
        self.shift = 1

    def test_initialization(self):
        """Test Windowizer initialization with valid and invalid parameters"""
        # Test valid parameters
        windowizer = Windowizer(window_size=10, shift=1)
        self.assertEqual(windowizer.window_size, 10)
        self.assertEqual(windowizer.shift, 1)
        self.assertIsNone(windowizer.backend)

        # Test default shift
        windowizer = Windowizer(window_size=5)
        self.assertEqual(windowizer.shift, 1)

        # Test invalid window_size
        with self.assertRaises(ValueError):
            Windowizer(window_size=0)
        with self.assertRaises(ValueError):
            Windowizer(window_size=-5)

        # Test invalid shift
        with self.assertRaises(ValueError):
            Windowizer(window_size=10, shift=0)
        with self.assertRaises(ValueError):
            Windowizer(window_size=10, shift=-1)

    def test_fit(self):
        """Test fit method"""
        windowizer = Windowizer(window_size=self.window_size, shift=self.shift)

        # Test fit without backend
        with self.assertRaises(ValueError):
            windowizer.fit(self.signal_1d)

        # Bind backend and test fit
        windowizer.backend = self.backend
        result = windowizer.fit(self.signal_1d)

        # fit should return self
        self.assertIs(result, windowizer)

        # Test fit with 2D data
        windowizer_2d = Windowizer(window_size=5, shift=2)
        windowizer_2d.backend = self.backend
        windowizer_2d.fit(self.signal_2d)

    def test_fit_invalid_dimensions(self):
        """Test fit with invalid input dimensions"""
        windowizer = Windowizer(window_size=5, shift=1)
        windowizer.backend = self.backend

        # 3D data should raise error
        with self.assertRaises(ValueError):
            windowizer.fit(np.random.rand(10, 5, 3))

        # 1D data is valid
        windowizer.fit(np.random.rand(100))

    def test_transform(self):
        """Test transform method with various inputs"""
        windowizer = Windowizer(window_size=3, shift=1)
        windowizer.backend = self.backend

        # Fit and transform 1D data
        windowizer.fit(self.simple_signal_1d)
        windows = windowizer.transform(self.simple_signal_1d)

        # For signal of length 5, window_size=3, shift=1
        # Number of windows = floor((5-3)/1) + 1 = 3
        self.assertEqual(windows.shape, (3, 3))

    def test_transform_2d(self):
        """Test transform with 2D input"""
        windowizer = Windowizer(window_size=5, shift=2)
        windowizer.backend = self.backend

        # Create 2D data: (n_samples, n_channels)
        data_2d = np.random.rand(20, 4)
        windowizer.fit(data_2d)

        windows = windowizer.transform(data_2d)

        # Number of windows = floor((20-5)/2) + 1 = 8
        # Shape should be (n_windows, window_size, n_channels)
        self.assertEqual(windows.shape, (8, 5, 4))

    def test_transform_invalid_dimensions(self):
        """Test transform with invalid input dimensions"""
        windowizer = Windowizer(window_size=5, shift=1)
        windowizer.backend = self.backend

        # Fit with 1D
        windowizer.fit(np.random.rand(100))

        # Transform with 3D should raise error
        with self.assertRaises(ValueError):
            windowizer.transform(np.random.rand(10, 5, 3))

    def test_transform_no_backend(self):
        """Test that transform raises ValueError when backend is not bound"""
        windowizer = Windowizer(window_size=5, shift=1)

        with self.assertRaises(ValueError):
            windowizer.transform(self.simple_signal_1d)

    def test_transform_output_values(self):
        """Test that transform produces correct window values"""
        windowizer = Windowizer(window_size=3, shift=1)
        windowizer.backend = self.backend

        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        windowizer.fit(signal)
        windows = windowizer.transform(signal)

        # Expected windows:
        # Window 0: [1, 2, 3]
        # Window 1: [2, 3, 4]
        # Window 2: [3, 4, 5]
        expected = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])

        np.testing.assert_almost_equal(windows, expected)

    def test_transform_with_shift(self):
        """Test transform with different shift values"""
        signal = np.arange(10)

        # Shift = 1
        windowizer = Windowizer(window_size=3, shift=1)
        windowizer.backend = self.backend
        windowizer.fit(signal)
        windows_1 = windowizer.transform(signal)
        self.assertEqual(windows_1.shape[0], 8)  # 10 - 3 + 1 = 8

        # Shift = 2
        windowizer = Windowizer(window_size=3, shift=2)
        windowizer.backend = self.backend
        windowizer.fit(signal)
        windows_2 = windowizer.transform(signal)
        self.assertEqual(windows_2.shape[0], 4)  # floor((10-3)/2) + 1 = 3 + 1 = 4

        # Shift = 3 (non-overlapping)
        windowizer = Windowizer(window_size=3, shift=3)
        windowizer.backend = self.backend
        windowizer.fit(signal)
        windows_3 = windowizer.transform(signal)
        self.assertEqual(windows_3.shape[0], 3)  # floor((10-3)/3) + 1 = 2 + 1 = 3

    def test_fit_transform(self):
        """Test fit_transform convenience method"""
        windowizer = Windowizer(window_size=5, shift=2)
        windowizer.backend = self.backend

        result = windowizer.fit_transform(self.signal_1d)

        expected_n_windows = (len(self.signal_1d) - 5) // 2 + 1
        self.assertEqual(result.shape, (expected_n_windows, 5))

    # ==================== Inverse Transform Tests ====================

    def test_inverse_transform_1d_input(self):
        """Test inverse_transform with 1D input (window averages only)"""
        windowizer = Windowizer(window_size=3, shift=1)
        windowizer.backend = self.backend

        # Create window averages from signal [1, 2, 3, 4, 5]
        # Windows: [1,2,3] avg=2, [2,3,4] avg=3, [3,4,5] avg=4
        window_averages = np.array([2.0, 3.0, 4.0])

        # Expected: each point gets average of windows it belongs to
        # point 0: window 0 -> 2
        # point 1: windows 0,1 -> (2+3)/2 = 2.5
        # point 2: windows 0,1,2 -> (2+3+4)/3 = 3
        # point 3: windows 1,2 -> (3+4)/2 = 3.5
        # point 4: window 2 -> 4
        expected_output = np.array([2.0, 2.5, 3.0, 3.5, 4.0])

        result = windowizer.inverse_transform(window_averages, original_length=5)

        np.testing.assert_almost_equal(result, expected_output)

    def test_inverse_transform_2d_input(self):
        """Test inverse_transform with 2D input (full window values)"""
        windowizer = Windowizer(window_size=3, shift=1)
        windowizer.backend = self.backend

        # Windows from signal [1, 2, 3, 4, 5]
        windows = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])

        result = windowizer.inverse_transform(windows, original_length=5)

        # Each point should be average of its occurrences in windows
        # point 0: only in window 0 at pos 0 -> 1
        # point 1: window 0 pos 1=2, window 1 pos 0=2 -> (2+2)/2 = 2
        # point 2: window 0 pos 2=3, window 1 pos 1=3, window 2 pos 0=3 -> 3
        # point 3: window 1 pos 2=4, window 2 pos 1=4 -> (4+4)/2 = 4
        # point 4: only in window 2 pos 2 -> 5
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_almost_equal(result, expected)

    def test_inverse_transform_3d_input(self):
        """Test inverse_transform with 3D input (multi-channel)"""
        windowizer = Windowizer(window_size=3, shift=1)
        windowizer.backend = self.backend

        # 3 windows, window_size=3, 2 channels
        windows = np.random.rand(3, 3, 2)

        result = windowizer.inverse_transform(windows, original_length=5)

        # Result should be (5, 2)
        self.assertEqual(result.shape, (5, 2))

        # Values should be within window range
        self.assertTrue(np.all(result >= windows.min()))
        self.assertTrue(np.all(result <= windows.max()))

    def test_inverse_transform_roundtrip_1d(self):
        """Test that transform -> inverse_transform reconstructs original 1D signal"""
        windowizer = Windowizer(window_size=self.window_size, shift=self.shift)
        windowizer.backend = self.backend

        windows = windowizer.transform(self.signal_1d)
        reconstructed = windowizer.inverse_transform(windows, original_length=len(self.signal_1d))

        np.testing.assert_almost_equal(reconstructed, self.signal_1d, decimal=10)

    def test_inverse_transform_roundtrip_2d(self):
        """Test that transform -> inverse_transform reconstructs original 2D signal"""
        windowizer = Windowizer(window_size=self.window_size, shift=self.shift)
        windowizer.backend = self.backend

        windows = windowizer.transform(self.signal_2d)
        reconstructed = windowizer.inverse_transform(windows, original_length=len(self.signal_2d))

        np.testing.assert_almost_equal(reconstructed, self.signal_2d, decimal=10)

    def test_inverse_transform_original_length_handling(self):
        """Test inverse_transform with explicit and implicit original_length"""
        windowizer = Windowizer(window_size=3, shift=1)
        windowizer.backend = self.backend

        window_averages = np.array([2.0, 3.0, 4.0])

        # With explicit length (min: 3*2+3-1=8, max: 3*1+3-1=5... wait let's calculate)
        # N_windows=3, shift=1, window_size=3
        # min_length = 1*(3-1) + 3 = 5
        # max_length = 1*3 + 3 - 1 = 5
        # So original_length must be 5
        result_explicit = windowizer.inverse_transform(window_averages, original_length=5)
        self.assertEqual(result_explicit.shape, (5,))

        # Without original_length (uses max: shift * N_windows + window_size - 1 = 5)
        result_implicit = windowizer.inverse_transform(window_averages)
        self.assertEqual(result_implicit.shape, (5,))
        np.testing.assert_array_equal(result_explicit, result_implicit)

    def test_inverse_transform_original_length_range(self):
        """Test inverse_transform with shift=2 to get a range of valid lengths"""
        windowizer = Windowizer(window_size=3, shift=2)
        windowizer.backend = self.backend

        window_averages = np.array([2.0, 3.0, 4.0])

        # N_windows=3, shift=2, window_size=3
        # min_length = 2*(3-1) + 3 = 7
        # max_length = 2*3 + 3 - 1 = 8

        # Valid lengths: 7 and 8
        result_7 = windowizer.inverse_transform(window_averages, original_length=7)
        self.assertEqual(result_7.shape, (7,))

        result_8 = windowizer.inverse_transform(window_averages, original_length=8)
        self.assertEqual(result_8.shape, (8,))

        # Invalid lengths
        with self.assertRaises(ValueError):
            windowizer.inverse_transform(window_averages, original_length=6)
        with self.assertRaises(ValueError):
            windowizer.inverse_transform(window_averages, original_length=9)

    def test_inverse_transform_window_size_validation(self):
        """Test that inverse_transform validates window size for 2D+ inputs"""
        windowizer = Windowizer(window_size=3, shift=1)
        windowizer.backend = self.backend

        # Wrong window size should raise error for 2D input
        windows_wrong_size = np.random.rand(3, 5)

        with self.assertRaises(ValueError):
            windowizer.inverse_transform(windows_wrong_size)
