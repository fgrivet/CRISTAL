"""
Comprehensive test suite for the NumpyBackend class.
Tests all methods with 1D, 2D, and 3D arrays where applicable.
"""

import numpy as np
import torch

from cristal.backend.numpy_backend import NumpyBackend

from .base_test_backend import BaseTestBackend


class TestNumpyBackend(BaseTestBackend):
    """Test suite for NumpyBackend implementation."""

    @classmethod
    def setUpClass(cls):
        """Set up the backend for testing."""
        cls.backend = NumpyBackend()

    def int32(self):
        return np.int32

    def int64(self):
        return np.int64

    def _all(self, cond):
        return bool(np.all(cond))

    # ===== Type Tests =====

    def test_is_array_like(self):
        """Test type checking with different array types."""
        # 1D array
        arr_1d = np.zeros(5)
        self.assertTrue(self.backend.is_array_like(arr_1d))

        # 2D array
        arr_2d = np.zeros((3, 4))
        self.assertTrue(self.backend.is_array_like(arr_2d))

        # 3D array
        arr_3d = np.zeros((2, 3, 4))
        self.assertTrue(self.backend.is_array_like(arr_3d))

        # Non-array types
        self.assertFalse(self.backend.is_array_like([1, 2, 3]))
        self.assertFalse(self.backend.is_array_like(torch.Tensor([1, 2, 3])))
        self.assertFalse(self.backend.is_array_like(5))
        self.assertFalse(self.backend.is_array_like(None))

    def test_to_array_like(self):
        """Test conversion to array_like with different dtypes."""
        # Convert list
        result = self.backend.to_array_like([1, 2, 3])
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

        # Convert with dtype
        result = self.backend.to_array_like([1, 2, 3], dtype=np.float32)
        self.assertEqual(result.dtype, np.float32)

        # Convert nested list
        result = self.backend.to_array_like([[1, 2], [3, 4]])
        self.assertEqual(result.shape, (2, 2))
