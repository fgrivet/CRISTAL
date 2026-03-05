# File: cristal/tests/backend/test_numpy_backend.py
import unittest
import numpy as np
import torch
from unittest.mock import patch

# Import the backend classes
from cristal.backend.numpy_backend import NumpyBackend


class TestNumpyBackend(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.backend = NumpyBackend(dtype=np.float64)

    def test_init(self):
        """Test backend initialization"""
        self.assertEqual(self.backend.default_dtype, np.float64)

    def test_is_array_like(self):
        """Test is_array_like method"""
        arr = np.array([1, 2, 3])
        self.assertTrue(self.backend.is_array_like(arr))
        self.assertFalse(self.backend.is_array_like([1, 2, 3]))
        self.assertFalse(self.backend.is_array_like(5))

    def test_to_array_like(self):
        """Test to_array_like method"""
        # Test with list
        result = self.backend.to_array_like([1, 2, 3])
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)

        # Test with dtype conversion
        result = self.backend.to_array_like([1, 2, 3], dtype=np.float32)
        self.assertEqual(result.dtype, np.float32)

    def test_to_numpy(self):
        """Test to_numpy method"""
        arr = np.array([1, 2, 3])
        result = self.backend.to_numpy(arr)
        np.testing.assert_array_equal(result, arr)

    def test_zeros(self):
        """Test zeros creation"""
        result = self.backend.zeros((2, 3))
        expected = np.zeros((2, 3))
        np.testing.assert_array_equal(result, expected)

        # Test with dtype
        result = self.backend.zeros((2, 3), dtype=np.float32)
        self.assertEqual(result.dtype, np.float32)

    def test_ones(self):
        """Test ones creation"""
        result = self.backend.ones((2, 3))
        expected = np.ones((2, 3))
        np.testing.assert_array_equal(result, expected)

        # Test with dtype
        result = self.backend.ones((2, 3), dtype=np.float32)
        self.assertEqual(result.dtype, np.float32)

    def test_eye(self):
        """Test eye creation"""
        result = self.backend.eye(3)
        expected = np.eye(3)
        np.testing.assert_array_equal(result, expected)

        # Test with dtype
        result = self.backend.eye(3, dtype=np.float32)
        self.assertEqual(result.dtype, np.float32)

    def test_full(self):
        """Test full creation"""
        result = self.backend.full((2, 3), 5)
        expected = np.full((2, 3), 5)
        np.testing.assert_array_equal(result, expected)

        # Test with dtype
        result = self.backend.full((2, 3), 5, dtype=np.float32)
        self.assertEqual(result.dtype, np.float32)

    def test_arange(self):
        """Test arange creation"""
        result = self.backend.arange(0, 5)
        expected = np.arange(0, 5)
        np.testing.assert_array_equal(result, expected)

        # Test with dtype
        result = self.backend.arange(0, 5, dtype=np.float32)
        self.assertEqual(result.dtype, np.float32)

    def test_set_seed(self):
        """Test set_seed method"""
        self.backend.set_seed(42)
        result1 = self.backend.random((2, 2))

        self.backend.set_seed(42)
        result2 = self.backend.random((2, 2))

        np.testing.assert_array_equal(result1, result2)

    def test_random(self):
        """Test random sampling"""
        result = self.backend.random((2, 3))
        self.assertEqual(result.shape, (2, 3))
        self.assertTrue(np.all(result >= 0) and np.all(result < 1))

    def test_randn(self):
        """Test normal distribution sampling"""
        result = self.backend.randn(0, 1, (2, 3))
        self.assertEqual(result.shape, (2, 3))

    def test_randint(self):
        """Test integer random sampling"""
        result = self.backend.randint(0, 10, (2, 3))
        self.assertEqual(result.shape, (2, 3))
        self.assertTrue(np.all(result >= 0) and np.all(result < 10))

    def test_swap(self):
        """Test swap operation"""
        arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = self.backend.swap(arr, 0, 1)
        expected = np.swapaxes(arr, 0, 1)
        np.testing.assert_array_equal(result, expected)

    def test_concat(self):
        """Test concatenation"""
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[5, 6], [7, 8]])
        result = self.backend.concat([arr1, arr2], axis=0)
        expected = np.concatenate([arr1, arr2], axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_stack(self):
        """Test stacking"""
        arr1 = np.array([1, 2])
        arr2 = np.array([3, 4])
        result = self.backend.stack([arr1, arr2], axis=0)
        expected = np.stack([arr1, arr2], axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_broadcast(self):
        """Test broadcasting"""
        arr = np.array([1, 2, 3])
        result = self.backend.broadcast(arr, (2, 3))
        expected = np.broadcast_to(arr, (2, 3))
        np.testing.assert_array_equal(result, expected)

    def test_sum(self):
        """Test sum operation"""
        arr = np.array([[1, 2], [3, 4]])
        result = self.backend.sum(arr)
        self.assertEqual(result, 10)

        result = self.backend.sum(arr, axis=0)
        expected = np.sum(arr, axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_prod(self):
        """Test product operation"""
        arr = np.array([[1, 2], [3, 4]])
        result = self.backend.prod(arr)
        self.assertEqual(result, 24)

        result = self.backend.prod(arr, axis=0)
        expected = np.prod(arr, axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_cumsum(self):
        """Test cumulative sum operation"""
        arr = np.array([[1, 2], [3, 4]])
        result = self.backend.cumsum(arr)
        expected = np.cumsum(arr)
        np.testing.assert_array_equal(result, expected)

    def test_min(self):
        """Test min operation"""
        arr = np.array([[1, 2], [3, 4]])
        result = self.backend.min(arr)
        self.assertEqual(result, 1)

        result = self.backend.min(arr, axis=0)
        expected = np.min(arr, axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_max(self):
        """Test max operation"""
        arr = np.array([[1, 2], [3, 4]])
        result = self.backend.max(arr)
        self.assertEqual(result, 4)

        result = self.backend.max(arr, axis=0)
        expected = np.max(arr, axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_argmin(self):
        """Test argmin operation"""
        arr = np.array([[1, 2], [3, 4]])
        result = self.backend.argmin(arr)
        self.assertEqual(result, 0)

        result = self.backend.argmin(arr, axis=0)
        expected = np.argmin(arr, axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_argmax(self):
        """Test argmax operation"""
        arr = np.array([[1, 2], [3, 4]])
        result = self.backend.argmax(arr)
        self.assertEqual(result, 3)

        result = self.backend.argmax(arr, axis=0)
        expected = np.argmax(arr, axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_mean(self):
        """Test mean operation"""
        arr = np.array([[1, 2], [3, 4]])
        result = self.backend.mean(arr)
        self.assertEqual(result, 2.5)

        result = self.backend.mean(arr, axis=0)
        expected = np.mean(arr, axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_std(self):
        """Test standard deviation operation"""
        arr = np.array([[1, 2], [3, 4]])
        result = self.backend.std(arr)
        expected = np.std(arr)
        np.testing.assert_array_equal(result, expected)

    def test_norm(self):
        """Test norm operation"""
        arr = np.array([[1, 2], [3, 4]])
        result = self.backend.norm(arr)
        expected = np.linalg.norm(arr)
        self.assertAlmostEqual(result, expected)

    def test_norm2D(self):
        """Test 2D norm operation"""
        arr = np.array([[1, 2], [3, 4]])
        result = self.backend.norm2D(arr)
        expected = np.einsum("ij,ij->i", arr, arr)
        np.testing.assert_array_equal(result, expected)

    def test_einsum(self):
        """Test einsum operation"""
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[5, 6], [7, 8]])
        result = self.backend.einsum("ij,jk->ik", arr1, arr2)
        expected = np.einsum("ij,jk->ik", arr1, arr2)
        np.testing.assert_array_equal(result, expected)

    def test_where_with_values(self):
        """Test where with values"""
        condition = np.array([[True, False], [False, True]])
        true_val = np.array([[1, 1], [1, 1]])
        false_val = np.array([[0, 0], [0, 0]])
        result = self.backend.where(condition, true_val, false_val)
        expected = np.where(condition, true_val, false_val)
        np.testing.assert_array_equal(result, expected)

    def test_where_without_values(self):
        """Test where without values"""
        condition = np.array([[True, False], [False, True]])
        result = self.backend.where(condition)
        expected = np.where(condition)
        self.assertEqual(len(result), len(expected))
        for i in range(len(result)):
            np.testing.assert_array_equal(result[i], expected[i])

    def test_clip(self):
        """Test clip operation"""
        arr = np.array([1, 2, 3, 4, 5])
        result = self.backend.clip(arr, 2, 4)
        expected = np.clip(arr, 2, 4)
        np.testing.assert_array_equal(result, expected)

    def test_fill_diagonal(self):
        """Test fill diagonal operation"""
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = self.backend.fill_diagonal(arr, 0)
        expected = arr.copy()
        np.fill_diagonal(expected, 0)
        np.testing.assert_array_equal(result, expected)

    def test_diag(self):
        """Test diag operation"""
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = self.backend.diag(arr)
        expected = np.diag(arr)
        np.testing.assert_array_equal(result, expected)

    def test_nan(self):
        """Test nan creation"""
        result = self.backend.nan()
        self.assertTrue(np.isnan(result))

    def test_isnan(self):
        """Test isnan operation"""
        arr = np.array([1, np.nan, 3])
        result = self.backend.isnan(arr)
        expected = np.isnan(arr)
        np.testing.assert_array_equal(result, expected)

    def test_pow(self):
        """Test power operation"""
        arr = np.array([1, 2, 3])
        result = self.backend.pow(arr, 2)
        expected = np.power(arr, 2)
        np.testing.assert_array_equal(result, expected)

    def test_sqrt(self):
        """Test square root operation"""
        arr = np.array([1, 4, 9])
        result = self.backend.sqrt(arr)
        expected = np.sqrt(arr)
        np.testing.assert_array_equal(result, expected)

    def test_abs(self):
        """Test absolute operation"""
        arr = np.array([-1, 2, -3])
        result = self.backend.abs(arr)
        expected = np.abs(arr)
        np.testing.assert_array_equal(result, expected)

    def test_exp(self):
        """Test exponential operation"""
        arr = np.array([0, 1, 2])
        result = self.backend.exp(arr)
        expected = np.exp(arr)
        np.testing.assert_array_equal(result, expected)

    def test_log(self):
        """Test logarithm operation"""
        arr = np.array([1, np.e, np.e**2])
        result = self.backend.log(arr)
        expected = np.log(arr)
        np.testing.assert_array_equal(result, expected)

    def test_cos(self):
        """Test cosine operation"""
        arr = np.array([0, np.pi / 2, np.pi])
        result = self.backend.cos(arr)
        expected = np.cos(arr)
        np.testing.assert_array_equal(result, expected)

    def test_sin(self):
        """Test sine operation"""
        arr = np.array([0, np.pi / 2, np.pi])
        result = self.backend.sin(arr)
        expected = np.sin(arr)
        np.testing.assert_array_equal(result, expected)

    def test_tan(self):
        """Test tangent operation"""
        arr = np.array([0, np.pi / 4, np.pi / 2])
        result = self.backend.tan(arr)
        expected = np.tan(arr)
        np.testing.assert_array_equal(result, expected)

    def test_cosh(self):
        """Test hyperbolic cosine operation"""
        arr = np.array([0, 1, 2])
        result = self.backend.cosh(arr)
        expected = np.cosh(arr)
        np.testing.assert_array_equal(result, expected)

    def test_sinh(self):
        """Test hyperbolic sine operation"""
        arr = np.array([0, 1, 2])
        result = self.backend.sinh(arr)
        expected = np.sinh(arr)
        np.testing.assert_array_equal(result, expected)

    def test_tanh(self):
        """Test hyperbolic tangent operation"""
        arr = np.array([0, 1, 2])
        result = self.backend.tanh(arr)
        expected = np.tanh(arr)
        np.testing.assert_array_equal(result, expected)

    def test_inv(self):
        """Test matrix inversion"""
        arr = np.array([[4, 7], [2, 6]])
        result = self.backend.inv(arr)
        expected = np.linalg.inv(arr)
        np.testing.assert_array_almost_equal(result, expected)

    def test_pinv(self):
        """Test pseudo-inverse"""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        result = self.backend.pinv(arr)
        expected = np.linalg.pinv(arr)
        np.testing.assert_array_almost_equal(result, expected)

    def test_solve(self):
        """Test solving linear systems"""
        A = np.array([[3, 1], [1, 2]])
        b = np.array([9, 8])
        result = self.backend.solve(A, b)
        expected = np.linalg.solve(A, b)
        np.testing.assert_array_almost_equal(result, expected)

    def test_qr(self):
        """Test QR decomposition"""
        A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
        Q, R = self.backend.qr(A)
        reconstructed = Q @ R
        np.testing.assert_array_almost_equal(reconstructed, A)

    def test_vander(self):
        """Test Vandermonde matrix"""
        a = np.array([1, 2, 3])
        result = self.backend.vander(a, 2)
        expected = np.vander(a, 3)
        np.testing.assert_array_equal(result, expected)

    def test_lstsq(self):
        """Test least squares solution"""
        A = np.array([[1, 1], [1, 2], [1, 3]])
        B = np.array([1, 2, 3])
        result = self.backend.lstsq(A, B)
        expected = np.linalg.lstsq(A, B, rcond=None)[0]
        np.testing.assert_array_almost_equal(result, expected)
