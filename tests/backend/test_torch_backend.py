# File: cristal/tests/backend/test_torch_backend.py
import unittest
import numpy as np
import torch
from unittest.mock import patch

# Import the backend classes
from cristal.backend.torch_backend import TorchBackend


class TestTorchBackend(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Use CPU for testing
        self.backend = TorchBackend(dtype=torch.float64, device="cpu")

    def test_init(self):
        """Test backend initialization"""
        self.assertEqual(self.backend.default_dtype, torch.float64)
        self.assertEqual(self.backend.device, "cpu")

    def test_is_array_like(self):
        """Test is_array_like method"""
        arr = torch.tensor([1, 2, 3])
        self.assertTrue(self.backend.is_array_like(arr))
        self.assertFalse(self.backend.is_array_like([1, 2, 3]))
        self.assertFalse(self.backend.is_array_like(5))

    def test_to_array_like(self):
        """Test to_array_like method"""
        # Test with list
        result = self.backend.to_array_like([1, 2, 3])
        expected = torch.tensor([1, 2, 3])
        torch.testing.assert_close(result, expected)

        # Test with dtype conversion
        result = self.backend.to_array_like([1, 2, 3], dtype=torch.float32)
        self.assertEqual(result.dtype, torch.float32)

    def test_to_numpy(self):
        """Test to_numpy method"""
        arr = torch.tensor([1, 2, 3])
        result = self.backend.to_numpy(arr)
        np.testing.assert_array_equal(result, arr.cpu().numpy())

    def test_zeros(self):
        """Test zeros creation"""
        result = self.backend.zeros((2, 3))
        expected = torch.zeros((2, 3))
        torch.testing.assert_close(result, expected)

        # Test with dtype
        result = self.backend.zeros((2, 3), dtype=torch.float32)
        self.assertEqual(result.dtype, torch.float32)

    def test_ones(self):
        """Test ones creation"""
        result = self.backend.ones((2, 3))
        expected = torch.ones((2, 3))
        torch.testing.assert_close(result, expected)

        # Test with dtype
        result = self.backend.ones((2, 3), dtype=torch.float32)
        self.assertEqual(result.dtype, torch.float32)

    def test_eye(self):
        """Test eye creation"""
        result = self.backend.eye(3)
        expected = torch.eye(3)
        torch.testing.assert_close(result, expected)

        # Test with dtype
        result = self.backend.eye(3, dtype=torch.float32)
        self.assertEqual(result.dtype, torch.float32)

    def test_full(self):
        """Test full creation"""
        result = self.backend.full((2, 3), 5)
        expected = torch.full((2, 3), 5)
        torch.testing.assert_close(result, expected)

        # Test with dtype
        result = self.backend.full((2, 3), 5, dtype=torch.float32)
        self.assertEqual(result.dtype, torch.float32)

    def test_arange(self):
        """Test arange creation"""
        result = self.backend.arange(0, 5)
        expected = torch.arange(0, 5)
        torch.testing.assert_close(result, expected)

        # Test with dtype
        result = self.backend.arange(0, 5, dtype=torch.float32)
        self.assertEqual(result.dtype, torch.float32)

    def test_set_seed(self):
        """Test set_seed method"""
        self.backend.set_seed(42)
        result1 = self.backend.random((2, 2))

        self.backend.set_seed(42)
        result2 = self.backend.random((2, 2))

        torch.testing.assert_close(result1, result2)

    def test_random(self):
        """Test random sampling"""
        result = self.backend.random((2, 3))
        self.assertEqual(result.shape, (2, 3))
        self.assertTrue(torch.all(result >= 0) and torch.all(result < 1))

    def test_randn(self):
        """Test normal distribution sampling"""
        result = self.backend.randn(0, 1, (2, 3))
        self.assertEqual(result.shape, (2, 3))

    def test_randint(self):
        """Test integer random sampling"""
        result = self.backend.randint(0, 10, (2, 3))
        self.assertEqual(result.shape, (2, 3))
        self.assertTrue(torch.all(result >= 0) and torch.all(result < 10))

    def test_swap(self):
        """Test swap operation"""
        arr = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = self.backend.swap(arr, 0, 1)
        expected = torch.swapdims(arr, 0, 1)
        torch.testing.assert_close(result, expected)

    def test_concat(self):
        """Test concatenation"""
        arr1 = torch.tensor([[1, 2], [3, 4]])
        arr2 = torch.tensor([[5, 6], [7, 8]])
        result = self.backend.concat([arr1, arr2], axis=0)
        expected = torch.cat([arr1, arr2], dim=0)
        torch.testing.assert_close(result, expected)

    def test_stack(self):
        """Test stacking"""
        arr1 = torch.tensor([1, 2])
        arr2 = torch.tensor([3, 4])
        result = self.backend.stack([arr1, arr2], axis=0)
        expected = torch.stack([arr1, arr2], dim=0)
        torch.testing.assert_close(result, expected)

    def test_broadcast(self):
        """Test broadcasting"""
        arr = torch.tensor([1, 2, 3])
        result = self.backend.broadcast(arr, (2, 3))
        expected = arr.expand((2, 3))
        torch.testing.assert_close(result, expected)

    def test_sum(self):
        """Test sum operation"""
        arr = torch.tensor([[1, 2], [3, 4]])
        result = self.backend.sum(arr)
        self.assertEqual(result, 10)

        result = self.backend.sum(arr, axis=0)
        expected = torch.sum(arr, dim=0)
        torch.testing.assert_close(result, expected)

    def test_prod(self):
        """Test product operation"""
        arr = torch.tensor([[1, 2], [3, 4]])
        result = self.backend.prod(arr)
        self.assertEqual(result, 24)

        result = self.backend.prod(arr, axis=0)
        expected = torch.prod(arr, dim=0)
        torch.testing.assert_close(result, expected)

    def test_cumsum(self):
        """Test cumulative sum operation"""
        arr = torch.tensor([[1, 2], [3, 4]])
        result = self.backend.cumsum(arr)
        expected = torch.cumsum(arr, dim=None)
        torch.testing.assert_close(result, expected)

    def test_min(self):
        """Test min operation"""
        arr = torch.tensor([[1, 2], [3, 4]])
        result = self.backend.min(arr)
        self.assertEqual(result, 1)

        result = self.backend.min(arr, axis=0)
        expected = torch.amin(arr, dim=0)
        torch.testing.assert_close(result, expected)

    def test_max(self):
        """Test max operation"""
        arr = torch.tensor([[1, 2], [3, 4]])
        result = self.backend.max(arr)
        self.assertEqual(result, 4)

        result = self.backend.max(arr, axis=0)
        expected = torch.amax(arr, dim=0)
        torch.testing.assert_close(result, expected)

    def test_argmin(self):
        """Test argmin operation"""
        arr = torch.tensor([[1, 2], [3, 4]])
        result = self.backend.argmin(arr)
        self.assertEqual(result, 0)

        result = self.backend.argmin(arr, axis=0)
        expected = torch.argmin(arr, dim=0)
        torch.testing.assert_close(result, expected)

    def test_argmax(self):
        """Test argmax operation"""
        arr = torch.tensor([[1, 2], [3, 4]])
        result = self.backend.argmax(arr)
        self.assertEqual(result, 3)

        result = self.backend.argmax(arr, axis=0)
        expected = torch.argmax(arr, dim=0)
        torch.testing.assert_close(result, expected)

    def test_mean(self):
        """Test mean operation"""
        arr = torch.tensor([[1, 2], [3, 4]])
        result = self.backend.mean(arr)
        self.assertEqual(result, 2.5)

        result = self.backend.mean(arr, axis=0)
        expected = torch.mean(arr, dim=0)
        torch.testing.assert_close(result, expected)

    def test_std(self):
        """Test standard deviation operation"""
        arr = torch.tensor([[1, 2], [3, 4]])
        result = self.backend.std(arr)
        expected = torch.std(arr)
        torch.testing.assert_close(result, expected)

    def test_norm(self):
        """Test norm operation"""
        arr = torch.tensor([[1, 2], [3, 4]])
        result = self.backend.norm(arr)
        expected = torch.norm(arr)
        self.assertAlmostEqual(result.item(), expected.item())

    def test_norm2D(self):
        """Test 2D norm operation"""
        arr = torch.tensor([[1, 2], [3, 4]])
        result = self.backend.norm2D(arr)
        expected = (arr**2).sum(dim=1)
        torch.testing.assert_close(result, expected)

    def test_einsum(self):
        """Test einsum operation"""
        arr1 = torch.tensor([[1, 2], [3, 4]])
        arr2 = torch.tensor([[5, 6], [7, 8]])
        result = self.backend.einsum("ij,jk->ik", arr1, arr2)
        expected = torch.einsum("ij,jk->ik", arr1, arr2)
        torch.testing.assert_close(result, expected)

    def test_where_with_values(self):
        """Test where with values"""
        condition = torch.tensor([[True, False], [False, True]])
        true_val = torch.tensor([[1, 1], [1, 1]])
        false_val = torch.tensor([[0, 0], [0, 0]])
        result = self.backend.where(condition, true_val, false_val)
        expected = torch.where(condition, true_val, false_val)
        torch.testing.assert_close(result, expected)

    def test_where_without_values(self):
        """Test where without values"""
        condition = torch.tensor([[True, False], [False, True]])
        result = self.backend.where(condition)
        expected = torch.where(condition)
        self.assertEqual(len(result), len(expected))
        for i in range(len(result)):
            torch.testing.assert_close(result[i], expected[i])

    def test_clip(self):
        """Test clip operation"""
        arr = torch.tensor([1, 2, 3, 4, 5])
        result = self.backend.clip(arr, 2, 4)
        expected = torch.clamp(arr, min=2, max=4)
        torch.testing.assert_close(result, expected)

    def test_fill_diagonal(self):
        """Test fill diagonal operation"""
        arr = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = self.backend.fill_diagonal(arr, 0)
        expected = arr.clone()
        expected.fill_diagonal_(0)
        torch.testing.assert_close(result, expected)

    def test_diag(self):
        """Test diag operation"""
        arr = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = self.backend.diag(arr)
        expected = torch.diag(arr)
        torch.testing.assert_close(result, expected)

    def test_nan(self):
        """Test nan creation"""
        result = self.backend.nan()
        self.assertTrue(torch.isnan(result))

    def test_isnan(self):
        """Test isnan operation"""
        arr = torch.tensor([1, float("nan"), 3])
        result = self.backend.isnan(arr)
        expected = torch.isnan(arr)
        torch.testing.assert_close(result, expected)

    def test_pow(self):
        """Test power operation"""
        arr = torch.tensor([1, 2, 3])
        result = self.backend.pow(arr, 2)
        expected = torch.pow(arr, 2)
        torch.testing.assert_close(result, expected)

    def test_sqrt(self):
        """Test square root operation"""
        arr = torch.tensor([1, 4, 9])
        result = self.backend.sqrt(arr)
        expected = torch.sqrt(arr)
        torch.testing.assert_close(result, expected)

    def test_abs(self):
        """Test absolute operation"""
        arr = torch.tensor([-1, 2, -3])
        result = self.backend.abs(arr)
        expected = torch.abs(arr)
        torch.testing.assert_close(result, expected)

    def test_exp(self):
        """Test exponential operation"""
        arr = torch.tensor([0, 1, 2])
        result = self.backend.exp(arr)
        expected = torch.exp(arr)
        torch.testing.assert_close(result, expected)

    def test_log(self):
        """Test logarithm operation"""
        arr = torch.tensor([1, torch.e, torch.e**2])
        result = self.backend.log(arr)
        expected = torch.log(arr)
        torch.testing.assert_close(result, expected)

    def test_cos(self):
        """Test cosine operation"""
        arr = torch.tensor([0, torch.pi / 2, torch.pi])
        result = self.backend.cos(arr)
        expected = torch.cos(arr)
        torch.testing.assert_close(result, expected)

    def test_sin(self):
        """Test sine operation"""
        arr = torch.tensor([0, torch.pi / 2, torch.pi])
        result = self.backend.sin(arr)
        expected = torch.sin(arr)
        torch.testing.assert_close(result, expected)

    def test_tan(self):
        """Test tangent operation"""
        arr = torch.tensor([0, torch.pi / 4, torch.pi / 2])
        result = self.backend.tan(arr)
        expected = torch.tan(arr)
        torch.testing.assert_close(result, expected)

    def test_cosh(self):
        """Test hyperbolic cosine operation"""
        arr = torch.tensor([0, 1, 2])
        result = self.backend.cosh(arr)
        expected = torch.cosh(arr)
        torch.testing.assert_close(result, expected)

    def test_sinh(self):
        """Test hyperbolic sine operation"""
        arr = torch.tensor([0, 1, 2])
        result = self.backend.sinh(arr)
        expected = torch.sinh(arr)
        torch.testing.assert_close(result, expected)

    def test_tanh(self):
        """Test hyperbolic tangent operation"""
        arr = torch.tensor([0, 1, 2])
        result = self.backend.tanh(arr)
        expected = torch.tanh(arr)
        torch.testing.assert_close(result, expected)

    def test_inv(self):
        """Test matrix inversion"""
        arr = torch.tensor([[4, 7], [2, 6]], dtype=torch.float64)
        result = self.backend.inv(arr)
        expected = torch.linalg.inv(arr)
        torch.testing.assert_close(result, expected)

    def test_pinv(self):
        """Test pseudo-inverse"""
        arr = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float64)
        result = self.backend.pinv(arr)
        expected = torch.linalg.pinv(arr)
        torch.testing.assert_close(result, expected)

    def test_solve(self):
        """Test solving linear systems"""
        A = torch.tensor([[3, 1], [1, 2]], dtype=torch.float64)
        b = torch.tensor([9, 8], dtype=torch.float64)
        result = self.backend.solve(A, b)
        expected = torch.linalg.solve(A, b)
        torch.testing.assert_close(result, expected)

    def test_qr(self):
        """Test QR decomposition"""
        A = torch.tensor([[12, -51, 4], [6, 167, -68], [-4, 24, -41]], dtype=torch.float64)
        Q, R = self.backend.qr(A)
        reconstructed = Q @ R
        torch.testing.assert_close(reconstructed, A)

    def test_vander(self):
        """Test Vandermonde matrix"""
        a = torch.tensor([1, 2, 3], dtype=torch.float64)
        result = self.backend.vander(a, 2)
        expected = torch.vander(a, 3)
        torch.testing.assert_close(result, expected)

    def test_lstsq(self):
        """Test least squares solution"""
        A = torch.tensor([[1, 1], [1, 2], [1, 3]], dtype=torch.float64)
        B = torch.tensor([1, 2, 3], dtype=torch.float64)
        result = self.backend.lstsq(A, B)
        expected = torch.linalg.lstsq(A, B).solution
        torch.testing.assert_close(result, expected)
