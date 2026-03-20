"""
Comprehensive test suite for the BaseBackend class.
Each TestBackend should implement a few methods to test all the Backend's functionnality.
Tests all methods with 1D, 2D, and 3D arrays where applicable.
"""

import unittest
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy

from cristal.backend.base_backend import Backend
from cristal.types import DTypeLike


class BaseTestBackendMeta(ABCMeta):
    """Métaclass qui empêche l'exécution de TestBaseBackend."""

    def __call__(cls, *args, **kwargs):
        if cls.__name__ == "BaseTestBackend":
            raise TypeError("BaseTestBackend cannot be instantiated directly")
        return super().__call__(*args, **kwargs)


class BaseTestBackend(unittest.TestCase, metaclass=BaseTestBackendMeta):
    """Test suite for each Backend implementation."""

    @classmethod
    @abstractmethod
    def setUpClass(cls):
        """Set up the backend for testing."""
        cls.backend: Backend

    def setUp(self):
        """Set up the seed before each test."""
        self.backend.set_seed(42)

    @abstractmethod
    def int32(self) -> DTypeLike: ...

    @abstractmethod
    def int64(self) -> DTypeLike: ...

    @abstractmethod
    def _all(self, cond) -> bool: ...

    # Define assertion for arrays because in this setup, all desired results are computed in numpy so we need to convert the actual result to numpy before testing
    # The to_numpy method is already robust and tested elsewhere
    def assert_array_equal(self, actual, desired, *args, **kwargs):
        actual_np = self.backend.to_numpy(actual)
        np.testing.assert_array_equal(actual_np, desired, *args, **kwargs)

    def assert_almost_equal(self, actual, desired, *args, **kwargs):
        actual_np = self.backend.to_numpy(actual)
        np.testing.assert_almost_equal(actual_np, desired, *args, **kwargs)

    # ===== Type Tests =====

    @abstractmethod
    def test_is_array_like(self):
        """Test type checking with different array types."""

    @abstractmethod
    def test_to_array_like(self):
        """Test conversion to array_like with different dtypes."""

    def test_to_numpy(self):
        """Test conversion to numpy array."""
        arr = self.backend.to_array_like([0, 1, 2, 3, 4])
        result = self.backend.to_numpy(arr)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.arange(5))

        arr2 = self.backend.to_array_like([[0, 1], [2, 3], [4, 5]])
        result = self.backend.to_numpy(arr2)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([[0, 1], [2, 3], [4, 5]]))

    # ===== Creation Tests =====

    def test_zeros(self):
        """Test zeros creation with 1D, 2D, 3D arrays."""
        # 1D
        arr_1d = self.backend.zeros(5)
        self.assertEqual(arr_1d.shape, (5,))
        self.assert_array_equal(arr_1d, np.array([0, 0, 0, 0, 0]))

        # 2D
        arr_2d = self.backend.zeros((3, 4))
        self.assertEqual(arr_2d.shape, (3, 4))
        self.assert_array_equal(arr_2d, np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))

        # 3D
        arr_3d = self.backend.zeros((2, 3, 4))
        self.assertEqual(arr_3d.shape, (2, 3, 4))
        self.assert_array_equal(arr_3d, np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]))

    def test_ones(self):
        """Test ones creation with 1D, 2D, 3D arrays."""
        # 1D
        arr_1d = self.backend.ones(5)
        self.assert_array_equal(arr_1d, np.array([1, 1, 1, 1, 1]))

        # 2D
        arr_2d = self.backend.ones((3, 4))
        self.assert_array_equal(arr_2d, np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]))

        # 3D
        arr_3d = self.backend.ones((2, 3, 4))
        self.assert_array_equal(arr_3d, np.array([[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]))

    def test_eye(self):
        """Test identity matrix creation."""
        # 3x3 matrix
        mat_3x3 = self.backend.eye(3)
        self.assert_array_equal(mat_3x3, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

        # 5x5 matrix
        mat_5x5 = self.backend.eye(5)
        self.assert_array_equal(mat_5x5, np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]))

    def test_full(self):
        """Test array filled with value."""
        # 1D
        arr_1d = self.backend.full(5, 7)
        self.assert_array_equal(arr_1d, np.array([7, 7, 7, 7, 7]))

        # 2D
        arr_2d = self.backend.full((3, 4), 8)
        self.assert_array_equal(arr_2d, np.array([[8, 8, 8, 8], [8, 8, 8, 8], [8, 8, 8, 8]]))

        # 3D
        arr_3d = self.backend.full((2, 3, 4), 9)
        self.assert_array_equal(arr_3d, np.array([[[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]], [[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]]]))

    def test_arange(self):
        """Test arange with different argument combinations."""
        # Single argument
        arr_1 = self.backend.arange(5)
        self.assert_array_equal(arr_1, np.array([0, 1, 2, 3, 4]))

        # Two arguments
        arr_2 = self.backend.arange(1, 5)
        self.assert_array_equal(arr_2, np.array([1, 2, 3, 4]))

        # Three arguments
        arr_3 = self.backend.arange(1, 5, 2)
        self.assert_array_equal(arr_3, np.array([1, 3]))

        # With dtype
        arr_4 = self.backend.arange(5, dtype=self.int32())
        self.assertEqual(arr_4.dtype, self.int32())

    def test_copy(self):
        """Test array copying."""
        arr = self.backend.zeros(5)
        arr[0] = 5
        arr_copy = self.backend.copy(arr)

        arr_copy[0] = 10
        self.assertEqual(arr[0], 5)  # Original unchanged
        self.assertEqual(arr_copy[0], 10)  # Copy changed

    def test_set_seed(self):
        """Test seed reproducibility."""
        self.backend.set_seed(42)
        arr1 = self.backend.random((3, 3))

        self.backend.set_seed(42)
        arr2 = self.backend.random((3, 3))

        self.assert_array_equal(arr1, arr2)

    def test_random(self):
        """Test random uniform distribution."""
        arr_1d = self.backend.random(5)
        self.assertTrue(self._all((0 <= arr_1d) & (arr_1d < 1)))

        arr_2d = self.backend.random((3, 4))
        self.assertTrue(self._all((0 <= arr_2d) & (arr_2d < 1)))

        arr_3d = self.backend.random((2, 3, 4))
        self.assertTrue(self._all((0 <= arr_3d) & (arr_3d < 1)))

    def test_randn(self):
        """Test random normal distribution."""
        arr = self.backend.randn(0, 1, (3, 3))
        self.assertEqual(arr.shape, (3, 3))

        # Test with array mean
        arr2 = self.backend.randn(self.backend.to_array_like([1, 2]), 1, (2, 2))
        self.assertEqual(arr2.shape, (2, 2))

    def test_randint(self):
        """Test random integer distribution."""
        arr = self.backend.randint(0, 5, (3, 3))
        self.assertTrue(self._all((0 <= arr) & (arr < 5)))
        self.assertTrue(arr.dtype == self.int64())

        # Test with array boundaries
        arr2 = self.backend.randint(self.backend.to_array_like([0, 3]), 5, (8, 2))
        self.assertTrue(self._all((0 <= arr2) & (arr2 < 5)))
        self.assertTrue(arr2.dtype == self.int64())

        # Test with array boundaries
        arr3 = self.backend.randint(self.backend.to_array_like([0, 3]), self.backend.to_array_like([2, 5]), (8, 2))
        self.assertTrue(self._all((0 <= arr3) & (arr3 < 5)))
        self.assertTrue(arr3.dtype == self.int64())

        # Test with array boundaries
        arr4 = self.backend.randint(1, self.backend.to_array_like([2, 5]), (8, 2))
        self.assertTrue(self._all((1 <= arr4) & (arr4 < 5)))
        self.assertTrue(arr4.dtype == self.int64())

    def test_randint_value_error(self):
        """Test randint with invalid parameters."""
        with self.assertRaises(ValueError):
            self.backend.randint(5, 0, (2, 2))

        with self.assertRaises(ValueError):
            self.backend.randint(self.backend.to_array_like([0, 3]), 2, (8, 2))

        with self.assertRaises(ValueError):
            self.backend.randint(self.backend.to_array_like([2, 3]), self.backend.to_array_like([0, 5]), (8, 2))

        with self.assertRaises(ValueError):
            self.backend.randint(self.backend.to_array_like([2, 3, 4]), self.backend.to_array_like([0, 5, 8]), (8, 2))

    def test_swap_1d(self):
        """Test swap on 1D arrays (single axis)."""
        arr = self.backend.ones(5)
        result = self.backend.swap(arr, 0, 0)
        self.assert_array_equal(result, arr)

    def test_swap_2d(self):
        """Test swap on 2D arrays."""
        arr = self.backend.arange(6).reshape(2, 3)
        result = self.backend.swap(arr, 0, 1)
        expected = np.array([[0, 3], [1, 4], [2, 5]])
        self.assert_array_equal(result, expected)

    def test_swap_3d(self):
        """Test swap on 3D arrays."""
        arr = self.backend.arange(24).reshape(2, 3, 4)
        result = self.backend.swap(arr, 1, 2)
        expected = np.array([[[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]], [[12, 16, 20], [13, 17, 21], [14, 18, 22], [15, 19, 23]]])
        self.assert_array_equal(result, expected)

    def test_concat_1d(self):
        """Test concatenation of 1D arrays."""
        arr1 = self.backend.arange(3)
        arr2 = self.backend.arange(3, 6)
        result = self.backend.concat([arr1, arr2], axis=0)
        self.assert_array_equal(result, np.array([0, 1, 2, 3, 4, 5]))

    def test_concat_2d_axis0(self):
        """Test concatenation of 2D arrays along axis 0."""
        arr1 = self.backend.ones((2, 3))
        arr2 = self.backend.ones((3, 3)) * 2
        result = self.backend.concat([arr1, arr2], axis=0)
        self.assertEqual(result.shape, (5, 3))
        self.assert_array_equal(result, np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]))

    def test_concat_2d_axis1(self):
        """Test concatenation of 2D arrays along axis 1."""
        arr1 = self.backend.ones((3, 2))
        arr2 = self.backend.ones((3, 2)) * 2
        result = self.backend.concat([arr1, arr2], axis=1)
        self.assertEqual(result.shape, (3, 4))
        self.assert_array_equal(result, np.array([[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]))

    def test_concat_3d(self):
        """Test concatenation of 3D arrays."""
        arr1 = self.backend.ones((2, 3, 4))
        arr2 = self.backend.ones((2, 3, 4)) * 2
        result = self.backend.concat([arr1, arr2], axis=1)
        self.assert_array_equal(
            result,
            np.array(
                [
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0],
                    ],
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0],
                    ],
                ]
            ),
        )

    def test_concat_invalid(self):
        """Test concat with invalid shapes."""
        arr1 = self.backend.ones((3, 4))
        arr2 = self.backend.ones((2, 4))
        with self.assertRaises(ValueError):
            self.backend.concat([arr1, arr2], axis=1)

    def test_stack_1d(self):
        """Test stacking of 1D arrays."""
        arr1 = self.backend.arange(3)
        arr2 = self.backend.arange(3, 6)
        result = self.backend.stack([arr1, arr2], axis=0)
        self.assert_array_equal(result, np.array([[0, 1, 2], [3, 4, 5]]))

    def test_stack_2d(self):
        """Test stacking of 2D arrays."""
        arr1 = self.backend.ones((3, 4))
        arr2 = self.backend.ones((3, 4)) * 2
        result = self.backend.stack([arr1, arr2], axis=1)
        desired = np.array(
            [
                [
                    [
                        1,
                        1,
                        1,
                        1,
                    ],
                    [
                        2,
                        2,
                        2,
                        2,
                    ],
                ],
                [
                    [
                        1,
                        1,
                        1,
                        1,
                    ],
                    [
                        2,
                        2,
                        2,
                        2,
                    ],
                ],
                [
                    [
                        1,
                        1,
                        1,
                        1,
                    ],
                    [
                        2,
                        2,
                        2,
                        2,
                    ],
                ],
            ]
        )
        self.assert_array_equal(result, desired)

    def test_stack_3d(self):
        """Test stacking of 3D arrays."""
        arr1 = self.backend.ones((2, 3, 4))
        arr2 = self.backend.ones((2, 3, 4)) * 2
        result = self.backend.stack([arr1, arr2], axis=2)
        desired = np.array(
            [
                [
                    [
                        [
                            1,
                            1,
                            1,
                            1,
                        ],
                        [
                            2,
                            2,
                            2,
                            2,
                        ],
                    ],
                    [
                        [
                            1,
                            1,
                            1,
                            1,
                        ],
                        [
                            2,
                            2,
                            2,
                            2,
                        ],
                    ],
                    [
                        [
                            1,
                            1,
                            1,
                            1,
                        ],
                        [
                            2,
                            2,
                            2,
                            2,
                        ],
                    ],
                ],
                [
                    [
                        [
                            1,
                            1,
                            1,
                            1,
                        ],
                        [
                            2,
                            2,
                            2,
                            2,
                        ],
                    ],
                    [
                        [
                            1,
                            1,
                            1,
                            1,
                        ],
                        [
                            2,
                            2,
                            2,
                            2,
                        ],
                    ],
                    [
                        [
                            1,
                            1,
                            1,
                            1,
                        ],
                        [
                            2,
                            2,
                            2,
                            2,
                        ],
                    ],
                ],
            ]
        )
        self.assert_array_equal(result, desired)

    def test_broadcast_1d(self):
        """Test broadcasting 1D array."""
        arr = self.backend.arange(3)
        result = self.backend.broadcast(arr, (4, 3))
        desired = np.array(
            [
                [
                    0,
                    1,
                    2,
                ],
                [
                    0,
                    1,
                    2,
                ],
                [
                    0,
                    1,
                    2,
                ],
                [
                    0,
                    1,
                    2,
                ],
            ]
        )
        self.assert_array_equal(result, desired)

    def test_broadcast_2d(self):
        """Test broadcasting 2D array."""
        arr = self.backend.ones((3, 1))
        result = self.backend.broadcast(arr, (3, 4))
        desired = np.array(
            [
                [
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                ],
            ]
        )
        self.assert_array_equal(result, desired)

    def test_broadcast_3d(self):
        """Test broadcasting 3D array."""
        arr = self.backend.ones((1, 2, 1))
        result = self.backend.broadcast(arr, (3, 2, 4))
        desired = np.array(
            [
                [
                    [
                        1,
                        1,
                        1,
                        1,
                    ],
                    [
                        1,
                        1,
                        1,
                        1,
                    ],
                ],
                [
                    [
                        1,
                        1,
                        1,
                        1,
                    ],
                    [
                        1,
                        1,
                        1,
                        1,
                    ],
                ],
                [
                    [
                        1,
                        1,
                        1,
                        1,
                    ],
                    [
                        1,
                        1,
                        1,
                        1,
                    ],
                ],
            ]
        )
        self.assert_array_equal(result, desired)

    def test_broadcast_invalid(self):
        """Test broadcasting with invalid shapes."""
        arr = self.backend.ones((2, 3))
        with self.assertRaises(ValueError):
            self.backend.broadcast(arr, (2, 5))

    # ===== Sum =====
    def test_sum_1d_no_axis(self):
        """Test sum on 1D array without axis."""
        arr = self.backend.arange(1, 6)
        result = self.backend.sum(arr)
        self.assertEqual(result, 15.0)

    def test_sum_1d_with_axis(self):
        """Test sum on 1D array with axis."""
        arr = self.backend.arange(5)
        result = self.backend.sum(arr, axis=0)
        self.assert_array_equal(result, 10)

    def test_sum_2d_no_axis(self):
        """Test sum on 2D array without axis."""
        arr = self.backend.arange(1, 10).reshape(3, 3)
        result = self.backend.sum(arr)
        self.assertEqual(result, 45.0)

    def test_sum_2d_axis0(self):
        """Test sum on 2D array along axis 0."""
        arr = self.backend.arange(1, 10).reshape(3, 3)
        result = self.backend.sum(arr, axis=0)
        self.assert_array_equal(result, np.array([12, 15, 18]))

    def test_sum_2d_axis1(self):
        """Test sum on 2D array along axis 1."""
        arr = self.backend.arange(1, 10).reshape(3, 3)
        result = self.backend.sum(arr, axis=1)
        self.assert_array_equal(result, np.array([6, 15, 24]))

    def test_sum_2d_keepdims(self):
        """Test sum with keepdims on 2D array."""
        arr = self.backend.arange(1, 10).reshape(3, 3)
        result = self.backend.sum(arr, axis=0, keepdims=True)
        self.assert_array_equal(result, np.array([[12, 15, 18]]))

    def test_sum_3d(self):
        """Test sum on 3D array."""
        arr = self.backend.ones((2, 3, 4))
        result = self.backend.sum(arr)
        self.assertEqual(result, 24.0)

    # ===== Prod =====
    def test_prod_1d(self):
        """Test product on 1D array."""
        arr = self.backend.arange(1, 6)
        result = self.backend.prod(arr)
        self.assertEqual(result, 120.0)

    def test_prod_2d_axis0(self):
        """Test product on 2D array along axis 0."""
        arr = self.backend.ones((3, 3)) * 2
        result = self.backend.prod(arr, axis=0)
        self.assert_array_equal(result, np.array([8.0, 8.0, 8.0]))

        result_2 = self.backend.prod(arr, keepdims=True)
        self.assert_array_equal(result_2, np.array([[512.0]]))

    # ===== Cumsum =====
    def test_cumsum_1d(self):
        """Test cumulative sum on 1D array."""
        arr = self.backend.arange(1, 6)
        result = self.backend.cumsum(arr)
        self.assert_array_equal(result, np.array([1, 3, 6, 10, 15]))

    def test_cumsum_2d_axis0(self):
        """Test cumulative sum on 2D array along axis 0."""
        arr = self.backend.arange(1, 10).reshape(3, 3)
        result = self.backend.cumsum(arr, axis=0)
        self.assert_array_equal(result, np.array([[1, 2, 3], [5, 7, 9], [12, 15, 18]]))

    # ===== Min/Max =====
    def test_min_1d(self):
        """Test minimum on 1D array."""
        arr = self.backend.arange(5)
        result = self.backend.min(arr)
        self.assertEqual(result, 0.0)

    def test_max_1d(self):
        """Test maximum on 1D array."""
        arr = self.backend.arange(5)
        result = self.backend.max(arr)
        self.assertEqual(result, 4.0)

    def test_min_max_2d_axis(self):
        """Test min/max on 2D array along axis."""
        arr = self.backend.arange(1, 10).reshape(3, 3)
        min_result = self.backend.min(arr, axis=0)
        max_result = self.backend.max(arr, axis=0)
        self.assert_array_equal(min_result, np.array([1, 2, 3]))
        self.assert_array_equal(max_result, np.array([7, 8, 9]))

        min_result_1 = self.backend.min(arr, axis=1)
        max_result_1 = self.backend.max(arr, axis=1)
        self.assert_array_equal(min_result_1, np.array([1, 4, 7]))
        self.assert_array_equal(max_result_1, np.array([3, 6, 9]))

    # ===== Argmin/Argmax =====
    def test_argmin_1d(self):
        """Test argmin on 1D array."""
        arr = self.backend.arange(5)
        result = self.backend.argmin(arr)
        self.assertEqual(result, 0)

    def test_argmin_2d_axis(self):
        """Test argmin on 2D array along axis."""
        arr = self.backend.arange(1, 10).reshape(3, 3)
        result = self.backend.argmin(arr, axis=0)
        self.assert_array_equal(result, np.array([0, 0, 0]))

    def test_argmax_1d(self):
        """Test argmin on 1D array."""
        arr = self.backend.arange(5)
        result = self.backend.argmax(arr)
        self.assertEqual(result, 4)

    def test_argmax_2d_axis(self):
        """Test argmin on 2D array along axis."""
        arr = self.backend.arange(1, 10).reshape(3, 3)
        result = self.backend.argmax(arr, axis=0)
        self.assert_array_equal(result, np.array([2, 2, 2]))

    # ===== Mean =====
    def test_mean_1d(self):
        """Test mean on 1D array."""
        arr = self.backend.arange(5, dtype=float)
        result = self.backend.mean(arr)
        self.assertEqual(result, 2.0)

    def test_mean_2d_axis(self):
        """Test mean on 2D array along axis."""
        arr = self.backend.arange(1, 10, dtype=float).reshape(3, 3)
        result = self.backend.mean(arr, axis=0)
        self.assert_array_equal(result, np.array([4, 5, 6]))

    # ===== Std =====
    def test_std_1d(self):
        """Test standard deviation on 1D array."""
        arr = self.backend.arange(3, dtype=float)
        result = self.backend.std(arr)
        self.assert_almost_equal(result, np.std(self.backend.to_numpy(arr)))

    def test_std_2d_axis(self):
        """Test standard deviation on 2D array along axis."""
        arr = self.backend.ones((3, 3)) * self.backend.to_array_like([1, 2, 3])
        result = self.backend.std(arr)
        self.assert_array_equal(result, np.std(self.backend.to_numpy(arr)))

        result2 = self.backend.std(arr, axis=0)
        self.assert_array_equal(result2, np.std(self.backend.to_numpy(arr), axis=0))

    # ===== Norm =====
    def test_norm_fro(self):
        """Test Frobenius norm."""
        arr = self.backend.to_array_like(self.backend.arange(1, 10).reshape(3, 3), float)
        result = self.backend.norm(arr)
        self.assert_almost_equal(result, np.linalg.norm(self.backend.to_numpy(arr)))

    def test_norm_1(self):
        """Test L1 norm."""
        arr = self.backend.to_array_like(self.backend.arange(1, 10).reshape(3, 3), float)
        result = self.backend.norm(arr, p=1)
        self.assert_almost_equal(result, np.linalg.norm(self.backend.to_numpy(arr), ord=1))

    def test_norm_keepdims(self):
        """Test norm with keepdims."""
        arr = self.backend.to_array_like(self.backend.ones((3, 3)), float)
        result = self.backend.norm(arr, p="fro", keepdims=True)
        self.assertEqual(result.shape, (1, 1))

    def test_norm2d(self):
        """Test 2-norm for each row of 2D array."""
        arr = self.backend.to_array_like(self.backend.arange(1, 10).reshape(3, 3), float)
        result = self.backend.norm2D(arr)
        expected = np.sum(self.backend.to_numpy(arr) ** 2, axis=1)
        self.assert_array_equal(result, expected)

    def test_norm2d_invalid(self):
        """Test norm2D with non-2D array."""
        arr_1d = self.backend.ones(5)
        with self.assertRaises(ValueError):
            self.backend.norm2D(arr_1d)

    def test_norm2d_3d(self):
        """Test norm2D with 3D array."""
        arr_3d = self.backend.ones((2, 3, 4))
        with self.assertRaises(ValueError):
            self.backend.norm2D(arr_3d)

    def test_einsum_matrix_mult(self):
        """Test matrix multiplication using einsum."""
        A = self.backend.to_array_like(self.backend.arange(1, 5).reshape(2, 2), float)
        B = self.backend.to_array_like(self.backend.arange(5, 9).reshape(2, 2), float)

        result = self.backend.einsum("ij,jk->ik", A, B)
        expected = np.einsum("ij,jk->ik", A, B)
        self.assert_array_equal(result, expected)

    def test_einsum_trace(self):
        """Test trace using einsum."""
        A = self.backend.to_array_like(self.backend.arange(1, 10).reshape(3, 3), float)

        result = self.backend.einsum("ii->", A)
        expected = np.einsum("ii->", A)
        self.assert_almost_equal(result, expected)

    def test_einsum_outer(self):
        """Test outer product using einsum."""
        a = self.backend.to_array_like(self.backend.arange(3), float)
        b = self.backend.to_array_like(self.backend.arange(3), float)

        result = self.backend.einsum("i,j->ij", a, b)
        expected = np.einsum("i,j->ij", a, b)
        self.assert_array_equal(result, expected)

    def test_einsum_3d(self):
        """Test einsum with 3D arrays."""
        A = self.backend.ones((2, 3, 4))
        B = self.backend.ones((3, 4))

        result = self.backend.einsum("ijk,jk->ik", A, B)
        expected = np.einsum("ijk,jk->ik", A, B)
        self.assert_array_equal(result, expected)

    def test_where_condition_true(self):
        """Test where with true values."""
        arr = self.backend.arange(3)
        condition = arr > 1
        result = self.backend.where(condition, 10, 0)
        self.assert_array_equal(result, [0, 0, 10])

    def test_where_indices(self):
        """Test where returning indices."""
        arr = self.backend.arange(5)
        condition = arr > 2
        result = self.backend.where(condition)
        self.assert_array_equal(result[0], [3, 4])

    def test_clip_simple(self):
        """Test clip with simple values."""
        arr = self.backend.arange(-2, 4)
        result = self.backend.clip(arr, 0, 2)
        self.assert_array_equal(result, [0, 0, 0, 1, 2, 2])

    def test_clip_min_only(self):
        """Test clip with only min."""
        arr = self.backend.arange(-2, 4)
        result = self.backend.clip(arr, min_=0)
        self.assert_array_equal(result, [0, 0, 0, 1, 2, 3])

    def test_clip_max_only(self):
        """Test clip with only max."""
        arr = self.backend.arange(0, 5)
        result = self.backend.clip(arr, max_=2)
        self.assert_array_equal(result, [0, 1, 2, 2, 2])

    def test_fill_diagonal_2d(self):
        """Test fill_diagonal on 2D array."""
        arr = self.backend.zeros((3, 3))
        result = self.backend.fill_diagonal(arr, 5)
        expected = np.diag([5, 5, 5])
        self.assert_array_equal(result, expected)

    def test_fill_diagonal_invalid(self):
        """Test fill_diagonal on non-2D array."""
        arr_1d = self.backend.zeros(5)
        with self.assertRaises(ValueError):
            self.backend.fill_diagonal(arr_1d, 5)

    def test_diag_extract(self):
        """Test diag extracting diagonal from 2D array."""
        arr = self.backend.to_array_like(self.backend.arange(1, 10).reshape(3, 3), float)
        result = self.backend.diag(arr)
        self.assert_array_equal(result, np.diag(arr))

    def test_diag_construct(self):
        """Test diag constructing diagonal matrix."""
        diag_arr = self.backend.to_array_like(self.backend.arange(1, 4), float)
        result = self.backend.diag(diag_arr)
        expected = np.diag(diag_arr)
        self.assert_array_equal(result, expected)

    def test_isnan_simple(self):
        """Test isnan with NaN values."""
        arr = self.backend.to_array_like([1.0, float("nan"), 3.0])
        result = self.backend.isnan(arr)
        self.assert_array_equal(result, [False, True, False])

    def test_isnan_all_nan(self):
        """Test isnan with all NaN array."""
        arr = self.backend.to_array_like([float("nan")] * 5)
        result = self.backend.isnan(arr)
        self.assert_array_equal(result, [True] * 5)

    def test_pow_scalar(self):
        """Test power with scalar exponent."""
        arr = self.backend.to_array_like(self.backend.arange(1, 5), float)
        result = self.backend.pow(arr, 2)
        self.assert_array_equal(result, arr**2)

    def test_pow_array(self):
        """Test power with array exponent."""
        base = self.backend.to_array_like(self.backend.arange(1, 4), float)
        exp = self.backend.to_array_like([1, 2, 3])
        result = self.backend.pow(base, exp)
        self.assert_array_equal(result, base**exp)

    def test_sqrt_simple(self):
        """Test sqrt with positive values."""
        arr = self.backend.to_array_like([1.0, 4.0, 9.0])
        result = self.backend.sqrt(arr)
        self.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_sqrt_negative(self):
        """Test sqrt with negative values."""
        arr = self.backend.to_array_like([-1.0, 4.0])
        with self.assertRaises(ValueError):
            self.backend.sqrt(arr)

    def test_abs_simple(self):
        """Test absolute value."""
        arr = self.backend.to_array_like([-3, -1, 0, 1, 3])
        result = self.backend.abs(arr)
        self.assert_array_equal(result, [3, 1, 0, 1, 3])

    def test_exp_simple(self):
        """Test exponential."""
        arr = self.backend.to_array_like([0.0, 1.0])
        result = self.backend.exp(arr)
        self.assert_almost_equal(result, np.exp(self.backend.to_numpy(arr)))

    def test_log_simple(self):
        """Test logarithm."""
        arr = self.backend.to_array_like([1.0, np.e])
        result = self.backend.log(arr)
        self.assert_almost_equal(result, np.log(self.backend.to_numpy(arr)))

    def test_log_negative(self):
        """Test logarithm with negative values."""
        arr = self.backend.to_array_like([-1.0, 1.0])
        with self.assertRaises(ValueError):
            self.backend.log(arr)

    # ===== Trigonometric =====
    def test_cos(self):
        """Test cosine."""
        arr = self.backend.to_array_like([0.0, np.pi / 2])
        result = self.backend.cos(arr)
        self.assert_almost_equal(result, np.cos(self.backend.to_numpy(arr)))

    def test_sin(self):
        """Test sine."""
        arr = self.backend.to_array_like([0.0, np.pi / 2])
        result = self.backend.sin(arr)
        self.assert_almost_equal(result, np.sin(self.backend.to_numpy(arr)))

    def test_tan(self):
        """Test tangent."""
        arr = self.backend.to_array_like([0.0, np.pi / 4])
        result = self.backend.tan(arr)
        self.assert_almost_equal(result, np.tan(self.backend.to_numpy(arr)))

    # ===== Hyperbolic =====
    def test_cosh(self):
        """Test hyperbolic cosine."""
        arr = self.backend.to_array_like([0.0, 1.0])
        result = self.backend.cosh(arr)
        self.assert_almost_equal(result, np.cosh(self.backend.to_numpy(arr)))

    def test_sinh(self):
        """Test hyperbolic sine."""
        arr = self.backend.to_array_like([0.0, 1.0])
        result = self.backend.sinh(arr)
        self.assert_almost_equal(result, np.sinh(self.backend.to_numpy(arr)))

    def test_tanh(self):
        """Test hyperbolic tangent."""
        arr = self.backend.to_array_like([0.0, 1.0])
        result = self.backend.tanh(arr)
        self.assert_almost_equal(result, np.tanh(self.backend.to_numpy(arr)))

    def test_inv_square(self):
        """Test matrix inverse."""
        A = self.backend.to_array_like([[3.0, 1.0], [1.0, 2.0]])
        result = self.backend.inv(A)
        expected = np.linalg.inv(A)
        self.assert_almost_equal(result, expected)

    def test_pinv_non_square(self):
        """Test pseudoinverse."""
        A = self.backend.to_array_like([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = self.backend.pinv(A)
        expected = np.linalg.pinv(A)
        self.assert_almost_equal(result, expected)

    def test_solve_linear(self):
        """Test solving linear system."""
        A = self.backend.to_array_like([[3.0, 1.0], [1.0, 2.0]])
        b = self.backend.to_array_like([9.0, 8.0])
        result = self.backend.solve(A, b)
        expected = np.linalg.solve(A, b)
        self.assert_almost_equal(result, expected)

    def test_solve_triangular_upper(self):
        """Test solving triangular system (upper)."""
        A = self.backend.to_array_like([[1.0, 2.0], [0.0, 1.0]])
        b = self.backend.to_array_like([3.0, 4.0])
        result = self.backend.solve_triangular(A, b, upper=True)
        expected = scipy.linalg.solve_triangular(A, b, lower=False)
        self.assert_almost_equal(result, expected)

    def test_solve_triangular_lower(self):
        """Test solving triangular system (lower)."""
        A = self.backend.to_array_like([[1.0, 0.0], [2.0, 1.0]])
        b = self.backend.to_array_like([3.0, 7.0])
        result = self.backend.solve_triangular(A, b, upper=False)
        expected = scipy.linalg.solve_triangular(A, b, lower=True)
        self.assert_almost_equal(result, expected)

    def test_cholesky(self):
        """Test Cholesky decomposition."""
        A = self.backend.to_array_like([[4.0, 6.0], [6.0, 17.0]])
        result = self.backend.cholesky(A, allow_adding_reg=False)
        expected = np.linalg.cholesky(A)
        self.assert_almost_equal(result, expected)

    def test_cholesky_reg(self):
        """Test Cholesky decomposition with regularization."""
        A = self.backend.to_array_like([[1e-11, 0], [0, -1e-11]])
        result = self.backend.cholesky(A)
        expected = np.linalg.cholesky(self.backend.to_numpy(A) + 1e-10 * np.eye(2))
        self.assert_almost_equal(result, expected)

    def test_cholesky_invalid(self):
        """Test Cholesky decomposition with regularization."""
        A = self.backend.to_array_like([[1, 2], [2, 1]])
        with self.assertRaises(ValueError):
            self.backend.cholesky(A, allow_adding_reg=False)
        # Even with small regularization
        with self.assertRaises(ValueError):
            self.backend.cholesky(A)

    def test_inverse_cholesky(self):
        """Test inverse Cholesky decomposition."""
        A = self.backend.to_array_like([[4.0, 6.0], [6.0, 17.0]])
        result = self.backend.inverse_cholesky(A, allow_adding_reg=False)
        expected = np.linalg.inv(A)
        self.assert_almost_equal(result, expected)

    def test_inverse_cholesky_reg(self):
        """Test inverse Cholesky decomposition with regularization."""
        A = self.backend.to_array_like([[1e-11, 0], [0, -1e-11]])
        result = self.backend.inverse_cholesky(A)
        expected = np.linalg.inv(self.backend.to_numpy(A) + 1e-10 * np.eye(2))
        self.assert_almost_equal(result, expected)

    def test_inverse_cholesky_invalid(self):
        """Test Cholesky decomposition with regularization."""
        A = self.backend.to_array_like([[1, 2], [2, 1]])
        with self.assertRaises(ValueError):
            self.backend.inverse_cholesky(A, allow_adding_reg=False)
        # Even with small regularization
        with self.assertRaises(ValueError):
            self.backend.inverse_cholesky(A)

    def test_qr_reduced(self):
        """Test QR decomposition (reduced)."""
        A = self.backend.to_array_like([[1.0, 2.0], [3.0, 4.0]])
        Q, R = self.backend.qr(A, mode="reduced")
        A_reconstructed = self.backend.einsum("ij,jk->ik", Q, R)
        self.assert_almost_equal(A_reconstructed, A)

    def test_qr_complete(self):
        """Test QR decomposition (complete)."""
        A = self.backend.to_array_like([[1.0, 2.0], [3.0, 4.0]])
        Q, R = self.backend.qr(A, mode="complete")
        A_reconstructed = self.backend.einsum("ij,jk->ik", Q, R)
        self.assert_almost_equal(A_reconstructed, A)

    def test_vander(self):
        """Test Vandermonde matrix."""
        a = self.backend.to_array_like([1.0, 2.0, 3.0])
        result = self.backend.vander(a, degree=3, increasing=True)
        expected = np.vander(a, N=4, increasing=True)
        self.assert_array_equal(result, expected)

    def test_lstsq(self):
        """Test least squares solution."""
        A = self.backend.to_array_like([[1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        b = self.backend.to_array_like([2.0, 0.0, 4.0])
        result = self.backend.lstsq(A, b)
        expected = np.linalg.lstsq(A, b, rcond=None)[0]
        self.assert_almost_equal(result, expected)
