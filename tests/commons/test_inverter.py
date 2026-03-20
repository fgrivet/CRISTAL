"""
Unit tests for Inverter class
"""

import unittest

import numpy as np

from cristal.backend import NumpyBackend
from cristal.commons.inverter import IMPLEMENTED_INVERTER, Inverter


class TestInverter(unittest.TestCase):
    """Test the Inverter class functionality"""

    def setUp(self):
        np.random.seed(42)
        # Use 3x3 matrices to have the exact inverse and compare fairly
        self.mat_2D = self._make_matrix_pd(np.random.rand(3, 3))
        self.mat_3D = self._make_matrix_pd(np.random.rand(10, 3, 3))

        self.inv_mat_2D = self._inverse_matrix_3x3(self.mat_2D)
        self.inv_mat_2D_eps = self._inverse_matrix_3x3(self.mat_2D + 1e-8 * np.eye(3))
        self.inv_mat_3D = np.stack([self._inverse_matrix_3x3(m) for m in self.mat_3D], axis=0)
        self.inv_mat_3D_eps = np.stack([self._inverse_matrix_3x3(m + 1e-8 * np.eye(3)) for m in self.mat_3D], axis=0)

    def _make_matrix_pd(self, mat):
        mat = (mat + np.swapaxes(mat, -1, -2)) / 2  # Ensure symmetry
        mat += np.eye(mat.shape[-1])  # Ensure positivness
        return mat

    def _inverse_matrix_3x3(self, mat):
        """
        Compute the exact inverse of a 3x3 matrix.
        """
        a, b, c = mat[0, 0], mat[0, 1], mat[0, 2]
        d, e, f = mat[1, 0], mat[1, 1], mat[1, 2]
        g, h, i = mat[2, 0], mat[2, 1], mat[2, 2]

        # Calcul du déterminant
        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

        if det == 0:
            raise ValueError("Matrix is not invertible (det nul).")

        # Calcul de l'inverse
        inv = np.zeros((3, 3))
        inv[0, 0] = (e * i - f * h) / det
        inv[0, 1] = -(b * i - c * h) / det
        inv[0, 2] = (b * f - c * e) / det
        inv[1, 0] = -(d * i - f * g) / det
        inv[1, 1] = (a * i - c * g) / det
        inv[1, 2] = -(a * f - c * d) / det
        inv[2, 0] = (d * h - e * g) / det
        inv[2, 1] = -(a * h - b * g) / det
        inv[2, 2] = (a * e - b * d) / det

        return inv

    def test_inverter_initialization(self):
        """Test Inverter initialization with valid parameters"""
        # Test valid inverter methods
        for method in IMPLEMENTED_INVERTER.__args__:
            inverter = Inverter(method=method)
            self.assertEqual(inverter.method, method)
            self.assertIsNone(inverter.eps)

            # Test no backend bound
            self.assertRaises(ValueError, inverter, self.mat_2D)

        # Test valid eps
        inverter = Inverter(eps=1e-12)
        self.assertEqual(inverter.eps, 1e-12)

        # Test invalid method
        self.assertRaises(ValueError, Inverter, method="invalid_method")

        # Test invalid eps
        self.assertRaises(ValueError, Inverter, eps=0)
        self.assertRaises(ValueError, Inverter, eps=-3)

    def test_inverter(self):
        """Test Inverter method"""
        backend = NumpyBackend()

        for method in IMPLEMENTED_INVERTER.__args__:
            # Create inverter
            inverter = Inverter(method=method)

            # Bind backend
            inverter.backend = backend

            # Invert matrices
            inv_mat_2D = inverter(self.mat_2D)
            inv_mat_3D = inverter(self.mat_3D)

            # Check that the inverse is correct
            np.testing.assert_almost_equal(inv_mat_2D, self.inv_mat_2D, decimal=8, err_msg=f"{method} 2D mat")
            np.testing.assert_almost_equal(inv_mat_3D, self.inv_mat_3D, decimal=8, err_msg=f"{method} 3D mat")

    def test_inverter_with_eps(self):
        """Test Inverter method with regularization eps"""
        backend = NumpyBackend()

        for method in IMPLEMENTED_INVERTER.__args__:
            # Create inverter
            inverter = Inverter(method=method)

            # Bind backend
            inverter.backend = backend

            # Invert matrices
            inv_mat_2D = inverter(self.mat_2D, eps=1e-12)
            inv_mat_3D = inverter(self.mat_3D, eps=1e-12)

            # Check that the inverse is correct
            np.testing.assert_almost_equal(inv_mat_2D, self.inv_mat_2D, decimal=8, err_msg=f"{method} 2D mat with eps")
            np.testing.assert_almost_equal(inv_mat_3D, self.inv_mat_3D, decimal=8, err_msg=f"{method} 3D mat with eps")
