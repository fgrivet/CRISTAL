"""
Unit tests for Incrementer class
"""

import unittest

import numpy as np

from cristal.backend.numpy_backend import NumpyBackend
from cristal.commons.incrementer import IMPLEMENTED_INCREMENTERS, Incrementer
from cristal.commons.inverter import Inverter
from cristal.commons.polynomial_basis import PolynomialBasis


class TestIncrementer(unittest.TestCase):
    """Test the Incrementer class functionality"""

    def test_incrementer_initialization(self):
        """Test Incrementer initialization with valid parameters"""
        # Test valid incrementer methods
        for method in IMPLEMENTED_INCREMENTERS.__args__:
            incrementer = Incrementer(method=method)
            self.assertEqual(incrementer.method, method)
            backend = NumpyBackend()
            inverter = Inverter("solve")

            # Test no backend bound
            self.assertRaises(ValueError, incrementer, np.zeros((2, 2)), 2, np.zeros((2, 2)), 2)
            # Test no inverter bound
            incrementer.backend = backend
            self.assertRaises(ValueError, incrementer, np.zeros((2, 2)), 2, np.zeros((2, 2)), 2)
            # Test no polynomial basis bound
            incrementer.inverter = inverter
            self.assertRaises(ValueError, incrementer, np.zeros((2, 2)), 2, np.zeros((2, 2)), 2)

        # Test invalid method
        self.assertRaises(ValueError, Incrementer, method="invalid_method")

    def test_incrementer_30_new(self):
        """Test Incrementer method"""
        # Dependencies
        backend = NumpyBackend()
        inverter = Inverter("solve")
        inverter.backend = backend
        polynomial_basis = PolynomialBasis("monomials")
        polynomial_basis.backend = backend

        # Simulate a vector of N = 1_000 points in dimension d = 4 and 30 new data
        N, N_add, d = 1_000, 30, 4
        np.random.seed(42)
        X = np.random.rand(N, d)
        X_new = np.random.rand(N_add, d)

        # Compute the vector basis for the moments matrices (the polynomial_basis are already tested)
        n = 5
        V = polynomial_basis.vandermonde_nd(X, n)
        V_new = polynomial_basis.vandermonde_nd(X_new, n)

        # Compute the moments matrices
        M = (V.T @ V) / N
        M_new = (V.T @ V + V_new.T @ V_new) / (N + N_add)

        # Compute the inverse of the moments matrices (the inverters are already tested)
        M_inv = inverter(M)
        M_new_inv = inverter(M_new)

        # Test the incrementer
        for method in IMPLEMENTED_INCREMENTERS.__args__:
            incrementer = Incrementer(method=method)

            # Bind dependencies
            incrementer.backend = backend
            incrementer.inverter = inverter
            incrementer.polynomial_basis = polynomial_basis

            # Test the incrementer
            # incrementer with inverse method requires M whereas sherman and woodbury need M_inv
            if method == "inverse":
                new_M, new_N, new_M_inv = incrementer(M, N, X_new, n)
                np.testing.assert_almost_equal(new_M, M_new, decimal=12)
            else:
                new_N, new_M_inv = incrementer(M_inv, N, X_new, n)

            # Test that the new M_inv is the good one
            np.testing.assert_allclose(new_M_inv, M_new_inv, rtol=1e-4, atol=1e-4, err_msg=f"{method} new_M_inv M_new_inv")
            # Test that the new N is the good one
            self.assertEqual(new_N, N + N_add)

    def test_incrementer_1_new(self):
        """Test Incrementer method"""
        # Dependencies
        backend = NumpyBackend()
        inverter = Inverter("solve")
        inverter.backend = backend
        polynomial_basis = PolynomialBasis("monomials")
        polynomial_basis.backend = backend

        # Simulate a vector of N = 1_000 points in dimension d = 4 and 1 new data
        N, N_add, d = 500, 1, 4
        np.random.seed(42)
        X = np.random.rand(N, d)
        X_new = np.random.rand(N_add, d)

        # Compute the vector basis for the moments matrices (the polynomial_basis are already tested)
        n = 5
        V = polynomial_basis.vandermonde_nd(X, n)
        V_new = polynomial_basis.vandermonde_nd(X_new, n)

        # Compute the moments matrices
        M = (V.T @ V) / N
        M_new = (V.T @ V + V_new.T @ V_new) / (N + N_add)

        # Compute the inverse of the moments matrices (the inverters are already tested)
        M_inv = inverter(M)
        M_new_inv = inverter(M_new)

        # Test the incrementer
        for method in IMPLEMENTED_INCREMENTERS.__args__:
            incrementer = Incrementer(method=method)

            # Bind dependencies
            incrementer.backend = backend
            incrementer.inverter = inverter
            incrementer.polynomial_basis = polynomial_basis

            # Test the incrementer
            # incrementer with inverse method requires M whereas sherman and woodbury need M_inv
            if method == "inverse":
                new_M, new_N, new_M_inv = incrementer(M, N, X_new, n)
                np.testing.assert_almost_equal(new_M, M_new, decimal=12)
            else:
                new_N, new_M_inv = incrementer(M_inv, N, X_new, n)
            self.assertEqual(new_N, N + N_add)

            # Test that the new M_inv is the good one
            np.testing.assert_allclose(new_M_inv, M_new_inv, rtol=1e-4, atol=1e-4, err_msg=f"{method} new_M_inv M_new_inv")
