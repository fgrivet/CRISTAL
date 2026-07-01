"""
Unit tests for Solver class
"""

import unittest

import numpy as np

from cristal.backend.numpy_backend import NumpyBackend
from cristal.commons.inverter import Inverter
from cristal.commons.solver import IMPLEMENTED_SOLVERS, Solver


class TestSolver(unittest.TestCase):
    """Test the Solver class functionality"""

    def setUp(self):
        np.random.seed(42)
        self.N = 5
        # Create symmetric positive definite matrix
        self.V = np.random.rand(self.N, 3, 3)

        # Create test vector
        self.v = np.random.rand(3)
        self.v = np.stack([self.v for _ in range(self.N)], axis=0).reshape(self.N, 3, 1)

        # Compute expected results manually
        self.exp_result = self._compute_expected_result(self.V, self.v, self.N)

    def _compute_expected_result(self, V, v, N):
        """Manually compute expected result for validation"""
        inverter = Inverter()
        inverter.backend = NumpyBackend()
        G = V.swapaxes(-1, -2) @ V / N
        G = (G + G.swapaxes(-1, -2)) / 2
        G_inv = inverter(G)
        return (v.swapaxes(-1, -2) @ G_inv @ v)[:, 0, 0]

    def test_solver_initialization(self):
        """Test Solver initialization with valid parameters"""
        for method in IMPLEMENTED_SOLVERS.__args__:
            solver = Solver(solver=method)
            self.assertEqual(solver.solver, method)
            self.assertIsNone(solver.eps)

            # Test no backend bound
            self.assertRaises(ValueError, solver, self.V, self.v, self.N)

        # Test invalid method
        self.assertRaises(ValueError, Solver, solver="invalid_method")

    def test_solver(self):
        """Test Solver method"""
        backend = NumpyBackend()

        for method in IMPLEMENTED_SOLVERS.__args__:
            # Create inverter
            solver = Solver(solver=method)

            # Bind backend
            solver.backend = backend

            # Compute G if method is not QR
            if method == "qr":
                G = self.V
            else:
                G = self.V.swapaxes(1, 2) @ self.V / self.N

            # Solve the system
            result = solver(G, self.v, self.N)

            # Check that the inverse is correct
            np.testing.assert_almost_equal(result, self.exp_result, decimal=8, err_msg=f"{method}")

    def test_solver_with_eps(self):
        """Test Solver method with regularization eps"""
        backend = NumpyBackend()

        for method in IMPLEMENTED_SOLVERS.__args__:
            # Create inverter
            solver = Solver(solver=method, eps=1e-12)

            # Bind backend
            solver.backend = backend

            # Compute G if method is not QR
            if method == "qr":
                G = self.V
            else:
                G = self.V.swapaxes(1, 2) @ self.V / self.N

            # Solve the system
            result = solver(G, self.v, self.N)

            # Check that the inverse is correct
            np.testing.assert_almost_equal(result, self.exp_result, decimal=5, err_msg=f"{method}")
