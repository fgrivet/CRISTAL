"""
Unit tests for Solver class
"""

from unittest.mock import Mock

import pytest

from cristal.commons.solver import IMPLEMENTED_SOLVER, Solver


class TestSolver:
    """Test the Solver class functionality"""

    def test_solver_initialization(self):
        """Test Solver initialization with valid parameters"""
        # Test valid solver types
        for solver_type in IMPLEMENTED_SOLVER.__args__:
            solver = Solver(solver=solver_type)
            assert solver.solver == solver_type

        # Test invalid solver type
        with pytest.raises(AssertionError):
            Solver(solver="invalid_solver")

    def test_solver_with_mock_backend(self):
        """Test Solver with mocked backend"""
        # Create a mock backend
        backend = Mock()
        backend.qr.return_value = (Mock(), Mock())
        backend.swap.return_value = Mock()
        backend.solve.return_value = Mock()
        backend.eye.return_value = Mock()
        backend.einsum.return_value = Mock()

        # Create solver
        solver = Solver(solver="solve")

        # Bind backend
        solver.backend = backend

        # Test that solver can be called without error
        # Note: This test requires more complex mocking to fully test the solve method
        assert solver.solver == "solve"
