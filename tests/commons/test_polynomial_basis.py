"""
Unit tests for PolynomialBasis class
"""

from unittest.mock import Mock

import numpy as np
import pytest

from cristal.commons.polynomial_basis import IMPLEMENTED_POLYNOMIAL_BASIS, PolynomialBasis


class TestPolynomialBasis:
    """Test the PolynomialBasis class functionality"""

    def test_polynomial_basis_initialization(self):
        """Test PolynomialBasis initialization with valid parameters"""
        # Test valid polynomial bases
        for basis in IMPLEMENTED_POLYNOMIAL_BASIS.__args__:
            poly_basis = PolynomialBasis(basis=basis)
            assert poly_basis.basis == basis

        # Test invalid basis
        with pytest.raises(AssertionError):
            PolynomialBasis(basis="invalid_basis")

    def test_generate_multi_indices_combinations(self):
        """Test multi-indices combinations generation"""
        # Create a mock backend
        backend = Mock()
        backend.to_array_like.return_value = np.array([[0, 0], [1, 0], [0, 1]])

        poly_basis = PolynomialBasis(basis="chebyshev")
        poly_basis.backend = backend

        # Test with simple case
        result = poly_basis.generate_multi_indices_combinations(1, 2)
        assert result is not None
        assert result.shape == (3, 2)

    def test_vandermonde_1d(self):
        """Test 1D Vandermonde matrix generation"""
        # Create a mock backend
        backend = Mock()
        backend.arange.return_value = np.arange(3)
        backend.pow.return_value = np.array([[1, 1, 1], [1, 2, 4], [1, 3, 9]])
        backend.zeros.return_value = np.zeros((3, 3))
        backend.clip.return_value = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        poly_basis = PolynomialBasis(basis="chebyshev")
        poly_basis.backend = backend

        # Test with simple input
        X = np.array([[0.0], [1.0], [2.0]])
        result = poly_basis.vandermonde_1d(X, 2, 1.0)
        assert result is not None

    def test_vandermonde_nd(self):
        """Test ND Vandermonde matrix generation"""
        # Create a mock backend
        backend = Mock()
        backend.stack.return_value = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
        backend.reshape.return_value = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
        backend.ones.return_value = np.ones((2, 3))
        backend.prod.return_value = np.array([[1, 1], [1, 1]])
        backend.pow.return_value = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])

        poly_basis = PolynomialBasis(basis="chebyshev")
        poly_basis.backend = backend

        # Test with simple input
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = poly_basis.vandermonde_nd(X, 1)
        assert result is not None

    def test_make_v(self):
        """Test make_v method"""
        # Create a mock backend
        backend = Mock()
        backend.zeros.return_value = np.zeros(3)
        backend.ones.return_value = np.ones(3)

        poly_basis = PolynomialBasis(basis="chebyshev")
        poly_basis.backend = backend

        # Test with simple input
        result = poly_basis.make_v(2, int)
        assert result is not None
        assert len(result) == 3
