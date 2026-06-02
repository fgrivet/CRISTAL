"""
Unit tests for PolynomialBasis class
"""

import unittest
from functools import lru_cache

import numpy as np

from cristal.backend.numpy_backend import NumpyBackend
from cristal.commons.polynomial_basis import IMPLEMENTED_POLYNOMIAL_BASIS, PolynomialBasis

# TODO : Test new normalization in chebyshev


class TestPolynomialBasis(unittest.TestCase):
    """Test the PolynomialBasis class functionality"""

    def monomials(self, x, n):
        """Monomials function not optimized but reliable for the tests"""
        return x**n

    @lru_cache(maxsize=None)
    def chebyshev(self, x, n):
        """Chebyshev function not optimized but reliable for the tests"""
        if n == 0:
            return 1
        elif n == 1:
            return x
        return 2 * x * self.chebyshev(x, n - 1) - self.chebyshev(x, n - 2)

    def vandermonde_1d(self, x, n, func):
        """Vandermonde 1d function not optimized but reliable for the tests"""
        M, N = x.shape
        res = np.zeros((M, N, n + 1))
        for i in range(M):
            for j in range(N):
                for k in range(n + 1):
                    res[i, j, k] = func(x[i, j], k)
        return res

    def vandermonde_nd(self, x, n, func, comb):
        """Vandermonde nd function not optimized but reliable for the tests"""
        M, d = x.shape
        res = np.ones((M, len(comb)))
        for i in range(M):
            for k, c in enumerate(comb):
                for j in range(d):
                    # res_{i,k} = prod( T_{c_j}(x_{i,j}) )
                    res[i, k] *= func(x[i, j], c[j])
        return res

    def test_polynomial_basis_initialization(self):
        """Test PolynomialBasis initialization with valid parameters"""
        # Test valid polynomial bases
        for basis in IMPLEMENTED_POLYNOMIAL_BASIS.__args__:
            poly_basis = PolynomialBasis(basis=basis)
            self.assertEqual(poly_basis.basis, basis)
            self.assertTrue(poly_basis.normalize)

            # Test no backend bound
            self.assertRaises(ValueError, poly_basis.generate_multi_indices_combinations, 2, 2)
            self.assertRaises(ValueError, poly_basis.vandermonde_1d, np.zeros((2, 2)), 2, 2)
            self.assertRaises(ValueError, poly_basis.vandermonde_nd, np.zeros((2, 2)), 2)
            self.assertRaises(ValueError, poly_basis.make_v, 2, int)

        poly_basis = PolynomialBasis(normalize=False)
        self.assertFalse(poly_basis.normalize)

        # Test invalid basis
        self.assertRaises(ValueError, PolynomialBasis, basis="invalid_basis")

    def test_generate_multi_indices_combinations(self):
        """Test multi-indices combinations generation"""
        backend = NumpyBackend()

        # Same result for all basis
        for basis in IMPLEMENTED_POLYNOMIAL_BASIS.__args__:
            poly_basis = PolynomialBasis(basis=basis)
            poly_basis.backend = backend

            # Test with a 1D case
            res_1D = poly_basis.generate_multi_indices_combinations(5, 1)
            np.testing.assert_equal(res_1D, np.array([0, 1, 2, 3, 4, 5]).reshape(6, 1), "1D case")

            # Test with a 3D case
            res_3D = poly_basis.generate_multi_indices_combinations(2, 3)
            desired_res_3D = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]])
            np.testing.assert_equal(res_3D, desired_res_3D, "3D case")

            # Test invalid values
            self.assertRaises(ValueError, poly_basis.generate_multi_indices_combinations, 0, 3)
            self.assertRaises(ValueError, poly_basis.generate_multi_indices_combinations, -3, 1)
            self.assertRaises(ValueError, poly_basis.generate_multi_indices_combinations, 5, 0)
            self.assertRaises(ValueError, poly_basis.generate_multi_indices_combinations, 2, -4)

    def test_vandermonde_1d(self):
        """Test 1D Vandermonde matrix generation"""
        backend = NumpyBackend()
        X = np.array([[-1.5, -1], [-0.5, 0], [0.5, 1], [1.5, 0.8]])
        n = 6

        # Test monomials basis
        mon_basis = PolynomialBasis(basis="monomials")
        mon_basis.backend = backend

        # Without normalization
        result_mon = mon_basis.vandermonde_1d(X, n, 5.0, normalize=False)
        desired_result_mon = self.vandermonde_1d(X, n, self.monomials)
        np.testing.assert_equal(result_mon, desired_result_mon, "Monomials basis")

        # With normalization (by default None so True)
        result_mon = mon_basis.vandermonde_1d(X, n, 5.0)  # Same result because normalization is only for Chebyshev basis
        np.testing.assert_equal(result_mon, desired_result_mon, "Monomials basis")

        # Test chebyshev basis
        cheb_basis = PolynomialBasis(basis="chebyshev")
        cheb_basis.backend = backend

        # Without normalization
        result_cheb = cheb_basis.vandermonde_1d(X, n, 5.0, normalize=False)
        desired_result_cheb = self.vandermonde_1d(X, n, self.chebyshev)
        np.testing.assert_almost_equal(result_cheb, desired_result_cheb, 12, "Chebyshev basis")

        # With normalization (by default None so True)
        result_cheb = cheb_basis.vandermonde_1d(X, n, 2.0)
        desired_result_cheb = self.vandermonde_1d(X / 4 - 1, n, self.chebyshev)
        np.testing.assert_almost_equal(result_cheb, desired_result_cheb, 12, "Chebyshev basis normalized")

        # Test invalid values
        # Wrong X shape
        self.assertRaises(ValueError, mon_basis.vandermonde_1d, X.flatten(), n, 5.0, normalize=False)
        self.assertRaises(ValueError, cheb_basis.vandermonde_1d, X.flatten(), n, 5.0, normalize=False)

        # Wrong n values
        self.assertRaises(ValueError, mon_basis.vandermonde_1d, X, -2, 5.0, normalize=False)
        self.assertRaises(ValueError, cheb_basis.vandermonde_1d, X, 0, 5.0, normalize=False)

        # Wrong d values
        self.assertRaises(ValueError, mon_basis.vandermonde_1d, X, n, 0, normalize=False)
        self.assertRaises(ValueError, cheb_basis.vandermonde_1d, X, n, -1, normalize=False)

    def test_vandermonde_nd(self):
        """Test ND Vandermonde matrix generation"""
        backend = NumpyBackend()
        X = np.array([[-1.5, -1], [-0.5, 0], [0.5, 1], [1.5, 0.8]])
        n = 6

        # Test monomials basis
        mon_basis = PolynomialBasis(basis="monomials")
        mon_basis.backend = backend
        comb = mon_basis.generate_multi_indices_combinations(n, X.shape[1])

        result_mon = mon_basis.vandermonde_nd(X, n)
        desired_result_mon = self.vandermonde_nd(X, n, self.monomials, comb)
        np.testing.assert_equal(result_mon, desired_result_mon, "Monomials basis")

        # Test chebyshev basis
        cheb_basis = PolynomialBasis(basis="chebyshev")
        cheb_basis.backend = backend

        result_cheb = cheb_basis.vandermonde_nd(X, n)
        desired_result_cheb = self.vandermonde_nd(X, n, self.chebyshev, comb)
        np.testing.assert_almost_equal(result_cheb, desired_result_cheb, 12, "Chebyshev basis")

        # Test invalid values
        # Wrong X shape
        self.assertRaises(ValueError, mon_basis.vandermonde_nd, X.flatten(), n)
        self.assertRaises(ValueError, cheb_basis.vandermonde_nd, X.flatten(), n)

        # Wrong n values
        self.assertRaises(ValueError, mon_basis.vandermonde_nd, X, -2)
        self.assertRaises(ValueError, cheb_basis.vandermonde_nd, X, 0)

    def test_make_v(self):
        """Test make_v method"""
        backend = NumpyBackend()

        # Test monomials basis
        mon_basis = PolynomialBasis(basis="monomials")
        mon_basis.backend = backend
        # Even number
        np.testing.assert_array_equal(mon_basis.make_v(8, int), np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]), "Monomials basis even n")
        # Odd number
        np.testing.assert_array_equal(mon_basis.make_v(5, int), np.array([1, 0, 0, 0, 0, 0]), "Monomials basis odd n")

        # Test chebyshev basis
        cheb_basis = PolynomialBasis(basis="chebyshev")
        cheb_basis.backend = backend
        # Even number
        np.testing.assert_array_equal(cheb_basis.make_v(8, int), np.array([1, -1, 1, -1, 1, -1, 1, -1, 1], dtype=int), "Chebyshev basis even n")
        # Odd number
        np.testing.assert_array_equal(cheb_basis.make_v(5, int), np.array([1, -1, 1, -1, 1, -1], dtype=int), "Chebyshev basis odd n")

        # Test invalid values
        self.assertRaises(ValueError, mon_basis.make_v, 0, int)
        self.assertRaises(ValueError, cheb_basis.make_v, -2, int)
