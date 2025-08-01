"""
Polynomials basis classes and generator for polynomial combinations.
"""

from collections.abc import Callable
from math import comb, factorial
from multiprocessing import Pool

import numpy as np

from cristal.helper_classes.base import BasePolynomialBasis

__all__ = [
    "IMPLEMENTED_POLYNOMIAL_BASIS",
    "MonomialsBasis",
    "ChebyshevT1Basis",
    "ChebyshevT2Basis",
    "ChebyshevUBasis",
    "LegendreBasis",
    "PolynomialsBasisGenerator",
]


class MonomialsBasis(BasePolynomialBasis):
    """
    Class for generating monomials as a polynomial basis.
    1, x, x^2, ..., x^n
    """

    @staticmethod
    def func(x: np.ndarray, monomials_matrix: np.ndarray) -> np.ndarray:
        """Compute the monomials of the input data.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        monomials_matrix : np.ndarray
            The monomials matrix. Should be of shape (s(n), d) where s(n) is the number of monomials and d is the dimension of the input data.

        Returns
        -------
        np.ndarray
            The computed monomials.
        """
        x_repeated = np.tile(x, (monomials_matrix.shape[0], 1))
        return np.power(x_repeated, monomials_matrix)


class ChebyshevT1Basis(BasePolynomialBasis):
    """
    Class for generating first order Chebyshev polynomials as a polynomial basis.
    Orthonormal on [-1, 1] with respect to the Lebesgue measure with 1 / sqrt(1 - x**2) as weight.
    1, x, 2x^2 - 1, ..., T_n(x) = cos(n * arccos(x))
    """

    @staticmethod
    def func(x: np.ndarray, monomials_matrix: np.ndarray) -> np.ndarray:
        """Compute the first order Chebyshev polynomials.
        Orthonormal on [-1, 1] with respect to the Lebesgue measure with 1 / sqrt(1 - x**2) as weight.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        monomials_matrix : np.ndarray
            The matrix of monomials. Should be of shape (s(n), d) where s(n) is the number of monomials and d is the dimension of the input data.

        Returns
        -------
        np.ndarray
            The computed Chebyshev T_1 polynomials.
        """
        x_repeated = np.tile(x, (monomials_matrix.shape[0], 1))

        # Mask for each condition on x
        mask_1 = x_repeated < -1
        mask_2 = x_repeated > 1
        mask_3 = (x_repeated >= -1) & (x_repeated <= 1)

        # Initialize the result array
        result = np.zeros_like(monomials_matrix, dtype=float)

        # Compute for each condition
        # Case x < -1
        result[mask_1] = (
            (-1) ** monomials_matrix[mask_1]
            * np.cosh(monomials_matrix[mask_1] * np.arccosh(-x_repeated[mask_1]))
            / np.where(monomials_matrix[mask_1] == 0, np.sqrt(np.pi), np.sqrt(np.pi / 2))
        )

        # Case x > 1
        result[mask_2] = np.cosh(monomials_matrix[mask_2] * np.arccosh(x_repeated[mask_2])) / np.where(
            monomials_matrix[mask_2] == 0, np.sqrt(np.pi), np.sqrt(np.pi / 2)
        )

        # Case -1 <= x <= 1
        result[mask_3] = np.cos(monomials_matrix[mask_3] * np.arccos(x_repeated[mask_3])) / np.where(
            monomials_matrix[mask_3] == 0, np.sqrt(np.pi), np.sqrt(np.pi / 2)
        )

        return result


# TODO : Make it work with vectorized input
class ChebyshevT2Basis(BasePolynomialBasis):
    """
    Class for generating second order Chebyshev polynomials as a polynomial basis.
    Orthonormal on [-1, 1] with respect to the Lebesgue measure with 1 / sqrt(1 - x**2) as weight.
    1, 2x, 4x^2 - 1, ..., U_n(x) = sin((n + 1) * arccos(x)) / sin(arccos(x))
    U_n(x) = sum_{k=0}^{floor(n/2)} (-1)^k * C(n-k, k) * (2x)^{n-2k}
    """

    @staticmethod
    def func(x: np.ndarray, monomials_matrix: np.ndarray) -> np.ndarray:
        """Compute the Chebyshev T_2 polynomials.
        Orthonormal on [-1, 1] with respect to the Lebesgue measure with 1 / sqrt(1 - x**2) as weight.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        monomials_matrix : np.ndarray
            The matrix of monomials. Should be of shape (s(n), d) where s(n) is the number of monomials and d is the dimension of the input data.

        Returns
        -------
        np.ndarray
            The computed Chebyshev T_2 polynomials.
        """
        f = np.vectorize(ChebyshevT2Basis.chebyshev_t_2)
        return f(x, monomials_matrix)

    @staticmethod
    def chebyshev_t_2(x: np.ndarray, n: int) -> int:
        """Compute the Chebyshev T_2 polynomial.
        Orthonormal on [-1, 1] with respect to the Lebesgue measure with 1 / sqrt(1 - x**2) as weight.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        n : int
            The degree of the polynomial.

        Returns
        -------
        int
            The computed Chebyshev T_2 polynomial.
        """
        if n == 0:
            return 1 / np.sqrt(np.pi)
        else:
            return (n / np.sqrt(np.pi / 2)) * np.sum(
                [(-2) ** i * (factorial(n + i - 1) / (factorial(n - i) * factorial(2 * i))) * (1 - x) ** i for i in range(n + 1)]
            )


# TODO Make it work with vectorized input
class ChebyshevUBasis(BasePolynomialBasis):
    """
    Class for generating Chebyshev U polynomials as a polynomial basis.
    Orthonormal on [-1, 1] with respect to the Lebesgue measure with 1 / sqrt(1 - x**2) as weight.
    1, 2x, 4x^2 - 1, ..., U_n(x) = sin((n + 1) * arccos(x)) / sin(arccos(x))
    U_n(x) = sum_{k=0}^{floor(n/2)} (-1)^k * C(n-k, k) * (2x)^{n-2k}
    """

    @staticmethod
    def func(x: np.ndarray, monomials_matrix: np.ndarray) -> np.ndarray:
        """Compute the Chebyshev U polynomials.
        Orthonormal on [-1, 1] with respect to the Lebesgue measure with 1 / sqrt(1 - x**2) as weight.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        monomials_matrix : np.ndarray
            The matrix of monomials. Should be of shape (s(n), d) where s(n) is the number of monomials and d is the dimension of the input data.

        Returns
        -------
        np.ndarray
            The computed Chebyshev U polynomials.
        """
        f = np.vectorize(ChebyshevUBasis.chebyshev_u)
        return f(x, monomials_matrix)

    @staticmethod
    def chebyshev_u(x: np.ndarray, n: int) -> int:
        """
        Compute the Chebyshev U_n polynomial.
        Orthonormal on [-1, 1] with respect to the Lebesgue measure with 1 / sqrt(1 - x**2) as weight.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        n : int
            The degree of the polynomial.

        Returns
        -------
        int
            The computed Chebyshev U_n polynomial.
        """
        if n == 0:
            return np.sqrt(2 / np.pi)
        else:
            return np.sqrt(2 / np.pi) * np.sum(
                [(-2) ** i * (factorial(n + i - 1) / (factorial(n - i) * factorial(2 * i + 1))) * (1 - x) ** i for i in range(n + 1)]
            )


# TODO Make it work with vectorized input
class LegendreBasis(BasePolynomialBasis):
    """
    Class for generating Legendre polynomials as a polynomial basis.
    Orthonormal on [-1, 1] with respect to the Lebesgue measure.
    1, x, (3x^2 - 1) / 2, ..., P_n(x) = (1 / 2^n) * sum_{k=0}^{floor(n/2)} C(n, k) * (2x)^{n-2k}
    """

    @staticmethod
    def func(x: np.ndarray, monomials_matrix: np.ndarray) -> np.ndarray:
        """Compute the Legendre polynomials.
        Orthonormal on [-1, 1] with respect to the Lebesgue measure.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        monomials_matrix : np.ndarray
            The matrix of monomials. Should be of shape (s(n), d) where s(n) is the number of monomials and d is the dimension of the input data.

        Returns
        -------
        np.ndarray
            The computed Legendre polynomials.
        """
        f = np.vectorize(LegendreBasis.legendre)
        return f(x, monomials_matrix)

    @staticmethod
    def legendre(x: np.ndarray, n: int) -> int:
        """Compute the Legendre polynomial.
        Orthonormal on [-1, 1] with respect to the Lebesgue measure.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        n : int
            The degree of the polynomial.

        Returns
        -------
        int
            The computed Legendre polynomial.
        """
        return np.sqrt((2 * n + 1) / 2) * np.sum([comb(n, i) * comb(n + i, i) * ((x - 1) / 2) ** i for i in range(n + 1)])


IMPLEMENTED_POLYNOMIAL_BASIS: dict[str, type[BasePolynomialBasis]] = {
    "monomials": MonomialsBasis,
    "chebyshev_t_1": ChebyshevT1Basis,
    "chebyshev_t_2": ChebyshevT2Basis,
    "chebyshev_u": ChebyshevUBasis,
    "legendre": LegendreBasis,
}


class PolynomialsBasisGenerator:
    """Class for generating polynomial combinations and applying polynomial basis functions.

    Methods
    -------
    generate_combinations(max_degree, dimensions)
        Generate all combinations of monomials of a given degree and dimensions.
    apply_combinations(x, m, basis_func)
        Applies the polynomial basis to the input data.
    make_design_matrix(x, monomials_matrix, basis_func, allow_parallelization=False)
        Compute the design matrix for the given data points and monomials.
    """

    @staticmethod
    def generate_combinations(max_degree: int, dimensions: int) -> np.ndarray:
        """Generate all combinations of monomials of a given degree and dimensions.

        Parameters
        ----------
        max_degree : int
            The maximum degree of the monomials.
        dimensions : int
            The number of dimensions.

        Returns
        -------
        np.ndarray
            An array of shape (s(n), d) containing the combinations of monomials,
            where s(n) is the number of monomials and d is the number of dimensions.
        """

        def helper(remaining_dimensions, remaining_degree, combination):
            if remaining_dimensions == 0:
                combinations.append(combination)
                return
            for value in range(0, remaining_degree + 1):
                helper(remaining_dimensions - 1, remaining_degree - value, combination + [value])

        combinations = []
        helper(dimensions, max_degree, [])
        return np.asarray(sorted(combinations, key=lambda e: (np.sum(list(e)), list(-1 * np.array(list(e))))), dtype=np.int8)

    @staticmethod
    def apply_combinations(x: np.ndarray, m: np.ndarray, basis_class: type[BasePolynomialBasis]) -> np.ndarray:
        """Applies the polynomial basis to the input data.

        Parameters
        ----------
        x : np.ndarray
            The input data to transform.
        m : np.ndarray
            The monomials to use for the transformation.
        basis_class : type[BasePolynomialBasis]
            The basis class to apply.

        Returns
        -------
        np.ndarray
            The transformed data.
        """
        result = basis_class.func(x, m)
        return np.prod(result, axis=1).reshape(-1, 1)

    @staticmethod
    def make_design_matrix(
        x: np.ndarray, monomials_matrix: np.ndarray, basis_class: type[BasePolynomialBasis], allow_parallelization: bool = False
    ) -> np.ndarray:
        """Compute the design matrix for the given data points and monomials.
        If s(n) > 500 and N > 1000, it will use a parallelized approach to compute the design matrix.

        Parameters
        ----------
        x : np.ndarray
            The input data. Should be of shape (n_samples, n_features).
        monomials_matrix : np.ndarray
            The monomials matrix. Should be of shape (s(n), d).
        basis_class : type[BasePolynomialBasis]
            The basis class to use for the transformation.
        allow_parallelization : bool, optional
            Whether to allow parallelization, by default False

        Returns
        -------
        np.ndarray
            The design matrix for the given data points and monomials.
        """

        def compute_row(xx):
            return PolynomialsBasisGenerator.apply_combinations(xx, monomials_matrix, basis_class)[:, 0]

        if allow_parallelization and x.shape[0] > 1000 and monomials_matrix.shape[0] > 500:
            with Pool() as pool:
                results = pool.map(compute_row, x)
            X = np.array(results)
        else:
            X = np.empty((x.shape[0], monomials_matrix.shape[0]), dtype=x.dtype)
            for idx, xx in enumerate(x):
                X[idx] = compute_row(xx)
        return X
