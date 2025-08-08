"""
Legendre polynomial family.
"""

from math import comb

import numpy as np

from .base import BasePolynomialFamily


# TODO Make it work with vectorized input
class LegendreFamily(BasePolynomialFamily):
    """
    Class for generating Legendre polynomials as a polynomial basis.
    Orthonormal on [-1, 1] with respect to the Lebesgue measure.

    .. math::

        P_0(x) = 1, ~
        P_1(x) = x, ~
        P_2(x) = (3x^2 - 1) / 2, ~
        \\cdots, ~
        P_n(x) = (1 / 2^n) \\times \\sum_{k=0}^{floor(n/2)} C(n, k) \\times (2x)^{n-2k}
    """

    @staticmethod
    def apply(x: np.ndarray, multidegree_combinations: np.ndarray) -> np.ndarray:
        """Compute the Legendre polynomials.
        Orthonormal on [-1, 1] with respect to the Lebesgue measure.

        Parameters
        ----------
        x : np.ndarray (1, d) or (d,)
            The input data.
        multidegree_combinations : np.ndarray (s(n), d)
            The multidegree combinations. Should be of shape (s(n), d)
            where s(n) is the number of combinations and d is the dimension of the input data.

        Returns
        -------
        np.ndarray (s(n), d)
            The computed Chebyshev T_1 polynomials.

        Example
        -------

        .. code-block:: python

            LegendreFamily.apply(np.array([2, 7]), np.array([[0, 0], [1, 0], [0, 1]]))
            array([[P_0(2), P_0(7)],
                [P_1(2), P_0(7)],
                [P_0(2), P_1(7)]])
        """
        f = np.vectorize(LegendreFamily.legendre)
        return f(x, multidegree_combinations)

    @staticmethod
    def legendre(x: np.ndarray, n: int) -> int:
        """Compute the Legendre polynomial.
        Orthonormal on [-1, 1] with respect to the Lebesgue measure.

        Parameters
        ----------
        x : np.ndarray (1, d) or (d,)
            The input data.
        n : int
            The degree of the polynomial.

        Returns
        -------
        int
            The computed Legendre polynomial P_n(x).
        """
        return np.sqrt((2 * n + 1) / 2) * np.sum([comb(n, i) * comb(n + i, i) * ((x - 1) / 2) ** i for i in range(n + 1)])
