"""
Monomials polynomial family.
"""

import numpy as np

from .base import BasePolynomialFamily


class MonomialsFamily(BasePolynomialFamily):
    """
    Class for generating monomials as a polynomial basis.

    .. math::

        U_0(x) = 1, ~
        U_1(x) = x, ~
        U_2(x) = x^2, ~
        \\cdots, ~
        U_n(x) = x^n
    """

    @staticmethod
    def apply(x: np.ndarray, multidegree_combinations: np.ndarray) -> np.ndarray:
        """Apply the polynomial basis function to the input data.

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
            The computed Monomials.

        Example
        -------

        .. code-block:: python

            MonomialsFamily.apply(np.array([2, 7]), np.array([[0, 0], [1, 0], [0, 1]]))
            array([[U_0(2), U_0(7)],
                [U_1(2), U_0(7)],
                [U_0(2), U_1(7)]])
        """
        x_repeated = np.tile(x, (multidegree_combinations.shape[0], 1))
        return np.power(x_repeated, multidegree_combinations)
