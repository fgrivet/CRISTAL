"""
Chebyshev polynomial families of the first and second kind.
"""

from math import factorial

import numpy as np

from .base import BasePolynomialFamily


class ChebyshevT1Family(BasePolynomialFamily):
    """
    Class for generating first order Chebyshev polynomials as a polynomial basis.
    Orthonormal on [-1, 1] with respect to the Lebesgue measure with :math:`1 / \\sqrt{1-x^2}` as weight.

    .. math::

        T_0(x) = 1, ~
        T_1(x) = x, ~
        T_2(x) = 2x^2 - 1, ~
        \\cdots, ~
        T_n(x) = cos(n \\times arccos(x))
    """

    @staticmethod
    def apply(x: np.ndarray, multidegree_combinations: np.ndarray) -> np.ndarray:
        """Compute the first order Chebyshev polynomials.
        Orthonormal on [-1, 1] with respect to the Lebesgue measure with :math:`1 / \\sqrt{1-x^2}` as weight.

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

            ChebyshevT1Family.apply(np.array([2, 7]), np.array([[0, 0], [1, 0], [0, 1]]))
            array([[T_0(2), T_0(7)],
                [T_1(2), T_0(7)],
                [T_0(2), T_1(7)]])
        """
        x_repeated = np.tile(x, (multidegree_combinations.shape[0], 1))

        # Mask for each condition on x
        mask_1 = x_repeated < -1
        mask_2 = x_repeated > 1
        mask_3 = (x_repeated >= -1) & (x_repeated <= 1)

        # Initialize the result array
        result = np.zeros_like(multidegree_combinations, dtype=float)

        # Compute for each condition
        # Case x < -1
        result[mask_1] = (
            (-1) ** multidegree_combinations[mask_1]
            * np.cosh(multidegree_combinations[mask_1] * np.arccosh(-x_repeated[mask_1]))
            / np.where(multidegree_combinations[mask_1] == 0, np.sqrt(np.pi), np.sqrt(np.pi / 2))
        )

        # Case x > 1
        result[mask_2] = np.cosh(multidegree_combinations[mask_2] * np.arccosh(x_repeated[mask_2])) / np.where(
            multidegree_combinations[mask_2] == 0, np.sqrt(np.pi), np.sqrt(np.pi / 2)
        )

        # Case -1 <= x <= 1
        result[mask_3] = np.cos(multidegree_combinations[mask_3] * np.arccos(x_repeated[mask_3])) / np.where(
            multidegree_combinations[mask_3] == 0, np.sqrt(np.pi), np.sqrt(np.pi / 2)
        )

        return result


# TODO : Make it work with vectorized input
class ChebyshevT2Family(BasePolynomialFamily):
    """
    Class for generating second order Chebyshev polynomials as a polynomial basis.
    Orthonormal on [-1, 1] with respect to the Lebesgue measure with :math:`1 / \\sqrt{1-x^2}` as weight.

    .. math::

        U_0(x) = 1, ~
        U_1(x) = 2x, ~
        U_2(x) = 4x^2 - 1, ~
        \\cdots, ~
        U_n(x) = sin((n + 1) \\times arccos(x)) / sin(arccos(x))


    Or equivalently:

    .. math::

        U_n(x) = \\sum_{k=0}^{floor(n/2)} (-1)^k \\times C(n-k, k) \\times (2x)^{n-2k}
    """

    @staticmethod
    def apply(x: np.ndarray, multidegree_combinations: np.ndarray) -> np.ndarray:
        """Compute the second order Chebyshev polynomials.
        Orthonormal on [-1, 1] with respect to the Lebesgue measure with :math:`1 / \\sqrt{1-x^2}` as weight.

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
            The computed Chebyshev T_2 polynomials.

        Example
        -------

        .. code-block:: python

            ChebyshevT2Family.apply(np.array([2, 7]), np.array([[0, 0], [1, 0], [0, 1]]))
            array([[U_0(2), U_0(7)],
                [U_1(2), U_0(7)],
                [U_0(2), U_1(7)]])
        """
        f = np.vectorize(ChebyshevT2Family.chebyshev_t_2)
        return f(x, multidegree_combinations)

    @staticmethod
    def chebyshev_t_2(x: np.ndarray, n: int) -> int:
        """Compute the Chebyshev T_2 polynomial.
        Orthonormal on [-1, 1] with respect to the Lebesgue measure with :math:`1 / \\sqrt{1-x^2}` as weight.

        Parameters
        ----------
        x : np.ndarray (1, d) or (d,)
            The input data.
        n : int
            The degree of the polynomial.

        Returns
        -------
        int
            The computed Chebyshev T_2 polynomial U_n(x).
        """
        if n == 0:
            to_return = 1 / np.sqrt(np.pi)
        else:
            to_return = (n / np.sqrt(np.pi / 2)) * np.sum(
                [(-2) ** i * (factorial(n + i - 1) / (factorial(n - i) * factorial(2 * i))) * (1 - x) ** i for i in range(n + 1)]
            )
        return to_return


# TODO Make it work with vectorized input
class ChebyshevUFamily(BasePolynomialFamily):
    """
    Class for generating U order Chebyshev polynomials as a polynomial basis.
    Orthonormal on [-1, 1] with respect to the Lebesgue measure with :math:`1 / \\sqrt{1-x^2}` as weight.

    .. math::

        U_0(x) = 1, ~
        U_1(x) = 2x, ~
        U_2(x) = 4x^2 - 1, ~
        \\cdots, ~
        U_n(x) = sin((n + 1) \\times arccos(x)) / sin(arccos(x))

    Or equivalently:

    .. math::

        U_n(x) = \\sum_{k=0}^{floor(n/2)} (-1)^k \\times C(n-k, k) \\times (2x)^{n-2k}
    """

    @staticmethod
    def apply(x: np.ndarray, multidegree_combinations: np.ndarray) -> np.ndarray:
        """Compute the U order Chebyshev polynomials.
        Orthonormal on [-1, 1] with respect to the Lebesgue measure with :math:`1 / \\sqrt{1-x^2}` as weight.

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
            The computed Chebyshev U polynomials.

        Example
        -------

        .. code-block:: python

            ChebyshevUFamily.apply(np.array([2, 7]), np.array([[0, 0], [1, 0], [0, 1]]))
            array([[U_0(2), U_0(7)],
                [U_1(2), U_0(7)],
                [U_0(2), U_1(7)]])
        """
        f = np.vectorize(ChebyshevUFamily.chebyshev_u)
        return f(x, multidegree_combinations)

    @staticmethod
    def chebyshev_u(x: np.ndarray, n: int) -> int:
        """
        Compute the Chebyshev U_n polynomial.
        Orthonormal on [-1, 1] with respect to the Lebesgue measure with :math:`1 / \\sqrt{1-x^2}` as weight.

        Parameters
        ----------
        x : np.ndarray (1, d) or (d,)
            The input data.
        n : int
            The degree of the polynomial.

        Returns
        -------
        int
            The computed Chebyshev U polynomial U_n(x).
        """
        if n == 0:
            to_return = np.sqrt(2 / np.pi)
        else:
            to_return = np.sqrt(2 / np.pi) * np.sum(
                [(-2) ** i * (factorial(n + i - 1) / (factorial(n - i) * factorial(2 * i + 1))) * (1 - x) ** i for i in range(n + 1)]
            )
        return to_return
