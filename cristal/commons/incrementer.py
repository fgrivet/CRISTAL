"""Contains the :class:`Incrementer <cristal.commons.incrementer.Incrementer>` class used in dynamic detectors."""

from typing import Generic, cast, get_args

from ..backend.base_backend import Backend
from ..types import IMPLEMENTED_INCREMENTERS, ArrayLike, DTypeLike
from .base_commons import BaseCommons
from .inverter import Inverter
from .polynomial_basis import PolynomialBasis


class Incrementer(BaseCommons, Generic[ArrayLike, DTypeLike]):
    """Class to increment the moment matrix :attr:`M` constructed from :attr:`N` points with new data points :attr:`X`.

    Attributes
    ----------
    method : :class:`IMPLEMENTED_INCREMENTERS <cristal.types.IMPLEMENTED_INCREMENTERS>`
        The incrementing method to use.
    backend : :class:`Backend <cristal.backend.base_backend.Backend>`
        The backend to use for the computation.
    inverter: :class:`Inverter <cristal.commons.inverter.Inverter>`
        The inversion class to use.
    polynomial_basis: :class:`PolynomialBasis <cristal.commons.polynomial_basis.PolynomialBasis>`:
        The polynomial basis used to generate the moment matrix.

    Examples
    --------
    >>> incrementer = Incrementer(method="woodbury")
    >>> incrementer.backend = NumpyBackend()
    >>> incrementer.inverter = Inverter()
    >>> incrementer.polynomial_basis = PolynomialBasis()
    >>> incrementer(M, N, X, n) # Increment by the points X the inverse of the moment matrix M computed with N points and a polynomial basis of degree n
    """

    requires = ["backend", "inverter", "polynomial_basis"]

    def __init__(self, method: IMPLEMENTED_INCREMENTERS = "woodbury"):
        """Class constructor.
        Define the incrementing :attr:`method` and bind the :attr:`backend`, :attr:`inverter`, and :attr:`polynomial_basis`.

        Parameters
        ----------
        method : :class:`IMPLEMENTED_INCREMENTERS <cristal.types.IMPLEMENTED_INCREMENTERS>`, optional
            The incrementing method to use, by default :const:`woodbury`.

        Raises
        ------
        ValueError
            If the incrementing :const:`method` is not valid.

        See Also
        --------
        cristal.types.IMPLEMENTED_INCREMENTERS : For more details on how the methods work.
        """
        if method not in get_args(IMPLEMENTED_INCREMENTERS):
            raise ValueError(f"method must be in {IMPLEMENTED_INCREMENTERS}. Got {method}.")
        self.method = method
        """The incrementing metric to use.

        See Also
        --------
        cristal.types.IMPLEMENTED_INCREMENTERS : For more details on how the methods work.
        """

        # Attributes bound in the configuration __init__
        self.backend: Backend[ArrayLike, DTypeLike]  #: The backend to use for the computation.
        self.inverter: Inverter[ArrayLike, DTypeLike]  #: The inversion class to use.
        self.polynomial_basis: PolynomialBasis[ArrayLike, DTypeLike]  #: The polynomial basis used to generate the moment matrix.

    def increment(self, M: ArrayLike, N: int, X: ArrayLike, n: int) -> tuple[ArrayLike, int, ArrayLike] | tuple[int, ArrayLike]:
        """Increment the moment matrix :attr:`M` constructed from :attr:`N` points with new data points :attr:`X`.

        Parameters
        ----------
        M : ArrayLike
            The moment matrix if :attr:`method` is :const:`inverse`, else the inverse of the moment matrix.
        N : int
            The number of points used to construct the moment matrix :attr:`M`.
        X : ArrayLike
            The new data points of shape (N_samples, d).
        n : int
            The degree of the polynomial basis.

        Returns
        -------
        tuple[ArrayLike, int, ArrayLike] | tuple[int, ArrayLike]
            The updated moment matrix (only if :attr:`method` is :const:`inverse`).

            The new number of data in the moment matrix :math:`N + N_{samples}`.

            The updated inverse moment matrix.

        Examples
        --------
        >>> incrementer = Incrementer(method="woodbury")
        >>> incrementer.backend = NumpyBackend()
        >>> incrementer.inverter = Inverter()
        >>> incrementer.polynomial_basis = PolynomialBasis()
        >>> incrementer(M, N, X, n) # Increment by the points X the inverse of the moment matrix M computed with N points and a polynomial basis of degree n
        """

        if self.backend is None:
            raise ValueError("A backend must be bound to the Incrementer class before using it.")
        if self.inverter is None:
            raise ValueError("An inverter must be bound to the Incrementer class before using it.")
        if self.polynomial_basis is None:
            raise ValueError("A polynomial basis must be bound to the Incrementer class before using it.")

        # Create a copy of the original matrices to avoid side effects
        M = self.backend.copy(M)
        X = self.backend.copy(X)

        N_prime: int = X.shape[0]
        new_N = N_prime + N

        # nd because increment is impossible for the univariate version
        V = self.polynomial_basis.vandermonde_nd(X, n)

        # Inverse
        if self.method == "inverse":
            # Revert the mean of the moment matrix
            M *= N
            # Add new data
            M += V.T @ V
            # Recompute the mean of the moment matrix
            M /= new_N
            # Inverse the new moment matrix
            M_inv = self.inverter(M)
            return M, new_N, M_inv

        # Revert the mean of the inverse moment matrix
        M /= N

        # Sherman
        if self.method == "sherman":
            # Add point by point using Sherman-Morrison formula
            for row in V:
                v = cast(ArrayLike, row).reshape(-1, 1)
                # Compute the left-hand side of the Sherman-Morrison formula: M^-1 u
                left = M @ v
                # Compute the denominator: v^T M^-1 u
                denom = v.T @ left
                # Divide the left-hand side by (1 + denom)
                # Reduce the division cost from O(N^2) to O(N) by using the fact that left is a vector and numerator is a matrix
                left_div = left / (1 + denom)
                M -= left_div @ left.T
            # Compute the mean of the updated inverse moment matrix
            return new_N, M * new_N

        # Woodbury
        # Define the identity matrix C
        C = self.backend.eye(N_prime)
        # Compute the product V @ M^-1
        V_M_inv = V @ M
        # Compute the sum C^-1 + V @ M^-1 @ U
        sum_ = C + V_M_inv @ V.T
        # Compute the inverse of the sum
        if N_prime == 1:
            sum_inv = 1 / sum_
        else:
            sum_inv = self.inverter(sum_)
        # Compute M^-1 @ U @ (C^-1 + V @ M^-1 @ U)^-1
        # M is symmetric and U = V^T so M^-1 @ U = (V @ M^-1)^T
        M_inv_U_sum_inv = V_M_inv.T @ sum_inv
        # Compute the product M^-1 @ U @ (C^-1 + V @ M^-1 @ U)^-1 @ V^T @ M^-1
        prod = M_inv_U_sum_inv @ V_M_inv
        M -= prod
        # Compute the mean of the updated inverse moment matrix
        return new_N, M * new_N

    def __call__(self, M: ArrayLike, N: int, X: ArrayLike, n: int) -> tuple[ArrayLike, int, ArrayLike] | tuple[int, ArrayLike]:
        """Increment the moment matrix :attr:`M` constructed from :attr:`N` points with new data points :attr:`X`.

        .. hint::

            This function is a wrapper for :func:`increment`.

        Parameters
        ----------
        M : ArrayLike
            The moment matrix if :attr:`method` is :const:`inverse`, else the inverse of the moment matrix.
        N : int
            The number of points used to construct the moment matrix :attr:`M`.
        X : ArrayLike
            The new data points of shape (N_samples, d).
        n : int
            The degree of the polynomial basis.

        Returns
        -------
        tuple[ArrayLike, int, ArrayLike] | tuple[int, ArrayLike]
            The updated moment matrix (only if :attr:`method` is :const:`inverse`).

            The new number of data in the moment matrix :math:`N + N_{samples}`.

            The updated inverse moment matrix.
        """
        return self.increment(M=M, N=N, X=X, n=n)
