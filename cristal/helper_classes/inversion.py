"""
Inversion methods for matrices.
"""

import numpy as np
from scipy.linalg import inv, lapack, pinv, solve

from cristal.helper_classes.base import BaseInverter

__all__ = [
    "IMPLEMENTED_INVERSION_OPTIONS",
    "InvInverter",
    "PseudoInverter",
    "PDInverter",
    "FPDInverter",
]


class InvInverter(BaseInverter):
    """
    Inverter for matrices using the standard inverse method from SciPy.
    """

    @staticmethod
    def invert(matrix: np.ndarray) -> np.ndarray:
        """Compute the inverse of a matrix.

        Parameters
        ----------
        matrix : np.ndarray
            The input matrix to invert.

        Returns
        -------
        np.ndarray
            The inverse of the input matrix.
        """
        return inv(matrix)


class PseudoInverter(BaseInverter):
    """
    Inverter for matrices using the Moore-Penrose pseudo-inverse method from SciPy.
    """

    @staticmethod
    def invert(matrix: np.ndarray) -> np.ndarray:
        """Compute the pseudo-inverse of a matrix.

        Parameters
        ----------
        matrix : np.ndarray
            The input matrix to invert.

        Returns
        -------
        np.ndarray
            The pseudo-inverse of the input matrix.
        """
        return pinv(matrix)  # type: ignore


class PDInverter(BaseInverter):
    """
    Inverter for positive definite matrices.
    See https://stackoverflow.com/a/40709871 for more details.
    """

    @staticmethod
    def invert(matrix: np.ndarray) -> np.ndarray:
        """Compute the inverse of a positive definite matrix. According to: https://stackoverflow.com/a/40709871

        Parameters
        ----------
        matrix : np.ndarray
            The input matrix to invert.

        Returns
        -------
        np.ndarray
            The inverse of the input matrix.
        """
        n = matrix.shape[0]
        I = np.identity(n)
        return solve(matrix, I, assume_a="pos", overwrite_b=True)


class FPDInverter(BaseInverter):
    """
    Inverter for positive definite matrices using Cholesky decomposition in LAPACK.
    See https://stackoverflow.com/a/58719188 for more details.
    """

    inds_cache = {}

    @staticmethod
    def upper_triangular_to_symmetric(ut: np.ndarray) -> None:
        """Convert an upper triangular matrix to a symmetric matrix.

        Parameters
        ----------
        ut : np.ndarray
            The upper triangular matrix to convert.
        """
        n = ut.shape[0]
        try:
            inds = FPDInverter.inds_cache[n]
        except KeyError:
            inds = np.tri(n, k=-1, dtype=bool)
            FPDInverter.inds_cache[n] = inds
        ut[inds] = ut.T[inds]

    @staticmethod
    def invert(matrix: np.ndarray) -> np.ndarray:
        """Compute the inverse of a positive definite matrix. According to: https://stackoverflow.com/a/58719188

        Parameters
        ----------
        matrix : np.ndarray
            The input matrix to invert.

        Returns
        -------
        np.ndarray
            The inverse of the input matrix.

        Raises
        ------
        ValueError
            If the input matrix is not positive definite / if the Cholesky decomposition fails.
        """
        cholesky, info = lapack.dpotrf(matrix)  # pylint: disable=no-member # type: ignore
        if info != 0:
            print("Error in dpotrf: ", info)
            # Matrix is probably not positive definite, we try to make it positive definite
            # by adding a small regularization term from 10^-10 to 10^-4
            for eps in range(10, 3, -1):
                cholesky, info = lapack.dpotrf(matrix + 10**-eps * np.eye(matrix.shape[0]))  # pylint: disable=no-member # type: ignore
                # We stop if the factorization succeeds
                if info == 0:
                    break
                else:
                    print(f"Error in dpotrf: {info} for eps = {eps}")
            # If none worked, we raise an error
            if info != 0:
                raise ValueError(f"dpotrf failed on input {matrix}")
        inv_matrix, info = lapack.dpotri(cholesky)  # pylint: disable=no-member # type: ignore
        if info != 0:
            raise ValueError(f"dpotri failed on input {cholesky}")
        FPDInverter.upper_triangular_to_symmetric(inv_matrix)
        return inv_matrix


IMPLEMENTED_INVERSION_OPTIONS: dict[str, type[BaseInverter]] = {
    "inv": InvInverter,
    "pinv": PseudoInverter,
    "pd_inv": PDInverter,
    "fpd_inv": FPDInverter,
}
