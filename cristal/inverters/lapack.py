"""
Inverters for SPD matrices using LAPACK.
"""

import logging

import numpy as np
from scipy.linalg import lapack

from .base import BaseInverter

logger = logging.getLogger(__name__)


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
        ut : np.ndarray (n, n)
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
        matrix : np.ndarray (n, n)
            The input matrix to invert.

        Returns
        -------
        np.ndarray (n, n)
            The inverse of the input matrix.

        Raises
        ------
        ValueError
            If the input matrix is not positive definite / if the Cholesky decomposition fails.
        """
        cholesky, info = lapack.dpotrf(matrix)  # type: ignore # pylint: disable=no-member
        if info != 0:
            logger.error("Error in dpotrf: %s", info)
            # Matrix is probably not positive definite, we try to make it positive definite
            # by adding a small regularization term from 10^-10 to 10^-4
            for eps in range(10, 3, -1):
                cholesky, info = lapack.dpotrf(matrix + 10**-eps * np.eye(matrix.shape[0]))  # type: ignore # pylint: disable=no-member
                # We stop if the factorization succeeds
                if info == 0:
                    break
                logger.error("Error in dpotrf: %s for eps = %d", info, eps)
            # If none worked, we raise an error
            if info != 0:
                raise ValueError(f"dpotrf failed on input {matrix}")
        inv_matrix, info = lapack.dpotri(cholesky)  # type: ignore # pylint: disable=no-member
        if info != 0:
            raise ValueError(f"dpotri failed on input {cholesky}")
        FPDInverter.upper_triangular_to_symmetric(inv_matrix)
        return inv_matrix
