"""
Inverters for matrices using SciPy's linear algebra module.
"""

import numpy as np
import scipy.linalg as la

from .base import BaseInverter


class InvInverter(BaseInverter):
    """
    Inverter for matrices using the standard inverse method from SciPy.
    """

    @staticmethod
    def invert(matrix: np.ndarray) -> np.ndarray:
        """Compute the inverse of a matrix.

        Parameters
        ----------
        matrix : np.ndarray (n, n)
            The input matrix to invert.

        Returns
        -------
        np.ndarray (n, n)
            The inverse of the input matrix.
        """
        return la.inv(matrix)


class PseudoInverter(BaseInverter):
    """
    Inverter for matrices using the Moore-Penrose pseudo-inverse method from SciPy.
    """

    @staticmethod
    def invert(matrix: np.ndarray) -> np.ndarray:
        """Compute the pseudo-inverse of a matrix.

        Parameters
        ----------
        matrix : np.ndarray (n, n)
            The input matrix to invert.

        Returns
        -------
        np.ndarray (n, n)
            The pseudo-inverse of the input matrix.
        """
        return la.pinv(matrix)  # type: ignore


class PDInverter(BaseInverter):
    """
    Inverter for positive definite matrices. \
    See https://stackoverflow.com/a/40709871 for more details.
    """

    @staticmethod
    def invert(matrix: np.ndarray) -> np.ndarray:
        """Compute the inverse of a positive definite matrix. According to: https://stackoverflow.com/a/40709871

        Parameters
        ----------
        matrix : np.ndarray (n, n)
            The input matrix to invert.

        Returns
        -------
        np.ndarray (n, n)
            The inverse of the input matrix.
        """
        n = matrix.shape[0]
        I = np.identity(n)
        return la.solve(matrix, I, assume_a="pos", overwrite_b=True)
